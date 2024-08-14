from dataclasses import dataclass
from qdax.core.emitters.emitter import Emitter
from functools import partial
import jax
from qdax.core.neuroevolution.networks import MLPPPO
import optax
from flax.training.train_state import TrainState
import jax.numpy as jnp
from wrappers_qd import VecEnv, NormalizeVecRewward
from qdax.core.neuroevolution.buffers.buffer import PPOTransition



@dataclass
class PurePPOConfig:
    LR: float = 1e-3
    NUM_ENVS: int = 2048
    NUM_STEPS: int = 10
    TOTAL_TIMESTEPS: int = 5e7
    UPDATE_EPOCHS: int = 4
    NUM_MINIBATCHES: int = 32
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_EPS: float = 0.2
    ENTROPY_COEFF: float = 0.0
    VF_COEFF: float = 0.5
    MAX_GRAD_NORM: float = 0.5
    ACTIVATION: str = 'tanh'
    ANNEAL_LR: bool = False
    NORMALIZE_ENV: bool = True
    

class PurePPOEmitter(Emitter):
    def __init__(self, config: PurePPOConfig, actor_critic, env, repertoire):
        env = VecEnv(env)
        env = NormalizeVecRewward(env, config.GAMMA)
        
        self._config = config,
        self._env = env
        self._repertoire = repertoire
        self._actor_critic = MLPPPO(
             action_dim=self._env.actio_size,
             activation=self._config.ACTIVATION,
             no_neurons=self._config.no_neurons
         )
        
    @partial(jax.jit, static_argnames=("self",))
    def emit(self, rng):

         num_updates = self._config.TOTAL_TIMESTEPS // self._config.NUM_ENVS // self._config.NUM_STEPS
         rng, _rng = jax.random.split(rng)
         init_x = jnp.zeros(self._env.observation_size)
         params = self._actor_critic.init(_rng, init_x)
         tx = optax.chain(
             optax.clip_by_global_norm(self._config.MAX_GRAD_NORM),
                optax.adam(self._config.LR, eps=1e-5)
         )
         
         train_state = TrainState.create(
                apply_fn=self._actor_critic.apply,
                params=params,
                tx=tx
         )
         
         rng, _rng = jax.random.split(rng)
         
         train_state = jax.lax.scan(
             self._update_step,
             train_state,
             None,
             length=num_updates,
         )
         
         return train_state
     
    @partial(jax.jit, static_argnames=("self",))
    def _env_step(self, state, train_state, key):
        rng, rng_ = jax.random.split(key)
        pi, value = self._actor_critic.apply(train_state.params, state.env_state.obs)
        action = pi.sample(seed=rng_)
        log_prob = pi.log_prob(action)
        
        #rng, rng_ = jax.random.split(rng)
        #rng_step = jax.random.split(rng_, num=self._config.NUM_ENVS)
        next_state = self._env.step(state, action)
        transition = PPOTransition(
            obs=state.env_state.obs,
            next_obs=next_state.env_state.obs,
            rewards=next_state.env_state.reward,
            dones=next_state.env_state.done,
            truncations=next_state.env_state.info["truncation"],
            actions=action,
            state_desc=state.env_state.info["state_descriptor"],
            next_state_desc=next_state.env_state.info["state_descriptor"],
            val=value,
            logp=log_prob,
        )
        
        return (next_state, train_state, rng), transition
    
    @partial(jax.jit, static_argnames=("self",))
    def _sample(self, state, train_state, rng):
        
        def _scan_env_step(carry, _):
            return self._env_step(*carry)
        
        (state, train_state, rng), transitions = jax.lax.scan(
            _scan_env_step,
            (state, train_state, rng),
            None,
            length=self._config.NUM_STEPS,
        )
        
        return state, train_state, rng, transitions
    
    @partial(jax.jit, static_argnames=("self",))
    def _calulate_single_gae(self, reward, value, next_value, done, prev_gae):
        delta = reward + self._config.GAMMA * next_value * (1 - done) - value
        return delta + self._config.GAMMA * self._config.GAE_LAMBDA * (1 - done) * prev_gae
    
    @partial(jax.jit, static_argnames=("self",))
    def _calculate_gae(self, traj_batch, last_val):
        
        def _scan_get_advs(carry, x):
            gae, next_val = carry
            done, value, reward = (
                x.dones,
                x.val,
                x.rewards,
            )
            #delta = reward + self._config.GAMMA * next_val * (1 - done) - value
            #gae = delta + self._config.GAMMA * self._config.GAE_LAMBDA * (1 - done) * gae
            gae = self._calulate_single_gae(reward, value, next_val, done, gae)
            
            return (gae, value), gae
        
        _, advs = jax.lax.scan(
            _scan_get_advs,
            (jnp.zeros_like(last_val), last_val),
            (traj_batch),
            reverse=True,
            unroll=16,
        )
         
        return advs, advs + traj_batch.val
    
    
    @partial(jax.jit, static_argnames=("self",))
    def _loss_fn(self, traj_batch, gae, targets, train_state):
        pi, value = self._actor_critic.apply(train_state.params, traj_batch.obs)
        log_prob = pi.log_prob(traj_batch.actions)
        
        value_pred_clipped = traj_batch.val + (
            value - traj_batch.val
        ).clip(-self._config.CLIP_EPS, self._config.CLIP_EPS)
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.mean(jnp.maximum(value_losses, value_losses_clipped))
        
        ratio = jnp.exp(log_prob - traj_batch.logp)
        gae = (gae - jnp.mean(gae)) / (jnp.std(gae) + 1e-8)
        loss_actor1 = ratio * gae
        loss_actor2 = jnp.clip(ratio, 1 - self._config.CLIP_EPS, 1 + self._config.CLIP_EPS) * gae
        
        loss_actor = -jnp.mean(jnp.minimum(loss_actor1, loss_actor2))
        entropy = jnp.mean(pi.entropy())
        
        return loss_actor + self._config.VF_COEFF * value_loss - self._config.ENTROPY_COEFF * entropy, (value_loss, loss_actor, entropy)
    
    @partial(jax.jit, static_argnames=("self",))
    def _update_minibatch(self, train_state, batch_info):
        traj_batch, advs, targets = batch_info
        
        grad_fn = jax.value_and_grad(self._loss_fn, has_aux=True)
        total_loss, grad = grad_fn(
            train_state.params,
            traj_batch,
            advs,
            targets,
        )
        train_state = train_state.apply_gradients(grad)
        return train_state, total_loss
    
    
    @partial(jax.jit, static_argnames=("self",))
    def _update_epoch(self, update_state):
        train_state, traj_batch, advs, targets, rng = update_state
        rng, _rng = jax.random.split(rng)
        batch_size = (self._config.NUM_ENVS * self._config.NUM_STEPS // self._config.NUM_MINIBATCHES) * self._config.NUM_MINIBATCHES
        
        assert (
            batch_size == self._config["NUM_ENVS"] * self._config["NUM_STEPS"]
        ), "batch size must be equal to number of steps * number of envs"
        permutation = jax.random.permutation(_rng, batch_size)
        batch = (traj_batch, advs, targets)
        batch = jax.tree_util.tree_map(
            lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
        )
        shuffled_batch = jax.tree_util.tree_map(
            lambda x: jnp.take(x, permutation, axis=0), batch
        )
        minibatches = jax.tree_util.tree_map(
            lambda x: jnp.reshape(
                x, [self._config.NUM_MINIBATCHES, -1] + list(x.shape[1:])
            ),
            shuffled_batch,
        )
        
        def _scan_update_minibatch(carry, x):
            return self._update_minibatch(*x)
        
        train_state, losses = jax.lax.scan(
            _scan_update_minibatch,
            train_state,
            minibatches,
        )
        
        update_state = (train_state, traj_batch, advs, targets, rng)
        
        return update_state, losses
        
        
        
    
    