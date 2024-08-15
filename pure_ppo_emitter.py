from dataclasses import dataclass
from qdax.core.emitters.emitter import Emitter
from functools import partial
import jax
from qdax.core.neuroevolution.networks.networks import MLPPPO
import optax
#from flax.training.train_state import TrainState
import jax.numpy as jnp
from wrappers_qd import VecEnv, NormalizeVecRewward
from qdax.core.neuroevolution.buffers.buffer import PPOTransition
from get_env import get_env
from qdax.tasks.brax_envs import reset_based_scoring_function_brax_envs as scoring_function
from qdax.environments import behavior_descriptor_extractor
from typing import Any
from qdax.core.emitters.emitter import EmitterState


@dataclass
class PurePPOConfig:
    NO_AGENTS: int = 1
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
    #ACTIVATION: str = 'tanh'
    ANNEAL_LR: bool = False
    NORMALIZE_ENV: bool = True
    NO_ADD: int = 1
    #NO_NEURONS: int = 64
    
    
class PurePPOEmitterState(EmitterState):
    rng: Any

class PurePPOEmitter():
    def __init__(self, config: PurePPOConfig, policy_net, env, scoring_function):
        env = VecEnv(env)
        env = NormalizeVecRewward(env, config.GAMMA)
        
        self._config = config
        self._env = env
        #self._actor_critic = MLPPPO(
        #     action_dim=self._env.action_size,
        #     activation=self._config.ACTIVATION,
        #     no_neurons=self._config.NO_NEURONS,
        # )
        self._actor_critic = policy_net
        self._scoring_function = scoring_function
        '''
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
        
        self._train_state = train_state
        '''
        #rng, _rng = jax.random.split(rng)
        #init_x = jnp.zeros(self._env.observation_size)
        #params = self._actor_critic.init(_rng, init_x)
        self._tx = optax.chain(
            optax.clip_by_global_norm(self._config.MAX_GRAD_NORM),
            optax.adam(self._config.LR, eps=1e-5)
        )
        
        
    @property
    def batch_size(self) -> int:
        """
        Returns:
            int: the batch size emitted by the emitter.
        """
        return self._config.NO_AGENTS
    
    @property
    def use_all_data(self) -> bool:
        """Whther to use all data or not when used along other emitters.
        """
        return False
    
        
    @partial(jax.jit, static_argnames=("self",))
    def init(self, random_key, repertoire, genotypes, fitnesses, descriptors, extra_scores):
        rng, _rng = jax.random.split(random_key)
        emitter_state = PurePPOEmitterState(rng=_rng)
        
        return emitter_state, rng
    
    @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        emitter_state: PurePPOEmitterState,
        repertoire,
        genotypes,
        fitnesses,
        descriptors,
        extra_scores,
    ):
        
        return emitter_state
        
    @partial(jax.jit, static_argnames=("self",))
    def emit(self, repertoire, emitter_state, rng):
    
        #rng, _rng = jax.random.split(rng)
        #init_x = jnp.zeros(self._env.observation_size)
        #params = self._actor_critic.init(_rng, init_x)
        
        parent, rng = repertoire.sample(
            rng,
            num_samples=self.batch_size,
        )
        
        params = jax.tree_util.tree_map(lambda x: x[0, ...], parent)

        opt_state = self._tx.init(params)
        #train_state = TrainState.create(
        #    apply_fn=self._actor_critic.apply,
        #    params=self._params,
        #    tx=self._tx,
        #)
        
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, num=self._config.NUM_ENVS)
        state = self._env.reset(reset_rng)
        
        def _scan_update_step(carry, _):
            return self._update_step(*carry)
        
        (state, params, opt_state, repertoire, rng), _ = jax.lax.scan(
            _scan_update_step,
            (state, params, opt_state, repertoire, rng),
            None,
            length=self._config.NO_ADD,
        )
        
        params = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], params)
        return params, {}, rng
     
    @partial(jax.jit, static_argnames=("self",))
    def _env_step(self, state, params, key):
        rng, rng_ = jax.random.split(key)
        pi, _, value = self._actor_critic.apply(params, state.env_state.obs)
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
        
        return (next_state, params, rng), transition
    
    @partial(jax.jit, static_argnames=("self",))
    def _sample(self, state, params, rng):
        
        def _scan_env_step(carry, _):
            return self._env_step(*carry)
        
        (state, params, rng), transitions = jax.lax.scan(
            _scan_env_step,
            (state, params, rng),
            None,
            length=self._config.NUM_STEPS,
        )
        
        return state, params, rng, transitions
    
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
    def _loss_fn(self, params, traj_batch, gae, targets):
        pi, _, value = self._actor_critic.apply(params, traj_batch.obs)
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
    def _update_minibatch(self, params, opt_state, batch_info):
        traj_batch, advs, targets = batch_info
        
        grad_fn = jax.value_and_grad(self._loss_fn, has_aux=True)
        total_loss, grad = grad_fn(
            params,
            traj_batch,
            advs,
            targets,
        )
        updates, opt_state = self._tx.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        #train_state = train_state.apply_gradients(grad)
        return (params, opt_state), total_loss
    
    
    @partial(jax.jit, static_argnames=("self",))
    def _update_epoch(self, update_state):
        params, opt_state, traj_batch, advs, targets, rng = update_state
        rng, _rng = jax.random.split(rng)
        batch_size = (self._config.NUM_ENVS * self._config.NUM_STEPS // self._config.NUM_MINIBATCHES) * self._config.NUM_MINIBATCHES
        
        assert (
            batch_size == self._config.NUM_ENVS * self._config.NUM_STEPS
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
            return self._update_minibatch(*carry, x)     
          
        (params, opt_state), losses = jax.lax.scan(
            _scan_update_minibatch,
            (params, opt_state),
            minibatches,
        )
        
        update_state = (params, opt_state, traj_batch, advs, targets, rng)
        
        return update_state, losses
        
        
        
    @partial(jax.jit, static_argnames=("self",))
    def _one_update(self, state, params, opt_state, rng):
        
        state, params, rng, traj_batch = self._sample(state, params, rng)
        _, _, last_val = self._actor_critic.apply(params, state.env_state.obs)
        advs, targets = self._calculate_gae(traj_batch, last_val)
        update_state = (params, opt_state, traj_batch, advs, targets, rng)
        
        def _scan_update_epoch(carry, _):
            return self._update_epoch(carry)
        
        (params, opt_state, traj_batch, advs, targets, rng), losses = jax.lax.scan(
            _scan_update_epoch,
            update_state,
            None,
            length=self._config.UPDATE_EPOCHS,
        )
        
        return (state, params, opt_state, rng), losses
    
    @partial(jax.jit, static_argnames=("self",))
    def _update_step(self, state, params, opt_state, repertoire, rng):
        num_updates = self._config.TOTAL_TIMESTEPS // self._config.NUM_ENVS // self._config.NUM_STEPS
        
        def _scan_one_update(carry, _):
            return self._one_update(*carry)
        
        (state, params, opt_state, rng), losses = jax.lax.scan(
            _scan_one_update,
            (state, params, opt_state, rng),
            None,
            length=num_updates // self._config.NO_ADD,
        )
        
        params = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], params)
        fitnesses, descriptors, extra_scores, rng = self._scoring_function(params, rng)
        
        repertoire = repertoire.add(params, descriptors, fitnesses, extra_scores)
        params = jax.tree_util.tree_map(lambda x: x[0, ...], params)
        
        return (state, params, opt_state, repertoire, rng), losses
        
        
        
if __name__ == "__main__":
    config = PurePPOConfig(
        LR=1e-3,
        NUM_ENVS=2048,
        NUM_STEPS=10,
        TOTAL_TIMESTEPS=5e7,
        UPDATE_EPOCHS=4,
        NUM_MINIBATCHES=32,
        GAMMA=0.99,
        GAE_LAMBDA=0.95,
        CLIP_EPS=0.2,
        ENTROPY_COEFF=0.0,
        VF_COEFF=0.5,
        MAX_GRAD_NORM=0.5,
        ACTIVATION="tanh",
        ANNEAL_LR=False,
        NORMALIZE_ENV=True,
        NO_ADD=10,
    )
    env = get_env("ant_uni")
    emitter = PurePPOEmitter(config, env)
    rng = jax.random.PRNGKey(5)
    params = emitter.emit(rng)
    params = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], params)
    
    
    
    reset_fn = jax.jit(env.reset)
    
    
    @jax.jit
    def play_step_fn(env_state, params, key):
        rng, rng_ = jax.random.split(key)
        pi, action, val = emitter._actor_critic.apply(params, env_state.obs)
        action_ = pi.sample(seed=rng_)
        log_prob = pi.log_prob(action_)
        
        #rng, rng_ = jax.random.split(rng)
        #rng_step = jax.random.split(rng_, num=self._config.NUM_ENVS)
        next_env_state = env.step(env_state, action)
        transition = PPOTransition(
            obs=env_state.obs,
            next_obs=next_env_state.obs,
            rewards=next_env_state.reward,
            dones=next_env_state.done,
            truncations=next_env_state.info["truncation"],
            actions=action,
            state_desc=env_state.info["state_descriptor"],
            next_state_desc=next_env_state.info["state_descriptor"],
            val=val,
            logp=log_prob,
        )
        
        return (next_env_state, params, rng), transition
        
    bd_extraction_fn = behavior_descriptor_extractor['ant_uni']
    socring_fn = partial(
        scoring_function,
        episode_length=1000,
        play_reset_fn=reset_fn,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )
    
    rng = jax.random.PRNGKey(1000)
    fitnesses, _, _, _ = socring_fn(params, rng)
    
    print(fitnesses)
    