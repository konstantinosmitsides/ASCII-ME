from dataclasses import dataclass
from qdax.core.emitters.emitter import Emitter
from functools import partial
import jax
from qdax.core.neuroevolution.networks import MLPPPO
import optax
from flax.training.train_state import TrainState
import jax.numpy as jnp
from wrappers_qd import VecEnv, NormalizeVecRewward
from qdax.core.neuroevolution.buffers.buffer import QDMCTransition



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
        
    @partial(jax.jit, static_argnames=("self",))
    def emit(self, rng):
         actor_critic = MLPPPO(
             action_dim=self._env.actio_size,
             activation=self._config.ACTIVATION,
             no_neurons=self._config.no_neurons
         )
         num_updates = self._config.TOTAL_TIMESTEPS // self._config.NUM_ENVS // self._config.NUM_STEPS
         rng, _rng = jax.random.split(rng)
         init_x = jnp.zeros(self._env.observation_size)
         params = actor_critic.init(_rng, init_x)
         tx = optax.chain(
             optax.clip_by_global_norm(self._config.MAX_GRAD_NORM),
                optax.adam(self._config.LR, eps=1e-5)
         )
         
         train_state = TrainState.create(
                apply_fn=actor_critic.apply,
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
    def _env_step(self, state, params, key):
        rng, rng_ = jax.random.split(key)
        pi, value = self._actor_critic.apply(params, state.env_state.obs)
        action = pi.sample(seed=rng_)
        log_prob = pi.log_prob(action)
        
        rng, rng_ = jax.random.split(rng)
        rng_step = jax.random.split(rng_, num=self._config.NUM_ENVS)
        next_state = self._env.step(state, action, rng_step)
        
