from brax.v1.envs import env as brax_env
import jax
from flax import struct
from typing import Any
import jax.numpy as jnp


class VecEnv(brax_env.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self.env.resest)
        self.step = jax.vmap(self.env.step)
        
@struct.dataclass
class NormalizeVecRewEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float
    env_state: Any
    

class NormalizeVecRewward(brax_env.Wrapper):
    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma
        
    def reset(self, key):
        state = self.env.reset(key)
        batch_count = state.obs.shape[0]
        state = NormalizeVecRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            env_state=state
        )
        
        return state
    
    def step(self, state, action):
        env_state = self.env.step(state.env_state, action)
        
        return_val = state.return_val * self.gamma * (1 - env_state.done) + env_state.reward
        
        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = state.env_state.obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count
        
        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        
        new_reward = env_state.reward / jnp.sqrt(new_var + 1e-8)
        env_state = env_state.replace(reward=new_reward)
        
        
        return NormalizeVecRewEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            env_state=env_state
        )