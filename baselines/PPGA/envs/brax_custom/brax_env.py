import functools
from typing import Union, Optional, Tuple, List, Any
from typing import ClassVar, Optional


import gym
import brax
#import brax.envs
from brax.v1.envs.env import Env
from brax.v1.envs.to_torch import JaxToTorchWrapper
#from baselines.PPGA.envs import brax_custom
from jax.dlpack import to_dlpack

import torch

v = torch.ones(1, device='cuda')  # init torch cuda before jax

import jax
import jax.numpy as jnp

#from brax.envs.base import PipelineEnv
import gym
from gym import spaces
from gym.vector import utils
import jax
import numpy as np
from google.protobuf import text_format

from qdax.environments_v1.base_wrappers import QDEnv
from qdax.environments_v1 import create
 
from baselines.PPGA.envs.brax_custom.custom_wrappers.clip_wrappers import ObservationClipWrapper, RewardClipWrapper, \
     ActionClipWrapper
# from baselines.qdax.environments import QDEnv, create


_to_custom_env = {
    'ant': {'custom_env_name': 'brax_custom-ant-v0',
            'kwargs': {
                'clip_actions': (-1, 1),
            }},
    'humanoid': {'custom_env_name': 'brax_custom-humanoid-v0',
                 'kwargs': {
                     'clip_actions': (-1, 1),
                 }},
    'walker2d': {'custom_env_name': 'brax-custom-walker2d-v0',
                 'kwargs': {
                     'clip_actions': (-1, 1),
                 }},
    'halfcheetah': {'custom_env_name': 'brax-custom-halfcheetah-v0',
                    'kwargs': {
                        'clip_actions': (-1, 1),
                    }},
}

# class JaxToTorchWrapper(gym.Wrapper):
#   """Wrapper that converts Jax tensors to PyTorch tensors."""

#   def __init__(self,
#                env,
#                device: Optional[torch.device] = None):
#     """Creates a Wrapper around a `GymWrapper` or `VectorGymWrapper` that outputs PyTorch tensors."""
#     super().__init__(env)
#     self.device: Optional[torch.device] = device

#   def observation(self, observation):
#     return jax_to_torch(observation, device=self.device)

#   def action(self, action):
#     return torch_to_jax(action)

#   def reward(self, reward):
#     return jax_to_torch(reward, device=self.device)

#   def done(self, done):
#     return jax_to_torch(done, device=self.device)

#   def info(self, info):
#     return jax_to_torch(info, device=self.device)

#   def reset(self):
#     obs = super().reset()
#     return self.observation(obs)

#   def step(self, action):
#     action = self.action(action)
#     obs, reward, done, info = super().step(action)
#     obs = self.observation(obs)
#     reward = self.reward(reward)
#     done = self.done(done)
#     info = self.info(info)
#     return obs, reward, done, info


class PPGAWrapper(QDEnv):
    def __init__(self, env: QDEnv):
        # Convert config to string format that text_format.Parse can handle
        config_str = text_format.MessageToString(env.sys.config)
        # Initialize with just config
        super().__init__(config=config_str)
        # Set sys after initialization
        self.sys = env.sys
        self.env = env
        self._env_name = env.name
        # Store the original observation size
        self._original_obs_size = env.observation_size

    @property
    def observation_size(self) -> int:
        # Return the original observation size to maintain consistency
        return self._original_obs_size

    @property
    def state_descriptor_length(self) -> int:
        return self.env.state_descriptor_length

    @property
    def state_descriptor_name(self) -> str:
        return self.env.state_descriptor_name

    @property
    def state_descriptor_limits(self) -> Tuple[List, List]:
        return self.env.state_descriptor_limits

    @property
    def behavior_descriptor_length(self) -> int:
        return self.env.behavior_descriptor_length

    @property
    def behavior_descriptor_limits(self) -> Tuple[List, List]:
        return self.env.behavior_descriptor_limits

    @property
    def name(self) -> str:
        return self._env_name

    def reset(self, rng: jnp.ndarray):
        state = self.env.reset(rng)
        if self._env_name in ["ant_omni", "anttrap_omni"]:
            state.info["measures"] = state.info['first_qp'].vel[:2]
        else:
            state.info["measures"] = state.info["state_descriptor"]
        return state

    def step(self, state, action: jnp.ndarray):
        state = self.env.step(state, action)
        if self._env_name in ["ant_omni", "anttrap_omni"]:
            state.info["measures"] = state.info['first_qp'].vel[:2]
        else:
            state.info["measures"] = state.info["state_descriptor"]
        return state

    @property
    def unwrapped(self) -> QDEnv:
        return self.env.unwrapped

    def __getattr__(self, name: str) -> Any:
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)

class VectorGymWrapperCustom(gym.vector.VectorEnv):
  """A wrapper that converts batched Brax Env to one that follows Gym VectorEnv API."""

  # Flag that prevents `gym.register` from misinterpreting the `_step` and
  # `_reset` as signs of a deprecated gym Env API.
  _gym_disable_underscore_compat: ClassVar[bool] = True

  def __init__(self,
               env: Env,
               seed: int = 0,
               backend: Optional[str] = None):
    self._env = env
    self.metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 1 / self._env.sys.config.dt
    }
    if not hasattr(self._env, 'batch_size'):
      raise ValueError('underlying env must be batched')

    self.num_envs = self._env.batch_size
    self.seed(seed)
    self.backend = backend
    self._state = None

    # Observation space
    obs = np.inf * np.ones(self._env.observation_size, dtype='float32')
    # jax.debug.print("OBS: {}", self._env.observation_size)
    self.single_observation_space = spaces.Box(-obs, obs, dtype='float32')
    self.observation_space = utils.batch_space(self.single_observation_space, self.num_envs)

    # Action space - use action_size instead of actuator
    action = np.ones(self._env.action_size, dtype='float32')
    self.single_action_space = spaces.Box(-action, action, dtype='float32')
    self.action_space = utils.batch_space(self.single_action_space, self.num_envs)

    def reset(key):
      key1, key2 = jax.random.split(key)
      state = self._env.reset(key2)
      return state, state.obs, key1

    self._reset = jax.jit(reset, backend=self.backend)

    def step(state, action):
      state = self._env.step(state, action)
      info = {**state.metrics, **state.info}
      return state, state.obs, state.reward, state.done, info

    self._step = jax.jit(step, backend=self.backend)

  def reset(self):
    self._state, obs, self._key = self._reset(self._key)
    return obs

  def step(self, action):
    self._state, obs, reward, done, info = self._step(self._state, action)
    return obs, reward, done, info

  def seed(self, seed: int = 0):
    self._key = jax.random.PRNGKey(seed)

  def __getattr__(self, name: str) -> Any:
    if name == "__setstate__":
        raise AttributeError(name)
    return getattr(self._env, name)


def create_gym_env(env,
                   batch_size: Optional[int] = None,
                   seed: int = 0,
                   backend: Optional[str] = None,
                   **kwargs) -> Union[gym.Env, gym.vector.VectorEnv]:
    """Creates a `gym.Env` or `gym.vector.VectorEnv` from a Brax environment."""
    if batch_size is None:
        raise NotImplementedError('batch_size=None is not supported yet.')
    if batch_size <= 0:
        raise ValueError(
            '`batch_size` should either be None or a positive integer.')
    return VectorGymWrapperCustom(env, seed=seed, backend=backend)

def make_vec_env_brax_ppga(task_name, batch_size, backend, episode_length, clip_obs_rew: bool, seed, **kwargs):
    env_name = task_name
    
    if env_name == "walker2d_uni":
       vec_env = create(
          env_name=env_name,
          batch_size=batch_size,
          episode_length=episode_length,
       )

    elif env_name == "hopper_uni":
        vec_env = create(
          env_name=env_name,
          batch_size=batch_size,
          episode_length=episode_length,
       )

    elif env_name == "ant_uni":
        vec_env = create(
          env_name=env_name,
          batch_size=batch_size,
          episode_length=episode_length,
          use_contact_forces=False,
          exclude_current_positions_from_observation=True
       )
        
    elif env_name == "ant_omni":
        vec_env = create(
          env_name=env_name,
          batch_size=batch_size,
          episode_length=episode_length,
          use_contact_forces=False,
          exclude_current_positions_from_observation=False,
       )
    elif env_name == "anttrap_omni":
        vec_env = create(
          env_name=env_name,
          batch_size=batch_size,
          episode_length=episode_length,
          use_contact_forces=False,
          exclude_current_positions_from_observation=False,
       )
        

    else:
        raise ValueError("Invalid environment name.")
    
    # jax.debug.print("OBS: {}", vec_env.observation_size)
    

    if clip_obs_rew:  # TODO: is this done in practice?
        clip_obs = (-10, 10)
        clip_rewards = (-10, 10)
    else:
        clip_obs = None
        clip_rewards = None

    clip_actions = (-1, 1)

    if clip_obs:
        vec_env = ObservationClipWrapper(vec_env, obs_min=clip_obs[0], obs_max=clip_obs[1])
    
    #vec_env = RewardClipWrapper(vec_env, rew_min=clip_rewards[0], rew_max=clip_rewards[1])
    vec_env = RewardClipWrapper(vec_env)
    
    if clip_actions:
        vec_env = ActionClipWrapper(vec_env, a_min=clip_actions[0], a_max=clip_actions[1])

    vec_env = PPGAWrapper(vec_env)
    # jax.debug.print("OBS: {}", vec_env.observation_size)

    vec_env = create_gym_env(vec_env, batch_size=batch_size, seed=seed)  # Should we also add the backend?

    vec_env = JaxToTorchWrapper(vec_env, device='cuda')

    return vec_env
