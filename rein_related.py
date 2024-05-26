from dataclasses import dataclass
from typing import Any, Union, Tuple, Callable, Optional
from functools import partial
import time

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training.train_state import TrainState
import optax

from qdax import environments

from IPython.display import HTML
from brax.io import html


@dataclass
class Config:
	batch_size: int = 512   # number of trajectories to sample per trainng step
	learning_rate: float = 1e-3  # learning rate for the optimizer
	discount_rate: float = 0.999   # discount factor for future rewards
	temperature: float = 0.    # temperature for entropy regularization
	clip_param: float = 0.2     # clipping parameter for PPO
	max_norm_clip: float = 0.    # maximum norm for gradient clipping


_half_log2pi = 0.5 * jnp.log(2 * jnp.pi)
EPS = 1e-8
class Reinforce:

	def __init__(self, config, policy, env):
		self._config = config
		self._policy = policy
		self._env = env


	def init(self, random_key):
		# Initialize params
		random_key_1, random_key_2 = jax.random.split(random_key)
		fake_obs = jnp.zeros(shape=(self._env.observation_size,))
		params = self._policy.init(random_key_1, random_key_2, fake_obs)

		# Initialize optimizer
		tx = optax.adam(self._config.learning_rate)

		# Return train state
		return TrainState.create(apply_fn=self._policy.apply, params=params, tx=tx)

	@partial(jax.jit, static_argnames=("self",))
	def logp_fn(self, params, obs, action):
		""" Computes the log-probability of actions given observations using the policy.
		"""
		return self._policy.apply(params, obs, action, method=self._policy.logp)

	@partial(jax.jit, static_argnames=("self",))
	def entropy_fn(self, params, obs):
		""" Computes the entropy of the policy.
		"""
		return self._policy.apply(params, obs, method=self._policy.entropy)

	@partial(jax.jit, static_argnames=("self",))
	def sample_step(self, random_key, train_state, env_state):
		""" Samples one step in the environment and returns the next state, action
			and log-probability of the action.
		"""
		action, action_logp = train_state.apply_fn(train_state.params, random_key, env_state.obs)
		next_env_state = self._env.step(env_state, action)

		return next_env_state, action, action_logp

	@partial(jax.jit, static_argnames=("self",))
	def sample_trajectory(self, random_key, train_state):
		""" Samples a full trajectory using the environment and policy.
		"""
		random_keys = jax.random.split(random_key, self._env.episode_length+1)
		env_state_init = self._env.reset(random_keys[-1])

		def _scan_sample_step(carry, x):
			(train_state, env_state,) = carry
			(random_key,) = x

			next_env_state, action, action_logp = self.sample_step(random_key, train_state, env_state)
			return (train_state, next_env_state), (env_state.obs, action, action_logp, next_env_state.reward, env_state.done, env_state.info["state_descriptor"])

			# uses jax.lax.scan to iterate over time steps
		_, (obs, action, action_logp, reward, done, state_desc) = jax.lax.scan(
			_scan_sample_step,
			(train_state, env_state_init,),
			(random_keys[:self._env.episode_length],),
			length=self._env.episode_length,
		)

		# computes a mask to indicate the valid steps 
		mask = 1. - jnp.clip(jnp.cumsum(done), a_min=0., a_max=1.)

		return obs, action, action_logp, reward, state_desc, mask

	@partial(jax.jit, static_argnames=("self",))
	def get_done_index(self, mask):
		mask = jnp.expand_dims(mask, axis=-1)
		done_index = jnp.int32(jnp.sum(mask)) - 1
		return done_index

	@partial(jax.jit, static_argnames=("self",))
	def get_final_xy_position(self, state_desc, mask):
		done_index = self.get_done_index(mask)
		return state_desc[done_index]

	@partial(jax.jit, static_argnames=("self",))
	def get_feet_contact_rate(self, state_desc, mask):
		mask = jnp.expand_dims(mask, axis=-1)
		return jnp.sum(state_desc * mask, axis=0)/jnp.sum(mask)

	@partial(jax.jit, static_argnames=("self",))
	def get_return(self, reward):
		""" Computes the discounted return for each step in the trajectory
		"""

		def _body(carry, x):
			(next_return,) = carry
			(reward,) = x

			current_return = reward + self._config.discount_rate * next_return
			return (current_return,), (current_return,)
		
		_, (return_,) = jax.lax.scan(
			_body,
			(jnp.array(0.),),
			(reward,),
			length=self._env.episode_length,
			reverse=True,)

		return return_

	@partial(jax.jit, static_argnames=("self",))
	def standardize(self, return_):
		return jax.nn.standardize(return_, axis=0, variance=1., epsilon=EPS)

	@partial(jax.jit, static_argnames=("self",))
	def get_return_standardized(self, reward, mask):
		""" Standardizes the return values for stability in training.
		"""
		return_ = jax.vmap(self.get_return)(reward * mask)
		return self.standardize(return_)

	@partial(jax.jit, static_argnames=("self",))
	def clip_by_l2_norm(self, x: jnp.ndarray, max_norm: float) -> jnp.ndarray:
		"""Clip gradients to maximum l2 norm `max_norm`."""
		sum_sq = jnp.sum(jnp.vdot(x, x))
		nonzero = sum_sq > 0
		sum_sq_ones = jnp.where(nonzero, sum_sq, jnp.ones_like(sum_sq))
		norm = jnp.where(nonzero, jnp.sqrt(sum_sq_ones), sum_sq)

		return (x * max_norm) / jnp.maximum(norm, max_norm)

	@partial(jax.jit, static_argnames=("self",))
	def loss_reinforce(self, params, obs, action, logp, mask, return_standardized):
		""" REINFORCE loss function.
		"""
		logp_ = self.logp_fn(params, jax.lax.stop_gradient(obs), jax.lax.stop_gradient(action))
		return -jnp.mean(jnp.multiply(logp_ * mask, jax.lax.stop_gradient(return_standardized)))

	@partial(jax.jit, static_argnames=("self",))
	def loss_reinforce_with_is(self, params, obs, action, logp, mask, return_standardized):
		""" REINFORCE with importance sampling loss function.
		"""
		logp_ = self.logp_fn(params, jax.lax.stop_gradient(obs), jax.lax.stop_gradient(action))
		ratio = jnp.exp(logp_ - jax.lax.stop_gradient(logp))
		return -jnp.mean(jnp.multiply(ratio * mask, jax.lax.stop_gradient(return_standardized)))

	@partial(jax.jit, static_argnames=("self",))
	def loss_ppo(self, params, obs, action, logp, mask, return_standardized):
		""" PPO loss function.
		"""
		logp_ = self.logp_fn(params, jax.lax.stop_gradient(obs), jax.lax.stop_gradient(action))
		ratio = jnp.exp(logp_ - jax.lax.stop_gradient(logp))

		pg_loss_1 = jnp.multiply(ratio * mask, jax.lax.stop_gradient(return_standardized))
		pg_loss_2 = jax.lax.stop_gradient(return_standardized) * jax.lax.clamp(1. - self._config.clip_param, ratio, 1. + self._config.clip_param)
		return -jnp.mean(jnp.minimum(pg_loss_1, pg_loss_2))

	@partial(jax.jit, static_argnames=("self",))
	def train_step(self, random_key, train_state):
		# Sample trajectories
		random_keys = jax.random.split(random_key, self._config.batch_size)
		obs, action, logp, reward, _, mask = jax.vmap(self.sample_trajectory, in_axes=(0, None))(random_keys, train_state)

		# Add entropy term to reward
		reward += self._config.temperature * (-logp)

		# Compute standardized return
		return_standardized = self.get_return_standardized(reward, mask)

		# Compute loss and grads
		loss, grads = jax.value_and_grad(self.loss_reinforce)(train_state.params, obs, action, logp, mask, return_standardized)
		train_state = train_state.apply_gradients(grads=grads)

		metrics = {
			"loss": loss,
			"reward": reward * mask,
			"mask": mask,
		}

		return (train_state,), (metrics,)

	@partial(jax.jit, static_argnames=("self", "num_steps"))
	def train(self, random_key, train_state, num_steps):
		random_keys = jax.random.split(random_key, num_steps)

		def _scan_train_step(carry, x):
			(train_state,) = carry
			(random_key,) = x

			(train_state,), (metrics,) = self.train_step(random_key, train_state)

			return (train_state,), (metrics,)

		(train_state,), (metrics,) = jax.lax.scan(
			_scan_train_step,
			(train_state,),
			(random_keys,),
			length=num_steps,)

		return (train_state,), (metrics,)