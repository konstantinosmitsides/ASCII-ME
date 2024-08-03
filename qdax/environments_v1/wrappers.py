from typing import Dict

from brax.v1.envs.env import Wrapper, State, Env
import flax.struct
import jax
import jax.numpy as jnp
from brax.v1 import jumpy as jp


class CompletedEvalMetrics(flax.struct.PyTreeNode):
    current_episode_metrics: Dict[str, jp.ndarray]
    completed_episodes_metrics: Dict[str, jp.ndarray]
    completed_episodes: jp.ndarray
    completed_episodes_steps: jp.ndarray


class CompletedEvalWrapper(Wrapper):
    """Brax env with eval metrics for completed episodes."""

    STATE_INFO_KEY = "completed_eval_metrics"

    def reset(self, rng: jp.ndarray) -> State:
        reset_state = self.env.reset(rng)
        reset_state.metrics["reward"] = reset_state.reward
        eval_metrics = CompletedEvalMetrics(
            current_episode_metrics=jax.tree_util.tree_map(
                jp.zeros_like, reset_state.metrics
            ),
            completed_episodes_metrics=jax.tree_util.tree_map(
                lambda x: jp.zeros_like(jp.sum(x)), reset_state.metrics
            ),
            completed_episodes=jp.zeros(()),
            completed_episodes_steps=jp.zeros(()),
        )
        reset_state.info[self.STATE_INFO_KEY] = eval_metrics
        return reset_state

    def step(
        self, state: State, action: jp.ndarray
    ) -> State:
        state_metrics = state.info[self.STATE_INFO_KEY]
        if not isinstance(state_metrics, CompletedEvalMetrics):
            raise ValueError(f"Incorrect type for state_metrics: {type(state_metrics)}")
        del state.info[self.STATE_INFO_KEY]
        nstate = self.env.step(state, action)
        nstate.metrics["reward"] = nstate.reward
        # steps stores the highest step reached when done = True, and then
        # the next steps becomes action_repeat
        completed_episodes_steps = state_metrics.completed_episodes_steps + jp.sum(
            nstate.info["steps"] * nstate.done
        )
        current_episode_metrics = jax.tree_util.tree_map(
            lambda a, b: a + b, state_metrics.current_episode_metrics, nstate.metrics
        )
        completed_episodes = state_metrics.completed_episodes + jp.sum(nstate.done)
        completed_episodes_metrics = jax.tree_util.tree_map(
            lambda a, b: a + jp.sum(b * nstate.done),
            state_metrics.completed_episodes_metrics,
            current_episode_metrics,
        )
        current_episode_metrics = jax.tree_util.tree_map(
            lambda a, b: a * (1 - nstate.done) + b * nstate.done,
            current_episode_metrics,
            nstate.metrics,
        )

        eval_metrics = CompletedEvalMetrics(
            current_episode_metrics=current_episode_metrics,
            completed_episodes_metrics=completed_episodes_metrics,
            completed_episodes=completed_episodes,
            completed_episodes_steps=completed_episodes_steps,
        )
        nstate.info[self.STATE_INFO_KEY] = eval_metrics
        return nstate


class TimeAwarenessWrapper(Wrapper):
    """Wraps gym environments to add time in obs."""

    def __init__(self, env: Env) -> None:
        super().__init__(env)

    @property
    def observation_size(self) -> int:
        return super().observation_size + 1

    def reset(self, rng: jp.ndarray) -> State:
        state = self.env.reset(rng)
        return state.replace(obs=jp.concatenate([state.obs, jp.ones((1,))]))

    def step(self, state: State, action: jp.ndarray) -> State:
        state = self.env.step(state.replace(obs=state.obs[:-1]), action)
        return state.replace(obs=jp.concatenate([state.obs, (jp.array([self.episode_length]) - state.info["steps"])/self.episode_length]))

class ClipRewardWrapper(Wrapper):
    """Wraps gym environments to clip the reward to be greater than 0.

    Utilisation is simple: create an environment with Brax, pass
    it to the wrapper with the name of the environment, and it will
    work like before and will simply clip the reward to be greater than 0.
    """

    def __init__(self, env: Env) -> None:
        super().__init__(env)

    def reset(self, rng: jnp.ndarray) -> State:
        state = self.env.reset(rng)
        return state.replace(reward=jnp.clip(state.reward, a_min=0.))

    def step(self, state: State, action: jnp.ndarray) -> State:
        state = self.env.step(state, action)
        return state.replace(reward=jnp.clip(state.reward, a_min=0.))
    
    
class ClipActionWrapper(Wrapper):
  """Clips actions to the action space."""

  def __init__(self, env, low=-1.0, high=1.0):
    super().__init__(env)
    self.low = low
    self.high = high
    
  def step(self, state, action):
    action = jp.clip(action, self.low, self.high)
    return self.env.step(state, action)

@flax.struct.dataclass
class NormalizeVecObsEnvState:
    mean: jnp.ndarray = None
    var: jnp.ndarray = None
    count: float = 1e-4
    env_state: Env = None
    
class NormalizeVecObservationWrapper(Wrapper):
    def __init__(self, env, initial_stats=None):
        super().__init__(env)
        
        self.stats = NormalizeVecObsEnvState(
            mean=initial_stats[0],  # Initialized as None, set on first update
            var=initial_stats[1],
            count=initial_stats[2],
            env_state=None
        ) if initial_stats else NormalizeVecObsEnvState()

    def update_stats(self, obs, stats):
        if stats.mean is None or stats.var is None:
            mean = obs  # Directly use observation as mean if first observation
            var = jnp.zeros_like(obs)  # Initialize variance to zero if first observation
            count = 1.0
        else:
            mean, var, count = stats.mean, stats.var, stats.count
            delta = obs - mean
            tot_count = count + 1

            new_mean = mean + delta / tot_count
            m_a = var * count
            m_b = jnp.square(delta) * count / tot_count
            M2 = m_a + m_b
            var = M2 / tot_count
            mean = new_mean
            count = tot_count

        return NormalizeVecObsEnvState(mean, var, count, stats.env_state)
    '''
    def reset(self, key):
        env_state = self.env.reset(key)
        new_stats = self.update_stats(env_state.obs, self.stats.replace(env_state=env_state))
        normalized_obs = (env_state.obs - new_stats.mean) / jnp.sqrt(new_stats.var + 1e-8)
        updated_env_state = env_state.replace(obs=normalized_obs)
        return new_stats.replace(env_state=updated_env_state)

    def step(self, stats, action):
        obs, env_state, reward, done, info = self.env.step(stats.env_state, action)
        new_stats = self.update_stats(obs, stats.replace(env_state=env_state))
        normalized_obs = (obs - new_stats.mean) / jnp.sqrt(new_stats.var + 1e-8)
        updated_env_state = env_state.replace(obs=normalized_obs)
        return new_stats.replace(env_state=updated_env_state)
    '''
    
    def reset(self, key):
        env_state = self.env.reset(key)
        new_stats = self.update_stats(env_state.obs, self.stats.replace(env_state=env_state))
        normalized_obs = (env_state.obs - new_stats.mean) / jnp.sqrt(new_stats.var + 1e-8)
        updated_env_state = env_state.replace(obs=normalized_obs)
        self.stats = new_stats.replace(env_state=updated_env_state)
        return updated_env_state  # Only return the updated environment state

    def step(self, state, action):
        env_state = self.env.step(state, action)
        new_stats = self.update_stats(env_state.obs, self.stats.replace(env_state=env_state))
        normalized_obs = (env_state.obs - new_stats.mean) / jnp.sqrt(new_stats.var + 1e-8)
        updated_env_state = env_state.replace(obs=normalized_obs)
        self.stats = new_stats.replace(env_state=updated_env_state)
        return updated_env_state
