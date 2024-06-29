from dataclasses import dataclass
from functools import partial
from math import floor 
from typing import Callable, Tuple, Any

import jax
from jax import debug
import jax.numpy as jnp
import flax.linen as nn
import optax
from chex import ArrayTree
from qdax.core.containers.repertoire import Repertoire
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey
from qdax.environments.base_wrappers import QDEnv
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.buffers.trajectory_buffer import TrajectoryBuffer
from rein_related import *

from qdax.core.emitters.emitter import Emitter, EmitterState

@dataclass
class REINaiveConfig:
    """Configuration for the REINaive emitter.
    
    Args:
        rollout_number: num of rollouts for gradient estimate
        sample_sigma: std to sample the samples for gradient estimate  (IS THIS PARAMETER SPACE EXPLORATION?)
        sample_mirror: if True, use mirroring sampling
        sample_rank_norm: if True, use normalisation
        
        num_generations_sample: frequency of archive-sampling
        
        adam_optimizer: if True, use ADAM, if False, use SGD
        learning_rate: obvious
        l2_coefficient: coefficient for regularisation
        
        novelty_nearest_neighbors: num of nearest neigbors for novelty computation
        use_novelty_archive: if True, use novelty archive for novelty (default is to use the content of the reperoire)
        use_novelty_fifo: if True, use fifo archive for novelty (default is to use the content of the repertoire)
        fifo_size: size of the novelty fifo bugger if used
        
        proprtion_explore: proportion of explore
    """
    batch_size: int = 32
    num_rein_training_steps: int = 10
    buffer_size: int = 320000
    rollout_number: int = 100
    discount_rate: float = 0.99
    adam_optimizer: bool = True
    learning_rate: float = 1e-3
    temperature: float = 0.



class REINaiveEmitterState(EmitterState):
    """Contains replay buffer.
    """
    trajectory_buffer: TrajectoryBuffer
    random_key: RNGKey
    
    
class REINaiveEmitter(Emitter):
    """
    An emitter that uses gradients approximated through rollouts.
    It dedicates part of the process on REINFORCE for fitness gradients and part
    to exploration gradients.
    
    This scan version scans through parents isntead of performing all REINFORCE
    operations in parallel, to avoid memory overload issue.
    """
    
    def __init__(
        self,
        config: REINaiveConfig,
        policy_network: nn.Module,
        env: QDEnv,
    ) -> None:
        self._config = config
        self._policy = policy_network
        self._env = env
            
            
        # SET UP THE LOSSES
        
        # Init optimizers
        
        self._policies_optimizer = optax.adam(
            learning_rate=self._config.learning_rate
            )
        
    @property
    def batch_size(self) -> int:
        """
        Returns:
            int: the batch size emitted by the emitter.
        """
        return self._config.batch_size
    
    @property 
    def use_all_data(self) -> bool:
        """Whether to use all data or not when used along other emitters.
        """
        return True
    
    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        random_key: RNGKey,
        repertoire: Repertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> Tuple[REINaiveEmitterState, RNGKey]:
        """Initializes the emitter.

        Args:
            init_genotypes: The initial population.
            random_key: A random key.

        Returns:
            The initial state of the REINaiveEmitter, a new random key.
        """

        observation_size = self._env.observation_size
        action_size = self._env.action_size
        descriptor_size = self._env.state_descriptor_length
        
        # Init trajectory buffer
        dummy_transition = QDTransition.init_dummy(
            observation_dim=observation_size,
            action_dim=action_size,
            descriptor_dim=descriptor_size,
        )
        
        trajectory_buffer = TrajectoryBuffer.init(
            buffer_size=self._config.buffer_size,
            transition=dummy_transition,
            env_batch_size=self._config.batch_size * 2,
            episode_length=self._env.episode_length,
        )
        
        random_key, subkey = jax.random.split(random_key)
        emitter_state = REINaiveEmitterState(
            trajectory_buffer=trajectory_buffer,
            random_key=subkey,
        )
        
        return emitter_state, random_key
    
    @partial(jax.jit, static_argnames=("self",))
    def emit(
        self,
        repertoire: Repertoire,
        emitter_state: REINaiveEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Do a step of REINFORCE emission.

        Args:
            repertoire: the current repertoire of genotypes.
            emitter_state: the state of the emitter used
            random_key: random key

        Returns:
            A batch of offspring, the new emitter state and a new key.
        """
        
        batch_size = self._config.batch_size
        
        # sample parents
        parents, random_key = repertoire.sample(random_key, batch_size)
        
        offsprings_rein = self.emit_rein(emitter_state, parents)
        
        genotypes = offsprings_rein
        
        return genotypes, {}, random_key
    
    @partial(jax.jit, static_argnames=("self",))
    def emit_rein(
        self,
        emitter_state: REINaiveEmitterState,
        parents: Genotype,
    ) -> Genotype:
        """Emit the offsprings generated through REINFORCE mutation.

        Args:
            emitter_state: the state of the emitter used, contains
            the trahectory buffer.
            parents: the parents selected to be applied gradients 
            to mutate towards better performance.

        Returns:
            A new set of offspring.
        """
        
        # Do a step of REINFORCE emission
        mutation_fn = partial(
            self._mutation_function_rein,
            emitter_state=emitter_state,
        )
        offsprings = jax.vmap(mutation_fn)(parents)
        
        return offsprings
    
    @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self, emitter_state : REINaiveEmitterState,
        repertoire: Optional[Repertoire],
        genotypes: Optional[Genotype],
        fitnesses: Optional[Fitness],
        descriptors: Optional[Descriptor],
        extra_scores: ExtraScores,
    ) -> REINaiveEmitterState:
        
        assert "transitions" in extra_scores.keys(), "Missing transitions or wrong key"
        transitions = extra_scores["transitions"]
        
        replay_buffer = emitter_state.trajectory_buffer.insert(transitions)
        emitter_state = emitter_state.replace(trajectory_buffer=replay_buffer)
        
        return emitter_state
    
    @partial(jax.jit, static_argnames=("self",))
    def _mutation_function_rein(
        self,
        policy_params: Genotype,
        emitter_state: REINaiveEmitterState,
    ) -> Genotype:
        """Apply REINFORCE mutation to a policy via multiple steps of gradient descent.

        Args:
            policy_params: a policy, supposed to be a differentiable neuaral network.
            emitter_state: the current state of the emitter, containing among others,
            the trajectory buffer.

        Returns:
            The updated parameters of the neural network.
        """
        
        # Define new policy optimizer state
        policy_optimizer_state = self._policies_optimizer.init(policy_params)
        
        def scan_train_policy(
            carry: Tuple[REINaiveEmitterState, Genotype, optax.OptState],
            unused: Any,
        ) -> Tuple[Tuple[REINaiveEmitterState, Genotype, optax.OptState], Any]:
            """Scans through the parents and applies REINFORCE training.
            """
            
            emitter_state, policy_params, policy_optimizer_state = carry
            
            (
                new_emitter_state,
                new_policy_params,
                new_policy_optimizer_state,
            ) = self._train_policy_(
                emitter_state,
                policy_params,
                policy_optimizer_state,
            )
            return (
                new_emitter_state,
                new_policy_params,
                new_policy_optimizer_state,
            ), ()
            
        (emitter_state, policy_params, policy_optimizer_state,), _ = jax.lax.scan(
            scan_train_policy,
            (emitter_state, policy_params, policy_optimizer_state),
            (),
            length=self._config.num_rein_training_steps,
        )
        
        return policy_params
        
    @partial(jax.jit, static_argnames=("self",))
    def _train_policy_(
        self,
        emitter_state: REINaiveEmitterState,
        policy_params: Genotype,
        policy_optimizer_state: optax.OptState,
    ) -> Tuple[REINaiveEmitterState, Genotype, optax.OptState]:
        """Apply one gradient step to a policy (called policy_params).

        Args:
            emitter_state: the current state of the emitter.
            policy_params: the current parameters of the policy network.
            policy_optimizer_state: the current state of the optimizer.

        Returns:
            The updated state of the emitter, the updated policy parameters
            and the updated optimizer state.
        """

        random_keys = jax.random.split(emitter_state.random_key, self._config.rollout_number+1)
        obs, action, logp, reward, _, mask = jax.vmap(
            self._sample_trajectory, in_axes=(0, None))(random_keys[:-1], policy_params)
        
        #debug.print("obs.shape: {}", obs.shape)
        
        # Add entropy term to reward
        reward += self._config.temperature * (-logp)
        
        # Compute standardized return
        return_standardized = self.get_return_standardized(reward, mask)
        
        # update policy
        policy_optimizer_state, policy_params = self._update_policy(
            policy_params=policy_params,
            policy_optimizer_state=policy_optimizer_state,
            obs=obs,
            action=action,
            logp=logp,
            mask=mask,
            return_standardized=return_standardized,
        )
        
        new_emitter_state = emitter_state.replace(
            random_key=random_keys[-1]
        )
        #print(reward * mask)
        #print('-'*50)
        
        #average_reward = jnp.mean(jnp.sum(reward * mask, axis=-1))
        #av_mask = jnp.mean(jnp.sum(mask, axis=-1))
        #debug.print("Average Reward: {}", average_reward)
        #debug.print('-'*50)      
        #debug.print("Average mask: {}", av_mask)  
        return new_emitter_state, policy_params, policy_optimizer_state
    
    @partial(jax.jit, static_argnames=("self",))
    def _sample_trajectory(
        self,
        random_key: RNGKey,
        policy_params: Genotype,
    ):
        """Samples a full trajectory using the environment and policy.
        Args:
            random_key: a random key.
            policy_params: the current parameters of the policy network.
        Returns:
            A tuple of observation, action, log-probability, reward, state descriptor, and mask arrays.
        """
        random_keys = jax.random.split(random_key, self._env.episode_length + 1)
        env_state_init = self._env.reset(random_keys[-1])
        #debug.print("env_state_init: {}", env_state_init)
        #debug.print('-'*50)        
        
        def _scan_sample_step(carry, x):
            (policy_params, env_state,) = carry
            (random_key,) = x
            
            next_env_state, action, action_logp = self.sample_step(
                random_key, policy_params, env_state
            )
            return (policy_params, next_env_state), (
                env_state.obs,
                action,
                action_logp,
                next_env_state.reward,
                env_state.done,
                env_state.info["state_descriptor"],
            )
        print(f"Length : {self._env.episode_length}")
        _, (obs, action, action_logp, reward, done, state_desc) = jax.lax.scan(
            _scan_sample_step,
            (policy_params, env_state_init),
            (random_keys[:self._env.episode_length],),
            length=self._env.episode_length,
        )
        
        # compute a mask to indicate the valid steps
        mask = 1. - jnp.clip(jnp.cumsum(done), a_min=0., a_max=1.)        
        return obs, action, action_logp, reward, state_desc, mask
    
    @partial(jax.jit, static_argnames=("self",))
    def sample_step(
        self,
        random_key: RNGKey,
        policy_params: Genotype,
        env_state: Any,
    ) -> Tuple[Any, Any, Any]:
        """Samples a step using the environment and policy.

        Args:
            random_key: a random key.
            policy_params: the current parameters of the policy network.
            env_state: the current state of the environment.

        Returns:
            A tuple of the next environment state, the action, and log-probability of the action.
        """
        #print(f"policy_params type: {type(policy_params)}")
        #print(f"policy_params: {policy_params}")
        #print(f"env_state.obs: {env_state.obs}")
        #print(f"env_state.obs type: {type(env_state.obs)}")
        '''
        action, action_logp = self._policy.sample(
            policy_params, random_key, env_state.obs
        )
        '''
        action, action_logp = self._policy.apply(
            policy_params, random_key, env_state.obs, method=self._policy.sample
        )

        next_env_state = self._env.step(env_state, action)
        
        return next_env_state, action, action_logp
    
    @partial(jax.jit, static_argnames=("self",))
    def get_return_standardized(self, reward: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        """Compute the standardized return.

        Args:
            reward: the reward obtained.
            mask: the mask to indicate the valid steps.

        Returns:
            The standardized return.
        """
        # compute the return
        return_ = jax.vmap(self.get_return)(reward * mask)
        return self.standardize(return_)
    
    @partial(jax.jit, static_argnames=("self",))
    def get_return(self, reward):
        """Computes the discounted return for each step in the trajectory.
        Args:
            reward: the reward array.
        Returns:
            The discounted return array.
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
            reverse=True,
        )
        return return_
    
    @partial(jax.jit, static_argnames=("self",))
    def standardize(self, return_):
        """Standardizes the return values.
        Args:
            return_: the return array.
        Returns:
            The standardized return array.
        """
        #return (return_ - return_.mean()) / (return_.std() + 1e-8)
        return jax.nn.standardize(return_, axis=0, variance=1., epsilon=EPS)

    @partial(jax.jit, static_argnames=("self",))
    def _update_policy(
        self,
        policy_params: Genotype,
        policy_optimizer_state: optax.OptState,
        obs,
        action,
        logp,
        mask,
        return_standardized
    ):
        """Updates the policy parameters using the optimizer.
        Args:
            policy_params: the current parameters of the policy network.
            policy_optimizer_state: the current state of the optimizer.
            obs: observations from the environment.
            action: actions taken in the environment.
            logp: log-probabilities of the actions.
            mask: the mask array indicating valid steps.
            return_standardized: the standardized return values.
        Returns:
            The updated optimizer state and policy parameters.
        """
        def loss_fn(params):
            #logp_ = self._policy.logp(params, jax.lax.stop_gradient(obs), jax.lax.stop_gradient(action))
            logp_ = self._policy.apply(params, jax.lax.stop_gradient(obs), jax.lax.stop_gradient(action), method=self._policy.logp)
            #return -jnp.mean(logp_ * mask * return_standardized)
            return -jnp.mean(jnp.multiply(logp_ * mask, jax.lax.stop_gradient(return_standardized)))

        grads = jax.grad(loss_fn)(policy_params)
        updates, new_optimizer_state = self._policies_optimizer.update(grads, policy_optimizer_state)
        new_policy_params = optax.apply_updates(policy_params, updates)
        return new_optimizer_state, new_policy_params