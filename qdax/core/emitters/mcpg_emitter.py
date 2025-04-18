from dataclasses import dataclass
from functools import partial
from math import floor 
from typing import Callable, Tuple, Any, Optional

import jax
from jax import debug
import jax.numpy as jnp
import flax.linen as nn
import optax
from chex import ArrayTree
from qdax.core.containers.repertoire import Repertoire
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey
from qdax.environments.base_wrappers import QDEnv
from qdax.core.neuroevolution.buffers.buffer import QDTransition, QDMCTransition, PPOTransition
#from qdax.core.neuroevolution.buffers.trajectory_buffer import TrajectoryBuffer
import flashbax as fbx
import chex

from qdax.core.emitters.emitter import Emitter, EmitterState
from jax import profiler
import os

EPS = 1e-8

#profiler_dir = "Memory_Investigation"
#os.makedirs(profiler_dir, exist_ok=True)


@dataclass
class MCPGConfig:
    """Configuration for the REINaive emitter.
    """
    no_agents: int = 256
    buffer_sample_batch_size: int = 32
    buffer_add_batch_size: int = 256
    #batch_size: int = 1000*256
    #mini_batch_size: int = 1000*256
    no_epochs: int = 16
    learning_rate: float = 3e-4
    discount_rate: float = 0.99
    #buffer_size: int = 256000
    clip_param: float = 0.2
    
class MCPGEmitterState(EmitterState):
    """Contains the trajectory buffer.
    """
    #buffer: Any
    buffer_state: Any
    random_key: RNGKey
    
class MCPGEmitter(Emitter):
    
    def __init__(
        self,
        config: MCPGConfig,
        policy_net: nn.Module,
        env: QDEnv,
    ) -> None:
        
        self._config = config
        self._policy = policy_net
        self._env = env
        
        self._policy_opt = optax.adam(
            learning_rate=self._config.learning_rate
        )
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=self._env.episode_length,
            min_length_time_axis=self._env.episode_length,
            sample_batch_size=self._config.buffer_sample_batch_size,
            add_batch_size=self._config.buffer_add_batch_size,
            sample_sequence_length=self._env.episode_length,
            period=self._env.episode_length,
        )
        self._buffer = buffer
        
    @property
    def batch_size(self) -> int:
        """
        Returns:
            int: the batch size emitted by the emitter.
        """
        return self._config.no_agents
    
    @property
    def use_all_data(self) -> bool:
        """Whther to use all data or not when used along other emitters.
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
    ) -> Tuple[MCPGEmitterState, RNGKey]:
        """Initializes the emitter state.
        """
        obs_size = self._env.observation_size
        action_size = self._env.action_size
        descriptor_size = self._env.state_descriptor_length
        
        # Init trajectory buffer

        dummy_transition = PPOTransition.init_dummy(
            observation_dim=obs_size,
            action_dim=action_size,
            descriptor_dim=descriptor_size,
        )

        '''
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=self._env.episode_length,
            min_length_time_axis=self._env.episode_length,
            sample_batch_size=self._config.no_agents,
            add_batch_size=self._config.no_agents,
            sample_sequence_length=self._env.episode_length,
            period=self._env.episode_length,
        )
        '''
        
        buffer_state = self._buffer.init(dummy_transition)
        
        '''
        buffer = TrajectoryBuffer.init(
            buffer_size=self._config.buffer_size,
            transition=dummy_transition,
            env_batch_size=self._config.no_agents*2,
            episode_length=self._env.episode_length,
        )
        '''
        
        random_key, subkey = jax.random.split(random_key)
        emitter_state = MCPGEmitterState(
            #buffer=buffer,
            buffer_state=buffer_state,
            random_key=subkey,
        )
        #ßprint(emitter_state)
        
        return emitter_state, random_key
    
    @partial(jax.jit, static_argnames=("self",))
    def emit(
        self,
        repertoire: Repertoire,
        emitter_state: MCPGEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Do a step of MCPG emission.
        """
        
        no_agents = self._config.no_agents
        
        random_keys = jax.random.split(random_key, no_agents+2)
        
        
        # sample parents
        parents, random_key = repertoire.sample(
            random_key=random_keys[-1],
            num_samples=no_agents,
        )
        
        offsprings_mcpg = self.emit_mcpg(emitter_state, parents, random_keys[:no_agents])
        
        #jax.debug.breakpoint()
        
        return offsprings_mcpg, {}, random_keys[-2]
    
    @partial(jax.jit, static_argnames=("self",))
    def emit_mcpg(
        self,
        emitter_state: MCPGEmitterState,
        parents: Genotype,
        random_keys: ArrayTree,
    ) -> Genotype:
        """Emit the offsprings generated through MCPG mutation.
        """
        '''
        mutation_fn = partial(
            self._mutation_function_mcpg,
            emitter_state=emitter_state,
        )
        '''
        #profiler.start_trace(profiler_dir)
        
        offsprings = jax.vmap(self._mutation_function_mcpg, in_axes=(0, None, 0))(parents, emitter_state, random_keys)
        
        #profiler.stop_trace()
        
        return offsprings
    
    @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        emitter_state: MCPGEmitterState,
        repertoire: Optional[Repertoire],
        genotypes: Optional[Genotype],
        fitnesses: Optional[Fitness],
        descriptors: Optional[Descriptor],
        extra_scores: ExtraScores,
    ) -> MCPGEmitterState:
        """Update the emitter state.
        """
        
        random_key, _ = jax.random.split(emitter_state.random_key)
        
        assert "transitions" in extra_scores.keys(), "Missing transtitions or wrong key"
        transitions = extra_scores["transitions"]
        new_buffer_state = self._buffer.add(emitter_state.buffer_state, transitions)
        new_emitter_state = emitter_state.replace(random_key=random_key, buffer_state=new_buffer_state)
        
        return new_emitter_state
        
        # update the buffer
        '''
        replay_buffer = emitter_state.buffer.insert(transitions)
        emitter_state = emitter_state.replace(buffer=replay_buffer)
        
        return emitter_state
        '''
    
    @partial(jax.jit, static_argnames=("self",))
    def compute_mask(
        self,
        done,
    ):
        return 1. - jnp.clip(jnp.cumsum(done), a_min=0., a_max=1.)
    '''
    @partial(jax.jit, static_argnames=("self",))
    def compute_logps(
        self,
        policy_params,
        obs,
        actions,
    ):
        """Compute the log probabilities of the actions.
        """
        compute_logp = partial(
            self._policy.apply,
            params=policy_params,
            method=self._policy.logp,
        )
        
        return jax.vmap(compute_logp)(obs, actions)
    '''
    
    @partial(jax.jit, static_argnames=("self",))
    def compute_logps(self, policy_params, obs, actions):
        def compute_logp(single_obs, single_action):
            # Correctly handle operations on single_obs and single_action
            # Ensure no inappropriate method calls like .items() are made
            return self._policy.apply(policy_params, single_obs, single_action, method=self._policy.logp)

        # Use jax.vmap to apply compute_logp across batches of obs and actions
        return jax.vmap(compute_logp, in_axes=(0, 0))(obs, actions)
       
    @partial(jax.jit, static_argnames=("self",))
    def get_return(
        self,
        rewards,
    ):
        def _body(carry, x):
            (next_return,) = carry
            (rewards,) = x

            current_return = rewards + self._config.discount_rate * next_return
            return (current_return,), (current_return,)
        
        
        
        #jax.debug.print("rewards", rewards.shape)
        
        _, (return_,) = jax.lax.scan(
            _body,
            (jnp.array(0.),),
            (rewards,),
            length=self._env.episode_length,
            reverse=True,
        )
        
        return return_
    
    
    '''
    @partial(jax.jit, static_argnames=("self",))
    def get_return(self, rewards):
        def _body(carry, reward):
            next_return = carry  # carry should be unpacked directly if it's a single element
            current_return = reward + self._config.discount_rate * next_return
            return current_return, current_return  # Maintain the same shape and type

        initial_return = jnp.array(0.0)  # Ensure initial_return is correctly shaped as a scalar
        _, return_ = jax.lax.scan(
            _body,
            initial_return,
            rewards,  # Pass rewards directly without extra tuple wrapping
            length=int(self._env.episode_length),
            reverse=True,
        )

        return return_
    '''
    
    @partial(jax.jit, static_argnames=("self",))
    def standardize(
        self,
        return_,
    ):
        return jax.nn.standardize(return_, axis=0, variance=1, epsilon=EPS)
    
    @partial(jax.jit, static_argnames=("self",))
    def get_standardized_return(
        self,
        rewards,
        mask,
    ):
        #mask = jnp.expand_dims(mask, axis=-1)
        #valid_rewards = (rewards * mask)#.squeeze(axis=-1)
        #jax.debug.print("mask: {}", mask.shape)
        #jax.debug.print("rewards*mask: {}", (rewards * mask).shape)
        return_ = jax.vmap(self.get_return)(rewards * mask)
        return self.standardize(return_)
        #return return_
    
    '''
    @partial(jax.jit, static_argnames=("self",))
    def _mutation_function_mcpg(
        self,
        policy_params,
        emitter_state: MCPGEmitterState,
    ) -> Genotype:
        """Mutation function for MCPG.
        """
        
        policy_opt_state = self._policy_opt.init(policy_params)
        
        random_key = emitter_state.random_key
        buffer_state = emitter_state.buffer_state
        
        # NOW YOU DONT CARE BUT AT SOME POINT YOU MIGH NEED DIFFERENT RANDOM KEY FRO SAMPLING FOR EACH GENOTYPE
        batch = self._buffer.sample(buffer_state, random_key)
        
        trans = batch.experience

        
        obs = trans.obs
        actions = trans.actions
        rewards = trans.rewards
        dones = trans.dones
        
        mask = jax.vmap(self.compute_mask, in_axes=0)(dones)
        logps = trans.logp  
         
        standardized_returns = self.get_standardized_return(rewards, mask)
        
        def scan_train_policy(
            carry: Tuple[MCPGEmitterState, Genotype, optax.OptState],
            unused: Any,
        ) -> Tuple[Tuple[MCPGEmitterState, Genotype, optax.OptState], Any]:
            
            policy_params, policy_opt_state = carry
            
            (
                new_policy_params,
                new_policy_opt_state,
            ) = self._train_policy_(
                policy_params,
                policy_opt_state,
                obs,
                actions,
                standardized_returns,
                logps,
                mask
            )
            return (
                new_policy_params,
                new_policy_opt_state,
            ), None
            
        (policy_params, policy_opt_state), _ = jax.lax.scan(
            scan_train_policy,
            (policy_params, policy_opt_state),
            None,
            length=self._config.no_epochs,
        )
        
        return policy_params
        '''
        
    @partial(jax.jit, static_argnames=("self",))
    def _mutation_function_mcpg(
        self,
        policy_params,
        emitter_state: MCPGEmitterState,
        random_key: RNGKey,
    ) -> Genotype:
        """Mutation function for MCPG."""

        policy_opt_state = self._policy_opt.init(policy_params)
        
        # Directly sample batch and use necessary components
        batch = self._buffer.sample(emitter_state.buffer_state, random_key)
        trans = batch.experience
        #jax.debug.print("trans: {}", trans)
        mask = jax.vmap(self.compute_mask, in_axes=0)(trans.dones)
        standardized_returns = self.get_standardized_return(trans.rewards, mask)
        #jax.debug.print("standardized_returns: {}", standardized_returns)
        
        def scan_train_policy(
            carry: Tuple[MCPGEmitterState, Genotype, optax.OptState],
            unused: Any,
        ) -> Tuple[Tuple[MCPGEmitterState, Genotype, optax.OptState], Any]:
            
            policy_params, policy_opt_state = carry
            
            # Train policy with directly used transaction fields
            new_policy_params, new_policy_opt_state = self._train_policy_(
                policy_params,
                policy_opt_state,
                trans.obs,
                trans.actions,
                standardized_returns,
                trans.logp,
                mask
            )
            return (new_policy_params, new_policy_opt_state), None
            
        (policy_params, policy_opt_state), _ = jax.lax.scan(
            scan_train_policy,
            (policy_params, policy_opt_state),
            None,
            length=self._config.no_epochs,
        )
        
        return policy_params
        
        
    '''
    @partial(jax.jit, static_argnames=("self",))
    def _mutation_function_mcpg(
        self,
        policy_params,
        emitter_state: MCPGEmitterState,
    ) -> Genotype:
        """Mutation function for MCPG.
        """
        
        buffer = emitter_state.buffer
        
        policy_opt_state = self._policy_opt.init(policy_params)
        
        random_key = emitter_state.random_key
        
        #random_key, subkey = jax.random.split(emitter_state.random_key)
        sample_size = int(self._config.batch_size) // int(self._env.episode_length)
        #print(f"episodic_data_size: {int(buffer.current_episodic_data_size)}")
        #episodic_data_size = buffer.current_episodic_data_size.item()
        
        trans, random_key = buffer.sample(
            random_key=random_key,
            sample_size=sample_size,
            episodic_data_size=64,
            sample_traj=True,
        )
        new_emitter_state = emitter_state.replace(random_key=random_key)
        # trans has shape (episde_length*sample_size, transition_dim)
        
        obs = trans.obs.reshape(sample_size, self._env.episode_length, -1)
        actions = trans.actions.reshape(sample_size, self._env.episode_length, -1)
        rewards = trans.rewards.reshape(sample_size, self._env.episode_length, -1)
        #jax.debug.print("rewards shape: {}", rewards.shape)
        #print(f"rewards shape: {rewards.shape}")
        dones = trans.dones.reshape(sample_size, self._env.episode_length, -1)
        
        mask = jax.vmap(self.compute_mask, in_axes=0)(dones)
        logps = jax.vmap(self.compute_logps, in_axes=(None, 0, 0))(policy_params, obs, actions)
        
        standardized_returns = self.get_standardized_return(rewards, mask)
        
        def scan_train_policy(
            carry: Tuple[MCPGEmitterState, Genotype, optax.OptState],
            unused: Any,
        ) -> Tuple[Tuple[MCPGEmitterState, Genotype, optax.OptState], Any]:
            
            policy_params, policy_opt_state = carry
            
            (
                new_policy_params,
                new_policy_opt_state,
            ) = self._train_policy_(
                policy_params,
                policy_opt_state,
                obs,
                actions,
                standardized_returns,
                mask,
                logps,
            )
            
            return (
                new_policy_params,
                new_policy_opt_state,
            ), None

        (policy_params, policy_opt_state), _ = jax.lax.scan(
            scan_train_policy,
            (policy_params, policy_opt_state),
            None,
            length=self._config.no_epochs,
        )
        
        return policy_params
        
    '''
    
    @partial(jax.jit, static_argnames=("self",))
    def _train_policy_(
        self,
        #emitter_state: MCPGEmitterState,
        policy_params,
        policy_opt_state: optax.OptState,
        obs,
        actions,
        standardized_returns,
        logps,
        mask
    ) -> Tuple[MCPGEmitterState, Genotype, optax.OptState]:
        """Train the policy.
        """
        '''
        random_key, subkey = jax.random.split(emitter_state.random_key)
        buffer_state = emitter_state.buffer_state
        
        batch = self._buffer.sample(buffer_state, subkey)
        
        #sample_size = int(self._config.batch_size) // int(self._env.episode_length)
        
        trans = batch.experience
        
        #obs = trans.obs.reshape(self._config.no_agents, self._env.episode_length, -1)
        #actions = trans.actions.reshape(self._config.no_agents, self._env.episode_length, -1)
        #rewards = trans.rewards.reshape(self._config.no_agents, self._env.episode_length, -1)
        #jax.debug.print("rewards shape: {}", rewards.shape)
        #print(f"rewards shape: {rewards.shape}")
        #dones = trans.dones.reshape(self._config.no_agents, self._env.episode_length, -1)
        
        obs = trans.obs
        actions = trans.actions
        rewards = trans.rewards
        dones = trans.dones
        
        mask = jax.vmap(self.compute_mask, in_axes=0)(dones)
        logps = jax.vmap(self.compute_logps, in_axes=(None, 0, 0))(policy_params, obs, actions)
        
        standardized_returns = self.get_standardized_return(rewards, mask)
        '''
        
        def scan_update(carry, _):
            policy_params, policy_opt_state = carry
            grads = jax.grad(self.loss_ppo)(policy_params, obs, actions, logps, mask, standardized_returns)
            updates, new_policy_opt_state = self._policy_opt.update(grads, policy_opt_state)
            new_policy_params = optax.apply_updates(policy_params, updates)
            return (new_policy_params, new_policy_opt_state), None
        
        (final_policy_params, final_policy_opt_state), _ = jax.lax.scan(
            scan_update,
            (policy_params, policy_opt_state),
            None,
            length=1,
        )
        
        #new_emitter_state = emitter_state.replace(random_key=random_key)
        
        return final_policy_params, final_policy_opt_state
    
    '''
    @partial(jax.jit, static_argnames=("self",))
    def _train_policy_(
        self,
        policy_params,
        policy_opt_state,
        obs,
        actions,
        standardized_returns,
        mask,
        logps,
    ):
        """Train the policy.
        """
        
        def _scan_update(carry, _):
            policy_params, policy_opt_state = carry
            grads = jax.grad(self.loss_ppo)(policy_params, obs, actions, logps, mask, standardized_returns)
            updates, new_policy_opt_state = self._policy_opt.update(grads, policy_opt_state)
            new_policy_params = optax.apply_updates(policy_params, updates)
            return (new_policy_params, new_policy_opt_state), None
        
        (final_policy_params, final_policy_opt_state), _ = jax.lax.scan(
            _scan_update,
            (policy_params, policy_opt_state),
            None,
            length=1,
        )

        return final_policy_params, final_policy_opt_state
        
    '''
    
    @partial(jax.jit, static_argnames=("self",))
    def loss_ppo(
        self,
        params,
        obs,
        actions,
        logps,
        mask,
        standardized_returns,
    ):
        '''
        logps_ = self._policy.apply(
            params,
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(actions),
            method=self._policy.logp,
        )
        '''
        pi, _ = self._policy.apply(params, obs)
        logps_ = pi.log_prob(actions)
        
        ratio = jnp.exp(logps_ - jax.lax.stop_gradient(logps))
        
        pg_loss_1 = jnp.multiply(ratio * mask, jax.lax.stop_gradient(standardized_returns))
        pg_loss_2 = jax.lax.stop_gradient(standardized_returns) * jax.lax.clamp(1. - self._config.clip_param, ratio, 1. + self._config.clip_param) * mask
        
        #return -jnp.mean(jnp.minimum(pg_loss_1, pg_loss_2))
        return (-jnp.sum(jnp.minimum(pg_loss_1, pg_loss_2))) / jnp.sum(ratio * mask)
        
    
        
