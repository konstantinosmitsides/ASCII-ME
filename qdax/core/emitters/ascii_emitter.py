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
from qdax.core.neuroevolution.buffers.buffer import QDTransition, QDMCTransition #, PPOTransition
#from qdax.core.neuroevolution.buffers.trajectory_buffer import TrajectoryBuffer
import flashbax as fbx
import chex
from utils import find_magnitude_of_updates, concatenate_params, compute_cosine_similarity

from qdax.core.emitters.emitter import Emitter, EmitterState
from jax import profiler
import os

EPS = 1e-8




@dataclass
class ASCIIConfig:
    """Configuration for the ASCII emitter.
    """
    no_agents: int = 256
    buffer_sample_batch_size: int = 32
    buffer_add_batch_size: int = 256
    no_epochs: int = 16
    learning_rate: float = 3e-4
    discount_rate: float = 0.99
    clip_param: float = 0.2
    std: float = 0.5
    
class ASCIIEmitterState(EmitterState):
    """Contains the trajectory buffer.
    """
    buffer_state: Any
    random_key: RNGKey
    
class ASCIIEmitter_0(Emitter):
    
    def __init__(
        self,
        config: ASCIIConfig,
        policy_net: nn.Module,
        env: QDEnv,
    ) -> None:
        
        self._config = config
        self._policy = policy_net
        self._env = env
        
        #self._policy_opt = optax.chain(
        #    optax.zero_nans(),
        #    optax.clip_by_global_norm(0.5),
        #    optax.adam(learning_rate=self._config.learning_rate),
        #)
        
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
    ) -> Tuple[ASCIIEmitterState, RNGKey]:
        """Initializes the emitter state.
        """
        obs_size = self._env.observation_size
        action_size = self._env.action_size
        descriptor_size = self._env.state_descriptor_length
        
        # Init trajectory buffer

        dummy_transition = QDTransition.init_dummy(
            observation_dim=obs_size,
            action_dim=action_size,
            descriptor_dim=descriptor_size,
        )


        
        buffer_state = self._buffer.init(dummy_transition)
        

        
        random_key, subkey = jax.random.split(random_key)
        emitter_state = ASCIIEmitterState(
            buffer_state=buffer_state,
            random_key=subkey,
        )
        
        return emitter_state, random_key
    
    @partial(jax.jit, static_argnames=("self",))
    def emit(
        self,
        repertoire: Repertoire,
        emitter_state: ASCIIEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Do a step of ASCII emission.
        """
        
        no_agents = self._config.no_agents
        
        random_keys = jax.random.split(random_key, no_agents+2)
        
        
        # sample parents
        parents, returns, random_key, trajectories = repertoire.sample(
            random_key=random_keys[-1],
            num_samples=no_agents,
        )
        
        offsprings_ascii = self.emit_ascii(emitter_state, parents, returns, trajectories, random_keys[:no_agents])
        #jax.debug.breakpoint()
        #new_params = concatenate_params(offsprings_mcpg)
        #mean_new = jnp.mean(new_params)
        #old_params = concatenate_params(parents)
        #mean_old = jnp.mean(old_params)
        #update_magnitudes = find_magnitude_of_updates(new_params, old_params)
        #genotype_differences = jax.tree_util.tree_map(lambda x, y: y - x, parents, offsprings_mcpg)
        #magnitude_of_differences = jax.tree_util.tree_map(jnp.linalg.norm, genotype_differences)

        

        
        #first_parent = jax.tree_util.tree_map(lambda x: x[0], parents)
        #first_offspring = jax.tree_util.tree_map(lambda x: x[0], offsprings_mcpg)
        
        #jax.debug.breakpoint()
        
        #return offsprings_mcpg, {'update_magns_pg' : update_magnitudes}, random_keys[-2]
        return offsprings_ascii, {}, random_keys[-2]
    
    @partial(jax.jit, static_argnames=("self",))
    def emit_ascii(
        self,
        emitter_state: ASCIIEmitterState,
        parents: Genotype,
        returns: Any,
        trajectories: Any,
        random_keys: ArrayTree,
    ) -> Genotype:
        """Emit the offsprings generated through ASCII mutation.
        """
        '''
        mutation_fn = partial(
            self._mutation_function_ascii,
            emitter_state=emitter_state,
        )
        '''
        
        offsprings = jax.vmap(self._mutation_function_ascii, in_axes=(0, 0, 0, None, 0))(parents, returns, trajectories, emitter_state, random_keys)
        
        
        
        return offsprings
    
    @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        emitter_state: ASCIIEmitterState,
        repertoire: Optional[Repertoire],
        genotypes: Optional[Genotype],
        fitnesses: Optional[Fitness],
        descriptors: Optional[Descriptor],
        extra_scores: ExtraScores,
    ) -> ASCIIEmitterState:
        """Update the emitter state.
        """
        
        random_key, subkey = jax.random.split(emitter_state.random_key)
        
        #_, _, random_key, trajectories = repertoire.sample(
        #    random_key=random_key,
        #    num_samples=self._config.no_agents * 2,
        #)
        
        #jax.debug.breakpoint()
        
        #trajectory_samples = jax.tree_util.tree_map(
        #    lambda x: jax.random.choice(subkey, x, shape=(self._config.no_agents,)),
        #    extra_scores["transitions"],
        #)
        
        #assert "transitions" in extra_scores.keys(), "Missing transtitions or wrong key"
        transitions = extra_scores["transitions"]
        new_buffer_state = self._buffer.add(emitter_state.buffer_state, transitions)
        #new_buffer_state = self._buffer.add(new_buffer_state, trajectory_samples)
        new_emitter_state = emitter_state.replace(random_key=random_key, buffer_state=new_buffer_state)
        
        return new_emitter_state
        
    
    # @partial(jax.jit, static_argnames=("self",))
    # def compute_mask(
    #     self,
    #     done,
    # ):
    #     return 1. - jnp.clip(jnp.cumsum(done), a_min=0., a_max=1.)


    @partial(jax.jit, static_argnames=("self",))
    def _mutation_function_ascii(
        self,
        policy_params,
        returns,
        trajectories,
        emitter_state: ASCIIEmitterState,
        random_key: RNGKey,
    ) -> Genotype:
        """Mutation function for ASCII."""

        policy_opt_state = self._policy_opt.init(policy_params)
        
        # Directly sample batch and use necessary components
        batch = self._buffer.sample(emitter_state.buffer_state, random_key)
        trans = batch.experience
        #jax.debug.print("Obs_1 shape : {}", trajectories.obs.shape)
        #jax.debug.print("Obs_2 shape : {}", jnp.squeeze(trans.obs, axis=0).shape)
        scale_returns = compute_cosine_similarity(trajectories.obs, jnp.squeeze(trans.obs, axis=0))
        #jax.debug.print("scale_returns : {}", scale_returns)
        #mask = self.compute_mask(trans.dones)
        standardized_returns = scale_returns * (trans.rewards - returns) 

        
        def scan_train_policy(
            carry: Tuple[ASCIIEmitterState, Genotype, optax.OptState],
            unused: Any,
        ) -> Tuple[Tuple[ASCIIEmitterState, Genotype, optax.OptState], Any]:
            
            policy_params, policy_opt_state = carry
            
            # Train policy with directly used transaction fields
            new_policy_params, new_policy_opt_state = self._train_policy_(
                policy_params,
                policy_opt_state,
                trans.obs,
                trans.actions,
                standardized_returns,
                #trans.logp,
                trans.dones
            )
            return (new_policy_params, new_policy_opt_state), None
            
        (policy_params, policy_opt_state), _ = jax.lax.scan(
            scan_train_policy,
            (policy_params, policy_opt_state),
            None,
            length=self._config.no_epochs,
        )
        
        
        
        return policy_params
        
        
    @partial(jax.jit, static_argnames=("self",))
    def _train_policy_(
        self,
        #emitter_state: MCPGEmitterState,
        policy_params,
        policy_opt_state: optax.OptState,
        obs,
        actions,
        standardized_returns,
        #logps,
        mask
    ) -> Tuple[ASCIIEmitterState, Genotype, optax.OptState]:
        """Train the policy.
        """
        
        grads = jax.grad(self.loss)(policy_params, obs, actions, standardized_returns, mask)
        #jax.debug.print("grads : {}", grads)
        
        
        updates, new_policy_opt_state = self._policy_opt.update(grads, policy_opt_state)
        new_policy_params = optax.apply_updates(policy_params, updates)
        

        return new_policy_params, new_policy_opt_state
    
    
    @partial(jax.jit, static_argnames=("self",))
    def loss(
        self,
        params,
        obs,
        actions,
        #logps,
        standardized_returns,
        mask
    ):

        pi, _ = self._policy.apply(params, obs)
        logps_ = pi.log_prob(actions) 
        log_factor = (jnp.log(self._config.std) + 0.5 * jnp.log(2.0 * jnp.pi))
        #jnp.array([self._config.std]*self._env.action_size)
        
        #ratio = jnp.exp(logps_ - jax.lax.stop_gradient(logps))

        ratio = jnp.exp(logps_ + self._env.action_size * log_factor)  

        pg_loss_1 = jnp.multiply(ratio, jax.lax.stop_gradient(standardized_returns * (1.0 - mask)))
        pg_loss_2 = jax.lax.stop_gradient(standardized_returns * (1.0 - mask)) * jnp.maximum(ratio, 1. - self._config.clip_param)
        #pg_loss_2 = jax.lax.stop_gradient(standardized_returns * (1.0 - mask)) * jax.lax.clamp(1. - self._config.clip_param, ratio, 1. + self._config.clip_param) 
        
        #return (-jnp.sum(jnp.minimum(pg_loss_1, pg_loss_2))) / jnp.sum((jax.lax.stop_gradient(ratio) + EPS)  * (1.0 - mask)) 
        return -jnp.mean(jnp.minimum(pg_loss_1, pg_loss_2))
        
        

class ASCIIEmitter_05(Emitter):
    
    def __init__(
        self,
        config: ASCIIConfig,
        policy_net: nn.Module,
        env: QDEnv,
    ) -> None:
        
        self._config = config
        self._policy = policy_net
        self._env = env
        
        #self._policy_opt = optax.chain(
        #    optax.zero_nans(),
        #    optax.clip_by_global_norm(0.5),
        #    optax.adam(learning_rate=self._config.learning_rate),
        #)
        
        self._policy_opt = optax.adam(
            learning_rate=self._config.learning_rate
        )
        
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=self._env.episode_length,
            min_length_time_axis=self._env.episode_length,
            sample_batch_size=self._config.buffer_sample_batch_size,
            add_batch_size=int(self._config.buffer_add_batch_size*0.5),
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
    ) -> Tuple[ASCIIEmitterState, RNGKey]:
        """Initializes the emitter state.
        """
        obs_size = self._env.observation_size
        action_size = self._env.action_size
        descriptor_size = self._env.state_descriptor_length
        
        # Init trajectory buffer

        dummy_transition = QDTransition.init_dummy(
            observation_dim=obs_size,
            action_dim=action_size,
            descriptor_dim=descriptor_size,
        )


        
        buffer_state = self._buffer.init(dummy_transition)
        

        
        random_key, subkey = jax.random.split(random_key)
        emitter_state = ASCIIEmitterState(
            buffer_state=buffer_state,
            random_key=subkey,
        )
        
        return emitter_state, random_key
    
    @partial(jax.jit, static_argnames=("self",))
    def emit(
        self,
        repertoire: Repertoire,
        emitter_state: ASCIIEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Do a step of ASCII emission.
        """
        
        no_agents = self._config.no_agents
        
        random_keys = jax.random.split(random_key, no_agents+2)
        
        
        # sample parents
        parents, returns, random_key, trajectories = repertoire.sample(
            random_key=random_keys[-1],
            num_samples=no_agents,
        )
        
        offsprings_ascii = self.emit_ascii(emitter_state, parents, returns, trajectories, random_keys[:no_agents])
        #jax.debug.breakpoint()
        #new_params = concatenate_params(offsprings_mcpg)
        #mean_new = jnp.mean(new_params)
        #old_params = concatenate_params(parents)
        #mean_old = jnp.mean(old_params)
        #update_magnitudes = find_magnitude_of_updates(new_params, old_params)
        #genotype_differences = jax.tree_util.tree_map(lambda x, y: y - x, parents, offsprings_mcpg)
        #magnitude_of_differences = jax.tree_util.tree_map(jnp.linalg.norm, genotype_differences)

        

        
        #first_parent = jax.tree_util.tree_map(lambda x: x[0], parents)
        #first_offspring = jax.tree_util.tree_map(lambda x: x[0], offsprings_mcpg)
        
        #jax.debug.breakpoint()
        
        #return offsprings_mcpg, {'update_magns_pg' : update_magnitudes}, random_keys[-2]
        return offsprings_ascii, {}, random_keys[-2]
    
    @partial(jax.jit, static_argnames=("self",))
    def emit_ascii(
        self,
        emitter_state: ASCIIEmitterState,
        parents: Genotype,
        returns: Any,
        trajectories: Any,
        random_keys: ArrayTree,
    ) -> Genotype:
        """Emit the offsprings generated through ASCII mutation.
        """
        '''
        mutation_fn = partial(
            self._mutation_function_ascii,
            emitter_state=emitter_state,
        )
        '''
        
        offsprings = jax.vmap(self._mutation_function_ascii, in_axes=(0, 0, 0, None, 0))(parents, returns, trajectories, emitter_state, random_keys)
        
        
        
        return offsprings
    
    @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        emitter_state: ASCIIEmitterState,
        repertoire: Optional[Repertoire],
        genotypes: Optional[Genotype],
        fitnesses: Optional[Fitness],
        descriptors: Optional[Descriptor],
        extra_scores: ExtraScores,
    ) -> ASCIIEmitterState:
        """Update the emitter state.
        """
        
        random_key, subkey = jax.random.split(emitter_state.random_key)
        
        _, _, random_key, trajectories = repertoire.sample(
            random_key=random_key,
            num_samples=int(self._config.buffer_add_batch_size*0.5),
        )
        
        #jax.debug.breakpoint()
        
        trajectory_samples = jax.tree_util.tree_map(
            lambda x: jax.random.choice(subkey, x, shape=(int(self._config.buffer_add_batch_size*0.5),), replace=False),
            extra_scores["transitions"],
        )
        
        #assert "transitions" in extra_scores.keys(), "Missing transtitions or wrong key"
        #transitions = extra_scores["transitions"]
        new_buffer_state = self._buffer.add(emitter_state.buffer_state, trajectories)
        new_buffer_state = self._buffer.add(new_buffer_state, trajectory_samples)
        new_emitter_state = emitter_state.replace(random_key=random_key, buffer_state=new_buffer_state)
        
        return new_emitter_state
        
    
    @partial(jax.jit, static_argnames=("self",))
    def compute_mask(
        self,
        done,
    ):
        return 1. - jnp.clip(jnp.cumsum(done), a_min=0., a_max=1.)


    @partial(jax.jit, static_argnames=("self",))
    def _mutation_function_ascii(
        self,
        policy_params,
        returns,
        trajectories,
        emitter_state: ASCIIEmitterState,
        random_key: RNGKey,
    ) -> Genotype:
        """Mutation function for ASCII."""

        policy_opt_state = self._policy_opt.init(policy_params)
        
        # Directly sample batch and use necessary components
        batch = self._buffer.sample(emitter_state.buffer_state, random_key)
        trans = batch.experience
        #jax.debug.print("Obs_1 shape : {}", trajectories.obs.shape)
        #jax.debug.print("Obs_2 shape : {}", jnp.squeeze(trans.obs, axis=0).shape)
        scale_returns = compute_cosine_similarity(trajectories.obs, jnp.squeeze(trans.obs, axis=0))
        #jax.debug.print("scale_returns : {}", scale_returns)
        #mask = self.compute_mask(trans.dones)
        standardized_returns = scale_returns * (trans.rewards - returns) 

        
        def scan_train_policy(
            carry: Tuple[ASCIIEmitterState, Genotype, optax.OptState],
            unused: Any,
        ) -> Tuple[Tuple[ASCIIEmitterState, Genotype, optax.OptState], Any]:
            
            policy_params, policy_opt_state = carry
            
            # Train policy with directly used transaction fields
            new_policy_params, new_policy_opt_state = self._train_policy_(
                policy_params,
                policy_opt_state,
                trans.obs,
                trans.actions,
                standardized_returns,
                #trans.logp,
                trans.dones
            )
            return (new_policy_params, new_policy_opt_state), None
            
        (policy_params, policy_opt_state), _ = jax.lax.scan(
            scan_train_policy,
            (policy_params, policy_opt_state),
            None,
            length=self._config.no_epochs,
        )
        
        
        
        return policy_params
        
        
    @partial(jax.jit, static_argnames=("self",))
    def _train_policy_(
        self,
        #emitter_state: MCPGEmitterState,
        policy_params,
        policy_opt_state: optax.OptState,
        obs,
        actions,
        standardized_returns,
        #logps,
        mask
    ) -> Tuple[ASCIIEmitterState, Genotype, optax.OptState]:
        """Train the policy.
        """
        
        grads = jax.grad(self.loss)(policy_params, obs, actions, standardized_returns, mask)
        #jax.debug.print("grads : {}", grads)
        
        
        updates, new_policy_opt_state = self._policy_opt.update(grads, policy_opt_state)
        new_policy_params = optax.apply_updates(policy_params, updates)
        

        return new_policy_params, new_policy_opt_state
    
    
    @partial(jax.jit, static_argnames=("self",))
    def loss(
        self,
        params,
        obs,
        actions,
        #logps,
        standardized_returns,
        mask
    ):

        pi, _ = self._policy.apply(params, obs)
        logps_ = pi.log_prob(actions) 
        log_factor = (jnp.log(self._config.std) + 0.5 * jnp.log(2.0 * jnp.pi))
        #jnp.array([self._config.std]*self._env.action_size)
        
        #ratio = jnp.exp(logps_ - jax.lax.stop_gradient(logps))

        ratio = jnp.exp(logps_ + self._env.action_size * log_factor)  

        pg_loss_1 = jnp.multiply(ratio, jax.lax.stop_gradient(standardized_returns * (1.0 - mask)))
        pg_loss_2 = jax.lax.stop_gradient(standardized_returns * (1.0 - mask)) * jnp.maximum(ratio, 1. - self._config.clip_param)
        #pg_loss_2 = jax.lax.stop_gradient(standardized_returns * (1.0 - mask)) * jax.lax.clamp(1. - self._config.clip_param, ratio, 1. + self._config.clip_param) 
        
        #return (-jnp.sum(jnp.minimum(pg_loss_1, pg_loss_2))) / jnp.sum((jax.lax.stop_gradient(ratio) + EPS)  * (1.0 - mask)) 
        return -jnp.mean(jnp.minimum(pg_loss_1, pg_loss_2))


class ASCIIEmitter_1(Emitter):
    
    def __init__(
        self,
        config: ASCIIConfig,
        policy_net: nn.Module,
        env: QDEnv,
    ) -> None:
        
        self._config = config
        self._policy = policy_net
        self._env = env
        
        #self._policy_opt = optax.chain(
        #    optax.zero_nans(),
        #    optax.clip_by_global_norm(0.5),
        #    optax.adam(learning_rate=self._config.learning_rate),
        #)
        
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
    ) -> Tuple[ASCIIEmitterState, RNGKey]:
        """Initializes the emitter state.
        """
        obs_size = self._env.observation_size
        action_size = self._env.action_size
        descriptor_size = self._env.state_descriptor_length
        
        # Init trajectory buffer

        dummy_transition = QDTransition.init_dummy(
            observation_dim=obs_size,
            action_dim=action_size,
            descriptor_dim=descriptor_size,
        )


        
        buffer_state = self._buffer.init(dummy_transition)
        

        
        random_key, subkey = jax.random.split(random_key)
        emitter_state = ASCIIEmitterState(
            buffer_state=buffer_state,
            random_key=subkey,
        )
        
        return emitter_state, random_key
    
    @partial(jax.jit, static_argnames=("self",))
    def emit(
        self,
        repertoire: Repertoire,
        emitter_state: ASCIIEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Do a step of ASCII emission.
        """
        
        no_agents = self._config.no_agents
        
        random_keys = jax.random.split(random_key, no_agents+2)
        
        
        # sample parents
        parents, returns, random_key, trajectories = repertoire.sample(
            random_key=random_keys[-1],
            num_samples=no_agents,
        )
        
        offsprings_ascii = self.emit_ascii(emitter_state, parents, returns, trajectories, random_keys[:no_agents])
        #jax.debug.breakpoint()
        #new_params = concatenate_params(offsprings_mcpg)
        #mean_new = jnp.mean(new_params)
        #old_params = concatenate_params(parents)
        #mean_old = jnp.mean(old_params)
        #update_magnitudes = find_magnitude_of_updates(new_params, old_params)
        #genotype_differences = jax.tree_util.tree_map(lambda x, y: y - x, parents, offsprings_mcpg)
        #magnitude_of_differences = jax.tree_util.tree_map(jnp.linalg.norm, genotype_differences)

        

        
        #first_parent = jax.tree_util.tree_map(lambda x: x[0], parents)
        #first_offspring = jax.tree_util.tree_map(lambda x: x[0], offsprings_mcpg)
        
        #jax.debug.breakpoint()
        
        #return offsprings_mcpg, {'update_magns_pg' : update_magnitudes}, random_keys[-2]
        return offsprings_ascii, {}, random_keys[-2]
    
    @partial(jax.jit, static_argnames=("self",))
    def emit_ascii(
        self,
        emitter_state: ASCIIEmitterState,
        parents: Genotype,
        returns: Any,
        trajectories: Any,
        random_keys: ArrayTree,
    ) -> Genotype:
        """Emit the offsprings generated through ASCII mutation.
        """
        '''
        mutation_fn = partial(
            self._mutation_function_ascii,
            emitter_state=emitter_state,
        )
        '''
        
        offsprings = jax.vmap(self._mutation_function_ascii, in_axes=(0, 0, 0, None, 0))(parents, returns, trajectories, emitter_state, random_keys)
        
        
        
        return offsprings
    
    @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        emitter_state: ASCIIEmitterState,
        repertoire: Optional[Repertoire],
        genotypes: Optional[Genotype],
        fitnesses: Optional[Fitness],
        descriptors: Optional[Descriptor],
        extra_scores: ExtraScores,
    ) -> ASCIIEmitterState:
        """Update the emitter state.
        """
        
        random_key, subkey = jax.random.split(emitter_state.random_key)
        
        _, _, random_key, trajectories = repertoire.sample(
            random_key=random_key,
            num_samples=int(self._config.buffer_add_batch_size),
        )
        
        #jax.debug.breakpoint()
        
        #trajectory_samples = jax.tree_util.tree_map(
        #    lambda x: jax.random.choice(subkey, x, shape=(int(self._config.buffer_add_batch_size*0.5),)),
        #    extra_scores["transitions"], replace=False
        #)
        
        #assert "transitions" in extra_scores.keys(), "Missing transtitions or wrong key"
        #transitions = extra_scores["transitions"]
        new_buffer_state = self._buffer.add(emitter_state.buffer_state, trajectories)
        #new_buffer_state = self._buffer.add(new_buffer_state, trajectory_samples)
        new_emitter_state = emitter_state.replace(random_key=random_key, buffer_state=new_buffer_state)
        
        return new_emitter_state
        
    
    @partial(jax.jit, static_argnames=("self",))
    def compute_mask(
        self,
        done,
    ):
        return 1. - jnp.clip(jnp.cumsum(done), a_min=0., a_max=1.)


    @partial(jax.jit, static_argnames=("self",))
    def _mutation_function_ascii(
        self,
        policy_params,
        returns,
        trajectories,
        emitter_state: ASCIIEmitterState,
        random_key: RNGKey,
    ) -> Genotype:
        """Mutation function for ASCII."""

        policy_opt_state = self._policy_opt.init(policy_params)
        
        # Directly sample batch and use necessary components
        batch = self._buffer.sample(emitter_state.buffer_state, random_key)
        trans = batch.experience
        #jax.debug.print("Obs_1 shape : {}", trajectories.obs.shape)
        #jax.debug.print("Obs_2 shape : {}", jnp.squeeze(trans.obs, axis=0).shape)
        scale_returns = compute_cosine_similarity(trajectories.obs, jnp.squeeze(trans.obs, axis=0))
        #jax.debug.print("scale_returns : {}", scale_returns)
        #mask = self.compute_mask(trans.dones)
        standardized_returns = scale_returns * (trans.rewards - returns) 

        
        def scan_train_policy(
            carry: Tuple[ASCIIEmitterState, Genotype, optax.OptState],
            unused: Any,
        ) -> Tuple[Tuple[ASCIIEmitterState, Genotype, optax.OptState], Any]:
            
            policy_params, policy_opt_state = carry
            
            # Train policy with directly used transaction fields
            new_policy_params, new_policy_opt_state = self._train_policy_(
                policy_params,
                policy_opt_state,
                trans.obs,
                trans.actions,
                standardized_returns,
                #trans.logp,
                trans.dones
            )
            return (new_policy_params, new_policy_opt_state), None
            
        (policy_params, policy_opt_state), _ = jax.lax.scan(
            scan_train_policy,
            (policy_params, policy_opt_state),
            None,
            length=self._config.no_epochs,
        )
        
        
        
        return policy_params
        
        
    @partial(jax.jit, static_argnames=("self",))
    def _train_policy_(
        self,
        #emitter_state: MCPGEmitterState,
        policy_params,
        policy_opt_state: optax.OptState,
        obs,
        actions,
        standardized_returns,
        #logps,
        mask
    ) -> Tuple[ASCIIEmitterState, Genotype, optax.OptState]:
        """Train the policy.
        """
        
        grads = jax.grad(self.loss)(policy_params, obs, actions, standardized_returns, mask)
        #jax.debug.print("grads : {}", grads)
        
        
        updates, new_policy_opt_state = self._policy_opt.update(grads, policy_opt_state)
        new_policy_params = optax.apply_updates(policy_params, updates)
        

        return new_policy_params, new_policy_opt_state
    
    
    @partial(jax.jit, static_argnames=("self",))
    def loss(
        self,
        params,
        obs,
        actions,
        #logps,
        standardized_returns,
        mask
    ):

        pi, _ = self._policy.apply(params, obs)
        logps_ = pi.log_prob(actions) 
        log_factor = (jnp.log(self._config.std) + 0.5 * jnp.log(2.0 * jnp.pi))
        #jnp.array([self._config.std]*self._env.action_size)
        
        #ratio = jnp.exp(logps_ - jax.lax.stop_gradient(logps))

        ratio = jnp.exp(logps_ + self._env.action_size * log_factor)  

        pg_loss_1 = jnp.multiply(ratio, jax.lax.stop_gradient(standardized_returns * (1.0 - mask)))
        pg_loss_2 = jax.lax.stop_gradient(standardized_returns * (1.0 - mask)) * jnp.maximum(ratio, 1. - self._config.clip_param)
        #pg_loss_2 = jax.lax.stop_gradient(standardized_returns * (1.0 - mask)) * jax.lax.clamp(1. - self._config.clip_param, ratio, 1. + self._config.clip_param) 
        
        #return (-jnp.sum(jnp.minimum(pg_loss_1, pg_loss_2))) / jnp.sum((jax.lax.stop_gradient(ratio) + EPS)  * (1.0 - mask)) 
        return -jnp.mean(jnp.minimum(pg_loss_1, pg_loss_2))
    
    

        
class ASCIIEmitter_0_not(Emitter):
    
    def __init__(
        self,
        config: ASCIIConfig,
        policy_net: nn.Module,
        env: QDEnv,
    ) -> None:
        
        self._config = config
        self._policy = policy_net
        self._env = env
        
        #self._policy_opt = optax.chain(
        #    optax.zero_nans(),
        #    optax.clip_by_global_norm(0.5),
        #    optax.adam(learning_rate=self._config.learning_rate),
        #)
        
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
    ) -> Tuple[ASCIIEmitterState, RNGKey]:
        """Initializes the emitter state.
        """
        obs_size = self._env.observation_size
        action_size = self._env.action_size
        descriptor_size = self._env.state_descriptor_length
        
        # Init trajectory buffer

        dummy_transition = QDMCTransition.init_dummy(
            observation_dim=obs_size,
            action_dim=action_size,
            descriptor_dim=descriptor_size,
        )


        
        buffer_state = self._buffer.init(dummy_transition)
        

        
        random_key, subkey = jax.random.split(random_key)
        emitter_state = ASCIIEmitterState(
            buffer_state=buffer_state,
            random_key=subkey,
        )
        
        return emitter_state, random_key
    
    @partial(jax.jit, static_argnames=("self",))
    def emit(
        self,
        repertoire: Repertoire,
        emitter_state: ASCIIEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Do a step of ASCII emission.
        """
        
        no_agents = self._config.no_agents
        
        random_keys = jax.random.split(random_key, no_agents+2)
        
        
        # sample parents
        parents, returns, random_key, _ = repertoire.sample(
            random_key=random_keys[-1],
            num_samples=no_agents,
        )
        
        offsprings_ascii = self.emit_ascii(emitter_state, parents, returns, random_keys[:no_agents])
        #jax.debug.breakpoint()
        #new_params = concatenate_params(offsprings_mcpg)
        #mean_new = jnp.mean(new_params)
        #old_params = concatenate_params(parents)
        #mean_old = jnp.mean(old_params)
        #update_magnitudes = find_magnitude_of_updates(new_params, old_params)
        #genotype_differences = jax.tree_util.tree_map(lambda x, y: y - x, parents, offsprings_mcpg)
        #magnitude_of_differences = jax.tree_util.tree_map(jnp.linalg.norm, genotype_differences)

        

        
        #first_parent = jax.tree_util.tree_map(lambda x: x[0], parents)
        #first_offspring = jax.tree_util.tree_map(lambda x: x[0], offsprings_mcpg)
        
        #jax.debug.breakpoint()
        
        #return offsprings_mcpg, {'update_magns_pg' : update_magnitudes}, random_keys[-2]
        return offsprings_ascii, {}, random_keys[-2]
    
    @partial(jax.jit, static_argnames=("self",))
    def emit_mcpg(
        self,
        emitter_state: ASCIIEmitterState,
        parents: Genotype,
        returns: Any,
        random_keys: ArrayTree,
    ) -> Genotype:
        """Emit the offsprings generated through ASCII mutation.
        """
        '''
        mutation_fn = partial(
            self._mutation_function_ascii,
            emitter_state=emitter_state,
        )
        '''
        
        offsprings = jax.vmap(self._mutation_function_ascii, in_axes=(0, 0, None, 0))(parents, returns, emitter_state, random_keys)
        
        
        
        return offsprings
    
    @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        emitter_state: ASCIIEmitterState,
        repertoire: Optional[Repertoire],
        genotypes: Optional[Genotype],
        fitnesses: Optional[Fitness],
        descriptors: Optional[Descriptor],
        extra_scores: ExtraScores,
    ) -> ASCIIEmitterState:
        """Update the emitter state.
        """
        
        random_key, subkey = jax.random.split(emitter_state.random_key)
        
        #_, _, random_key, trajectories = repertoire.sample(
        #    random_key=random_key,
        #    num_samples=self._config.no_agents * 2,
        #)
        
        #jax.debug.breakpoint()
        
        #trajectory_samples = jax.tree_util.tree_map(
        #    lambda x: jax.random.choice(subkey, x, shape=(self._config.no_agents,)),
        #    extra_scores["transitions"],
        #)
        
        #assert "transitions" in extra_scores.keys(), "Missing transtitions or wrong key"
        transitions = extra_scores["transitions"]
        new_buffer_state = self._buffer.add(emitter_state.buffer_state, transitions)
        #new_buffer_state = self._buffer.add(new_buffer_state, trajectory_samples)
        new_emitter_state = emitter_state.replace(random_key=random_key, buffer_state=new_buffer_state)
        
        return new_emitter_state
        
    
    @partial(jax.jit, static_argnames=("self",))
    def compute_mask(
        self,
        done,
    ):
        return 1. - jnp.clip(jnp.cumsum(done), a_min=0., a_max=1.)


    @partial(jax.jit, static_argnames=("self",))
    def _mutation_function_ascii(
        self,
        policy_params,
        returns,
        emitter_state: ASCIIEmitterState,
        random_key: RNGKey,
    ) -> Genotype:
        """Mutation function for ASCII."""

        policy_opt_state = self._policy_opt.init(policy_params)
        
        # Directly sample batch and use necessary components
        batch = self._buffer.sample(emitter_state.buffer_state, random_key)
        trans = batch.experience
        #jax.debug.print("Obs_1 shape : {}", trajectories.obs.shape)
        #jax.debug.print("Obs_2 shape : {}", jnp.squeeze(trans.obs, axis=0).shape)
        #scale_returns = compute_cosine_similarity(trajectories.obs, jnp.squeeze(trans.obs, axis=0))
        #jax.debug.print("scale_returns : {}", scale_returns)
        #mask = self.compute_mask(trans.dones)
        standardized_returns = (trans.rewards - returns) 

        
        def scan_train_policy(
            carry: Tuple[ASCIIEmitterState, Genotype, optax.OptState],
            unused: Any,
        ) -> Tuple[Tuple[ASCIIEmitterState, Genotype, optax.OptState], Any]:
            
            policy_params, policy_opt_state = carry
            
            # Train policy with directly used transaction fields
            new_policy_params, new_policy_opt_state = self._train_policy_(
                policy_params,
                policy_opt_state,
                trans.obs,
                trans.actions,
                standardized_returns,
                trans.logp,
                trans.dones
            )
            return (new_policy_params, new_policy_opt_state), None
            
        (policy_params, policy_opt_state), _ = jax.lax.scan(
            scan_train_policy,
            (policy_params, policy_opt_state),
            None,
            length=self._config.no_epochs,
        )
        
        
        
        return policy_params
        
        
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
    ) -> Tuple[ASCIIEmitterState, Genotype, optax.OptState]:
        """Train the policy.
        """
        
        grads = jax.grad(self.loss_ppo)(policy_params, obs, actions, logps, standardized_returns, mask)
        #jax.debug.print("grads : {}", grads)
        
        
        updates, new_policy_opt_state = self._policy_opt.update(grads, policy_opt_state)
        new_policy_params = optax.apply_updates(policy_params, updates)
        

        return new_policy_params, new_policy_opt_state
    
    
    @partial(jax.jit, static_argnames=("self",))
    def loss_ppo(
        self,
        params,
        obs,
        actions,
        logps,
        standardized_returns,
        mask
    ):

        pi, _ = self._policy.apply(params, obs)
        logps_ = pi.log_prob(actions) 
        
        ratio = jnp.exp(logps_ - jax.lax.stop_gradient(logps))
        
        pg_loss_1 = jnp.multiply(ratio, jax.lax.stop_gradient(standardized_returns * (1.0 - mask)))
        pg_loss_2 = jax.lax.stop_gradient(standardized_returns * (1.0 - mask)) * jax.lax.clamp(1. - self._config.clip_param, ratio, 1. + self._config.clip_param) 
        
        #return (-jnp.sum(jnp.minimum(pg_loss_1, pg_loss_2))) / jnp.sum((jax.lax.stop_gradient(ratio) + EPS)  * (1.0 - mask)) 
        return -jnp.mean(jnp.minimum(pg_loss_1, pg_loss_2))



class ASCIIEmitter_1_not(Emitter):
    
    def __init__(
        self,
        config: ASCIIConfig,
        policy_net: nn.Module,
        env: QDEnv,
    ) -> None:
        
        self._config = config
        self._policy = policy_net
        self._env = env
        
        #self._policy_opt = optax.chain(
        #    optax.zero_nans(),
        #    optax.clip_by_global_norm(0.5),
        #    optax.adam(learning_rate=self._config.learning_rate),
        #)
        
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
    ) -> Tuple[ASCIIEmitterState, RNGKey]:
        """Initializes the emitter state.
        """
        obs_size = self._env.observation_size
        action_size = self._env.action_size
        descriptor_size = self._env.state_descriptor_length
        
        # Init trajectory buffer

        dummy_transition = QDMCTransition.init_dummy(
            observation_dim=obs_size,
            action_dim=action_size,
            descriptor_dim=descriptor_size,
        )


        
        buffer_state = self._buffer.init(dummy_transition)
        

        
        random_key, subkey = jax.random.split(random_key)
        emitter_state = ASCIIEmitterState(
            buffer_state=buffer_state,
            random_key=subkey,
        )
        
        return emitter_state, random_key
    
    @partial(jax.jit, static_argnames=("self",))
    def emit(
        self,
        repertoire: Repertoire,
        emitter_state: ASCIIEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Do a step of ASCII emission.
        """
        
        no_agents = self._config.no_agents
        
        random_keys = jax.random.split(random_key, no_agents+2)
        
        
        # sample parents
        parents, returns, random_key, _ = repertoire.sample(
            random_key=random_keys[-1],
            num_samples=no_agents,
        )
        
        offsprings_ascii = self.emit_ascii(emitter_state, parents, returns, random_keys[:no_agents])
        #jax.debug.breakpoint()
        #new_params = concatenate_params(offsprings_mcpg)
        #mean_new = jnp.mean(new_params)
        #old_params = concatenate_params(parents)
        #mean_old = jnp.mean(old_params)
        #update_magnitudes = find_magnitude_of_updates(new_params, old_params)
        #genotype_differences = jax.tree_util.tree_map(lambda x, y: y - x, parents, offsprings_mcpg)
        #magnitude_of_differences = jax.tree_util.tree_map(jnp.linalg.norm, genotype_differences)

        

        
        #first_parent = jax.tree_util.tree_map(lambda x: x[0], parents)
        #first_offspring = jax.tree_util.tree_map(lambda x: x[0], offsprings_mcpg)
        
        #jax.debug.breakpoint()
        
        #return offsprings_mcpg, {'update_magns_pg' : update_magnitudes}, random_keys[-2]
        return offsprings_ascii, {}, random_keys[-2]
    
    @partial(jax.jit, static_argnames=("self",))
    def emit_mcpg(
        self,
        emitter_state: ASCIIEmitterState,
        parents: Genotype,
        returns: Any,
        random_keys: ArrayTree,
    ) -> Genotype:
        """Emit the offsprings generated through ASCII mutation.
        """
        '''
        mutation_fn = partial(
            self._mutation_function_ascii,
            emitter_state=emitter_state,
        )
        '''
        
        offsprings = jax.vmap(self._mutation_function_ascii, in_axes=(0, 0, None, 0))(parents, returns, emitter_state, random_keys)
        
        
        
        return offsprings
    
    @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        emitter_state: ASCIIEmitterState,
        repertoire: Optional[Repertoire],
        genotypes: Optional[Genotype],
        fitnesses: Optional[Fitness],
        descriptors: Optional[Descriptor],
        extra_scores: ExtraScores,
    ) -> ASCIIEmitterState:
        """Update the emitter state.
        """
        
        random_key, subkey = jax.random.split(emitter_state.random_key)
        
        _, _, random_key, trajectories = repertoire.sample(
            random_key=random_key,
            num_samples=int(self._config.buffer_add_batch_size),
        )
        
        #jax.debug.breakpoint()
        
        #trajectory_samples = jax.tree_util.tree_map(
        #    lambda x: jax.random.choice(subkey, x, shape=(int(self._config.buffer_add_batch_size*0.5),)),
        #    extra_scores["transitions"], replace=False
        #)
        
        #assert "transitions" in extra_scores.keys(), "Missing transtitions or wrong key"
        #transitions = extra_scores["transitions"]
        new_buffer_state = self._buffer.add(emitter_state.buffer_state, trajectories)
        #new_buffer_state = self._buffer.add(new_buffer_state, trajectory_samples)
        new_emitter_state = emitter_state.replace(random_key=random_key, buffer_state=new_buffer_state)
        
        return new_emitter_state
        
    
    @partial(jax.jit, static_argnames=("self",))
    def compute_mask(
        self,
        done,
    ):
        return 1. - jnp.clip(jnp.cumsum(done), a_min=0., a_max=1.)


    @partial(jax.jit, static_argnames=("self",))
    def _mutation_function_ascii(
        self,
        policy_params,
        returns,
        emitter_state: ASCIIEmitterState,
        random_key: RNGKey,
    ) -> Genotype:
        """Mutation function for ASCII."""

        policy_opt_state = self._policy_opt.init(policy_params)
        
        # Directly sample batch and use necessary components
        batch = self._buffer.sample(emitter_state.buffer_state, random_key)
        trans = batch.experience
        #jax.debug.print("Obs_1 shape : {}", trajectories.obs.shape)
        #jax.debug.print("Obs_2 shape : {}", jnp.squeeze(trans.obs, axis=0).shape)
        #scale_returns = compute_cosine_similarity(trajectories.obs, jnp.squeeze(trans.obs, axis=0))
        #jax.debug.print("scale_returns : {}", scale_returns)
        #mask = self.compute_mask(trans.dones)
        standardized_returns = (trans.rewards - returns) 

        
        def scan_train_policy(
            carry: Tuple[ASCIIEmitterState, Genotype, optax.OptState],
            unused: Any,
        ) -> Tuple[Tuple[ASCIIEmitterState, Genotype, optax.OptState], Any]:
            
            policy_params, policy_opt_state = carry
            
            # Train policy with directly used transaction fields
            new_policy_params, new_policy_opt_state = self._train_policy_(
                policy_params,
                policy_opt_state,
                trans.obs,
                trans.actions,
                standardized_returns,
                trans.logp,
                trans.dones
            )
            return (new_policy_params, new_policy_opt_state), None
            
        (policy_params, policy_opt_state), _ = jax.lax.scan(
            scan_train_policy,
            (policy_params, policy_opt_state),
            None,
            length=self._config.no_epochs,
        )
        
        
        
        return policy_params
        
        
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
    ) -> Tuple[ASCIIEmitterState, Genotype, optax.OptState]:
        """Train the policy.
        """
        
        grads = jax.grad(self.loss_ppo)(policy_params, obs, actions, logps, standardized_returns, mask)
        #jax.debug.print("grads : {}", grads)
        
        
        updates, new_policy_opt_state = self._policy_opt.update(grads, policy_opt_state)
        new_policy_params = optax.apply_updates(policy_params, updates)
        

        return new_policy_params, new_policy_opt_state
    
    
    @partial(jax.jit, static_argnames=("self",))
    def loss_ppo(
        self,
        params,
        obs,
        actions,
        logps,
        standardized_returns,
        mask
    ):

        pi, _ = self._policy.apply(params, obs)
        logps_ = pi.log_prob(actions) 
        
        ratio = jnp.exp(logps_ - jax.lax.stop_gradient(logps))
        
        pg_loss_1 = jnp.multiply(ratio, jax.lax.stop_gradient(standardized_returns * (1.0 - mask)))
        pg_loss_2 = jax.lax.stop_gradient(standardized_returns * (1.0 - mask)) * jax.lax.clamp(1. - self._config.clip_param, ratio, 1. + self._config.clip_param) 
        
        #return (-jnp.sum(jnp.minimum(pg_loss_1, pg_loss_2))) / jnp.sum((jax.lax.stop_gradient(ratio) + EPS)  * (1.0 - mask)) 
        return -jnp.mean(jnp.minimum(pg_loss_1, pg_loss_2))
    
    
    
class ASCIIEmitter_0_exps(Emitter):
    
    def __init__(
        self,
        config: ASCIIConfig,
        policy_net: nn.Module,
        env: QDEnv,
    ) -> None:
        
        self._config = config
        self._policy = policy_net
        self._env = env
        
        #self._policy_opt = optax.chain(
        #    optax.zero_nans(),
        #    optax.clip_by_global_norm(0.5),
        #    optax.adam(learning_rate=self._config.learning_rate),
        #)
        
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
    ) -> Tuple[ASCIIEmitterState, RNGKey]:
        """Initializes the emitter state.
        """
        obs_size = self._env.observation_size
        action_size = self._env.action_size
        descriptor_size = self._env.state_descriptor_length
        
        # Init trajectory buffer
        
        #jax.debug.print("LAAAAAA : {}", obs_size)

        dummy_transition = QDMCTransition.init_dummy(
            observation_dim=obs_size,
            action_dim=action_size,
            descriptor_dim=descriptor_size,
        )


        
        buffer_state = self._buffer.init(dummy_transition)
        

        
        random_key, subkey = jax.random.split(random_key)
        emitter_state = ASCIIEmitterState(
            buffer_state=buffer_state,
            random_key=subkey,
        )
        
        return emitter_state, random_key
    
    @partial(jax.jit, static_argnames=("self",))
    def emit(
        self,
        repertoire: Repertoire,
        emitter_state: ASCIIEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Do a step of ASCII emission.
        """
        
        no_agents = self._config.no_agents
        
        random_keys = jax.random.split(random_key, no_agents+2)
        
        
        # sample parents
        parents, returns, random_key, trajectories = repertoire.sample(
            random_key=random_keys[-1],
            num_samples=no_agents,
        )
        
        offsprings_ascii = self.emit_ascii(emitter_state, parents, returns, trajectories, random_keys[:no_agents])
        #jax.debug.breakpoint()
        #new_params = concatenate_params(offsprings_mcpg)
        #mean_new = jnp.mean(new_params)
        #old_params = concatenate_params(parents)
        #mean_old = jnp.mean(old_params)
        #update_magnitudes = find_magnitude_of_updates(new_params, old_params)
        #genotype_differences = jax.tree_util.tree_map(lambda x, y: y - x, parents, offsprings_mcpg)
        #magnitude_of_differences = jax.tree_util.tree_map(jnp.linalg.norm, genotype_differences)

        

        
        #first_parent = jax.tree_util.tree_map(lambda x: x[0], parents)
        #first_offspring = jax.tree_util.tree_map(lambda x: x[0], offsprings_mcpg)
        
        #jax.debug.breakpoint()
        
        #return offsprings_mcpg, {'update_magns_pg' : update_magnitudes}, random_keys[-2]
        return offsprings_ascii, {}, random_keys[-2]
    
    @partial(jax.jit, static_argnames=("self",))
    def emit_mcpg(
        self,
        emitter_state: ASCIIEmitterState,
        parents: Genotype,
        returns: Any,
        trajectories: Any,
        random_keys: ArrayTree,
    ) -> Genotype:
        """Emit the offsprings generated through ASCII mutation.
        """
        '''
        mutation_fn = partial(
            self._mutation_function_ascii,
            emitter_state=emitter_state,
        )
        '''
        
        offsprings = jax.vmap(self._mutation_function_ascii, in_axes=(0, 0, 0, None, 0))(parents, returns, trajectories, emitter_state, random_keys)
        
        
        
        return offsprings
    
    @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        emitter_state: ASCIIEmitterState,
        repertoire: Optional[Repertoire],
        genotypes: Optional[Genotype],
        fitnesses: Optional[Fitness],
        descriptors: Optional[Descriptor],
        extra_scores: ExtraScores,
    ) -> ASCIIEmitterState:
        """Update the emitter state.
        """
        
        random_key, subkey = jax.random.split(emitter_state.random_key)
        
        #_, _, random_key, trajectories = repertoire.sample(
        #    random_key=random_key,
        #    num_samples=self._config.no_agents * 2,
        #)
        
        #jax.debug.breakpoint()
        
        #trajectory_samples = jax.tree_util.tree_map(
        #    lambda x: jax.random.choice(subkey, x, shape=(self._config.no_agents,)),
        #    extra_scores["transitions"],
        #)
        
        #assert "transitions" in extra_scores.keys(), "Missing transtitions or wrong key"
        transitions = extra_scores["transitions"]
        new_buffer_state = self._buffer.add(emitter_state.buffer_state, transitions)
        #new_buffer_state = self._buffer.add(new_buffer_state, trajectory_samples)
        new_emitter_state = emitter_state.replace(random_key=random_key, buffer_state=new_buffer_state)
        
        return new_emitter_state
        
    
    @partial(jax.jit, static_argnames=("self",))
    def compute_mask(
        self,
        done,
    ):
        return 1. - jnp.clip(jnp.cumsum(done), a_min=0., a_max=1.)
    
    # @partial(jax.jit, static_argnames=("self",))
    # def _mutation_function_mcpg_(
    #     self,
    #     policy_params,
    #     returns,
    #     trajectories,
    #     emitter_state: MCPGEmitterState,
    #     random_key: RNGKey,
    # ) -> Genotype:
        
    #     def scan_mutation(
    #         carry: Tuple[Genotype, RNGKey],
    #         unused: Any,
    #     ) -> Tuple[Tuple[Genotype, RNGKey], None]:
    #         policy_params, random_key = carry

    #         # Split the random key
    #         random_key, subkey = jax.random.split(random_key)

    #         # Apply the mutation function
    #         new_policy_params = self._mutation_function_mcpg(
    #             policy_params,
    #             returns,
    #             trajectories,
    #             emitter_state,
    #             subkey,
    #         )

    #         return (new_policy_params, random_key), None

    #     # Perform the scan over n iterations
    #     (policy_params, random_key), _ = jax.lax.scan(
    #         scan_mutation,
    #         init=(policy_params, random_key),
    #         xs=None,
    #         length=4,
    #     )

    #     return policy_params


    @partial(jax.jit, static_argnames=("self",))
    def _mutation_function_ascii(
        self,
        policy_params,
        returns,
        trajectories,
        emitter_state: ASCIIEmitterState,
        random_key: RNGKey,
    ) -> Genotype:
        """Mutation function for ASCII."""

        policy_opt_state = self._policy_opt.init(policy_params)
        
        # Directly sample batch and use necessary components
        batch = self._buffer.sample(emitter_state.buffer_state, random_key)
        trans = batch.experience
        #jax.debug.print("Obs_1 shape : {}", trajectories.obs.shape)
        #jax.debug.print("Obs_2 shape : {}", jnp.squeeze(trans.obs, axis=0).shape)
        scale_returns = compute_cosine_similarity(trajectories.obs, jnp.squeeze(trans.obs, axis=0))
        #jax.debug.print("scale_returns : {}", scale_returns)
        #mask = self.compute_mask(trans.dones)
        standardized_returns = scale_returns * (trans.rewards - returns) 

        
        def scan_train_policy(
            carry: Tuple[ASCIIEmitterState, Genotype, optax.OptState],
            unused: Any,
        ) -> Tuple[Tuple[ASCIIEmitterState, Genotype, optax.OptState], Any]:
            
            policy_params, policy_opt_state = carry
            
            # Train policy with directly used transaction fields
            new_policy_params, new_policy_opt_state = self._train_policy_(
                policy_params,
                policy_opt_state,
                trans.obs,
                trans.actions,
                standardized_returns,
                trans.logp,
                trans.dones
            )
            return (new_policy_params, new_policy_opt_state), None
            
        (policy_params, policy_opt_state), _ = jax.lax.scan(
            scan_train_policy,
            (policy_params, policy_opt_state),
            None,
            length=self._config.no_epochs,
        )
        
        
        
        return policy_params
        
        
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
    ) -> Tuple[ASCIIEmitterState, Genotype, optax.OptState]:
        """Train the policy.
        """
        
        grads = jax.grad(self.loss_ppo)(policy_params, obs, actions, logps, standardized_returns, mask)
        #jax.debug.print("grads : {}", grads)
        
        
        updates, new_policy_opt_state = self._policy_opt.update(grads, policy_opt_state)
        new_policy_params = optax.apply_updates(policy_params, updates)
        

        return new_policy_params, new_policy_opt_state
    
    
    @partial(jax.jit, static_argnames=("self",))
    def loss_ppo(
        self,
        params,
        obs,
        actions,
        logps,
        standardized_returns,
        mask
    ):

        pi, _ = self._policy.apply(params, obs)
        logps_ = pi.log_prob(actions) 
        
        ratio = jnp.exp(logps_ - jax.lax.stop_gradient(logps))
        
        pg_loss_1 = jnp.multiply(ratio, jax.lax.stop_gradient(standardized_returns * (1.0 - mask)))
        #pg_loss_2 = jax.lax.stop_gradient(standardized_returns * (1.0 - mask)) * jax.lax.clamp(1. - self._config.clip_param, ratio, 1. + self._config.clip_param) 
        
        #return (-jnp.sum(jnp.minimum(pg_loss_1, pg_loss_2))) / jnp.sum((jax.lax.stop_gradient(ratio) + EPS)  * (1.0 - mask)) 
        return -jnp.mean(pg_loss_1)