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




@dataclass
class MCPGConfig:
    """Configuration for the REINaive emitter.
    """
    no_agents: int = 256
    buffer_sample_batch_size: int = 32
    buffer_add_batch_size: int = 256
    no_epochs: int = 16
    learning_rate: float = 3e-4
    discount_rate: float = 0.99
    clip_param: float = 0.2
    
class MCPGEmitterState(EmitterState):
    """Contains the trajectory buffer.
    """
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

        dummy_transition = QDMCTransition.init_dummy(
            observation_dim=obs_size,
            action_dim=action_size,
            descriptor_dim=descriptor_size,
        )


        
        buffer_state = self._buffer.init(dummy_transition)
        

        
        random_key, subkey = jax.random.split(random_key)
        emitter_state = MCPGEmitterState(
            buffer_state=buffer_state,
            random_key=subkey,
        )
        
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
        parents, returns, random_key = repertoire.sample(
            random_key=random_keys[-1],
            num_samples=no_agents,
        )
        
        offsprings_mcpg = self.emit_mcpg(emitter_state, parents, returns, random_keys[:no_agents])
        
        return offsprings_mcpg, {}, random_keys[-2]
    
    @partial(jax.jit, static_argnames=("self",))
    def emit_mcpg(
        self,
        emitter_state: MCPGEmitterState,
        parents: Genotype,
        returns: Any,
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
        
        offsprings = jax.vmap(self._mutation_function_mcpg, in_axes=(0, 0, None, 0))(parents, returns, emitter_state, random_keys)
        
        
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
        
    
    @partial(jax.jit, static_argnames=("self",))
    def compute_mask(
        self,
        done,
    ):
        return 1. - jnp.clip(jnp.cumsum(done), a_min=0., a_max=1.)


    @partial(jax.jit, static_argnames=("self",))
    def _mutation_function_mcpg(
        self,
        policy_params,
        returns,
        emitter_state: MCPGEmitterState,
        random_key: RNGKey,
    ) -> Genotype:
        """Mutation function for MCPG."""

        policy_opt_state = self._policy_opt.init(policy_params)
        
        # Directly sample batch and use necessary components
        batch = self._buffer.sample(emitter_state.buffer_state, random_key)
        trans = batch.experience

        
        standardized_returns = (trans.rewards - returns) 

        
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
                #mask
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
        #mask
    ) -> Tuple[MCPGEmitterState, Genotype, optax.OptState]:
        """Train the policy.
        """
        
        grads = jax.grad(self.loss_ppo)(policy_params, obs, actions, logps, standardized_returns)
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
        #mask,
        standardized_returns,
    ):

        pi, _ = self._policy.apply(params, obs)
        logps_ = pi.log_prob(actions)
        
        return -jnp.mean(jnp.multiply(logps_, jax.lax.stop_gradient(standardized_returns)))
        
        
    
        
