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
from qdax.core.neuroevolution.buffers.buffer import QDMCTransition, ReplayBuffer

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
    grad_steps: int = 16
    learning_rate: float = 3e-4
    #discount_rate: float = 0.99
    buffer_size: int = 512000
    clip_param: float = 0.2
    max_grad_norm: float = 0.5
    
class MCPGEmitterState(EmitterState):
    """Contains the trajectory buffer.
    """
    #buffer: Any
    random_key: RNGKey
    replay_buffer: ReplayBuffer
    
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
        '''
        self._policy_opt = optax.adam(
            learning_rate=self._config.learning_rate
        )
        '''
        self._policy_opt = optax.chain(
            optax.clip_by_global_norm(self._config.max_grad_norm),
            optax.adam(learning_rate=self._config.learning_rate)
        )
        
 
        
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
        
        # Initialize replay buffer
        dummy_transition = QDMCTransition.init_dummy(
            observation_dim=obs_size,
            action_dim=action_size,
            descriptor_dim=descriptor_size,
        )  
        
        replay_buffer = ReplayBuffer.init(
            buffer_size=self._config.buffer_size, transition=dummy_transition
        )


        assert "transitions" in extra_scores.keys(), "Missing transitions or wrong key"
        transitions = extra_scores["transitions"]

        # add transitions in the replay buffer
        #jax.debug.print("transitions shape: {}", transitions.obs.shape)
        replay_buffer = replay_buffer.insert(transitions)
        
        # FLATTEN TRANSITIONS
        
        random_key, subkey = jax.random.split(random_key)
        emitter_state = MCPGEmitterState(
            random_key=subkey,
            replay_buffer=replay_buffer,
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
        parents, random_key = repertoire.sample(
            random_key=random_keys[-1],
            num_samples=no_agents,
        )
        
        offsprings_mcpg = self.emit_mcpg(emitter_state, parents, random_keys[:no_agents])
        
        return offsprings_mcpg, {}, random_keys[-2]
    
    @partial(jax.jit, static_argnames=("self",))
    def emit_mcpg(
        self,
        emitter_state: MCPGEmitterState,
        parents: Genotype,
        random_keys: ArrayTree,
    ) -> Genotype:
        

        #flattened_transitions = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1, *x.shape[2:])), emitter_state.transitions)


        
        offsprings = jax.vmap(self._mutation_function_mcpg, in_axes=(0, None, 0))(parents, emitter_state, random_keys)
        
        
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
        
        replay_buffer = emitter_state.replay_buffer.insert(transitions)
        new_emitter_state = emitter_state.replace(random_key=random_key, replay_buffer=replay_buffer)
        
        return new_emitter_state
        

    
    

        
    @partial(jax.jit, static_argnames=("self",))
    def _mutation_function_mcpg(
        self,
        policy_params,
        emitter_state: MCPGEmitterState,
        random_key: RNGKey,
    ) -> Genotype:
        """Mutation function for MCPG."""

        policy_opt_state = self._policy_opt.init(policy_params)
        
        

        
        def scan_train_policy(
            carry: Tuple[MCPGEmitterState, Genotype, optax.OptState],
            unused: Any,
        ) -> Tuple[Tuple[MCPGEmitterState, Genotype, optax.OptState], Any]:
            
            
            
            policy_params, policy_opt_state, random_key = carry
            random_key, subkey= jax.random.split(random_key)
            
            # Train policy with directly used transaction fields
            new_policy_params, new_policy_opt_state, random_key = self._train_policy_(
                emitter_state,
                policy_params,
                policy_opt_state,
                subkey,
            )
            
            return (new_policy_params, new_policy_opt_state, random_key), None
            
        (policy_params, policy_opt_state, random_key), _ = jax.lax.scan(
            scan_train_policy,
            (policy_params, policy_opt_state, random_key),
            None,
            length=self._config.grad_steps,
        )
        
        return policy_params
        
        

    
    @partial(jax.jit, static_argnames=("self",))
    def _train_policy_(
        self,
        emitter_state: MCPGEmitterState,
        policy_params,
        policy_opt_state: optax.OptState,
        random_key: RNGKey,
    ) -> Tuple[MCPGEmitterState, Genotype, optax.OptState]:
        """Train the policy.s
        """
        
        #random_key = emitter_state.random_key
        replay_buffer = emitter_state.replay_buffer
        trans, random_key = replay_buffer.sample(random_key, sample_size=self._config.buffer_sample_batch_size)
        
        
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
        '''
        
        policy_opt_state, policy_params = self._update_policy(
            policy_params,
            policy_opt_state,
            trans=trans,
        )
        
        
        
        #new_emitter_state = emitter_state.replace(random_key=random_key, replay_buffer=replay_buffer)
        
        return policy_params, policy_opt_state, random_key
    
    
    @partial(jax.jit, static_argnames=("self",))
    def _update_policy(
        self,
        policy_params,
        policy_opt_state,
        trans,
    ):
        
        grads = jax.grad(self.loss_ppo)(policy_params, trans.obs, trans.actions, trans.logp, trans.rewards, trans.dones)
        updates, new_policy_opt_state = self._policy_opt.update(grads, policy_opt_state)
        new_policy_params = optax.apply_updates(policy_params, updates)
        
        return new_policy_opt_state, new_policy_params
    

    
    @partial(jax.jit, static_argnames=("self",))
    def loss_ppo(
        self,
        params,
        obs,
        actions,
        logps,
        standardized_returns,
        mask,
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
        #jax.debug.print("mean ratio: {}", jnp.mean(ratio))
        #jax.debug.print("mean standardized_returns: {}", jnp.mean(standardized_returns))
        #print(ratio)
        
        pg_loss_1 = jnp.multiply(ratio, jax.lax.stop_gradient(standardized_returns))
        pg_loss_2 = jax.lax.stop_gradient(standardized_returns) * jax.lax.clamp(1. - self._config.clip_param, ratio, 1. + self._config.clip_param) 
        #jax.debug.print("pg_loss_1: {}", jnp.mean(pg_loss_1))
        #jax.debug.print("pg_loss_2: {}", jnp.mean(pg_loss_2))
        #jax.debug.print("denominator: {}", jnp.sum(ratio*(1-mask)))
        
        #return -jnp.mean(jnp.minimum(pg_loss_1, pg_loss_2))
        return (-jnp.sum(jnp.minimum(pg_loss_1, pg_loss_2))) / jnp.sum(ratio*(1-mask))
        
    
        
