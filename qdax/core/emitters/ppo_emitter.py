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
from qdax.core.neuroevolution.buffers.buffer import QDTransition, QDMCTransition, PPOTransition
#from qdax.core.neuroevolution.buffers.trajectory_buffer import TrajectoryBuffer
import flashbax as fbx
import chex
from rein_related import *
from flax.training.train_state import TrainState

from qdax.core.emitters.emitter import Emitter, EmitterState
from jax import profiler
import os

EPS = 1e-8

#profiler_dir = "Memory_Investigation"
#os.makedirs(profiler_dir, exist_ok=True)


@dataclass
class PPOConfig:
    """Configuration for the PPO emitter.
    """
    no_agents: int = 256
    buffer_sample_batch_size: int = 32
    buffer_add_batch_size: int = 256
    #batch_size: int = 1000*256
    #mini_batch_size: int = 1000*256
    no_epochs: int = 16
    learning_rate: float = 3e-4
    discount_rate: float = 0.99
    adam_optimizer: bool = True
    #buffer_size: int = 256000
    num_minibatches: int = 32
    vf_coef: float = 0.5
    clip_param: float = 0.2
    max_grad_norm: float = 0.5
    
class PPOEmitterState(EmitterState):
    """Contains the trajectory buffer.
    """
    #buffer: Any
    buffer_state: Any
    random_key: RNGKey
    
class PPOEmitter(Emitter):
    
    def __init__(
        self,
        config: PPOConfig,
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
    ) -> Tuple[PPOEmitterState, RNGKey]:
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
        emitter_state = PPOEmitterState(
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
        emitter_state: PPOEmitterState,
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
        emitter_state: PPOEmitterState,
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
        emitter_state: PPOEmitterState,
        repertoire: Optional[Repertoire],
        genotypes: Optional[Genotype],
        fitnesses: Optional[Fitness],
        descriptors: Optional[Descriptor],
        extra_scores: ExtraScores,
    ) -> PPOEmitterState:
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
    def compute_logps(self, policy_params, obs, actions):
        def compute_logp(single_obs, single_action):
            # Correctly handle operations on single_obs and single_action
            # Ensure no inappropriate method calls like .items() are made
            return self._policy.apply(policy_params, single_obs, single_action, method=self._policy.logp)

        # Use jax.vmap to apply compute_logp across batches of obs and actions
        return jax.vmap(compute_logp, in_axes=(0, 0))(obs, actions)
    
    
        
    @partial(jax.jit, static_argnames=("self",))
    def _mutation_function_mcpg(
        self,
        policy_params,
        emitter_state: PPOEmitterState,
        random_key: RNGKey,
    ) -> Genotype:
        """Mutation function for MCPG."""
        
        tx = optax.chain(
            optax.clip_by_global_norm(self._config.max_grad_norm),
            optax.adam(
                learning_rate=self._config.learning_rate,
                eps=1e-5,
            ),
        )        
        
        train_state = TrainState.create(
            apply_fn=self._policy.apply,
            params=policy_params,
            tx=tx,
        )
        
        # Directly sample batch and use necessary components
        batch = self._buffer.sample(emitter_state.buffer_state, random_key)
        traj_batch = batch.experience
        #trans = trans.ravel().reshape(1, trans.shape[0]*trans.shape[1])
        #mask = jax.vmap(self.compute_mask, in_axes=0)(trans.dones)
        #standardized_returns = self.get_standardized_return(trans.rewards, mask)
        
        def _update_epoch(update_state, unused):
            def _update_minibatch(train_state, batch_info):
                traj_batch = batch_info
                
                def _loss_fn(params, traj_batch):
                    pi, value = self._policy.apply(params, traj_batch.obs)
                    logp_ = pi.log_prob(traj_batch.actions)
                    
                    # see if you will clip the values
                    value_losses = jnp.square(value - traj_batch.target)
                    value_loss = 0.5 * jnp.mean(value_losses)
                    
                    ratio = jnp.exp(logp_ - traj_batch.logp)
                    gae = (traj_batch.val_adv - jnp.mean(traj_batch.val_adv)) / (jnp.std(traj_batch.val_adv) + EPS)
                    loss_actor1 = ratio * gae
                    loss_actor2 = jnp.clip(ratio, 1 - self._config.clip_param, 1 + self._config.clip_param) * gae
                    loss_actor = -jnp.mean(jnp.minimum(loss_actor1, loss_actor2))
                    
                    total_loss = loss_actor + self._config.vf_coef * value_loss
                    
                    return total_loss, (value_loss, loss_actor)
                
                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grad = grad_fn(
                    train_state.params,
                    traj_batch,
                )
                train_state = train_state.apply_gradients(grads=grad)
                return train_state, total_loss
            
            train_state, traj_batch, rng = update_state
            rng, _rng = jax.random.split(rng)
            batch_size = traj_batch.rewards.shape[0] * traj_batch.rewards.shape[1]
            #jax.debug.print("Batch size: {}", batch_size)
            
            permutation = jax.random.permutation(_rng, batch_size)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), traj_batch
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch   
            )
            
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [self._config.num_minibatches, -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )
            
            
            train_state, total_loss = jax.lax.scan(
                _update_minibatch,
                train_state,
                minibatches,
            )
            
            update_state = (train_state, traj_batch, rng)
            return update_state, total_loss
        
        update_state = (train_state, traj_batch, random_key)
        update_state, total_loss = jax.lax.scan(
            _update_epoch,
            update_state,
            None,
            length=self._config.no_epochs,
        )
        
        train_state = update_state[0]
        rng = update_state[-1]
        
        #runner_state = (train_state, rng)
        return train_state.params
        
            

    

        
    
        
