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
class MCPGConfig:
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
    no_agents: int = 256
    batch_size: int = 1000*256
    mini_batch_size: int = 1000*256
    no_epochs: 16
    learning_rate: float = 3e-4
    discount_rate: float = 0.99
    adam_optimizer: bool = True
    buffer_size: int = 256000
    clip_param: float = 0.2
    
class MCPGEmitterState(EmitterState):
    """Containes the trajectory buffer.
    """
    buffer: TrajectoryBuffer
    random_ket: RNGKey
    
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
        
    @property
    def batch_size(self) -> int:
        """
        Returns:
            int: the batch size emitted by the emitter.
        """
        return self._config.batch_size
    
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
        dummy_transition = QDTransition.init_dummy(
            observation_dim=obs_size,
            action_dim=action_size,
            descriptor_dim=descriptor_size,
        )
        
        buffer = TrajectoryBuffer.init(
            buffer_size=self._config.buffer_size,
            transition=dummy_transition,
            env_batch_size=self._config.batch_size*2,
            episode_length=self._env.episode_length,
        )
        
        random_key, subkey = jax.random.split(random_key)
        emitter_state = MCPGEmitterState(
            buffer=buffer,
            random_key=subkey,
        )
        
        return emitter_state, random_key
    
