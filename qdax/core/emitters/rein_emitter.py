from dataclasses import dataclass
from functools import partial
from math import floor 
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import optax
from chex import ArrayTree
from qdax.core.containers.repertoire import Repertoire
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey

from qdax.core.emitters.es_novelty_archives import (
    EmptyNoveltyArchive,
    NoveltyArchive,
    ParallelNoveltyArchive,
    RepertoireNoveltyArchive,
    SequentialNoveltyArchive,
    SequentialScanNoveltyArchive,
)
from qdax.core.emitters.emitter import Emitter, EmitterState

@dataclass
class REINNaiveConfig:
    """Configuration for the REINNaive emitter.
    
    Args:
        rollout_number: num of rollouts for gradient estimate
        sample_sigma: std to sample the samples for gradient estimate  (IS THIS PARAMETER SPACE EXPLORATION?)
        sample_mirror: if True, use mirroring sampling
        sample_rank_norm: if True, use normalisation
        
        num_generations_sample: frequency of archive-sampling
        
        adam_optimizer: if True, use ADAM, if False, use SGD
        learning_rate: obvious
        l2_coeffivient: coefficient for regularisation
        
        novelty_nearest_neighbors: num of nearest neigbors for novelty computation
        use_novelty_archive: if True, use novelty archive for novelty (default is to use the content of the reperoire)
        use_novelty_fifo: if True, use fifo archive for novelty (default is to use the content of the repertoire)
        fifo_size: size of the novelty fifo bugger if used
        
        proprtion_explore: proportion of explore
    """
    
    sample_number: int = 10
    sample_sigma: float = 0.02
    sample_mirror: bool = True
    sample_rank_norm: bool = True
    
    num_generations_sample: int = 10
    
    adam_optimizer: bool = True
    learning_rate: float = 0.01
    l2_coefficient: float = False
    
    novelty_nearest_neighbors: int = 10
    use_novelty_archive: bool = False
    use_novelty_fifo: bool = False
    fifo_size: int = 100000
    
    proportion_explore: float = 0.5
    