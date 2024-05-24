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
    
    rollout_number: int = 10
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
    
    proportion_explore: float = 0.0


class REINaiveEmitterState(EmitterState):
    """Emitter State for the REINaive emitter.
    
    Args:
        initial_optimizer_state: stored to re-initialize when sampling new parent
        optimizer_state: current optimizer state
        offspring: offspring generated through gradient estimate
        generation_count: generation counter used to update the novelty archive
        novelty_archive: used to compute novelty for explore
        random_key: key to handle stochastic operations
    """
    
    initial_optimizer_state: optax.OptState
    optimizer_states: ArrayTree
    offspring: Genotype
    generation_count: int
    novelty_archive: NoveltyArchive
    random_key: RNGKey
    
    
class REINAiveEmitter(Emitter):
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
        batch_size: int,
        scoring_fn: Callable[
            [Genotype, RNGKey],
            Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        num_descriptors: int,
        scan_batch_size: int = 1,
        scan_novelty: int = 1,
        total_generations: int = 1,
        num_centroids: int = 1,
    ) -> None:
        """Initialize the emitter.
        
        Args:
            config
            batch_size: number of individuals generated per generation.
            scoring_fn: used to evaluate the rollouts for the gradfient estimate.
            num_descriptors: dimensions of the descriptors, used to initialize
                the empty novelty archive.
                total_generations: total number of generations for which the emitter
                    will run, allow to initialize the novelty archive.
        """
        assert (
            batch_size % scan_batch_size == 0 or scan_batch_size > batch_size
        ), "ERROR!!! Batch-size should be divisible by snace-batch-size."
        total_rollouts = batch_size * config.sample_number
        assert(
            total_rollouts % scan_novelty == 0 or scan_novelty > total_rollouts
        ), "ERROR!!! Total number of rollouts should be divisible by scan-novelty"
        
        # Set up config 
        self._config = config
        self._config.use_explore = self._config.proportion_explore > 0
        
        # Set up other parameters
        self._batch_size = batch_size
        self._scoring_fn = scoring_fn
        self._scan_batch_size = (
            scan_batch_size if batch_size > scan_batch_size else batch_size
        )
        self._scan_novelty = (
            scan_novelty if total_rollouts > scan_novelty else total_rollouts
        )
        self._num_scan = self._batch_size // self._scan_batch_size
        self._num_descriptors = num_descriptors
        self._total_generations = total_generations
        self._num_centroids = num_centroids
        assert not (
            self._config.use_novelty_archive and self._config.use_novelty_fifo
        ), "!!! ERROR!!! Use both novelty archive and novelty fifo."
        
        # Create the score repartition based on proportion_explore
        number_explore = floor(self._batch_size * self._config.proportion_explore)
        self._non_scan_explore = jnp.concatenate(
            [
                jnp.ones(number_explore),
                jnp.zeros(self._batch_size - number_explore)
            ],
            axis=0,
        )
        self._explore = jnp.repeat(
            self._non_scan_explore, self._config.sample_number, axis=0
        )
        self._explore = jnp.reshape(self._explore, (self._num_scan, -1))
        
        # Initialize optimizer
        if self._config.adam_optimizer:
            self._optimizer = optax.adam(learning_rate=config.learning_rate)
        else:
            self._optimizer - optax.sgd(learning_rate=config.learning_rate)
            
        @property
        def batch_size(self) -> int:
            """
            Returns:
                the batch size emitter by the emitter.
            """
            return self._batch_size
        
        @partial(
            jax.jit,
            static_argnames=("self", "batch_size", "novelty_batch_size")
        )
        def _init_novelty_archive(
            self, batch_size: int, novelty_batch_size: int
        ) -> NoveltyArchive:
            """Init the novelty archive for the emitter.
            """
            
            if self._config.use_explore and self._config.use_novelty_archive:
                novelty_archive = SequentialNoveltyArchive.init(
                    self._total_generations * batch_size, self._num_descriptors
                )
            elif (
                self._config.use_explore
                and self._config.use_novelty_fifo
                and self._scan_novelty < novelty_batch_size
            ):
                novelty_archive = SequentialScanNoveltyArchive.init(
                    self._config.fifo_size,
                    self._num_descriptors,
                    scan_size=self._scan_novelty,
                )
            elif (
                self._config.use_explore
                and self._config.use_novelty_fifo
                and self._scan_novelty == 1
            ):
                novelty_archive = SequentialNoveltyArchive.init(
                    self._config.fifo_size, self._num_descriptors
                )
            elif self._config.use_explore and self._config.use_novelty_fifo:
                novelty_archive = ParallelNoveltyArchive.init(
                    self._config.fifo_size, self._num_descriptors
                )
            elif self._config.use_explore:
                novelty_archive = RepertoireNoveltyArchive.init(
                    self._num_centroids, self._num_descriptors
                )
            else:
                novelty_archive = EmptyNoveltyArchive.init(1, 1)
                
            return novelty_archive
        
        @partial(
            jax.jit,
            static_argnames=("self",),
        )
        def init(
            self,
            init_genotypes: Genotype,
            random_key: RNGKey,
        ) -> Tuple[REINaiveEmitterState, RNGKey]:
            """Initiliazes the emitter state.
            
            Args:
                init_genotypes: The initial population.
                random_key: A random key.
                
            Returns:
                The initial state of the emitter, a new random key.
            """
            
            # Initialize optimizer
            params = jax.tree_util.tree_map(lambda x: x[0], init_genotypes)
            initial_optimizer_state = self._optimizer.init(params)
            
            # One optimizer_state per lineage 
            # A lineage is essentially a chain of individuals connected by descent, 
            # tracing back from a current individual to its ancestors in previous generations.
            optimizer_states = jax.tree_util.tree_map(
                lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), self._batch_size, axis=0),
                initial_optimizer_state,
            )
            
            # Empty Novelty archive
            #novelty_archive = self._init_novelty_archive(
            #    self._batch_size, self._batch_size * self._config.sample_number
            #)
            
            return (
                REINaiveEmitterState(
                    initial_optimizer_state=initial_optimizer_state,
                    optimizer_states=optimizer_states,
                    offspring=init_genotypes,
                    generation_count=0,
                    #novelty_archive=novelty_archive,
                    random_key=random_key,
                ),
                random_key,
            )
            
        @partial(
            jax.jit,
            static_argnames=("self",),
        )
        def emit(
            self,
            repertoire: Repertoire,
            emitter_state: REINaiveEmitterState,
            random_key: RNGKey,
        ) -> Tuple[Genotype, RNGKey]:
            """Return the offsrping generated through gradient update.
            
            Args:
                repertoire: the MAP-ELites reperoire to sample from emitter_state
                random_key: a jax PRNG random key
                
            Returns:
                a batch of offsprings
                a new jax PRNG key
            """
            
            assert emitter_state is not None, "\n!!! ERROR!! No emitter state."
            return emitter_state.offspring, random_key
        
        @partial(
            jax.jit,
            static_argnames=("self",),
        )
        def _scores(
            self,
            final_policies_of_gen: Genotype,
            explore: jnp.ndarray,
            repertoire: Repertoire,
            emitter_state: REINaiveEmitterState,
            random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            """Compute the scores associated with each rollout.
            
            Args:
                final_policies_of_gen: the most updated policies of the generation
                explore: repartition of explore and exploit emitters
                reperoire: current repertoire
                emitter_state: current emitter state
                random_key: a jax PRNG key
                
            Returns:
                the gradients to apply and a new random key
            """
            
            # Evaluate rollouts
            fitnesses, descriptors, _, random_key = self._scoring_fn(
                final_policies_of_gen,
                random_key,
            )
            
            # Get corresponding score
            scores = jnp.where(
                explore,
                emitter_state.novelty_archive.novelty(
                    descriptors,
                    self._config.novelty_nearest_neighbors,
                ),
                fitnesses,
            )
            
            return scores, random_key
        
        
        # SEE WHAT YOU WILL DO WITH THE _pre_es_noise & _pre_es_apply FUNCTIONS
        @partial(
            jax.jit,
            static_argnames=("self",),
        )
        def logp_fn(
            self,
            
        )