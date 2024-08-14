"""Core components of the MAP-Elites algorithm."""
from __future__ import annotations

from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from dataclasses import dataclass, replace

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)

from jax import tree_util



@dataclass
class ObsNormalizer:
    size: int
    mean: jnp.ndarray = None
    var: jnp.ndarray = None
    count: jnp.ndarray = 1e-4
    
    def __post_init__(self):
        if self.mean is None:
            self.mean = jnp.zeros(self.size)
        if self.var is None:
            self.var = jnp.ones(self.size)
            
    def update(self, x):
        # Flatten the first two dimensions (x, y) to treat as a single batch dimension
        flat_x = x.reshape(-1, self.size)
        batch_mean = jnp.mean(flat_x, axis=0)
        batch_var = jnp.var(flat_x, axis=0)
        batch_count = flat_x.shape[0]

        new_mean, new_var, new_count = self._update_mean_var_count(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)
        
        return replace(self, mean=new_mean, var=new_var, count=new_count)

    def normalize(self, x):
        # Normalize maintaining the original shape, using broadcasting
        return (x - self.mean) / jnp.sqrt(self.var + 1e-8)

    def _update_mean_var_count(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count


    def tree_flatten(self):
        return ((self.mean, self.var, self.count), self.size)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        size = aux_data
        mean, var, count = children
        return cls(size=size, mean=mean, var=var, count=count)

# Register Normalizer as a pytree node with JAX
tree_util.register_pytree_node(
    ObsNormalizer,
    ObsNormalizer.tree_flatten,
    ObsNormalizer.tree_unflatten
)


@dataclass
class RewardNormalizer:
    size: int
    mean: jnp.ndarray = 0.0
    var: jnp.ndarray = 1.0
    count: jnp.ndarray = 1e-4
    return_val: jnp.ndarray = None
    
    def __post_init__(self):
        if self.return_val is None:
            self.return_val = jnp.zeros((self.size,))

         
    def update(self, reward, done, gamma=0.99):
        
        def _update_column_scan(carry, x):
            mean, var, count, return_val = carry
            (reward, done) = x
            
            jax.debug.print("Reward shape: {}", reward.shape)
            
            # Update the return value
            new_return_val = reward + gamma * return_val * (1 - done)
            
            # Update the mean, var, and count
            batch_mean = jnp.mean(new_return_val, axis=0)
            batch_var = jnp.var(new_return_val, axis=0)
            batch_count = new_return_val.shape[0]
            
            delta = batch_mean - mean
            tot_count = count + batch_count
            
            new_mean = mean + delta * batch_count / tot_count
            m_a = var * count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
            new_var = M2 / tot_count
            new_count = tot_count
            
            normalized_reward = reward / jnp.sqrt(new_var + 1e-8)
            
            return (new_mean, new_var, new_count, new_return_val), normalized_reward
        
        (new_mean, new_var, new_count, _), normalized_rewards = jax.lax.scan(
            _update_column_scan,
            (self.mean, self.var, self.count, self.return_val),
            (reward.T, done.T),
        )

        
        return replace(self, mean=new_mean, var=new_var, count=new_count), normalized_rewards.T






    def tree_flatten(self):
        return ((self.mean, self.var, self.count, self.return_val), self.size)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        size = aux_data
        mean, var, count, return_val = children
        return cls(size=size, mean=mean, var=var, count=count, return_val=return_val)
    
        
tree_util.register_pytree_node(
    RewardNormalizer,
    RewardNormalizer.tree_flatten,
    RewardNormalizer.tree_unflatten
)



class MAPElites:
    """Core elements of the MAP-Elites algorithm.

    Note: Although very similar to the GeneticAlgorithm, we decided to keep the
    MAPElites class independant of the GeneticAlgorithm class at the moment to keep
    elements explicit.

    Args:
        scoring_function: a function that takes a batch of genotypes and compute    # evaluate fitness & descriptor
            their fitnesses and descriptors
        emitter: an emitter is used to suggest offsprings given a MAPELites         # select sols? & update them
            repertoire. It has two compulsory functions. A function that takes
            emits a new population, and a function that update the internal state
            of the emitter.
        metrics_function: a function that takes a MAP-Elites repertoire and compute  # evaluate the MAP-Elites repertoire
            any useful metric to track its evolution
    """

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        emitter: Emitter,
        metrics_function: Callable[[MapElitesRepertoire], Metrics],
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function
        self._obs_normalizer =ObsNormalizer(self._emitter._env.observation_size)
        self._reward_normalizer = RewardNormalizer(self._emitter.batch_size)
        #self._obs_normalizer =ObsNormalizer(28)
      

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        genotypes: Genotype,
        centroids: Centroid,
        random_key: RNGKey,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], RNGKey]:
        """
        Initialize a Map-Elites repertoire with an initial population of genotypes.
        Requires the definition of centroids that can be computed with any method
        such as CVT or Euclidean mapping.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            centroids: tesselation centroids of shape (batch_size, num_descriptors)
            random_key: a random key used for stochastic operations.

        Returns:
            An initialized MAP-Elite repertoire with the initial state of the emitter,
            and a random key.
        """
        # score initial genotypes      
        fitnesses, descriptors, extra_scores, random_key, obs_normalizer, reward_normalizer = self._scoring_function(
            genotypes, random_key, self._obs_normalizer, self._reward_normalizer
        )
        
        #print(extra_scores["transitions"])

        # init the repertoire
        repertoire = MapElitesRepertoire.init(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            extra_scores=extra_scores,
        )

        # get initial state of the emitter
        '''
        emitter_state, random_key = self._emitter.init(
            random_key=random_key,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )
        '''
        
        
        
        
        emitter_state, random_key = self._emitter.init(
            random_key=random_key,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )
        
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores={**extra_scores}#, **extra_info},
        )

        return repertoire, emitter_state, random_key, obs_normalizer, reward_normalizer

    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
        obs_normalizer: Optional[ObsNormalizer],
        reward_normalizer: Optional[RewardNormalizer],
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], Metrics, RNGKey]:
        """
        Performs one iteration of the MAP-Elites algorithm.
        1. A batch of genotypes is sampled in the repertoire and the genotypes
            are copied.
        2. The copies are mutated and crossed-over
        3. The obtained offsprings are scored and then added to the repertoire.


        Args:
            repertoire: the MAP-Elites repertoire
            emitter_state: state of the emitter
            random_key: a jax PRNG random key

        Returns:
            the updated MAP-Elites repertoire
            the updated (if needed) emitter state
            metrics about the updated repertoire
            a new jax PRNG key
        """
        # generate offsprings with the emitter
        '''
        genotypes, extra_info, random_key = self._emitter.emit(
            repertoire, emitter_state, random_key
        )
        '''
        genotypes, _, random_key = self._emitter.emit(
            repertoire, emitter_state, random_key
        )

        # scores the offsprings
        fitnesses, descriptors, extra_scores, random_key, obs_normalizer, reward_normalizer = self._scoring_function(
            genotypes, random_key, obs_normalizer, reward_normalizer
        )
        #jax.debug.print("dones : {}", extra_scores['transitions'].dones)

        # add genotypes in the repertoire
        repertoire = repertoire.add(genotypes, descriptors, fitnesses, extra_scores)

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores={**extra_scores}#, **extra_info},
        )

        # update the metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics, random_key, obs_normalizer, reward_normalizer

    @partial(jax.jit, static_argnames=("self",))
    def scan_update(
        self,
        carry: Tuple[MapElitesRepertoire, Optional[EmitterState], RNGKey, Optional[ObsNormalizer], Optional[RewardNormalizer]],
        unused: Any,
    ) -> Tuple[Tuple[MapElitesRepertoire, Optional[EmitterState], RNGKey], Metrics]:
        """Rewrites the update function in a way that makes it compatible with the
        jax.lax.scan primitive.

        Args:
            carry: a tuple containing the repertoire, the emitter state and a
                random key.
            unused: unused element, necessary to respect jax.lax.scan API.

        Returns:
            The updated repertoire and emitter state, with a new random key and metrics.
        """
        repertoire, emitter_state, random_key, obs_normalizer,reward_normalizer = carry
        (repertoire, emitter_state, metrics, random_key, obs_normalizer, reward_normalizer) = self.update(
            repertoire,
            emitter_state,
            random_key,
            obs_normalizer,
            reward_normalizer
        )

        return (repertoire, emitter_state, random_key, obs_normalizer, reward_normalizer), metrics
