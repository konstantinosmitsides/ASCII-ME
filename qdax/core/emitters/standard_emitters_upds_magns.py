from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import ExtraScores, Genotype, RNGKey
from utils import find_magnitude_of_updates, concatenate_params


class MixingEmitter(Emitter):
    def __init__(
        self,
        mutation_fn: Callable[[Genotype, RNGKey], Tuple[Genotype, RNGKey]],
        variation_fn: Callable[[Genotype, Genotype, RNGKey], Tuple[Genotype, RNGKey]],
        variation_percentage: float,
        batch_size: int,
    ) -> None:
        self._mutation_fn = mutation_fn
        self._variation_fn = variation_fn
        self._variation_percentage = variation_percentage
        self._batch_size = batch_size

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit(
        self,
        repertoire: Repertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[Genotype, ExtraScores, RNGKey]:
        """
        Emitter that performs both mutation and variation. Two batches of
        variation_percentage * batch_size genotypes are sampled in the repertoire,
        copied and cross-over to obtain new offsprings. One batch of
        (1.0 - variation_percentage) * batch_size genotypes are sampled in the
        repertoire, copied and mutated.

        Note: this emitter has no state. A fake none state must be added
        through a function redefinition to make this emitter usable with MAP-Elites.

        Params:
            repertoire: the MAP-Elites repertoire to sample from
            emitter_state: void
            random_key: a jax PRNG random key

        Returns:
            a batch of offsprings
            a new jax PRNG key
        """
        n_variation = int(self._batch_size * self._variation_percentage)
        n_mutation = self._batch_size - n_variation

        if n_variation > 0:
            x1, _, random_key = repertoire.sample(random_key, n_variation)
            x2, _, random_key = repertoire.sample(random_key, n_variation)

            x_variation, random_key = self._variation_fn(x1, x2, random_key)

        if n_mutation > 0:
            x1, _, random_key = repertoire.sample(random_key, n_mutation)
            x_mutation, random_key = self._mutation_fn(x1, random_key)
            
        #if n_variation == 0 and n_mutation == 0:
        #    return {}, {}, random_key

        if n_variation == 0:
            genotypes = x_mutation
        elif n_mutation == 0:
            genotypes = x_variation
        else:
            genotypes = jax.tree_util.tree_map(
                lambda x_1, x_2: jnp.concatenate([x_1, x_2], axis=0),
                x_variation,
                x_mutation,
            )
            
        new_params = concatenate_params(genotypes)
        old_params = concatenate_params(x1)
        update_magnitudes = find_magnitude_of_updates(new_params, old_params)

        return genotypes, {"update_magns_ga" : update_magnitudes}, random_key

    @property
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        return self._batch_size
    
    @property
    def use_all_data(self) -> bool:
        """Whether to use all data or not when used along other emitters.

        Used when an emitter is used in a multi emitter setting.

        Some emitter only the information from the genotypes they emitted when
        they update their state (for instance, the CMA emitters); but other use data
        from genotypes emitted by others (for instance, QualityPGEmitter and
        DiversityPGEmitter). The meta emitters like MultiEmitter need to know which
        data to give the sub emitter when udapting them. This property is used at
        this moment.

        Default behavior is to used only the data related to what was emitted.

        Returns:
            Whether to pass only the genotypes (and their evaluations) that the emitter
            emitted when updating it or all the genotypes emitted by all the emitters.
        """
        return True
