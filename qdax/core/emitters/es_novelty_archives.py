from __future__ import annotations

from abc import abstractmethod
from functools import partial
from typing import Optional, Tuple

import flax
import jax
import jax.numpy as jnp
from qdax.types import Descriptor, Fitness


class NoveltyArchive(flax.struct.PyTreeNode):
    """Novelty Archive used by the ES emitters.

    Args:
        archive: content of the archive
        size: total size of the archive
        position: current position in the archive
        total_position: total number added to the archive
        scan_size: unused in most classes
    """

    archive: jnp.ndarray
    size: int = flax.struct.field(pytree_node=False)
    position: jnp.ndarray = flax.struct.field()
    total_position: jnp.ndarray = flax.struct.field()
    scan_size: int = flax.struct.field(pytree_node=False)

    @classmethod
    def init(
        cls,
        size: int,
        num_descriptors: int,
        scan_size: Optional[int] = 0,
    ) -> NoveltyArchive:
        archive = jnp.zeros((size, num_descriptors))
        return cls(
            archive=archive,
            size=size,
            position=jnp.array(0, dtype=int),
            total_position=jnp.array(0, dtype=int),
            scan_size=scan_size,
        )

    @abstractmethod
    def update(
        self,
        descriptors: Descriptor,
        repertoire_descriptors: Descriptor,
        repertoire_fitnesses: Fitness,
    ) -> NoveltyArchive:
        """Update the content of the novelty archive with newly generated descriptors.

        Args:
            descriptor: new descriptor generated by ES emitters
            repertoire_descriptors: descriptors in repertoire, available for computation
            repertoire_fitnesses: fitnesses in repertoire, available for computation
        Returns:
            The updated NoveltyArchive
        """
        pass

    @abstractmethod
    @partial(jax.jit, static_argnames=("num_nearest_neighbors",))
    def novelty(
        self,
        descriptors: Descriptor,
        num_nearest_neighbors: int,
    ) -> jnp.ndarray:
        """Compute the novelty of the given descriptors as the average distance
        to the k nearest neighbours in the archive.

        Args:
            descriptors: the descriptors to compute novelty for
            num_nearest_neighbors: k used to compute the k-nearest-neighbours
        Returns:
            the novelty of each descriptor in descriptors.
        """
        pass

    @partial(jax.jit, static_argnames=("num_nearest_neighbors",))
    def _single_novelty(
        self,
        descriptor: Descriptor,
        num_nearest_neighbors: int,
    ) -> jnp.ndarray:
        """Compute the novelty of one given descriptor as the average distance
        to the k nearest neighbours in the archive.

        Args:
            descriptor: the descriptor to compute novelty for
            num_nearest_neighbors: k used to compute the k-nearest-neighbours
        Returns:
            the novelty of descriptor.
        """

        # Compute all distances with archive content
        def distance(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            return jnp.sqrt(jnp.sum(jnp.square(x - y)))

        distances = jax.vmap(partial(distance, y=descriptor))(self.archive)

        # Filter distance with empty slot of archive
        indices = jnp.arange(0, self.size, step=1) < self.total_position + 1
        distances = jnp.where(indices, distances, jnp.inf)

        # Find k nearest neighbours
        _, indices = jax.lax.top_k(-distances, num_nearest_neighbors)

        # Compute novelty as average distance with k neirest neirghbours
        distances = jnp.where(distances == jnp.inf, jnp.nan, distances)
        novelty = jnp.nanmean(jnp.take_along_axis(distances, indices, axis=0), axis=0)
        return novelty

    @partial(jax.jit, static_argnames=("num_nearest_neighbors",))
    def _vectorised_novelty(
        self,
        descriptors: Descriptor,
        num_nearest_neighbors: int,
    ) -> jnp.ndarray:
        """Compute the novelty of the given descriptors as the average distance
        to the k nearest neighbours in the archive.

        Args:
            descriptors: the descriptors to compute novelty for
            num_nearest_neighbors: k used to compute the k-nearest-neighbours
        Returns:
            the novelty of each descriptor in descriptors.
        """

        # Compute all distances with archive content
        def distance(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            return jnp.sqrt(jnp.sum(jnp.square(x - y)))

        distances = jax.vmap(
            jax.vmap(partial(distance), in_axes=(None, 0)), in_axes=(0, None)
        )(descriptors, self.archive)

        # Filter distance with empty slot of archive
        indices = jnp.arange(0, self.size, step=1) < self.total_position + 1
        distances = jax.vmap(lambda distance: jnp.where(indices, distance, jnp.inf))(
            distances
        )

        # Find k nearest neighbours
        _, indices = jax.lax.top_k(-distances, num_nearest_neighbors)

        # Compute novelty as average distance with k neirest neirghbours
        distances = jnp.where(distances == jnp.inf, jnp.nan, distances)
        novelty = jnp.nanmean(jnp.take_along_axis(distances, indices, axis=1), axis=1)
        return novelty


class ParallelNoveltyArchive(NoveltyArchive):
    @jax.jit
    def update(
        self,
        descriptors: Descriptor,
        repertoire_descriptors: Descriptor,
        repertoire_fitnesses: Fitness,
    ) -> NoveltyArchive:
        """Update the content of the novelty archive with newly generated descriptors.

        Args:
            descriptor: new descriptor generated by ES emitters
            repertoire_descriptors: descriptors in repertoire, available for computation
            repertoire_fitnesses: fitnesses in repertoire, available for computation
        Returns:
            The updated NoveltyArchive
        """
        batch_size = descriptors.shape[0]

        # Find position to add to
        roll = jnp.minimum(0, self.size - self.position - batch_size)
        new_archive = jnp.roll(self.archive, roll, axis=0)
        new_position = self.position + roll

        # Add to archive
        new_archive = jax.lax.dynamic_update_slice_in_dim(
            new_archive,
            descriptors,
            new_position,
            axis=0,
        )
        new_position += batch_size
        new_total_position = self.total_position + batch_size

        return self.replace(
            archive=new_archive,
            position=new_position,
            total_position=new_total_position,
        )

    @partial(jax.jit, static_argnames=("num_nearest_neighbors",))
    def novelty(
        self,
        descriptors: Descriptor,
        num_nearest_neighbors: int,
    ) -> jnp.ndarray:
        return self._vectorised_novelty(descriptors, num_nearest_neighbors)


class RepertoireNoveltyArchive(NoveltyArchive):
    @jax.jit
    def update(
        self,
        descriptors: Descriptor,
        repertoire_descriptors: Descriptor,
        repertoire_fitnesses: Fitness,
    ) -> NoveltyArchive:
        """Update the content of the novelty archive with newly generated descriptors.

        Args:
            descriptor: new descriptor generated by ES emitters
            repertoire_descriptors: descriptors in repertoire, available for computation
            repertoire_fitnesses: fitnesses in repertoire, available for computation
        Returns:
            The updated NoveltyArchive
        """
        order = jnp.argsort(
            jnp.where(repertoire_fitnesses == -jnp.inf, jnp.inf, repertoire_fitnesses)
        )
        new_descriptors = jnp.take_along_axis(
            repertoire_descriptors, jnp.expand_dims(order, axis=1), axis=0
        )

        # Add to archive
        new_position = 0
        new_archive = jax.lax.dynamic_update_slice_in_dim(
            self.archive,
            new_descriptors,
            new_position,
            axis=0,
        )
        new_total_position = jnp.sum(repertoire_fitnesses != -jnp.inf)

        return self.replace(
            archive=new_archive,
            position=new_position,
            total_position=new_total_position,
        )

    @partial(jax.jit, static_argnames=("num_nearest_neighbors",))
    def novelty(
        self,
        descriptors: Descriptor,
        num_nearest_neighbors: int,
    ) -> jnp.ndarray:
        return self._vectorised_novelty(descriptors, num_nearest_neighbors)


class SequentialNoveltyArchive(NoveltyArchive):
    @jax.jit
    def update(
        self,
        descriptors: Descriptor,
        repertoire_descriptors: Descriptor,
        repertoire_fitnesses: Fitness,
    ) -> NoveltyArchive:
        """Update the content of the novelty archive with newly generated descriptors.

        Args:
            descriptor: new descriptor generated by ES emitters
            repertoire_descriptors: descriptors in repertoire, available for computation
            repertoire_fitnesses: fitnesses in repertoire, available for computation
        Returns:
            The updated NoveltyArchive
        """
        batch_size = descriptors.shape[0]

        # Find position to add to
        roll = jnp.minimum(0, self.size - self.position - batch_size)
        new_archive = jnp.roll(self.archive, roll, axis=0)
        new_position = self.position + roll

        # Add to archive
        new_archive = jax.lax.dynamic_update_slice_in_dim(
            new_archive,
            descriptors,
            new_position,
            axis=0,
        )
        new_position += batch_size
        new_total_position = self.total_position + batch_size

        return self.replace(
            archive=new_archive,
            size=self.size,
            position=new_position,
            total_position=new_total_position,
        )

    @partial(jax.jit, static_argnames=("num_nearest_neighbors",))
    def novelty(
        self,
        descriptors: Descriptor,
        num_nearest_neighbors: int,
    ) -> jnp.ndarray:
        @jax.jit
        def _compute_single_novelty(
            carry: int,
            unused: Tuple[()],
        ) -> Tuple[int, float]:
            """Compute the novelty of all individuals using lax.scan
            as this operation requires a lot of memory.
            """
            (counter) = carry
            descriptor = descriptors[counter]
            novelty = self._single_novelty(
                descriptor=descriptor,
                num_nearest_neighbors=num_nearest_neighbors,
            )
            return counter + 1, novelty

        # Compute novelty using lax.scan to avoid memory issue
        (_), (novelties) = jax.lax.scan(
            _compute_single_novelty,
            (0),
            (),
            length=descriptors.shape[0],
        )

        return novelties


class SequentialScanNoveltyArchive(SequentialNoveltyArchive):
    @partial(jax.jit, static_argnames=("num_nearest_neighbors",))
    def novelty(
        self,
        descriptors: Descriptor,
        num_nearest_neighbors: int,
    ) -> jnp.ndarray:
        @jax.jit
        def _compute_sub_novelty(
            carry: int,
            unused: Tuple[()],
            scan_descriptors: Descriptor,
        ) -> Tuple[int, float]:
            """Compute the novelty of all individuals using lax.scan
            as this operation requires a lot of memory.
            """
            (counter) = carry
            sub_descriptors = scan_descriptors[counter]
            novelty = self._vectorised_novelty(
                descriptors=sub_descriptors,
                num_nearest_neighbors=num_nearest_neighbors,
            )
            return counter + 1, novelty

        # Reshape descriptors
        num_scan = descriptors.shape[0] // self.scan_size
        scan_descriptors = jnp.reshape(
            descriptors, (num_scan, self.scan_size) + descriptors.shape[1:]
        )

        # Compute novelty using lax.scan to avoid memory issue
        compute_sub_novelty_fn = partial(
            _compute_sub_novelty, scan_descriptors=scan_descriptors
        )
        (_), (novelties) = jax.lax.scan(
            compute_sub_novelty_fn,
            (0),
            (),
            length=num_scan,
        )
        novelties = jnp.reshape(novelties, (descriptors.shape[0]))

        return novelties


class EmptyNoveltyArchive(NoveltyArchive):
    @jax.jit
    def update(
        self,
        descriptors: Descriptor,
        repertoire_descriptors: Descriptor,
        repertoire_fitnesses: Fitness,
    ) -> NoveltyArchive:
        return self

    @partial(jax.jit, static_argnames=("num_nearest_neighbors",))
    def novelty(
        self,
        descriptors: Descriptor,
        num_nearest_neighbors: int,
    ) -> jnp.ndarray:
        return jnp.zeros(descriptors.shape[0])