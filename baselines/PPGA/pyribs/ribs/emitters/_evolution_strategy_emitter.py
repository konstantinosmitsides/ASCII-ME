"""Provides the EvolutionStrategyEmitter."""
import itertools

import numpy as np

from baselines.PPGA.pyribs.ribs.emitters._emitter_base import EmitterBase
from baselines.PPGA.pyribs.ribs.emitters.opt import CMAEvolutionStrategy
from baselines.PPGA.pyribs.ribs.emitters.rankers import _get_ranker


class EvolutionStrategyEmitter(EmitterBase):
    """Adapts a distribution of solutions with CMA-ES.

    This emitter originates in `Fontaine 2020
    <https://arxiv.org/abs/1912.02400>`_. The multivariate Gaussian solution
    distribution begins at ``x0`` with standard deviation ``sigma0``. Based on
    how the generated solutions are ranked (see ``ranker``), CMA-ES then adapts
    the mean and covariance of the distribution.

    Args:
        archive (ribs.archives.ArchiveBase): An archive to use when creating and
            inserting solutions. For instance, this can be
            :class:`ribs.archives.GridArchive`.
        x0 (np.ndarray): Initial solution. Must be 1-dimensional.
        sigma0 (float): Initial step size / standard deviation.
        selection_rule ("mu" or "filter"): Method for selecting parents in
            CMA-ES. With "mu" selection, the first half of the solutions will be
            selected as parents, while in "filter", any solutions that were
            added to the archive will be selected.
        ranker (Callable or str): The ranker is a :class:`RankerBase` object
            that orders the solutions after they have been evaluated in the
            environment. This parameter may be a callable (e.g. a class or a
            lambda function) that takes in no parameters and returns an instance
            of :class:`RankerBase`, or it may be a full or abbreviated ranker
            name as described in :meth:`ribs.emitters.rankers.get_ranker`.
        restart_rule (int, "no_improvement", and "basic"): Method to use when
            checking for restarts. If given an integer, then the emitter will
            restart after this many iterations, where each iteration is a call
            to :meth:`tell`. With "basic", only the default CMA-ES convergence
            rules will be used, while with "no_improvement", the emitter will
            restart when none of the proposed solutions were added to the
            archive.
        bounds (None or array-like): Bounds of the solution space. As suggested
            in `Biedrzycki 2020
            <https://www.sciencedirect.com/science/article/abs/pii/S2210650219301622>`_,
            solutions are resampled until they fall within these bounds.  Pass
            None to indicate there are no bounds. Alternatively, pass an
            array-like to specify the bounds for each dim. Each element in this
            array-like can be None to indicate no bound, or a tuple of
            ``(lower_bound, upper_bound)``, where ``lower_bound`` or
            ``upper_bound`` may be None to indicate no bound.
        batch_size (int): Number of solutions to return in :meth:`ask`. If not
            passed in, a batch size will be automatically calculated using the
            default CMA-ES rules.
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
    Raises:
        ValueError: If ``restart_rule``, ``selection_rule``, or ``ranker`` is
        invalid.
    """

    def __init__(self,
                 archive,
                 x0,
                 sigma0,
                 ranker,
                 selection_rule="filter",
                 restart_rule="no_improvement",
                 bounds=None,
                 batch_size=None,
                 seed=None):
        self._rng = np.random.default_rng(seed)

        self._x0 = np.array(x0, dtype=archive.dtype)
        if self._x0.ndim != 1:
            raise ValueError(
                f"x0 has shape {self._x0.shape}, should be 1-dimensional.")

        self._sigma0 = sigma0
        EmitterBase.__init__(
            self,
            archive,
            len(self._x0),
            bounds,
        )

        if selection_rule not in ["mu", "filter"]:
            raise ValueError(f"Invalid selection_rule {selection_rule}")
        self._selection_rule = selection_rule

        self._restart_rule = restart_rule
        self._restarts = 0
        self._itrs = 0
        # Check if the restart_rule is valid.
        _ = self._check_restart(0)

        opt_seed = None if seed is None else self._rng.integers(10_000)
        self.opt = CMAEvolutionStrategy(sigma0, batch_size, self._solution_dim,
                                        "truncation", opt_seed,
                                        self.archive.dtype)
        self.opt.reset(self._x0)

        self._ranker = _get_ranker(ranker)
        self._ranker.reset(self, archive, self._rng)

        self._batch_size = self.opt.batch_size

    @property
    def x0(self):
        """numpy.ndarray: Initial solution for the optimizer."""
        return self._x0

    @property
    def batch_size(self):
        """int: Number of solutions to return in :meth:`ask`."""
        return self._batch_size

    @property
    def restarts(self):
        """int: The number of restarts for this emitter."""
        return self._restarts

    @property
    def itrs(self):
        """int: The number of iterations for this emitter, where each iteration
        is a call to :meth:`tell`."""
        return self._itrs

    def ask(self):
        """Samples new solutions from a multivariate Gaussian.

        The multivariate Gaussian is parameterized by the evolution strategy
        optimizer ``self.opt``.

        Returns:
            (batch_size, :attr:`solution_dim`) array -- a batch of new solutions
            to evaluate.
        """
        return self.opt.ask(self.lower_bounds, self.upper_bounds)

    def _check_restart(self, num_parents):
        """Emitter-side checks for restarting the optimizer.

        The optimizer also has its own checks.

        Args:
            num_parents (int): The number of solution to propagate to the next
                generation from the solutions generated by CMA-ES.

        Raises:
          ValueError: If :attr:`restart_rule` is invalid.
        """
        if isinstance(self._restart_rule, (int, np.integer)):
            return self._itrs % self._restart_rule == 0
        if self._restart_rule == "no_improvement":
            return num_parents == 0
        if self._restart_rule == "basic":
            return False
        raise ValueError(f"Invalid restart_rule {self._restart_rule}")

    def tell(self,
             solution_batch,
             objective_batch,
             measures_batch,
             status_batch,
             value_batch,
             metadata_batch=None):
        """Gives the emitter results from evaluating solutions.

        The solutions are ranked based on the `rank()` function defined by
        `self._ranker`. Then, the ranked solutions are passed to CMA-ES for
        adaptation.

        This function also checks for restart condition and restarts CMA-ES
        when needed.

        Args:
            solution_batch (numpy.ndarray): (batch_size, :attr:`solution_dim`)
                array of solutions generated by this emitter's :meth:`ask()`
                method.
            objective_batch (numpy.ndarray): 1D array containing the objective
                function value of each solution.
            measures_batch (numpy.ndarray): (batch_size, measure space
                dimension) array with the measure space coordinates of each
                solution.
            status_batch (numpy.ndarray): 1D array of
                :class:`ribs.archive.AddStatus` returned by a series of calls to
                archive's :meth:`add()` method.
            value_batch (numpy.ndarray): 1D array of floats returned by a series
                of calls to archive's :meth:`add()` method. For what these
                floats represent, refer to :meth:`ribs.archives.add()`.
            metadata_batch (numpy.ndarray): 1D object array containing a
                metadata object for each solution.
        """
        # Increase iteration counter.
        self._itrs += 1

        metadata_batch = itertools.repeat(
            None) if metadata_batch is None else metadata_batch

        # Count number of new solutions.
        new_sols = status_batch.astype(bool).sum()

        # Sort the solutions using ranker.
        indices, ranking_values = self._ranker.rank(
            self, self.archive, self._rng, solution_batch, objective_batch,
            measures_batch, status_batch, value_batch, metadata_batch)

        # Select the number of parents.
        num_parents = (new_sols if self._selection_rule == "filter" else
                       self._batch_size // 2)

        # Update Evolution Strategy.
        self.opt.tell(solution_batch[indices], num_parents)

        # Check for reset.
        if (self.opt.check_stop(ranking_values[indices]) or
                self._check_restart(new_sols)):
            new_x0 = self.archive.sample_elites(1).solution_batch[0]
            self.opt.reset(new_x0)
            self._ranker.reset(self, self.archive, self._rng)
            self._restarts += 1
