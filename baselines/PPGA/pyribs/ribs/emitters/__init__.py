"""Emitters output new candidate solutions in QD algorithms.

All emitters should inherit from :class:`EmitterBase`, except for emitters
designed for differentiable quality diversity (DQD), which should instead
inherit from :class:`DQDEmitterBase`.

.. note::
    Emitters provided here take on the data type of the archive passed to their
    constructor. For instance, if an archive has dtype ``np.float64``, then an
    emitter created with that archive will emit solutions with dtype
    ``np.float64``.

.. autosummary::
    :toctree:

    ribs.emitters.EvolutionStrategyEmitter
    ribs.emitters.GradientAborescenceEmitter
    ribs.emitters.GaussianEmitter
    ribs.emitters.IsoLineEmitter
    ribs.emitters.EmitterBase
    ribs.emitters.DQDEmitterBase
"""
from baselines.PPGA.pyribs.ribs.emitters._dqd_emitter_base import DQDEmitterBase
from baselines.PPGA.pyribs.ribs.emitters._emitter_base import EmitterBase
from baselines.PPGA.pyribs.ribs.emitters._evolution_strategy_emitter import EvolutionStrategyEmitter
from baselines.PPGA.pyribs.ribs.emitters._gaussian_emitter import GaussianEmitter
from baselines.PPGA.pyribs.ribs.emitters._gradient_aborescence_emitter import \
  GradientAborescenceEmitter
from baselines.PPGA.pyribs.ribs.emitters._proximal_policy_gradient_arborescence_emitter import PPGAEmitter
from baselines.PPGA.pyribs.ribs.emitters._iso_line_emitter import IsoLineEmitter

__all__ = [
    "EvolutionStrategyEmitter",
    "GradientAborescenceEmitter",
    "PPGAEmitter",
    "GaussianEmitter",
    "IsoLineEmitter",
    "EmitterBase",
    "DQDEmitterBase",
]
