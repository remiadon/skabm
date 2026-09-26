"""
Calibration — two different problems that share a word.

The ABM literature and the microsimulation literature both say "calibration"
and mean different things.  Both are here, in separate modules, and they
compose rather than compete:

``skabm.calibration.population``
    Fits the agent population ``X`` — one row per firm, household, bank — to
    accounting identities and known margins: IO coefficients, census shares,
    Basel III ratios.  Poledna et al. (2023) §4's sense, and survey sampling's
    (Deville & Särndal 1992).  ``make_dataset`` draws it, the constraint
    calibrators reconcile it.

``skabm.calibration.parameters``
    Fits the behavioural parameters ``θ`` — ``RDFSimulator(params=...)`` — to
    macro time series the model should reproduce.  The ABM literature's usual
    sense (Fagiolo et al. 2019).  skabm supplies the adapter; `black-it
    <https://github.com/bancaditalia/black-it>`_ supplies the search, via
    ``pip install "skabm[model-calibration]"``.

In sklearn vocabulary the split is the familiar one: population calibration is
a transformer on ``X``, model calibration is a search over ``get_params()``.
The ordering follows from that — a population is calibrated **once, outside**
any parameter search, and held frozen while ``θ`` moves inside it.

Both modules' public names are re-exported here, so ``from skabm.calibration
import make_dataset`` keeps working.
"""

from .parameters import noise_floor, simulator_model
from .population import (
    GeneticConstraintCalibration,
    MetropolisHastingsConstraintCalibration,
    energy,
    make_dataset,
    weighted_enum,
)

__all__ = [
    "GeneticConstraintCalibration",
    "MetropolisHastingsConstraintCalibration",
    "energy",
    "make_dataset",
    "noise_floor",
    "simulator_model",
    "weighted_enum",
]
