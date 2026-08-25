"""
Model calibration — fitting behavioural parameters to macro time series.

The ABM literature's usual sense of "calibration": search the free parameters
``θ`` of a model so its aggregate output matches series the model is supposed to
reproduce (Fagiolo et al. 2019).  Its sibling module, ``population``, fits the
agent *rows* instead; the two compose in the sklearn way — a population is a
transformer on ``X``, a parameter search is a search over ``get_params()`` — and
the population is drawn once, *outside* the search loop.

skabm does not implement the search.  `black-it
<https://github.com/bancaditalia/black-it>`_ already does it well, and this
module is the adapter: ``simulator_model`` turns an ``RDFSimulator``
configuration into the plain ``model(theta, N, seed) -> (N, D)`` callable
black-it (and most other calibration toolboxes) expect.  Nothing here imports
black-it, so the module works on the core install; the toolbox is only needed
where the returned callable is handed to a ``Calibrator``::

    pip install "skabm[model-calibration]"

``noise_floor`` is the diagnostic that belongs next to it.  A simulated loss is
an estimate, so a loss *gap* between two candidates means nothing until it
exceeds the spread one candidate shows across seed draws — and that spread is
what ``ensemble_size`` buys down.  Measuring it costs a few dozen simulations
and routinely saves a few thousand.

See ``notebooks/poledna/poledna.ipynb`` (chapter 4) for both in use, including
the identification reading of a calibration result.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import polars as pl

from skabm.behaviour.params import poledna_params

__all__ = ["simulator_model", "noise_floor"]


def simulator_model(
    populations: dict[str, pl.DataFrame],
    free: Sequence[str],
    summarise: Callable[[pl.DataFrame], Sequence[float]],
    params: dict | None = None,
    **simulator_kwargs,
) -> Callable[[Sequence[float], int, int], np.ndarray]:
    """Build a ``model(theta, N, seed) -> (N, D)`` callable from a simulator setup.

    Parameters
    ----------
    populations : dict[str, pl.DataFrame]
        The agent populations, keyword-style (``{"Firm": firms, ...}``), already
        calibrated and **frozen**.  Every candidate ``θ`` is evaluated against
        the same rows, which is what makes the losses comparable.
    free : Sequence[str]
        Names of the parameters ``θ`` indexes, in order.  Each must already
        exist in the merged parameter dict — a name no rule template reads would
        be searched over silently and change nothing, so it is rejected here.
    summarise : Callable[[pl.DataFrame], Sequence[float]]
        One per-tick state frame (as yielded by ``fit_iter``) to the ``D``
        observables.  Ordinary polars on the caller's side, exactly as in a
        hand-written run loop.
    params : dict, optional
        Base parameters the search perturbs.  Defaults to the canonical Poledna
        values; a partial dict is merged over them.
    **simulator_kwargs
        Passed to ``RDFSimulator`` (``update_rules``, ``udfs``, ``infer``, ...).
        ``n_periods``, ``random_seed`` and ``warm_start`` are owned by the
        returned callable and rejected here: black-it supplies the first two per
        call, and a warm start would leak one candidate's state into the next.

    Returns
    -------
    Callable
        ``model(theta, N, seed)`` returning an ``(N, D)`` float array — one row
        per tick, so a caller that differences the series must ask for ``N + 1``
        ticks and drop one.  Fits are always cold.

    Examples
    --------
    >>> model = simulator_model(          # doctest: +SKIP
    ...     populations={"Firm": firms, "Household": households},
    ...     free=("dividend_ratio", "rho"),
    ...     summarise=lambda s: [s["wealth"].drop_nulls().sum()],
    ...     params={"total_deposits": 2000.0},
    ... )
    >>> Calibrator(model=model, ...).calibrate(n_batches=6)   # doctest: +SKIP
    """
    from skabm.simulation import RDFSimulator

    owned = {"n_periods", "random_seed", "warm_start"} & set(simulator_kwargs)
    if owned:
        raise ValueError(
            f"{sorted(owned)} are set per call by the returned model, not here: "
            "N and seed come from the calibrator, and fits are always cold."
        )

    base = {**poledna_params, **(params or {})}
    unknown = [name for name in free if name not in base]
    if unknown:
        raise ValueError(
            f"free parameters {unknown} are not in the parameter set, so no rule "
            "template reads them and searching over them would change nothing. "
            f"Known names: {sorted(base)}"
        )

    free = tuple(free)

    def model(theta: Sequence[float], N: int, seed: int) -> np.ndarray:
        sim = RDFSimulator(
            params={**base, **dict(zip(free, (float(v) for v in theta)))},
            n_periods=int(N),
            random_seed=int(seed),
            **simulator_kwargs,
        )
        return np.asarray(
            [summarise(state) for state in sim.fit_iter(**populations)], dtype=float
        )

    model.free = free  # type: ignore[attr-defined]
    return model


def noise_floor(
    model: Callable[[Sequence[float], int, int], np.ndarray],
    theta: Sequence[float],
    real_data: np.ndarray,
    loss_function,
    ensemble_size: int,
    n_repeats: int = 3,
    seed0: int = 1,
) -> list[float]:
    """Evaluate ``loss_function`` at one ``theta`` over ``n_repeats`` disjoint seed sets.

    The spread of the returned values is the precision floor of the calibration:
    a loss difference smaller than it is seed noise, not evidence, whatever the
    sampler concludes.  Run it at the parameters you believe and at one clearly
    wrong candidate — if the two ranges overlap, the observables do not identify
    that parameter and no sampling budget will change it.

    ``loss_function`` is any black-it loss (its ``compute_loss`` takes an
    ``(ensemble_size, N, D)`` array and the real data); ``N`` is taken from
    ``real_data``.  Raising ``ensemble_size`` narrows the spread — that is the
    knob this measures.
    """
    N = len(real_data)
    losses = []
    for repeat in range(n_repeats):
        first = seed0 + repeat * ensemble_size
        simulated = np.stack(
            [model(theta, N, s) for s in range(first, first + ensemble_size)]
        )
        losses.append(
            float(np.ravel(loss_function.compute_loss(simulated, real_data))[0])
        )
    return losses
