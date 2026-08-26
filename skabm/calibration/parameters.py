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
    summarise: Callable[[pl.DataFrame], Sequence[float]] | None = None,
    params: dict | None = None,
    stop: Callable[[list[list[float]]], bool] | None = None,
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
    summarise : Callable[[pl.DataFrame], Sequence[float]], optional
        One per-tick state frame (as yielded by ``fit_iter``) to the ``D``
        observables.  **Usually unnecessary.**  The default (``None``) uses the
        observables the rule set already implies — every state predicate under
        the aggregate its own rules apply to it, plus the share of agents at
        each binding constraint — which is exactly the vector of per-tick
        summary statistics a method-of-moments loss consumes.  Their names, in
        column order, land on ``model.observables_``.

        Pass one only for a statistic the recorded set cannot express: anything
        distributional (a Gini coefficient, a percentile ratio) needs the
        per-agent frame, because the recorded observables are class-level
        scalars.
    stop : Callable[[list[list[float]]], bool], optional
        Early stopping.  Called with every summarised row so far (so
        ``rows[-1]`` is the newest); returning True ends that candidate's run
        and the result is padded back to ``N`` rows by carrying the last one
        forward.  A parameter search spends most of its time on candidates that
        were never going to fit, and a diverging one is usually detectable in a
        handful of ticks — on the Poledna rules an implausible ``growth_sigma``
        drives output negative by tick 5 of 60.

        The *criterion* stays yours, deliberately: "this run has gone bad" is a
        modelling claim, in the same category as naming a series GDP.  What is
        handled here is the mechanics, because the ``(N, D)`` contract is not
        optional and every caller would otherwise write the same pad.

        Carrying the last row forward is chosen over a sentinel so the loss
        stays finite and the diverged values remain in it — the candidate is
        then penalised on its own numbers rather than by a magic constant.  A
        criterion that fires too eagerly silently truncates good runs, which is
        worse than slow ones; ``model.ticks_`` reports how far the last call
        actually got, so a search can be checked rather than trusted.
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
        ticks and drop one.  Fits are always cold.  ``model.ticks_`` is how many
        ticks the last call ran before ``stop`` ended it, or ``N`` when it did
        not; ``model.free`` is the searched parameter names.

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

    if summarise is None and simulator_kwargs.get("track") is False:
        raise ValueError(
            "summarise=None reads the recorded observables, which track=False "
            "switches off. Pass a summarise callable, or leave track alone."
        )
    if summarise is None:
        simulator_kwargs["track"] = True

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
        rows: list[list[float]] = []
        for tick, state in enumerate(sim.fit_iter(**populations), start=1):
            if summarise is None:
                names, values = _recorded(sim, tick)
                model.observables_ = names  # type: ignore[attr-defined]
                rows.append(values)
            else:
                rows.append([float(v) for v in summarise(state)])
            if stop is not None and stop(rows):
                break
        model.ticks_ = len(rows)  # type: ignore[attr-defined]
        if rows:
            rows += [rows[-1]] * (int(N) - len(rows))
        return np.asarray(rows, dtype=float).reshape(len(rows), -1)

    model.free = free  # type: ignore[attr-defined]
    model.ticks_ = 0  # type: ignore[attr-defined]
    model.observables_ = ()  # type: ignore[attr-defined]
    return model


def _recorded(sim, tick: int) -> tuple[tuple[str, ...], list[float]]:
    """Tick *tick*'s recorded observables, name-sorted so columns line up.

    The simulator has already measured them — ``_observe`` writes one row per
    signal per tick — so this is a read, not a second pass over the graph.
    Sorting by name is what makes column ``j`` mean the same thing on every
    tick and across every candidate.
    """
    from skabm.history import TABLE

    if sim.connection_ is None:
        raise RuntimeError(
            "summarise=None needs recorded observables, and this rule set "
            "opened no history. It implies none, so pass a summarise callable."
        )
    rows = sim.connection_.execute(
        f"SELECT signal, level FROM {TABLE} WHERE t = ? ORDER BY signal", [int(tick)]
    ).fetchall()
    return tuple(name for name, _ in rows), [float(level) for _, level in rows]


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
