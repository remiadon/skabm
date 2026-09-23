"""calibration.parameters — the ``(N, D)`` adapter a parameter search consumes."""

import numpy as np
import polars as pl
import pytest
from maplib import Model

from skabm.calibration import noise_floor, simulator_model
from skabm.templates import firm_template, household_template

CAL_FIRMS = pl.DataFrame({"id": [f"firm_{i}" for i in range(6)]}).with_columns(
    output=pl.lit(100.0),
    price=pl.lit(1.0),
    alpha=pl.lit(12.0),
    size=pl.lit(10.0),
    margin=pl.lit(0.15),
    liquidity=pl.lit(50.0),
    profit=pl.lit(5.0),
    w_bar=pl.lit(30.0),
)
CAL_HH = pl.DataFrame(
    {
        "id": [f"hh_{i}" for i in range(12)],
        "psi": [0.9] * 12,
        "employer": [f"firm_{i % 6}" for i in range(12)],
    }
)


def cal_world() -> Model:
    world = Model()
    world.map(firm_template, CAL_FIRMS.with_iri())
    world.map(household_template, CAL_HH.with_iri("employer"))
    return world


CAL_PARAMS = {"total_deposits": 4.0e5}


def _summarise(state: pl.DataFrame) -> list[float]:
    active = state.drop_nulls("output")
    return [active["output"].sum(), active["price"].mean()]


def _model(**kwargs):
    return simulator_model(
        cal_world,
        free=["growth_sigma"],
        summarise=_summarise,
        params=CAL_PARAMS,
        **kwargs,
    )


def test_summarise_defaults_to_the_derived_observables():
    """The yielded observables *are* the per-tick summary statistics.

    A method-of-moments loss wants a fixed vector per tick, which is what the
    rule set already implies — so the common case passes no ``summarise`` and
    the columns come out named.
    """
    model = simulator_model(cal_world, free=["growth_sigma"], params=CAL_PARAMS)
    out = model([0.0], 6, 0)

    assert out.shape == (6, len(model.observables_))
    assert "sig__SUM__Firm__output" in model.observables_
    assert model.observables_ == tuple(sorted(model.observables_))  # stable columns
    assert np.isfinite(out).all()

    # track=False narrows the vector to the signals the rules read back
    narrow = simulator_model(
        cal_world, free=["growth_sigma"], params=CAL_PARAMS, track=False
    )
    assert narrow([0.0], 6, 0).shape == (6, 3)
    assert set(narrow.observables_) < set(model.observables_)


def test_simulator_model_shape_and_guards():
    """``model(theta, N, seed) -> (N, D)``, and the two ways to misuse it."""
    model = _model()
    out = model([0.0], 5, 0)
    assert out.shape == (5, 2) and model.ticks_ == 5 and model.free == ("growth_sigma",)
    # deterministic without shocks: same theta and seed, same array
    assert (model([0.0], 5, 0) == out).all()

    with pytest.raises(ValueError, match="set per call"):
        _model(n_periods=5)
    with pytest.raises(ValueError, match="not in the parameter set"):
        simulator_model(
            cal_world, free=["no_such_knob"], summarise=_summarise, params=CAL_PARAMS
        )


def test_early_stopping_preserves_the_shape_contract():
    """A stopped run is padded, so a calibrator still gets its ``(N, D)``.

    The padding carries the last row forward rather than writing a sentinel, so
    the diverged values stay in the array and the loss stays finite — the
    candidate is penalised on its own numbers.
    """
    diverged = simulator_model(
        cal_world,
        free=["growth_sigma"],
        summarise=_summarise,
        params=CAL_PARAMS,
        stop=lambda rows: rows[-1][0] <= 0,
    )
    out = diverged([0.9], 40, 0)

    assert out.shape == (40, 2)  # contract held
    assert diverged.ticks_ < 40  # ... but the run was cut short
    assert np.isfinite(out).all()  # no sentinel leaked in
    ran = diverged.ticks_
    assert (out[ran - 1] == out[-1]).all()  # the tail is the carried last row

    # a criterion that never fires costs nothing and changes nothing
    never = simulator_model(
        cal_world,
        free=["growth_sigma"],
        summarise=_summarise,
        params=CAL_PARAMS,
        stop=lambda rows: False,
    )
    assert (never([0.9], 40, 0) == _model()([0.9], 40, 0)).all()
    assert never.ticks_ == 40


def test_noise_floor_measures_seed_spread():
    """The precision floor: how much of a loss difference is just the seed.

    Uses a stub loss so the diagnostic is tested without pulling in a calibration
    toolbox — it only ever calls ``loss_function.compute_loss(simulated, real)``.
    """

    class MeanGap:
        def compute_loss(self, simulated, real_data):
            return float(np.abs(np.mean(simulated, axis=0) - real_data).mean())

    model = simulator_model(cal_world, free=["growth_sigma"], params=CAL_PARAMS)
    real = np.zeros((4, len(model([0.0], 4, 0)[0])))

    losses = noise_floor(
        model,
        theta=[0.02],
        real_data=real,
        loss_function=MeanGap(),
        ensemble_size=2,
        n_repeats=3,
    )
    assert len(losses) == 3 and all(np.isfinite(losses))
    # disjoint seed sets, so the repeats are not identical by construction
    assert len(set(losses)) > 1


def test_derived_summarise_needs_something_to_derive():
    """A rule set that implies no observable says so instead of returning (N, 0)."""
    model = simulator_model(
        cal_world,
        free=["growth_sigma"],
        params=CAL_PARAMS,
        init_rules=(),
        update_rules=(),
    )
    # no rules at all, so Firm is inert and warns before the failure we are after
    with (
        pytest.warns(UserWarning, match="not referenced"),
        pytest.raises(RuntimeError, match="implies no observable"),
    ):
        model([0.0], 3, 0)
