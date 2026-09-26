"""calibration.parameters — the ``(N, D)`` adapter a parameter search consumes."""

import numpy as np
import polars as pl
import pytest
from maplib import Model

from skabm import template
from skabm.behaviour.household import initial
from skabm.calibration import noise_floor, simulator_model

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
    world.map(template.firm, CAL_FIRMS.with_iri())
    households = initial(CAL_HH, CAL_FIRMS, total_deposits=4.0e5)
    world.map(template.household, households.with_iri("employer"))
    return world


@pytest.fixture
def cal_params(poledna_params):
    return poledna_params


def _summarise(state: pl.DataFrame) -> list[float]:
    active = state.drop_nulls("output")
    return [active["output"].sum(), active["price"].mean()]


def _model(cal_params, **kwargs):
    return simulator_model(
        cal_world,
        free=["growth_sigma"],
        summarise=_summarise,
        params=cal_params,
        **kwargs,
    )


def test_simulator_model_shape_and_guards(cal_params):
    """``model(theta, N, seed) -> (N, D)``, and the two ways to misuse it."""
    model = _model(cal_params)
    out = model([0.0], 5, 0)
    assert out.shape == (5, 2) and model.ticks_ == 5 and model.free == ("growth_sigma",)
    # deterministic without shocks: same theta and seed, same array
    assert (model([0.0], 5, 0) == out).all()

    with pytest.raises(ValueError, match="set per call"):
        _model(cal_params, n_periods=5)
    with pytest.raises(ValueError, match="not in the parameter set"):
        simulator_model(
            cal_world, free=["no_such_knob"], summarise=_summarise, params=cal_params
        )


def test_early_stopping_preserves_the_shape_contract(cal_params):
    """A stopped run is padded, so a calibrator still gets its ``(N, D)``.

    The padding carries the last row forward rather than writing a sentinel, so
    the diverged values stay in the array and the loss stays finite — the
    candidate is penalised on its own numbers.
    """
    diverged = simulator_model(
        cal_world,
        free=["growth_sigma"],
        summarise=_summarise,
        params=cal_params,
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
        params=cal_params,
        stop=lambda rows: False,
    )
    assert (never([0.9], 40, 0) == _model(cal_params)([0.9], 40, 0)).all()
    assert never.ticks_ == 40


def test_noise_floor_measures_seed_spread(cal_params):
    """The precision floor: how much of a loss difference is just the seed.

    Uses a stub loss so the diagnostic is tested without pulling in a calibration
    toolbox — it only ever calls ``loss_function.compute_loss(simulated, real)``.
    """

    class MeanGap:
        def compute_loss(self, simulated, real_data):
            return float(np.abs(np.mean(simulated, axis=0) - real_data).mean())

    model = _model(cal_params)
    real = np.zeros((4, 2))

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
