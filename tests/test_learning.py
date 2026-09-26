"""SAC learning (``behaviour.learning``): an ordinary rule on each signal's node, run
after every tick.  Forecasts are checked against ``sac_by_numpy``, the
two-pass textbook estimate, rather than golden numbers: the learner computes the same
thing from running sums, and that equivalence is the claim under test.
"""

import jax
import numpy as np
import polars as pl
import pytest
from maplib import Model

from skabm import template
from skabm.behaviour.learning import consumed, expect, sac, signal_name
from skabm.dsl import Agents, DSLError, jax_tick, sparql
from skabm.simulation import RDFSimulator
from skabm.sparql import _PREFIXES, DEF_NS

jax.config.update("jax_enable_x64", True)

_PREFIX = f"PREFIX def:<{DEF_NS}> PREFIX ex:<http://example.net/skabm#>"
OUTPUT = ("SUM", "Firm", "output")
Firm = Agents("Firm")

FIRMS = pl.DataFrame(
    {
        "id": ["firm_0", "firm_1"],
        "size": [10.0, 20.0],
        "alpha": [1e6, 2e6],  # capacity far above output: growth is never capped
        "w_bar": [30.0, 40.0],
        "tech_share": [0.4, 0.5],
        "output": [900.0, 3600.0],
        "price": [1.0, 1.0],
        "profit": [50.0, -10.0],
        "margin": [0.1, 0.05],
        "liquidity": [10.0, 10.0],
    }
).with_iri()
OPENING = 900.0 + 3600.0  # SUM of output at t=0


def firms() -> Model:
    """A fresh world per fit: a cold fit advances the graph it is given."""
    world = Model()
    world.map(template.firm, FIRMS)
    return world


@pytest.fixture
def params(poledna_params):
    return poledna_params


@pytest.fixture
def shocked(params):
    return {**params, "growth_sigma": 0.02}  # flat without shocks; see below


def sac_by_numpy(levels) -> float:
    """The Hommes & Zhu (2014) one-step forecast, written out longhand."""
    levels = np.asarray(levels, dtype=float)
    g = levels[1:] / levels[:-1] - 1.0
    a = g.mean()
    dev = g - a
    if len(dev) < 2:
        return float(a)
    num = sum(dev[k] * dev[k - 1] for k in range(1, len(dev)))
    b = float(np.clip(num / (dev * dev).sum(), -0.999, 0.999))
    return float(a + b * (g[-1] - a))


def test_reading_expect_is_what_consumes_a_signal():
    learned = {Firm.output: Firm.output * (1 + expect(*OUTPUT))}
    assert consumed([learned]) == {OUTPUT}
    assert consumed([{Firm.output: Firm.output * 1.1}]) == set()


def test_forecasts_live_on_signal_nodes(params):
    """Learning writes one forecast per signal the world can feed, upserted.

    The default rules also read household income, but this world has no
    households: no level, so no learner, and the expectation stays at the
    structural zero.  A cold refit restarts the world; a warm start continues it.
    """
    sim = RDFSimulator(params=params, n_periods=4).fit(firms())
    forecasts = f"{_PREFIX} SELECT ?s ?f WHERE {{ ?s def:forecast ?f }}"
    written = sim.model_.query(forecasts)
    assert {iri.strip("<>").rsplit("#", 1)[-1] for iri in written["s"]} == {
        "sig__SUM__Firm__output",  # the learners' nodes, one per signal read
        "sig__AVG__Firm__price",
    }

    sim.fit(firms())
    assert sim.model_.query(forecasts).height == 2  # upserted, not accumulated
    sim.warm_start = True
    assert [frame["t"][0] for frame in sim.fit_iter()] == [5, 6, 7, 8]


# ---------------------------------------------------------------------------
# behaviour.learning — SAC as running sums on the signal node
# ---------------------------------------------------------------------------


def test_sac_forecast_matches_the_textbook_formula_in_sparql_and_jax():
    """Running sums reproduce the two-pass estimate after every level, both ways.

    Including the degenerate ends: a lone level has no change to measure, so the
    forecast is the structural zero, and one growth observation has zero sample
    autocorrelation, so it is naive expectations.
    """
    levels = [100.0, 103.0, 101.0, 106.0, 104.0, 110.0]
    learner = sac(*OUTPUT)
    rule = sparql(learner, {})
    world = Model()
    world.map(template.firm, FIRMS.head(1))
    state = {
        "Firm": {"output": jax.numpy.zeros(1)}
    }  # the signal node grows from nothing
    tick = jax_tick([learner])

    for t, level in enumerate(levels):
        world.update(
            _PREFIXES + "DELETE { ?f def:output ?y } "
            f"INSERT {{ ?f def:output {level!r}e0 }} WHERE {{ ?f def:output ?y }}"
        )
        world.update(rule)
        state["Firm"]["output"] = state["Firm"]["output"] * 0 + level
        state = tick(state)

        expected = 0.0 if t == 0 else sac_by_numpy(levels[: t + 1])
        published = world.query(
            _PREFIXES + "SELECT ?f WHERE { ex:sig__SUM__Firm__output def:forecast ?f }"
        )["f"][0]
        assert published == pytest.approx(expected, abs=1e-12)
        assert float(state[signal_name(*OUTPUT)]["forecast"][0]) == pytest.approx(
            expected, abs=1e-12
        )
    assert sac_by_numpy(levels[:2]) == pytest.approx(0.03)
    with pytest.raises(DSLError, match="MAX"):
        sac("MAX", "Firm", "output")


def test_expectations_are_learned_end_to_end_from_an_empty_history(params, shocked):
    """The full loop, and the steady state that follows from having no prior.

    Nothing is assumed about growth, so a deterministic run never leaves its
    initial state: the opening forecast is zero, the realized growth it then
    learns from is zero, and it stays there.  Under shocks the AR(1) has
    something to estimate, and what gets published must equal the forecast
    recomputed by hand from the observed series.
    """

    def output(sim):
        run = pl.concat(sim.fit_iter(firms()))
        return run.group_by("t", maintain_order=True).agg(pl.col("output").sum())

    flat = output(RDFSimulator(params=params, n_periods=5))
    assert flat["output"].to_list() == [OPENING] * 5

    sim = RDFSimulator(params=shocked, n_periods=10, random_seed=7)
    levels = [OPENING, *output(sim)["output"]]
    published = sim.model_.query(
        f"{_PREFIX} SELECT ?f WHERE {{ ex:sig__SUM__Firm__output def:forecast ?f }}"
    )["f"][0]

    assert len(set(levels)) > 1
    assert published == pytest.approx(sac_by_numpy(levels))
