"""State history (``skabm.history``) and SAC learning (``behaviour.learning``).

The two are independent: the engine measures aggregates and, when asked, persists
them; learning is an ordinary rule on each signal's node, run after every tick.  The
tests split the same way.  Forecasts are checked against ``sac_by_numpy``, the
two-pass textbook estimate, rather than golden numbers: the learner computes the same
thing from running sums, and that equivalence is the claim under test.
"""

import jax
import numpy as np
import polars as pl
import pytest
from maplib import Model

from skabm import history as H
from skabm.behaviour.learning import expect, sac
from skabm.dsl import Agents, DSLError, jax_tick
from skabm.ottr import firm_template
from skabm.simulation import RDFSimulator
from skabm.sparql import _PREFIXES, DEF_NS, render

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
    world.map(firm_template, FIRMS)
    return world


@pytest.fixture
def params(poledna_params):
    return {**poledna_params, "total_deposits": 1000.0}


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


# ---------------------------------------------------------------------------
# skabm.history — generic; knows nothing about what is being learned
# ---------------------------------------------------------------------------


PLAIN = {Firm.output: Firm.output * 1.1}


def test_measuring_needs_no_database():
    """A database is only ever asked for, never implied by the rules.

    Reading ``expect`` is what makes a rule consume a signal, and a rule set yields
    every observable it implies either way.  A
    ``duckdb_connection`` is the one reason to open a database.
    """
    learned = {Firm.output: Firm.output * (1 + expect(*OUTPUT))}
    assert H.consumed([learned]) == {OUTPUT}
    assert H.consumed([PLAIN]) == set()

    bare = RDFSimulator(init_rules=(), update_rules=(PLAIN,), n_periods=2)
    run = pl.DataFrame(bare.fit_iter(firms()))
    assert bare.connection_ is None
    assert set(run.columns) == {
        "t",
        "sig__SUM__Firm__output",
        "sig__AVG__Firm__output",
    }
    with pytest.raises(RuntimeError, match="no history was kept"):
        bare.history()

    kept = RDFSimulator(
        init_rules=(),
        update_rules=(PLAIN,),
        n_periods=2,
        duckdb_connection=H.connect(None),
    ).fit(firms())
    assert set(kept.history()["signal"]) == set(run.columns) - {"t"}

    narrow = RDFSimulator(
        init_rules=(), update_rules=(PLAIN,), n_periods=1, track=False
    )
    *_, row = narrow.fit_iter(firms())
    assert set(row) == {"t"}  # this rule set consumes nothing, so nothing is left


def test_measurement_covers_every_observable_of_every_class_present(params):
    """One value per observable per tick, and an absent class is dropped.

    Firm-only, so every Household / Government / CentralBank observable the
    default rule set implies has nobody to aggregate: those columns are not in
    the run and no row is written for them, while the Firm ones are all there
    and not just the two the rules read back.
    """
    sim = RDFSimulator(params=params, n_periods=4, duckdb_connection=H.connect(None))
    run = pl.DataFrame(sim.fit_iter(firms()))
    history = H.state_frame(sim.connection_)
    recorded = set(history["signal"])

    assert set(run.columns) == {"t"} | {o.signal for o in sim.observables_}
    assert {o.signal for o in sim.observables_ if o.klass == "Firm"} == recorded
    assert recorded > {"sig__SUM__Firm__output", "sig__AVG__Firm__price"}
    assert not any(s.startswith("sig__SUM__Household") for s in recorded)
    assert not any(c.startswith("sig__SUM__Household") for c in run.columns)
    # the binding constraint nobody declared: alpha is far above output here
    assert run["sig__AVG__Firm__binds__output"].to_list() == [0.0] * 4
    # t=0 is recorded before the first tick, so four ticks leave five rows
    assert history.filter(pl.col("signal") == "sig__SUM__Firm__output").height == 5


def test_forecasts_live_on_signal_nodes_and_the_table_persists(tmp_path, params):
    """Learning writes one forecast per signal the world can feed, and the table
    of observables follows the world's lifecycle.

    The default rules also read household income, but this world has no
    households: no level, so no learner, and the expectation stays at the
    structural zero.  A cold refit restarts the world including its database, a
    warm start continues it, and a file-backed run outlives the process.
    """
    path = str(tmp_path / "history.duckdb")
    sim = RDFSimulator(params=params, n_periods=4, duckdb_connection=path)
    sim.fit(firms())

    forecasts = f"{_PREFIX} SELECT ?s ?f WHERE {{ ?s def:forecast ?f }}"
    written = sim.model_.query(forecasts)
    assert {iri.strip("<>").rsplit("#", 1)[-1] for iri in written["s"]} == {
        "sig__SUM__Firm__output",
        "sig__AVG__Firm__price",
    }

    sim.fit(firms())  # cold refit clears the previous run's rows
    assert H.state_frame(sim.connection_)["t"].max() == 4
    assert sim.model_.query(forecasts).height == 2  # upserted, not accumulated

    sim.warm_start = True  # ... and a warm start appends rather than restarting
    sim.fit()
    assert H.state_frame(sim.connection_)["t"].max() == 8

    # a fresh simulator continuing the world picks the tick count up from the file
    resumed = RDFSimulator(
        params=params, n_periods=2, warm_start=True, duckdb_connection=path
    )
    assert [row["t"] for row in resumed.fit_iter(sim.model_)] == [9, 10]

    import duckdb

    with duckdb.connect(path) as con:  # survived the simulator that wrote it
        assert con.execute(f"SELECT COUNT(*) FROM {H.TABLE}").fetchone()[0] > 0


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
    rule = render(learner, {})
    world = Model()
    world.map(firm_template, FIRMS.head(1))
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
        assert float(state[H.signal_name(*OUTPUT)]["forecast"][0]) == pytest.approx(
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
    flat = pl.DataFrame(RDFSimulator(params=params, n_periods=5).fit_iter(firms()))
    assert flat["sig__SUM__Firm__output"].to_list() == [OPENING] * 5

    sim = RDFSimulator(params=shocked, n_periods=10, random_seed=7)
    run = pl.DataFrame(sim.fit_iter(firms()))
    levels = [OPENING, *run["sig__SUM__Firm__output"]]
    published = sim.model_.query(
        f"{_PREFIX} SELECT ?f WHERE {{ ex:sig__SUM__Firm__output def:forecast ?f }}"
    )["f"][0]

    assert len(set(levels)) > 1
    assert published == pytest.approx(sac_by_numpy(levels))
