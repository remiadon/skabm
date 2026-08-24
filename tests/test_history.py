"""State history (``skabm.history``) and SAC learning (``behaviour.learning``).

The two are deliberately independent — the engine records aggregates and writes
back whatever a history rule computes, knowing nothing about SAC — so the tests
split the same way: the first block covers the generic machinery, the second the
learning rule layered on it.

Forecasts are checked against ``sac_by_numpy`` rather than golden numbers, and
the two invariants that fail *silently* get dedicated tripwires: the ``ORDER BY``
that puts rows in time order, and keeping the virtualization off the simulation
graph (registering one nulls every aggregate the behaviour rules use).
"""

import numpy as np
import polars as pl
import pytest

from skabm import history as H
from skabm.behaviour.learning import expect, register_sac, sac_learning
from skabm.rules import DEF_NS
from skabm.simulation import RDFSimulator

_PREFIX = f"PREFIX def:<{DEF_NS}> PREFIX ex:<http://example.net/skabm#>"
PARAMS = {"total_deposits": 1000.0}
SHOCKED = {**PARAMS, "growth_sigma": 0.02}  # flat without shocks; see below
OUTPUT = ("SUM", "Firm", "output")

FIRMS = pl.DataFrame(
    {
        "id": ["firm_0", "firm_1"],
        "size": [10, 20],
        "alpha": [1e6, 2e6],  # capacity far above output: growth is never capped
        "w_bar": [30.0, 40.0],
        "tech_share": [0.4, 0.5],
        "output": [900.0, 3600.0],
        "price": [1.0, 1.0],
        "profit": [50.0, -10.0],
        "margin": [0.1, 0.05],
        "liquidity": [10.0, 10.0],
    }
)


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


@pytest.fixture
def learner_over():
    """A learner over a hand-written series, inserted in whatever order is asked."""

    def build(levels, order=None):
        con = H.connect(None)
        con.execute(f"DELETE FROM {H.TABLE}")
        con.executemany(
            f"INSERT INTO {H.TABLE} VALUES (?, ?, ?)",
            [
                (H.signal_name(*OUTPUT), t, levels[t])
                for t in (range(len(levels)) if order is None else order)
            ],
        )
        learner = H.build_learner(con, [OUTPUT])
        register_sac(learner)
        return learner

    return build


# ---------------------------------------------------------------------------
# skabm.history — generic; knows nothing about what is being learned
# ---------------------------------------------------------------------------


def test_signals_drive_recording_and_nothing_else_opens_a_database():
    """Naming ``ex:sig__`` in rule text is the whole opt-in mechanism."""
    assert H.parse_signals([expect(*OUTPUT, out="g_e")]) == [OUTPUT]
    assert H.parse_signals(["no signals here"]) == []

    plain = f"""{_PREFIX}
    DELETE {{ ?f def:output ?y0 }} INSERT {{ ?f def:output ?y1 }}
    WHERE {{ ?f a ex:Firm ; def:output ?y0 . BIND(?y0 * 1.1e0 AS ?y1) }}"""
    bare = RDFSimulator(
        init_rules=(), update_rules=(plain,), history_rules=(), n_periods=2
    ).fit(Firm=FIRMS)
    assert bare.connection_ is None and bare.learner_ is None


def test_recording_covers_every_declared_signal_that_has_a_population():
    """One row per signal per tick, and an absent class is skipped, not faked."""
    # Firm-only: the default rule set also declares Household income, which has
    # no population here and so aggregates to nothing
    sim = RDFSimulator(params=PARAMS, n_periods=4).fit(Firm=FIRMS)
    history = H.state_frame(sim.connection_)

    assert set(history["signal"]) == {"sig__SUM__Firm__output", "sig__AVG__Firm__price"}
    # t=0 is recorded before the first tick, so four ticks leave five rows
    assert history.filter(pl.col("signal") == "sig__SUM__Firm__output").height == 5


def test_history_rule_results_are_upserted_as_predicates_on_their_subject(tmp_path):
    """The generic write-back contract, plus the storage lifecycle around it.

    A history rule projects a subject IRI first and value columns after it,
    which land as ``def:`` predicates — one current value per signal however
    many ticks run.  A cold refit restarts the world including its memory, a
    warm start continues it, and a file-backed run outlives the process.
    """
    path = str(tmp_path / "history.duckdb")
    sim = RDFSimulator(params=PARAMS, n_periods=4, duckdb_connection=path)
    sim.fit(Firm=FIRMS)

    written = sim.model_.query(f"{_PREFIX} SELECT ?s ?f WHERE {{ ?s def:forecast ?f }}")
    assert {iri.strip("<>").rsplit("#", 1)[-1] for iri in written["s"]} == {
        "sig__SUM__Firm__output",
        "sig__AVG__Firm__price",
    }

    sim.fit(Firm=FIRMS)  # cold refit clears the previous run's memory
    assert H.state_frame(sim.connection_)["t"].max() == 4
    # upserted, not accumulated: still one forecast per signal
    assert (
        sim.model_.query(f"{_PREFIX} SELECT ?f WHERE {{ ?s def:forecast ?f }}").height
        == 2
    )

    sim.warm_start = True  # ... and a warm start appends rather than restarting
    sim.fit()
    assert H.state_frame(sim.connection_)["t"].max() == 8

    import duckdb

    with duckdb.connect(path) as con:  # survived the simulator that wrote it
        assert con.execute(f"SELECT COUNT(*) FROM {H.TABLE}").fetchone()[0] > 0


def test_virtualization_is_kept_off_the_simulation_graph():
    """``add_virtualization`` nulls every graph-local aggregate on its model.

    That is why the learner is a separate ``Model``.  Pinned because violating
    it would silently no-op firm_sales and the Taylor rule rather than raise.
    """
    sim = RDFSimulator(params=PARAMS, n_periods=3).fit(Firm=FIRMS)
    total = sim.model_.query(
        f"{_PREFIX} SELECT (SUM(?y) AS ?total) WHERE {{ ?f a ex:Firm ; def:output ?y }}"
    )["total"][0]
    assert total is not None and total > 0


# ---------------------------------------------------------------------------
# behaviour.learning — SAC as one SPARQL SELECT plus one polars UDF
# ---------------------------------------------------------------------------


def test_sac_forecast_matches_the_textbook_formula(learner_over):
    """Including the degenerate ends: one growth observation, and none at all."""
    query = sac_learning.substitute({})
    levels = [100.0, 103.0, 101.0, 106.0, 104.0, 110.0]

    assert learner_over(levels).query(query)["forecast"][0] == pytest.approx(
        sac_by_numpy(levels)
    )
    # one growth observation has zero sample autocorrelation: naive expectations
    assert learner_over([100.0, 103.0]).query(query)["forecast"][0] == pytest.approx(
        0.03
    )
    # a lone level has no change to measure, so there is nothing to publish
    assert learner_over([100.0]).query(query)["forecast"][0] is None


def test_order_by_is_what_puts_the_series_in_time_order(learner_over):
    """Rows reach a UDF in database order; the sub-SELECT's ORDER BY fixes that.

    Pinned because removing it does not raise — it returns a forecast fitted to
    a shuffled series.
    """
    levels = [100.0, 103.0, 101.0, 106.0, 104.0, 110.0]
    query = sac_learning.substitute({})
    learner = learner_over(levels, order=[3, 0, 5, 1, 4, 2])

    assert learner.query(query)["forecast"][0] == pytest.approx(sac_by_numpy(levels))
    unordered = learner.query(query.replace("ORDER BY ?ext ?t", ""))["forecast"][0]
    assert unordered != pytest.approx(sac_by_numpy(levels))


def test_expectations_are_learned_end_to_end_from_an_empty_history():
    """The full loop, and the steady state that follows from having no prior.

    Nothing is assumed about growth, so a deterministic run never leaves its
    initial state: the opening forecast is zero, the realized growth it then
    learns from is zero, and it stays there.  Under shocks the AR(1) has
    something to estimate, and what gets published must equal the forecast
    recomputed by hand from the recorded series.
    """
    flat = RDFSimulator(params=PARAMS, n_periods=5).fit(Firm=FIRMS)
    levels = H.state_frame(flat.connection_).filter(
        pl.col("signal") == "sig__SUM__Firm__output"
    )["level"]
    assert levels.n_unique() == 1  # steady state: no prior to push it off

    sim = RDFSimulator(params=SHOCKED, n_periods=10, random_seed=7).fit(Firm=FIRMS)
    shocked = (
        H.state_frame(sim.connection_)
        .filter(pl.col("signal") == "sig__SUM__Firm__output")
        .sort("t")["level"]
        .to_list()
    )
    published = sim.model_.query(
        f"{_PREFIX} SELECT ?f WHERE {{ ex:sig__SUM__Firm__output def:forecast ?f }}"
    )["f"][0]

    assert len(set(shocked)) > 1
    assert published == pytest.approx(sac_by_numpy(shocked))
