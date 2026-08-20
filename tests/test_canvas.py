"""CANVAS behavioral expectations on the RDFSimulator machinery — a tiny suite.

The model lives in skabm/canvas.py: heterogeneous `Firm`s grow by a single
aggregate expectation held on a `Belief` singleton, and that expectation is
formed by two competing forecast heuristics (adaptive vs trend-following)
whose weights evolve with accuracy — the Brock-Hommes heuristic switch that
sets CANVAS apart from Poledna's constant drift.  The belief node's whole
initial state is derived in-graph by BELIEF_INIT, so the caller passes only a
bare `Belief={"id": ["belief"]}` frame (no Eurostat, no calibration).

The model is deterministic (no `pr:` draws), so assertions are exact and no
seed is needed.  Covered: the engine plug-in + per-tick state, the reused
labor-capacity cap under a behavioral drift, the weights forming a proper
distribution, the switch favoring the more accurate heuristic, in-graph belief
derivation + self-scoping when no Belief is passed, and determinism.
"""

import polars as pl

from skabm.canvas import (
    BELIEF_INIT,
    CANVAS_INIT_RULES,
    CANVAS_PARAMS,
    CANVAS_UPDATE_RULES,
    FIRM_GROWTH,
    state_extract,
)
from skabm.rules import DEF_NS, EX_NS
from skabm.simulation import RDFSimulator

_PREFIX = f"PREFIX ex:<{EX_NS}> PREFIX def:<{DEF_NS}>"

# Two firms, both with labor capacity alpha * size = 100; aggregate output
# starts at 135, leaving headroom so the behavioral drift bites before the cap.
FIRMS = pl.DataFrame(
    {
        "id": ["firm_0", "firm_1"],
        "output": [90.0, 45.0],
        "alpha": [10.0, 10.0],
        "size": [10, 10],
    }
)
# The belief singleton is a bare id — BELIEF_INIT derives every field in-graph.
BELIEF = pl.DataFrame({"id": ["belief"]})


def _belief(sim, *cols: str) -> dict:
    """Read named predicates off the singleton belief node."""
    sel = " ; ".join(f"def:{c} ?{c}" for c in cols)
    row = sim.model_.query(f"{_PREFIX} SELECT * WHERE {{ ?b a ex:Belief ; {sel} }}")
    return {c: row[c][0] for c in cols}


def _canvas(**kw) -> RDFSimulator:
    return RDFSimulator(
        init_rules=CANVAS_INIT_RULES,
        update_rules=CANVAS_UPDATE_RULES,
        params=CANVAS_PARAMS,
        state_extract=state_extract,
        **kw,
    )


def test_fit_iter_yields_state_per_tick():
    # Plugs into RDFSimulator unchanged: one state frame per tick, each with a
    # firm block (output) and the belief row (g_exp).  Firms grow on the first
    # tick (aggregate 135 -> 137.7 at the seeded 2% drift).
    sim = _canvas(n_periods=12)
    states = list(sim.fit_iter(Firm=FIRMS, Belief=BELIEF))
    assert len(states) == 12

    first = states[0]
    assert first["output"].sum() > FIRMS["output"].sum()  # 137.7 > 135
    # every tick carries exactly one belief row with a finite expectation
    for s in states:
        g = s.filter(pl.col("g_exp").is_not_null())
        assert g.height == 1 and g["g_exp"].is_finite().all()


def test_growth_respects_labor_capacity():
    # The Poledna capacity cap, reused verbatim under the behavioral drift: with
    # only FIRM_GROWTH and a constant high expectation (init_growth=0.5, no
    # BELIEF_UPDATE to move it), both firms grow but never past alpha * size.
    sim = RDFSimulator(
        init_rules=[BELIEF_INIT],
        update_rules=[FIRM_GROWTH],
        params={**CANVAS_PARAMS, "init_growth": 0.5},
        n_periods=10,
    ).fit(Firm=FIRMS, Belief=BELIEF)

    out = sim.model_.query(
        f"{_PREFIX} SELECT ?f ?y ?a ?n "
        "WHERE { ?f a ex:Firm ; def:output ?y ; def:alpha ?a ; def:size ?n }"
    )
    assert (out["y"] <= out["a"] * out["n"] + 1e-9).all()
    assert (out["y"] == 100.0).all()  # both driven up to the cap


def test_weights_form_a_distribution():
    # The heuristic weights are a genuine probability split every tick.
    sim = _canvas(n_periods=12)
    for s in sim.fit_iter(Firm=FIRMS, Belief=BELIEF):
        r = s.filter(pl.col("g_exp").is_not_null())
        w_ada, w_trend = r["w_ada"][0], r["w_trend"][0]
        assert w_ada + w_trend == 1.0
        assert 0.0 <= w_ada <= 1.0 and 0.0 <= w_trend <= 1.0


def test_switching_favors_the_accurate_heuristic():
    # The core mechanism, tested by consistency rather than by predicting a
    # winner: after a run, the heuristic with the smaller accumulated squared
    # error must carry the larger weight, and the split must have moved off the
    # even 0.5 start (switching is active).  Here the trend follower wins —
    # it tracks the deceleration toward capacity better — but the assertion
    # holds whichever heuristic leads.
    sim = _canvas(n_periods=12).fit(Firm=FIRMS, Belief=BELIEF)
    b = _belief(sim, "u_ada", "u_trend", "w_ada", "w_trend")

    assert (b["u_ada"] < b["u_trend"]) == (b["w_ada"] > b["w_trend"])
    assert abs(b["w_ada"] - 0.5) > 0.01  # realized w_ada ~ 0.387, well off 0.5


def test_belief_derived_in_graph_and_scopes():
    # BELIEF_INIT materializes the whole belief state from a bare {"id"} frame:
    # after zero ticks the node already carries g_exp = init_growth and
    # prev_output = the firms' aggregate output (135), with even weights.
    derived = _belief(
        _canvas(n_periods=0).fit(Firm=FIRMS, Belief=BELIEF),
        "g_exp",
        "prev_output",
        "w_ada",
    )
    assert derived == {"g_exp": 0.02, "prev_output": 135.0, "w_ada": 0.5}

    # Self-scoping holds for the new family: with no Belief, FIRM_GROWTH finds
    # no g_exp to bind and no-ops, so firms stay at their initial output.
    sim = _canvas(n_periods=5).fit(Firm=FIRMS)
    out = sim.model_.query(
        f"{_PREFIX} SELECT ?y WHERE {{ ?f a ex:Firm ; def:output ?y }}"
    )
    assert sorted(out["y"].to_list()) == [45.0, 90.0]


def test_deterministic_across_cold_fits():
    # No randomness anywhere in CANVAS, so two cold fits reproduce each other to
    # the last digit — no seed required (contrast schelling's pr:uniform runs).
    def path(sim):
        return [
            round(s["output"].sum(), 10)
            for s in sim.fit_iter(Firm=FIRMS, Belief=BELIEF)
        ]

    assert path(_canvas(n_periods=12)) == path(_canvas(n_periods=12))
