"""CANVAS-style behavioral expectations on the RDFSimulator machinery.

CANVAS (Hommes, He, Poledna, Siqueira & Zhang, *A Canadian behavioral
agent-based model for monetary policy*, JEDC 2024/25) is an explicit
next-generation of the Poledna et al. (2023) macro ABM this codebase
implements in ``skabm.rules``.  Its headline departure is **non-rational,
behavioral expectations**: instead of one representative agent forecasting
with a fixed rule, a small set of forecasting *heuristics* compete, and the
population weight on each evolves with its recent accuracy (the Brock-Hommes
"heuristic switching model").

This module sketches exactly that one difference — nothing else — to gauge
how far the architecture built for Poledna carries a genuinely different
expectation-formation story.  Same ``Template`` rules, same ``map_df``
mapping, same ``fit_iter`` loop; ``RDFSimulator`` learns nothing new, exactly
as with ``skabm.schelling``.

**Why this is the sharp test.**  ``skabm.rules`` documents that Poledna's
AR(1) expectations are *constants* because "SPARQL cannot run regressions" —
the paper's agents re-estimate their forecast rule on the model's own history
each quarter, and that in-graph estimation is the wall.  Heuristic switching
is a different animal: the heuristics are *fixed* and the switch is pure
arithmetic (squared forecast errors + a normalized weighting), no regression.
So the mechanism that distinguishes CANVAS from Poledna is one the graph *can*
express, and this sketch expresses it.

**The one deliberate simplification.**  Brock-Hommes weights heuristics by a
multinomial logit on accumulated performance — ``exp(beta * U_h)`` with an
intensity-of-choice ``beta``.  Plain SPARQL has no ``exp`` (the same
``log``/``exp`` gap ``skabm.rules`` already notes), so ``BELIEF_UPDATE`` uses
the exp-free counterpart: weight each heuristic by *inverse* accumulated
squared error, normalized.  Better forecasters still gain weight; only the
sharpness of the response is fixed rather than tuned by ``beta``.  The exact
logit is one UDF away — registering ``math:exp`` via ``model.add_udf`` exactly
as ``rules.register_polars_random`` registers the ``pr:`` draws — and the rule
body would change in one BIND; the seam does not move (cf. the ``geof:`` note
in ``skabm.schelling``).

Two agent classes:

* **``Firm``** — the Poledna firm scaffolding (``output``, ``alpha``,
  ``size``): heterogeneous by capacity ``alpha*size``, sharing one aggregate
  expectation.  ``FIRM_GROWTH`` is Poledna's ``FIRM_PRODUCTION`` capacity cap
  with the constant drift ``$growth_e`` replaced by the *behavioral*
  expectation read off the belief node.
* **``Belief``** — a singleton (the ``CentralBank``-singleton pattern) holding
  the macro expectation state.  Its whole initial state is *derived in-graph*
  by ``BELIEF_INIT`` from the firms' aggregate output, so the caller passes
  only a bare ``Belief=pl.DataFrame({"id": ["belief"]})`` — the same
  "population is free" trick as ``schelling.SETTLE`` / ``rules.FIRM_OWNERSHIP``.

The model is deterministic — no ``pr:`` draws — so a run is reproducible with
no seed at all (see ``tests/test_canvas.py``).
"""

from string import Template

import polars as pl

from skabm.rules import _PREFIXES

# ---------------------------------------------------------------------------
# Initialization rule — Model.insert, once, right after mapping
# ---------------------------------------------------------------------------

# The belief node's entire initial state, derived in-graph so the caller
# declares only a bare `Belief={"id": ["belief"]}` frame.  prev_output is the
# firms' initial aggregate output (a subselect, like TAYLOR_RULE reads the
# lagged aggregates); both heuristics start forecasting $init_growth, so the
# initial expectation g_exp = $init_growth reproduces Poledna's constant drift
# at t=0 and only diverges once performance data accrues.  Weights start even
# (5e-1 each) and errors at zero.
BELIEF_INIT = Template(
    _PREFIXES
    + """
    CONSTRUCT {
        ?b def:prev_output ?y0 .
        ?b def:growth 0e0 .
        ?b def:f_ada $init_growth .
        ?b def:f_trend $init_growth .
        ?b def:g_exp $init_growth .
        ?b def:u_ada 0e0 .
        ?b def:u_trend 0e0 .
        ?b def:w_ada 5e-1 .
        ?b def:w_trend 5e-1 .
    }
    WHERE {
        { SELECT (SUM(?y) AS ?y0) WHERE { ?f a ex:Firm ; def:output ?y } }
        ?b a ex:Belief .
    }
    """
)


# ---------------------------------------------------------------------------
# Update rules — Model.update, every tick (upsert pattern)
# ---------------------------------------------------------------------------

# Poledna's FIRM_PRODUCTION capacity cap (eq. 5 + 12), but the growth drift is
# the *behavioral* expectation g_exp read off the belief node rather than the
# constant $growth_e: output <- min(output * (1 + g_exp), alpha * size).  The
# `?b a ex:Belief` binding is required (not OPTIONAL), so with no Belief mapped
# the rule matches nothing and no-ops — the engine's self-scoping, unchanged.
FIRM_GROWTH = Template(
    _PREFIXES
    + """
    DELETE { ?f def:output ?y0 }
    INSERT { ?f def:output ?y1 }
    WHERE {
        ?b a ex:Belief ; def:g_exp ?g .
        ?f a ex:Firm ; def:output ?y0 ; def:alpha ?alpha ; def:size ?n .
        BIND(?y0 * (1e0 + ?g) AS ?y_desired)
        BIND(?alpha * ?n AS ?y_capacity)
        BIND(IF(?y_desired < ?y_capacity, ?y_desired, ?y_capacity) AS ?y1)
    }
    """
)

# The heuristic switch, as one upsert.  Realized aggregate growth
# g_now = SUM(output)/prev_output - 1 (subselect, run *after* FIRM_GROWTH so it
# reflects this tick's output).  The two stored forecasts f_ada / f_trend are
# the predictions made *last* tick for this one, so they are scored against
# g_now and their accumulated squared error decays with $memory (Brock-Hommes
# performance memory): u_h <- $memory * u_h + (f_h - g_now)^2.  Weights are the
# exp-free logit surrogate — inverse error, normalized, with $epsilon guarding
# the zero-error start: w_ada = (u_trend + eps) / ((u_ada + eps) + (u_trend + eps)).
# Then each heuristic posts its next forecast: adaptive is naive (expect the
# last realized growth), trend extrapolates the last change
# (g_now + $trend_kappa * (g_now - growth)).  The new expectation is the
# weight-blended forecast, and the lags shift (growth <- g_now, prev_output <-
# Y_now).  All arithmetic; no exp, no regression.
BELIEF_UPDATE = Template(
    _PREFIXES
    + """
    DELETE {
        ?b def:prev_output ?py . ?b def:growth ?g_prev .
        ?b def:f_ada ?fa0 . ?b def:f_trend ?ft0 . ?b def:g_exp ?ge0 .
        ?b def:u_ada ?ua0 . ?b def:u_trend ?ut0 .
        ?b def:w_ada ?wa0 . ?b def:w_trend ?wt0 .
    }
    INSERT {
        ?b def:prev_output ?y_now . ?b def:growth ?g_now .
        ?b def:f_ada ?fa1 . ?b def:f_trend ?ft1 . ?b def:g_exp ?ge1 .
        ?b def:u_ada ?ua1 . ?b def:u_trend ?ut1 .
        ?b def:w_ada ?wa1 . ?b def:w_trend ?wt1 .
    }
    WHERE {
        { SELECT (SUM(?y) AS ?y_now) WHERE { ?f a ex:Firm ; def:output ?y } }
        ?b a ex:Belief ;
            def:prev_output ?py ; def:growth ?g_prev ;
            def:f_ada ?fa0 ; def:f_trend ?ft0 ; def:g_exp ?ge0 ;
            def:u_ada ?ua0 ; def:u_trend ?ut0 ;
            def:w_ada ?wa0 ; def:w_trend ?wt0 .
        BIND(?y_now / ?py - 1e0 AS ?g_now)
        BIND($memory * ?ua0 + (?fa0 - ?g_now) * (?fa0 - ?g_now) AS ?ua1)
        BIND($memory * ?ut0 + (?ft0 - ?g_now) * (?ft0 - ?g_now) AS ?ut1)
        BIND((?ut1 + $epsilon) / (?ua1 + ?ut1 + 2e0 * $epsilon) AS ?wa1)
        BIND(1e0 - ?wa1 AS ?wt1)
        BIND(?g_now AS ?fa1)
        BIND(?g_now + $trend_kappa * (?g_now - ?g_prev) AS ?ft1)
        BIND(?wa1 * ?fa1 + ?wt1 * ?ft1 AS ?ge1)
    }
    """
)


# ---------------------------------------------------------------------------
# State extract — Model.query, one row per agent, no aggregation
# ---------------------------------------------------------------------------


def state_extract(model) -> pl.DataFrame:
    """Per-agent state: one row per ``Firm`` (``output``) and the singleton
    ``Belief`` (``g_exp``, the heuristic weights, last realized ``growth``).
    Macro summaries stay in polars on the caller side — same contract as
    ``rules.state_extract``."""
    return model.query(
        _PREFIXES
        + """
    SELECT ?agent ?output ?g_exp ?w_ada ?w_trend ?growth
    WHERE {
        { ?agent a ex:Firm ; def:output ?output }
        UNION { ?agent a ex:Belief ; def:g_exp ?g_exp ;
                def:w_ada ?w_ada ; def:w_trend ?w_trend ; def:growth ?growth }
    }
    """
    )


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

CANVAS_PARAMS = {
    "init_growth": 0.02,  # both heuristics' t=0 forecast (Poledna-style drift)
    "trend_kappa": 0.5,  # trend follower's extrapolation strength
    "memory": 0.9,  # Brock-Hommes performance memory (error decay)
    "epsilon": 1e-6,  # guards the zero-error start of the inverse-error weights
}

CANVAS_INIT_RULES = (BELIEF_INIT,)  # derives the whole belief state in-graph

CANVAS_UPDATE_RULES = (
    FIRM_GROWTH,  # firms grow by the behavioral expectation, capacity-capped
    BELIEF_UPDATE,  # score the heuristics, re-weight, re-forecast
)
