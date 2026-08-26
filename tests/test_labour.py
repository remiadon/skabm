"""Occupational mobility and automation on the RDFSimulator machinery — del
Rio-Chanona et al. (2021), J. R. Soc. Interface 18:20200898.

The model lives in skabm/behaviour/labour.py: occupations are the agents, the
job-transition network A_ij is a reified `Edge` population, and a one-row
`Clock` carries the calendar time the automation S-curve needs.  There is no
calibration layer here — the network is synthetic and the occupation frame is
seeded by `labour.occupations`.

The rules need `math:exp` (the urn-ball matching function and the S-curve are
both transcendental), which arrives through the `udfs=` hyperparameter as
`labour.LABOUR_UDFS`.

Covered: the two conservation identities that pin the transcription of eqs.
13-17 (labour force constant, urn-ball hire count), convergence to a steady
state, the duration-cohort invariant, and the two headline results — a
downward-sloping Beveridge curve under a demand cycle, and an automation
shock whose cost depends on the network it lands on.
"""

import math

import polars as pl
import pytest

from skabm.behaviour.labour import (
    LABOUR_UDFS,
    LABOUR_UPDATE_RULES,
    labour_flow,
    labour_params,
    mobility_network,
    occupations,
    reallocate_demand,
)
from skabm.rules import _PREFIXES
from skabm.simulation import RDFSimulator

NEVER = 1e6  # shock_start far enough away that the S-curve stays dormant


def ring(n: int = 8, reach: int = 3) -> pl.DataFrame:
    """A ring of occupations, each reachable from its `reach` nearest
    neighbours with a distance-decayed transition count."""
    return mobility_network(
        pl.DataFrame(
            [
                {
                    "src": f"occ_{i}",
                    "dst": f"occ_{(i + d) % n}",
                    "count": float(reach + 1 - abs(d)),
                }
                for i in range(n)
                for d in [*range(-reach, 0), *range(1, reach + 1)]
            ]
        )
    )


def clock() -> pl.DataFrame:
    return pl.DataFrame({"id": ["clock"], "t": [0.0]})


def world(n: int = 8, employment: float = 1000.0, **overrides):
    """A settled-ish labour market: populations plus a simulator wired for it."""
    occ = occupations(
        pl.DataFrame(
            {"id": [f"occ_{i}" for i in range(n)], "employment": [employment] * n}
        )
    )
    sim = RDFSimulator(
        init_rules=(),
        update_rules=LABOUR_UPDATE_RULES,
        params={**labour_params, "shock_start": NEVER, **overrides.pop("params", {})},
        udfs=LABOUR_UDFS,
        **overrides,
    )
    return sim, {"Occupation": occ, "Edge": ring(n), "Clock": clock()}


def macro(state: pl.DataFrame) -> dict:
    """The paper's aggregates, off the IR-derived per-agent extract."""
    e, u, v, ltu = state.select(
        pl.col("employment").sum(),
        pl.col("unemployment").sum(),
        pl.col("vacancies").sum(),
        pl.col("ltu").sum(),
    ).row(0)
    return {"e": e, "u": u, "v": v, "ltu": ltu, "u_rate": u / (e + u)}


def test_reallocate_demand_preserves_the_aggregate():
    # Eq. 18 moves labour demand between occupations without creating or
    # destroying any: that is what makes an automation shock a reallocation
    # rather than a slump, and it is what lets the resulting unemployment be
    # read as mismatch.
    demand = pl.Series([1000.0, 2000.0, 500.0, 4000.0])
    automation = pl.Series([0.9, 0.1, 0.5, 0.0])
    assert reallocate_demand(demand, automation).sum() == pytest.approx(demand.sum())


def test_reallocate_demand_moves_work_away_from_automatable_jobs():
    demand = pl.Series([1000.0, 1000.0])
    shifted = reallocate_demand(demand, pl.Series([0.8, 0.0]))
    assert shifted[0] < demand[0] and shifted[1] > demand[1]
    # with no automation anywhere, nothing moves
    assert reallocate_demand(demand, pl.Series([0.0, 0.0])).to_list() == pytest.approx(
        demand.to_list()
    )


def test_mobility_network_rows_sum_to_one():
    # Eqs. 20-21: P_ij is row-normalised and A_ij = (1-r)P_ij off-diagonal with
    # A_ii = r, so every occupation's outgoing weights are a distribution.
    edges = ring()
    rows = edges.group_by("src").agg(total=pl.col("weight").sum())
    assert rows["total"].to_list() == pytest.approx([1.0] * len(rows))
    # the diagonal is imposed, not observed
    diag = edges.filter(pl.col("src") == pl.col("dst"))
    assert diag["weight"].to_list() == pytest.approx([0.55] * len(diag))


def test_labour_force_is_conserved():
    # Eqs. 2-3: separations move workers from e to u and hires move them back,
    # so e + u is invariant.  This is also the tripwire for maplib's
    # right-associative arithmetic — `?e0 - ?w + ?hired` parsed as
    # `?e0 - (?w + ?hired)` leaks workers on every tick and nothing else fails.
    sim, pops = world(n=8, n_periods=25)
    totals = [macro(s)["e"] + macro(s)["u"] for s in sim.fit_iter(**pops)]
    assert totals == pytest.approx([totals[0]] * len(totals), rel=1e-12)
    assert totals[0] == pytest.approx(8 * 1000.0 * 1.05)


def test_hires_match_the_urn_ball_count():
    # Eq. 16 summed over origin occupations collapses to v_j (1 - exp(-s_j/v_j)):
    # the number of j's vacancies that drew at least one applicant.  This pins
    # the whole matching function, the v_j^2 in the numerator included.
    stop = LABOUR_UPDATE_RULES.index(labour_flow) + 1  # halt before clearing,
    sim, pops = world(n=6, n_periods=1)  # so v and s still hold what f used
    sim.set_params(update_rules=LABOUR_UPDATE_RULES[:stop])
    sim.fit(**pops)
    got = sim.model_.query(
        _PREFIXES
        + """
        SELECT ?j ?v ?s (SUM(?f) AS ?hires) WHERE {
            ?e a ex:Edge ; def:dst ?j ; def:flow ?f .
            ?j def:vacancies ?v ; def:applications ?s
        } GROUP BY ?j ?v ?s"""
    )
    assert len(got) == 6
    for v, s, hires in got.select("v", "s", "hires").iter_rows():
        assert hires == pytest.approx(v * (1 - math.exp(-s / v)))


def test_duration_cohorts_sum_to_unemployment():
    # The four duration stages partition the unemployed: this tick's separations
    # enter the first, each cohort that fails to match ages into the next, and
    # `ltu` absorbs everyone past 4 steps (= 27 weeks).  The sum is the invariant.
    sim, pops = world(n=6, n_periods=12)
    for _ in sim.fit_iter(**pops):
        pass
    per_occ = sim.model_.query(
        _PREFIXES
        + """
        SELECT ?o ?u ?a ?b ?c ?l WHERE {
            ?o a ex:Occupation ; def:unemployment ?u ; def:u_spell1 ?a ;
               def:u_spell2 ?b ; def:u_spell3 ?c ; def:ltu ?l
        }"""
    )
    assert len(per_occ) == 6
    for u, a, b, c, ltu in per_occ.select("u", "a", "b", "c", "l").iter_rows():
        assert a + b + c + ltu == pytest.approx(u)
        assert 0 < ltu < u  # long-term unemployment builds up but never absorbs all


def test_converges_to_a_steady_state():
    # With target demand flat the market settles: the paper initialises this way
    # before shocking anything.
    sim, pops = world(n=8, n_periods=120)
    path = [macro(s)["u_rate"] for s in sim.fit_iter(**pops)]
    assert path[-1] == pytest.approx(path[-2], rel=1e-6)
    assert 0.01 < path[-1] < 0.15  # a plausible natural rate, not a corner


def test_beveridge_curve_slopes_downward():
    # The paper's macro validation (fig. 3): cycling *aggregate* demand traces a
    # downward-sloping relation between unemployment and vacancies — when
    # vacancies are plentiful the unemployed match faster.
    sim, pops = world(n=8, n_periods=60)
    for _ in sim.fit_iter(**pops):
        pass
    sim.set_params(n_periods=1, warm_start=True)

    def cycle(t):
        return 1.0 + 0.05 * math.sin(2 * math.pi * t / 40)

    curve = []
    for tick in range(80):
        sim.model_.update(
            _PREFIXES
            + f"""
            DELETE {{ ?o def:demand_init ?d0 }}
            INSERT {{ ?o def:demand_init ?d1 }}
            WHERE {{ ?o a ex:Occupation ; def:demand_init ?d0 .
                     BIND(?d0 * {cycle(tick + 1) / cycle(tick):.9e} AS ?d1) }}"""
        )
        for state in sim.fit_iter():
            pass
        m = macro(state)
        curve.append({"u": m["u_rate"], "v": m["v"] / (m["e"] + m["v"])})
    assert pl.DataFrame(curve).select(pl.corr("u", "v")).item() < -0.3


def segmented(n: int = 12) -> pl.DataFrame:
    """Two occupational communities joined by a single thin bridge — the
    network structure that makes a reallocation shock bite."""
    half = n // 2
    rows = [
        {"src": f"occ_{i}", "dst": f"occ_{j}", "count": 10.0}
        for block in (range(half), range(half, n))
        for i in block
        for j in block
        if i != j
    ]
    rows += [
        {"src": f"occ_{half - 1}", "dst": f"occ_{half}", "count": 1.0},
        {"src": f"occ_{half}", "dst": f"occ_{half - 1}", "count": 1.0},
    ]
    return mobility_network(pl.DataFrame(rows))


def shock_outcome(edges: pl.DataFrame, n: int) -> tuple[dict, dict]:
    """Settle, then halve target demand for the first half of the occupations
    and raise it for the second half.  Aggregate demand is unchanged, so
    whatever unemployment results is mismatch, not a slump."""
    occ = occupations(
        pl.DataFrame({"id": [f"occ_{i}" for i in range(n)], "employment": [1000.0] * n})
    )
    sim = RDFSimulator(
        init_rules=(),
        update_rules=LABOUR_UPDATE_RULES,
        params={**labour_params, "shock_start": NEVER},
        udfs=LABOUR_UDFS,
        n_periods=60,
    )
    sim.fit(Occupation=occ, Edge=edges, Clock=clock())
    before = macro(sim.extract())

    sim.model_.update(
        _PREFIXES
        + f"""
        DELETE {{ ?o def:demand_final ?f0 }}
        INSERT {{ ?o def:demand_final ?f1 }}
        WHERE {{
            ?o a ex:Occupation ; def:demand_init ?d ; def:demand_final ?f0 .
            BIND(xsd:integer(STRAFTER(STR(?o), "#occ_")) AS ?i)
            BIND(IF(?i < {n // 2}, ?d * 0.5e0, ?d * 1.5e0) AS ?f1)
        }}"""
    )
    sim.set_params(
        n_periods=200, warm_start=True, params={**labour_params, "shock_start": 0.0}
    )
    sim.fit()
    return before, macro(sim.extract())


def test_network_structure_decides_what_the_shock_costs():
    # The paper's central claim, as a controlled experiment: the *same*
    # reallocation shock on two networks.  Where occupations are mutually
    # reachable the displaced simply move and unemployment barely notices;
    # where the network is segmented they have nowhere to go and it bites.
    before_seg, after_seg = shock_outcome(segmented(12), 12)
    before_open, after_open = shock_outcome(ring(12, reach=5), 12)

    # both start from the same natural rate — only the topology differs
    assert before_seg["u_rate"] == pytest.approx(before_open["u_rate"], rel=0.25)

    assert after_seg["u_rate"] > 2 * before_seg["u_rate"]
    assert after_seg["ltu"] > before_seg["ltu"]
    assert after_seg["u_rate"] > 2 * after_open["u_rate"]

    # aggregate demand never moved, so nothing was destroyed — this is mismatch
    for before, after in ((before_seg, after_seg), (before_open, after_open)):
        assert after["e"] + after["u"] == pytest.approx(before["e"] + before["u"])


def test_labour_rules_need_the_math_udf():
    # math:exp is not a SPARQL built-in and not in the default registrar set;
    # the rules that need it fail loudly rather than silently, which is what
    # makes `udfs` a hyperparameter worth having.
    sim, pops = world(n=4, n_periods=1)
    sim.set_params(udfs=())
    with pytest.raises(Exception, match="(?i)function"):
        sim.fit(**pops)
