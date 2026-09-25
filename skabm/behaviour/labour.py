"""Occupational mobility and automation, del Rio-Chanona et al. (2021), mean-field. Seven
rules, in this order.

1. Each step, the clock's t advances by one.

2. An occupation's target_demand has two cases. Before the shock, when (the total over
clocks of their t) times the parameter weeks_per_step / 52 - the parameter shock_start is
below 0, it becomes its demand_init. Otherwise it becomes its demand_init plus (its
demand_final minus its demand_init) divided by
(1 + exp(-the parameter shock_k times ((the total over clocks of their t) times the
parameter weeks_per_step / 52 - the parameter shock_start - the parameter
shock_halfway))).

3. Then an occupation's separations, openings and app_norm update together, in one rule.
Its separations become the parameter delta_u times its employment plus (1 - delta_u)
times the smaller of its employment and the parameter gamma times the larger of 0 and
(employment + vacancies - target_demand). Its openings become the parameter delta_v
times its employment plus (1 - delta_v) times the smaller of its employment and gamma
times the larger of 0 and (target_demand - employment - vacancies). Its app_norm becomes
the sum, over the edges whose source is this occupation, of the edge's weight times the
vacancies of the edge's destination.

4. An occupation's applications become its vacancies times the sum, over the edges whose
destination is this occupation, of the edge's weight times a share that is the source's
unemployment divided by the source's app_norm when the source's app_norm is above 0, and
0 otherwise.

5. An edge's flow becomes the edge's weight times the source's unemployment divided by
the source's app_norm, times the destination's vacancies squared divided by the
destination's applications, times (1 - exp(-the destination's applications divided by
the destination's vacancies)), when the destination's vacancies are above 0 and the
destination's applications are above 0 and the source's app_norm is above 0; and 0
otherwise.

6. An occupation's job_finding becomes 0 when its unemployment is not above 0, and
otherwise the sum of flow over the edges whose source is this occupation divided by its
unemployment.

7. Then an occupation's stocks and unemployment spells update together, in one rule.
Employment becomes employment minus separations plus the sum of flow over the edges
whose destination is this occupation. Unemployment becomes unemployment plus separations
minus the sum of flow over the edges whose source is this occupation. Vacancies become
vacancies plus openings minus the sum of flow over the edges whose destination is this
occupation. u_spell1 becomes separations; u_spell2 becomes u_spell1 times (1 -
job_finding); u_spell3 becomes u_spell2 times (1 - job_finding); and ltu becomes
(u_spell3 + ltu) times (1 - job_finding).
"""

from __future__ import annotations

import polars as pl
import sympy as sp

from skabm.dsl import Agents, sum_over, total
from skabm.sparql import register_math, register_polars_random

# the rules call math:exp: RDFSimulator(udfs=LABOUR_UDFS)
LABOUR_UDFS = (register_polars_random, register_math)

Occupation, Edge, Clock = Agents("Occupation"), Agents("Edge"), Agents("Clock")
weeks_per_step, shock_start, shock_halfway, shock_k = sp.symbols(
    "weeks_per_step shock_start shock_halfway shock_k"
)
delta_u, delta_v, gamma = sp.symbols("delta_u delta_v gamma")

PARAMETERS = {
    weeks_per_step: 6.75,  # one time step, in weeks, del Rio-Chanona et al. (2021) Table 1
    shock_start: 20.0,  # scenario knob: years before the S-curve begins (past the burn-in)
    shock_halfway: 15.0,  # scenario knob: with shock_k, "30 years, mostly within 10"
    shock_k: 0.79,  # scenario knob: S-curve steepness
    delta_u: 0.0160,  # spontaneous separation rate, del Rio-Chanona et al. (2021) Table 1
    delta_v: 0.0120,  # spontaneous vacancy-opening rate, del Rio-Chanona et al. (2021) Table 1
    gamma: 0.160,  # speed of adjustment to target demand, del Rio-Chanona et al. (2021) Table 1
}

# del Rio-Chanona et al. (2021), J. R. Soc. Interface 18:20200898
clock_tick = {Clock.t: Clock.t + 1}
# eq. 19 (the exponent's sign as the text implies)
years = total(Clock.t) * weeks_per_step / 52
since = years - shock_start
initial, final = Occupation.demand_init, Occupation.demand_final
labour_target = {
    Occupation.target_demand: sp.Piecewise(
        (initial, since < 0),
        (
            initial
            + (final - initial) / (1 + sp.exp(-shock_k * (since - shock_halfway))),
            True,
        ),
    )
}
# eqs. 7, 9-12
employed, target = Occupation.employment, Occupation.target_demand
realised = employed + Occupation.vacancies
excess = sp.Min(employed, gamma * sp.Max(0, realised - target))
shortfall = sp.Min(employed, gamma * sp.Max(0, target - realised))
labour_demand = {
    Occupation.separations: delta_u * employed + (1 - delta_u) * excess,
    Occupation.openings: delta_v * employed + (1 - delta_v) * shortfall,
    Occupation.app_norm: sum_over(Edge.src, Edge.weight * Edge.dst.vacancies),
}
# eq. 17
origin = Edge.src
share = sp.Piecewise(
    (origin.unemployment / origin.app_norm, origin.app_norm > 0), (0, True)
)
applications = {
    Occupation.applications: Occupation.vacancies
    * sum_over(Edge.dst, Edge.weight * share)
}
# eq. 16
u_i, z_i = Edge.src.unemployment, Edge.src.app_norm
v_j, s_j = Edge.dst.vacancies, Edge.dst.applications
labour_flow = {
    Edge.flow: sp.Piecewise(
        (
            u_i * v_j**2 * Edge.weight * (1 - sp.exp(-s_j / v_j)) / (s_j * z_i),
            (v_j > 0) & (s_j > 0) & (z_i > 0),
        ),
        (0, True),
    )
}
# eqs. 13-15, Methods (long-term unemployment)
hired, matched = sum_over(Edge.dst, Edge.flow), sum_over(Edge.src, Edge.flow)
unemployed, separated = Occupation.unemployment, Occupation.separations
job_finding = {
    Occupation.job_finding: sp.Piecewise(
        (matched / unemployed, unemployed > 0), (0, True)
    )
}
stay = 1 - Occupation.job_finding
labour_market_clearing = {
    Occupation.employment: employed - separated + hired,
    Occupation.unemployment: unemployed + separated - matched,
    Occupation.vacancies: Occupation.vacancies + Occupation.openings - hired,
    Occupation.u_spell1: separated,
    Occupation.u_spell2: Occupation.u_spell1 * stay,
    Occupation.u_spell3: Occupation.u_spell2 * stay,
    Occupation.ltu: (Occupation.u_spell3 + Occupation.ltu) * stay,
}

STAY_PROBABILITY = 0.55  # r of eq. 21, baked into the edge weights by mobility_network

RULES = [
    clock_tick,
    labour_target,
    labour_demand,
    applications,
    labour_flow,
    job_finding,
    labour_market_clearing,
]


def mobility_network(
    transitions: pl.DataFrame, stay: float = STAY_PROBABILITY
) -> pl.DataFrame:
    """Build the ``Edge`` population from observed occupational transitions.

    Eqs. 20-21: row-normalise the transition counts T_ij into P_ij, then
    A_ij = (1 - r) P_ij off-diagonal and A_ii = r on it, so *r* of all job
    changes stay within the occupation.  In the paper T_ij is counted from
    seven years of CPS monthly panel data over 464 four-digit occupations.

    ``transitions`` needs ``src`` / ``dst`` / ``count`` columns of bare
    occupation ids; the result is ready for ``sim.fit(Edge=...)`` in any order,
    since ``template.edge`` declares both as links.  Self-transitions in
    the input are dropped — the diagonal is set by *stay*, not observed.
    """
    off = (
        transitions.filter(pl.col("src") != pl.col("dst"))
        .group_by("src", "dst")
        .agg(pl.col("count").sum())
        .with_columns(
            weight=(1.0 - stay) * pl.col("count") / pl.col("count").sum().over("src")
        )
        .drop("count")
    )
    diag = pl.DataFrame({"src": transitions["src"].unique().sort()}).with_columns(
        dst=pl.col("src"), weight=pl.lit(stay)
    )
    return (
        pl.concat([off, diag], how="diagonal")
        .with_columns(id=pl.format("edge_{}_{}", pl.col("src"), pl.col("dst")))
        .select("id", "src", "dst", "weight")
    )


def reallocate_demand(demand, automation):
    """Post-automation target demand — eq. 18.

    Automation is modelled as a *reallocation*, not a destruction, of labour
    demand: an occupation's hours fall by its automation level, the surviving
    hours are then shared equally across the whole workforce, and the resulting
    head-counts are the new targets.  Aggregate demand is unchanged by
    construction, so any unemployment that follows is mismatch — workers in the
    wrong place — rather than a slump.

    Takes and returns either polars Series or expressions, so it works both on
    a frame in hand and inside a ``with_columns``.
    """
    hours = demand * (1.0 - automation)
    return hours * demand.sum() / hours.sum()


def occupations(
    employment: pl.DataFrame, unemployment_rate: float = 0.05
) -> pl.DataFrame:
    """Seed an ``Occupation`` population from an employment column.

    Takes ``id`` / ``employment`` (the paper uses BLS occupational employment)
    and fills in the rest of the state a cold fit needs: an initial unemployed
    pool and vacancy stock, the flow accumulators at zero, and both demand
    levels at the initial realised demand — which is a *no-shock* scenario.

    The whole initial unemployed pool is seeded into the *shortest* duration
    cohort, so the duration stages sum to ``unemployment`` from t=0 and stay
    that way (the chain preserves the sum); long-term unemployment then builds
    up over the first four steps rather than being assumed.  Set ``demand_final`` to the post-automation reallocation of
    eq. 18 to give the model something to chase.
    """
    e = pl.col("employment")
    return (
        employment.select("id", "employment")
        .with_columns(
            unemployment=e * unemployment_rate,
            vacancies=e * unemployment_rate,
        )
        .with_columns(
            demand_init=e + pl.col("vacancies"),
            demand_final=e + pl.col("vacancies"),
            target_demand=e + pl.col("vacancies"),
            separations=pl.lit(0.0),
            openings=pl.lit(0.0),
            applications=pl.lit(0.0),
            app_norm=pl.lit(0.0),
            job_finding=pl.lit(0.0),
            u_spell1=pl.col("unemployment"),
            u_spell2=pl.lit(0.0),
            u_spell3=pl.lit(0.0),
            ltu=pl.lit(0.0),
        )
    )
