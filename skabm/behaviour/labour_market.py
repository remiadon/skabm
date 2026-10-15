"""Occupations and the job transitions between them: workers separated and hired along
a network of observed moves, vacancies opening and closing towards a target demand.

Urn-ball matching on an occupational mobility network, with an S-curve target for a
demand shock such as automation: del Rio-Chanona et al. (2021), J. R. Soc. Interface 18,
20200898, in its mean-field form.
"""

from __future__ import annotations

import polars as pl
import sympy as sp

from skabm.dsl import Agents, sum_over, total

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
clock = {Clock.t: Clock.t + 1}
# eq. 19 (the exponent's sign as the text implies)
years = total(Clock.t) * weeks_per_step / 52
since = years - shock_start
initial, final = Occupation.demand_init, Occupation.demand_final
s_curve_target = {
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
employed, wanted = Occupation.employment, Occupation.target_demand
realised = employed + Occupation.vacancies
excess = sp.Min(employed, gamma * sp.Max(0, realised - wanted))
shortfall = sp.Min(employed, gamma * sp.Max(0, wanted - realised))
separations_openings = {
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
urn_ball_flow = {
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
stock_flow_clearing = {
    Occupation.employment: employed - separated + hired,
    Occupation.unemployment: unemployed + separated - matched,
    Occupation.vacancies: Occupation.vacancies + Occupation.openings - hired,
    Occupation.u_spell1: separated,
    Occupation.u_spell2: Occupation.u_spell1 * stay,
    Occupation.u_spell3: Occupation.u_spell2 * stay,
    Occupation.ltu: (Occupation.u_spell3 + Occupation.ltu) * stay,
}


STAY_PROBABILITY = 0.55  # r of del Rio-Chanona et al. (2021) eq. 21, baked into the edge weights by mobility_network


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
