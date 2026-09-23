"""
Labour-market behaviour templates — occupational mobility on a job-transition network,
after del Rio-Chanona, Mealy, Beguerisse-Díaz, Lafond & Farmer (2021), *Occupational
mobility and automation*, J. R. Soc. Interface 18(174):20200898.

A third model family on the same machinery, and the first whose *specification* is a
network flow problem: workers do not consume or price, they queue.  ``Occupation`` holds
the stocks e/u/v of eqs. 2-4 plus the demand it chases; ``Edge`` is the mobility network
A_ij *reified* — one agent per edge with ``def:src`` / ``def:dst`` links and a
``def:weight``, since RDF triples carry none — and a one-row ``Clock`` carries the
calendar time the automation shock (eq. 19) needs.

**The event order is the rule order.**  The paper computes every flow from the state at
*t* and applies them simultaneously at *t+1* (fig. 2), so the staged-synchronous regime
the engine imposes is the paper's own timing rather than an approximation, as it is for
Poledna's random-sequential markets.  Intermediate flows are materialised as triples for
later rules in the same tick; ``labour_market_clearing`` moves the stocks.  The urn-ball
function (eq. 16) and the S-curve shock (eq. 19) are transcendental, so ``math:exp``
arrives as a UDF on the random draws' seam: ``RDFSimulator(udfs=LABOUR_UDFS, ...)``.

**Parenthesise every arithmetic chain.**  maplib's SPARQL evaluates same-precedence
operators *right*-associatively, so ``?e - ?w + ?f`` comes out as ``?e - (?w + ?f)``,
silently.  The conservation laws below are bracketed for that reason, and
``tests/test_labour.py`` pins the labour force as the tripwire.

**Deterministic mean-field, not agent-level.**  The paper gives both the stochastic
processes over individual workers (eqs. 2-12) and their large-population limit (13-17);
the limit is what every figure comes from and what is implemented here, so nodes are
occupations and 140M workers is 464 agents.
"""

from __future__ import annotations

import polars as pl

from skabm.behaviour import DefaultTemplate
from skabm.sparql import _PREFIXES, register_math, register_polars_random

# The labour rules call math:exp; the model family is otherwise deterministic,
# but pr:* stays registered so a stochastic variant can drop straight in.
LABOUR_UDFS = (register_polars_random, register_math)


# ---------------------------------------------------------------------------
# clock_tick — advance calendar time (update)
# ---------------------------------------------------------------------------
# Poledna's dynamics are autonomous: every rule reads only the current state.
# An automation shock is not — d† is a function of t (eq. 19) — so time has to
# exist somewhere in the graph.  It lives on a one-row ``Clock`` population,
# and this rule is the only thing that touches it.

clock_tick = DefaultTemplate(
    _PREFIXES
    + """
DELETE { ?c def:t ?t0 }
INSERT { ?c def:t ?t1 }
WHERE {
    ?c a ex:Clock ;
        def:t ?t0 .
    BIND(?t0 + 1e0 AS ?t1)
}
"""
)
clock_tick.metadata = {
    "@id": "clock_tick",
    "@type": "Behaviour",
    "agentClass": "ex:Clock",
    "source": "del Rio-Chanona et al. (2021), J. R. Soc. Interface 18:20200898",
}

# ---------------------------------------------------------------------------
# target_demand — S-curve automation shock (update)
# ---------------------------------------------------------------------------
# Eq. 19: d†_{i,t} = d_{i,0} for t < t_s, and afterwards
#   d†_{i,t} = d_{i,0} + (d†_i - d_{i,0}) / (1 + exp(-k (t - t_0)))
# with t in *years* (t_0 = $shock_halfway years after the shock starts, and
# k = 0.79 puts the target within 1e-4 of the post-shock level by year 30).
# The published equation prints the exponent without its minus sign; the
# text — target demand converges to d†, and sits at the midpoint at t_0 —
# fixes the sign as written here.
#
# ``def:demand_init`` is d_{i,0} (steady-state demand at the shock date) and
# ``def:demand_final`` the post-shock reallocated demand d† of eq. 18: the
# automation scenario enters as *data*, one column of the Occupation frame,
# never as rule logic.  Placeholders: ``weeks_per_step``, ``shock_start``,
# ``shock_halfway``, ``shock_k``.

target_demand = DefaultTemplate(
    _PREFIXES
    + """
DELETE { ?o def:target_demand ?d0 }
INSERT { ?o def:target_demand ?d1 }
WHERE {
    { SELECT ((?t * $weeks_per_step) / 52e0 AS ?years)
      WHERE { ?c a ex:Clock ; def:t ?t } }
    ?o a ex:Occupation ;
        def:target_demand ?d0 ;
        def:demand_init ?di ;
        def:demand_final ?df .
    BIND(?years - $shock_start AS ?since)
    BIND(math:exp(0e0 - $shock_k * (?since - $shock_halfway)) AS ?decay)
    BIND(IF(?since < 0e0, ?di, ?di + (?df - ?di) / (1e0 + ?decay)) AS ?d1)
}
""",
    {
        "weeks_per_step": 6.75,  # one time step, in weeks, del Rio-Chanona et al. (2021) Table 1
        "shock_start": 20.0,  # scenario knob: years before the S-curve begins (past the burn-in)
        "shock_halfway": 15.0,  # scenario knob: with shock_k, "30 years, mostly within 10"
        "shock_k": 0.79,  # scenario knob: S-curve steepness
    },
)
target_demand.metadata = {
    "@id": "target_demand",
    "@type": "Behaviour",
    "agentClass": "ex:Occupation",
    "source": "del Rio-Chanona et al. (2021), J. R. Soc. Interface 18:20200898, eq. 19",
}

# ---------------------------------------------------------------------------
# labour_adjustment — separations and vacancy openings (update)
# ---------------------------------------------------------------------------
# The bracketed terms of eqs. 13 and 15.  Realised demand is d_{i,t} = e + v;
# each occupation separates workers when it overshoots its target and opens
# vacancies when it undershoots, at speed γ (eqs. 9-10), on top of a
# state-independent churn rate δ_u / δ_v (eqs. 11-12):
#   ω = δ_u e + (1 - δ_u) γ max(0, d - d†)
#   ν = δ_v e + (1 - δ_v) γ max(0, d† - d)
# α_u and α_v are probabilities, so the state-dependent term is capped at e
# (paper, footnote 3) — the cap only ever binds under a violent shock.
# Placeholders: ``delta_u``, ``delta_v``, ``gamma``.

labour_adjustment = DefaultTemplate(
    _PREFIXES
    + """
DELETE { ?o def:separations ?w0 . ?o def:openings ?n0 }
INSERT { ?o def:separations ?w1 . ?o def:openings ?n1 }
WHERE {
    ?o a ex:Occupation ;
        def:employment ?e ;
        def:vacancies ?v ;
        def:target_demand ?dt ;
        def:separations ?w0 ;
        def:openings ?n0 .
    BIND(?e + ?v AS ?d)
    BIND(IF(?d > ?dt, $gamma * (?d - ?dt), 0e0) AS ?excess)
    BIND(IF(?dt > ?d, $gamma * (?dt - ?d), 0e0) AS ?shortfall)
    BIND(IF(?excess > ?e, ?e, ?excess) AS ?excess_capped)
    BIND(IF(?shortfall > ?e, ?e, ?shortfall) AS ?shortfall_capped)
    BIND($delta_u * ?e + (1e0 - $delta_u) * ?excess_capped AS ?w1)
    BIND($delta_v * ?e + (1e0 - $delta_v) * ?shortfall_capped AS ?n1)
}
""",
    {
        "delta_u": 0.0160,  # spontaneous separation rate, del Rio-Chanona et al. (2021) Table 1
        "delta_v": 0.0120,  # spontaneous vacancy-opening rate, del Rio-Chanona et al. (2021) Table 1
        "gamma": 0.160,  # speed of adjustment to target demand, del Rio-Chanona et al. (2021) Table 1
    },
)
labour_adjustment.metadata = {
    "@id": "labour_adjustment",
    "@type": "Behaviour",
    "agentClass": "ex:Occupation",
    "source": "del Rio-Chanona et al. (2021), J. R. Soc. Interface 18:20200898, eqs. 9-12",
}

# ---------------------------------------------------------------------------
# application_norm — denominator of the search distribution (update)
# ---------------------------------------------------------------------------
# Eq. 7: an unemployed worker last employed in i applies to j with probability
# q_ij = v_j A_ij / Σ_k v_k A_ik.  This rule materialises that denominator on
# each occupation, so the two rules that need it (applications, labour_flow)
# read one triple instead of re-deriving a sum over neighbours.  An occupation
# whose whole neighbourhood has zero vacancies gets 0 and is skipped downstream
# — its unemployed have nowhere to apply this tick.

application_norm = DefaultTemplate(
    _PREFIXES
    + """
DELETE { ?o def:app_norm ?z0 }
INSERT { ?o def:app_norm ?z1 }
WHERE {
    ?o a ex:Occupation .
    OPTIONAL { ?o def:app_norm ?z0 }
    OPTIONAL {
        { SELECT ?o (SUM(?vk * ?a) AS ?z)
          WHERE {
              ?edge a ex:Edge ; def:src ?o ; def:dst ?k ; def:weight ?a .
              ?k def:vacancies ?vk .
          } GROUP BY ?o }
    }
    BIND(IF(BOUND(?z), ?z, 0e0) AS ?z1)
}
"""
)
application_norm.metadata = {
    "@id": "application_norm",
    "@type": "Behaviour",
    "agentClass": "ex:Occupation",
    "source": "del Rio-Chanona et al. (2021), J. R. Soc. Interface 18:20200898, eq. 7",
}

# ---------------------------------------------------------------------------
# applications — job applications received per occupation (update)
# ---------------------------------------------------------------------------
# Eq. 17: s_j = Σ_i u_i v_j A_ij / (Σ_k v_k A_ik).  Each unemployed worker
# sends exactly one application per time step, so s_j is the size of the urn
# pool competing for occupation j's vacancies — the quantity that decides how
# many of them get filled.

applications = DefaultTemplate(
    _PREFIXES
    + """
DELETE { ?o def:applications ?s0 }
INSERT { ?o def:applications ?s1 }
WHERE {
    ?o a ex:Occupation ;
        def:vacancies ?vj .
    OPTIONAL { ?o def:applications ?s0 }
    OPTIONAL {
        { SELECT ?o (SUM((?ui * ?a) / ?zi) AS ?share)
          WHERE {
              ?edge a ex:Edge ; def:dst ?o ; def:src ?i ; def:weight ?a .
              ?i def:unemployment ?ui ; def:app_norm ?zi .
              FILTER(?zi > 0e0)
          } GROUP BY ?o }
    }
    BIND(IF(BOUND(?share), ?share * ?vj, 0e0) AS ?s1)
}
"""
)
applications.metadata = {
    "@id": "applications",
    "@type": "Behaviour",
    "agentClass": "ex:Occupation",
    "source": "del Rio-Chanona et al. (2021), J. R. Soc. Interface 18:20200898, eq. 17",
}

# ---------------------------------------------------------------------------
# labour_flow — urn-ball matching, per edge (update)
# ---------------------------------------------------------------------------
# Eq. 16, the heart of the model:
#   f_ij = u_i v_j^2 A_ij (1 - exp(-s_j / v_j)) / (s_j Σ_k v_k A_ik)
# Note the *square* on v_j — summing f_ij over origins i collapses the whole
# expression to v_j (1 - exp(-s_j / v_j)), the urn-ball count of vacancies that
# drew at least one ball, which is the check that the exponent is right.
# Every vacancy is an urn and every applicant a ball; a vacancy that draws no
# ball stays open.  (1 - exp(-s_j/v_j)) is the share of j's vacancies that
# receive at least one application, so the factor in front of it splits those
# hires back across the origin occupations in proportion to who applied.
#
# The flow is written onto the *edge*, which is what makes the matching one
# rule rather than a nest of aggregates: ``labour_market_clearing`` then sums
# the same predicate two ways, in-edges for hires and out-edges for exits.
# Self-loops (A_ii = r, the probability a job-changer stays in occupation) are
# ordinary edges and need no special case.

labour_flow = DefaultTemplate(
    _PREFIXES
    + """
DELETE { ?edge def:flow ?f0 }
INSERT { ?edge def:flow ?f1 }
WHERE {
    ?edge a ex:Edge ;
        def:src ?i ;
        def:dst ?j ;
        def:weight ?a .
    OPTIONAL { ?edge def:flow ?f0 }
    ?i def:unemployment ?ui ;
       def:app_norm ?zi .
    ?j def:vacancies ?vj ;
       def:applications ?sj .
    BIND(IF(?vj > 0e0 && ?sj > 0e0 && ?zi > 0e0,
            (?ui * ?vj * ?vj * ?a * (1e0 - math:exp(0e0 - ?sj / ?vj))) / (?sj * ?zi),
            0e0) AS ?f1)
}
"""
)
labour_flow.metadata = {
    "@id": "labour_flow",
    "@type": "Behaviour",
    "agentClass": "ex:Edge",
    "source": "del Rio-Chanona et al. (2021), J. R. Soc. Interface 18:20200898, eq. 16",
}

# ---------------------------------------------------------------------------
# labour_market_clearing — the conservation laws (update)
# ---------------------------------------------------------------------------
# Eqs. 13-15, the only rule that moves a stock:
#   e_{t+1} = e_t - ω + Σ_j f_ji      (separations out, hires in)
#   u_{t+1} = u_t + ω - Σ_j f_ij      (separations in, matches out)
#   v_{t+1} = v_t + ν - Σ_j f_ji      (openings in, filled vacancies out)
# Every term on the right is a triple written earlier this tick from state at
# t, so the update is genuinely simultaneous.  Workers are conserved by
# construction: what leaves e enters u and vice versa.
#
# ``def:job_finding`` — the share of an occupation's unemployed matched this
# tick — is a by-product needed by ``unemployment_duration``; it is cheaper to
# emit here, where the outflow is already in hand, than to re-derive it.

labour_market_clearing = DefaultTemplate(
    _PREFIXES
    + """
DELETE { ?o def:employment ?e0 . ?o def:unemployment ?u0 .
         ?o def:vacancies ?v0 . ?o def:job_finding ?phi0 }
INSERT { ?o def:employment ?e1 . ?o def:unemployment ?u1 .
         ?o def:vacancies ?v1 . ?o def:job_finding ?phi1 }
WHERE {
    ?o a ex:Occupation ;
        def:employment ?e0 ;
        def:unemployment ?u0 ;
        def:vacancies ?v0 ;
        def:separations ?w ;
        def:openings ?n .
    OPTIONAL { ?o def:job_finding ?phi0 }
    OPTIONAL {
        { SELECT ?o (SUM(?f) AS ?inflow)
          WHERE { ?edge a ex:Edge ; def:dst ?o ; def:flow ?f } GROUP BY ?o }
    }
    OPTIONAL {
        { SELECT ?o (SUM(?f) AS ?outflow)
          WHERE { ?edge a ex:Edge ; def:src ?o ; def:flow ?f } GROUP BY ?o }
    }
    BIND(IF(BOUND(?inflow), ?inflow, 0e0) AS ?hired)
    BIND(IF(BOUND(?outflow), ?outflow, 0e0) AS ?matched)
    BIND((?e0 - ?w) + ?hired AS ?e1)
    BIND((?u0 + ?w) - ?matched AS ?u1)
    BIND((?v0 + ?n) - ?hired AS ?v1)
    BIND(IF(?u0 > 0e0, ?matched / ?u0, 0e0) AS ?phi1)
}
"""
)
labour_market_clearing.metadata = {
    "@id": "labour_market_clearing",
    "@type": "Behaviour",
    "agentClass": "ex:Occupation",
    "source": "del Rio-Chanona et al. (2021), J. R. Soc. Interface 18:20200898, eqs. 13-15",
}

# ---------------------------------------------------------------------------
# unemployment_duration — short- vs long-term unemployment (update)
# ---------------------------------------------------------------------------
# Long-term unemployment (> 27 weeks) is one of the paper's two headline
# outputs, and the mean-field state (e, u, v) has no memory of how long anyone
# has been queuing.  A four-stage cohort chain gives it one: this tick's
# separations enter ``def:u_spell1``, and each cohort that fails to match
# (survival 1 - φ, from ``labour_market_clearing``) ages into the next.
# ``def:ltu`` is the absorbing fourth stage — unemployed for four steps or
# more, i.e. 4 x 6.75 = 27 weeks, which is what the time-step length is
# calibrated to make exact.
#
# NOTE: the paper derives long-term unemployment analytically in its
# supplementary material (eq. S36); this cohort chain is a reconstruction from
# the same job-finding rate, not a transcription of that derivation.  The
# stages sum to ``def:unemployment`` by construction, which is the invariant
# worth testing.

unemployment_duration = DefaultTemplate(
    _PREFIXES
    + """
DELETE { ?o def:u_spell1 ?a0 . ?o def:u_spell2 ?b0 .
         ?o def:u_spell3 ?c0 . ?o def:ltu ?l0 }
INSERT { ?o def:u_spell1 ?a1 . ?o def:u_spell2 ?b1 .
         ?o def:u_spell3 ?c1 . ?o def:ltu ?l1 }
WHERE {
    ?o a ex:Occupation ;
        def:separations ?w ;
        def:job_finding ?phi ;
        def:u_spell1 ?a0 ;
        def:u_spell2 ?b0 ;
        def:u_spell3 ?c0 ;
        def:ltu ?l0 .
    BIND(1e0 - ?phi AS ?survive)
    BIND(?w AS ?a1)
    BIND(?a0 * ?survive AS ?b1)
    BIND(?b0 * ?survive AS ?c1)
    BIND((?c0 + ?l0) * ?survive AS ?l1)
}
"""
)
unemployment_duration.metadata = {
    "@id": "unemployment_duration",
    "@type": "Behaviour",
    "agentClass": "ex:Occupation",
    "source": "del Rio-Chanona et al. (2021), J. R. Soc. Interface 18:20200898, Methods (long-term unemployment)",
}


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

# ``r`` of eq. 21 — the probability a job-changer stays in her own occupation —
# is not a rule parameter: it is baked into A_ij by ``mobility_network`` before
# the graph is ever built.
STAY_PROBABILITY = 0.55

LABOUR_UPDATE_RULES = (
    clock_tick,  # calendar time
    target_demand,  # eq. 19 — automation S-curve
    labour_adjustment,  # eqs. 9-12 — separations and openings
    application_norm,  # eq. 7 — search denominator
    applications,  # eq. 17 — applications received
    labour_flow,  # eq. 16 — urn-ball matching, per edge
    labour_market_clearing,  # eqs. 13-15 — conservation laws
    unemployment_duration,  # short- vs long-term unemployment
)


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
    since ``templates.edge_template`` declares both as links.  Self-transitions in
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


# ---------------------------------------------------------------------------
# Not implemented: the agent-level formulation (eqs. 2-12)
# ---------------------------------------------------------------------------
#
# The paper's stochastic version replaces every expectation above with a draw.
# Three of the four pieces are already reachable here: with individual Worker
# agents a Binomial is a Bernoulli per worker (``pr:uniform``); the categorical
# application draw is a cumulative interval per out-edge plus one uniform per
# worker; and picking one applicant per vacancy is the rank-join of
# ``schelling.RELOCATE``, partitioned by occupation.  What resists is *creating*
# vacancies — CONSTRUCT mints IRIs only from existing bindings, so opening v new
# ones means keeping a dormant pool and flipping a flag on v of them, which puts
# a capacity assumption in the graph.  Hence the mean-field version first.
