"""Homogeneous-agent ABMs on the skabm stack (grid SIR + Schelling happiness).

skabm's reference model (Poledna) is *heterogeneous*: many agent classes,
typed relations (employer, owns), interaction through those edges and global
aggregates.  The AMBER examples are the opposite pole — *homogeneous*: one
agent class, embedded in a grid or continuous space, interacting with spatial
neighbors (Schelling, virus/SIR, forest fire, Game of Life, flocking).

The claim this file demonstrates: the same stack — ``rules.map_df`` to lift a
population into a maplib graph, SPARQL DELETE/INSERT upserts as the dynamics,
``RDFSimulator`` as the tick engine — runs homogeneous ABMs with *less* code
than Poledna, not more.  Homogeneity removes work: one ``ex:Person`` class, no
init-rule scoping, no relation ontology.  What used to be an interaction matrix
or a KD-tree neighbor query becomes a coordinate self-join in the WHERE clause;
what used to be an agent ``step()`` method becomes one upsert rule.

Two of AMBER's homogeneous examples, mapped onto skabm:

  * **Grid SIR** (AMBER ``virus_spread_simulation``) — runs end to end below.
    A cellular contact process: S becomes I next to any I; I recovers to R
    after a fixed duration.  Deterministic, local, one agent class — a perfect
    fit for a set-based SPARQL rule engine.

  * **Schelling** (AMBER ``schelling_vectorized``) — the *happiness* half runs
    below and is trivial in SPARQL; the *relocation* half is exactly the
    random-sequential-assignment wall skabm documents (README "Limitations"),
    so it is shown as the rule it would be and left to the numerical backend.

The dividing line is not heterogeneous-vs-homogeneous, it is
deterministic-local-state-transition (fits: SIR, forest fire, Life, voter
models, Schelling-happiness) vs continuous-space movement / spatial randomness
/ vector kinematics (walls: random walk, random relocation, flocking, Monte
Carlo) — the same walls the Poledna port hits, for the same reasons (no
seedable RAND, no random-sequential matching, synchronous-only activation).
"""

import polars as pl
import polars_random as pr

from maplib import Model

from skabm.rules import map_df, _PREFIXES
from skabm.simulation import RDFSimulator

# --- Homogeneous population: an N x N grid of Persons -------------------------
# A grid is *structure*, not a calibrated marginal population, so it is a plain
# polars cartesian product rather than a make_dataset draw.  One agent class,
# one row per cell: id, integer coordinates, S/I/R state, a recovery timer.
# polars-random seeds patient-zero (one infected cell) reproducibly.
N = 21
grid = pl.int_range(N, eager=True)
people = (
    grid.to_frame("x")
    .join(grid.to_frame("y"), how="cross")
    .with_columns(
        pl.format("p_{}_{}", pl.col("x"), pl.col("y")).alias("id"),
        pl.col("x").cast(pl.Float64),
        pl.col("y").cast(pl.Float64),
        # one seed infection, drawn reproducibly; everyone else susceptible
        pr.uniform(0, 1).alias("_r"),
        pl.lit(0.0).alias("timer"),
    )
    .with_columns(
        pl.when(pl.col("_r") == pl.col("_r").max())
        .then(pl.lit("I"))
        .otherwise(pl.lit("S"))
        .alias("state")
    )
    .drop("_r")
    .select("id", "x", "y", "state", "timer")
)

# --- The dynamics: two SPARQL upserts, in tick order -------------------------
# RECOVERY runs first so CONTAGION's WHERE still sees this tick's not-yet-
# recovered infectives as I (a SPARQL UPDATE evaluates WHERE against the
# pre-update graph, so within one rule everything is synchronous; ordering the
# two rules is skabm's StagedActivation).  RECOVERY_TIME = 4 quarters here.
#
# Neighborhood is a coordinate self-join with a squared-distance FILTER: Moore
# radius 1 is dx^2 + dy^2 <= 2.  SPARQL has no sqrt, so we compare *squared*
# distance to a squared radius — which is exactly what you want anyway.  No
# neighbor edges are precomputed; the grid topology lives entirely in the two
# coordinate predicates.  (This is an O(n^2) self-join — fine for a demo grid,
# and the roadmap's polars/JAX backend is where a KD-tree or edge index goes.)
RECOVERY = _PREFIXES + """
DELETE { ?a def:state ?s0 . ?a def:timer ?t0 }
INSERT { ?a def:state ?s1 . ?a def:timer ?t1 }
WHERE {
    ?a a ex:Person ; def:state ?s0 ; def:timer ?t0 .
    FILTER(?s0 = "I")
    BIND(IF(?t0 >= 4e0, "R", "I") AS ?s1)
    BIND(?t0 + 1e0 AS ?t1)
}
"""

# Deterministic transmission: an S with >=1 infected Moore neighbor becomes I.
# AMBER's model multiplies by a per-contact transmission_rate; a seedable RAND
# would gate the INSERT on pr.uniform(0,1) < rate, but maplib SPARQL has none,
# so — exactly as the Poledna port drops stochastic shocks — this degenerates
# to "any exposure infects".  The topology and the state machine are the
# faithful part; the probability is the documented simplification.
CONTAGION = _PREFIXES + """
DELETE { ?a def:state ?s0 }
INSERT { ?a def:state "I" }
WHERE {
    ?a a ex:Person ; def:state ?s0 ; def:x ?xa ; def:y ?ya .
    FILTER(?s0 = "S")
    FILTER EXISTS {
        ?b def:state "I" ; def:x ?xb ; def:y ?yb .
        FILTER( (?xa-?xb)*(?xa-?xb) + (?ya-?yb)*(?ya-?yb) <= 2e0 )
    }
}
"""

# --- Run it on the ordinary RDFSimulator -------------------------------------
# No init rules (the population is already in its initial state), no params
# (the rules carry no $placeholders).  fit() maps the population and runs one
# tick; warm_start=True then keeps ticking the same model_ so we can snapshot
# the S/I/R counts between ticks with a plain polars group-by.  This is the
# same fit / warm_start / model_ lifecycle the Poledna demo uses — the engine
# is domain-agnostic; only the rules and the population change.
sim = RDFSimulator(init_rules=(), update_rules=(RECOVERY, CONTAGION), n_periods=1)


def sir_counts(model: Model) -> dict[str, int]:
    q = model.query(
        _PREFIXES
        + "SELECT ?st (COUNT(?a) AS ?n) WHERE "
        "{ ?a a ex:Person ; def:state ?st } GROUP BY ?st"
    )
    return {st: n for st, n in zip(q["st"], q["n"])}


print("Grid SIR (deterministic contact process, one seed):")
sim.fit(Person=people)
print(f"  t=1  {sir_counts(sim.model_)}")
sim.set_params(warm_start=True)
for t in range(2, 40):
    sim.fit()
    counts = sir_counts(sim.model_)
    print(f"  t={t:<2} {counts}")
    if counts.get("I", 0) == 0:
        print(f"  epidemic over at t={t} (no infectives left)")
        break

# --- Schelling: the half that fits, and the wall -----------------------------
# Segregation needs two behaviors: (1) compute happiness from the neighborhood,
# (2) relocate unhappy agents to a random empty cell.  (1) is a self-join +
# group-by, as natural as the SIR neighborhood above.  (2) is not expressible
# in pure SPARQL for the same reason Poledna's goods/labor markets aren't: it
# is *random sequential assignment* — draw a random empty cell per unhappy
# agent, first-come-first-served, no two agents landing on the same cell.  That
# needs seedable randomness and an ordering no single UPDATE can impose.  So
# skabm computes who is unhappy declaratively and hands the *move* to a
# numerical backend (roadmap) or the calibration layer — the same division of
# labor the README draws for search-and-matching.
SCHELLING_HAPPINESS = _PREFIXES + """
DELETE { ?a def:happy ?h0 }
INSERT { ?a def:happy ?h1 }
WHERE {
    ?a a ex:HH .
    OPTIONAL { ?a def:happy ?h0 }
    { SELECT ?a (COUNT(?nb) AS ?tot) (SUM(IF(?gn = ?ga, 1, 0)) AS ?sim) WHERE {
        ?a a ex:HH ; def:x ?xa ; def:y ?ya ; def:group ?ga .
        ?nb def:x ?xb ; def:y ?yb ; def:group ?gn .
        FILTER( (?xa-?xb)*(?xa-?xb) + (?ya-?yb)*(?ya-?yb) <= 2e0 )
        FILTER( !(?xa=?xb && ?ya=?yb) )
    } GROUP BY ?a }
    BIND(IF(?tot > 0 && ?sim + ?sim >= ?tot, "yes", "no") AS ?h1)
}
"""

# A tiny line of households; a_1 sits between a same-group and an other-group
# neighbor (50% similar -> happy at the 0.5 threshold), the ends see only their
# own group.  This is the reporter Schelling drives relocation from.
households = pl.DataFrame(
    {
        "id": ["a_0", "a_1", "a_2", "a_3"],
        "x": [0.0, 1.0, 2.0, 3.0],
        "y": [0.0, 0.0, 0.0, 0.0],
        "group": ["red", "red", "blue", "blue"],
    }
)
m = Model()
map_df(m, households, "HH")
m.update(SCHELLING_HAPPINESS)
happy = m.query(
    _PREFIXES + "SELECT ?a ?happy WHERE { ?a a ex:HH ; def:happy ?happy } ORDER BY ?a"
)
print("\nSchelling happiness (>=50% same-group neighbors) — the half SPARQL owns:")
print(happy)
print(
    "Relocation of the unhappy is the random-sequential-assignment wall "
    "(README Limitations) — left to the numerical backend."
)
