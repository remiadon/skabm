"""Schelling's spatial segregation model as a maplib knowledge graph.

A deliberate stress test of the architecture built for the Poledna macro ABM
(``skabm.rules`` + ``skabm.simulation.RDFSimulator``): same ``Template``
rules, same ``map_df`` mapping, same ``fit_iter`` loop, an entirely
different model family.  Nothing here is macro-economic, and nothing in
``RDFSimulator`` had to learn about space.

**The grid is a population, not a module.**  Where Mesa has
``DiscreteSpace`` and AMBER has ``model.grid`` / ``model.empty_spots``,
here a cell is just another agent class::

    sim.fit(Cell=cells, Person=persons)

``Cell`` carries ``x``/``y``; ``Person`` carries ``group`` and a
``location`` link into a cell.  Occupancy is the *absence* of an inbound
``def:location`` edge (``FILTER NOT EXISTS``) rather than a sentinel ``-1``
in a nested list, and the Moore neighbourhood is a ``def:neighbor``
relation CONSTRUCTed once at init from the coordinates — topology as an
init rule, exactly like ``rules.FIRM_OWNERSHIP`` constructs firm ownership.
The consequence is that "space" gains nothing special: rules traverse
``?p def:location ?c . ?c def:neighbor ?cn . ?n def:location ?cn`` the same
way Poledna's traverse ``?hh def:employer ?f``.

Two things Schelling needs that Poledna did not, and that pure SPARQL does
not give:

* **Randomness.**  ``rules`` documents "no stochastic shocks: SPARQL has no
  seedable RAND".  Schelling is meaningless without it, so randomness is
  carried *in the graph* as agent state: every agent holds a ``def:rng``
  counter advanced each tick by ``LCG_TICK``, a Park-Miller minimal
  standard generator (a=16807, m=2^31-1) written in double arithmetic —
  every intermediate stays under 2^53, so the recurrence is exact.  The
  *seed* enters as one calibration-layer column (``polars_random``), since
  SPARQL has no RAND to synthesize it in-graph.  This generalizes: the same
  ``def:rng`` trick would restore Poledna's AR(1) shocks.

  TODO: when maplib ships ``Model.add_udf`` (documented but not in 0.20.19),
  register polars-random's Series functions as SPARQL UDFs — then a single
  ``BIND(<urn:pr:uniform>(1e0, 2147483646e0) AS ?rng)`` in an init rule
  replaces the seed column outright, and puts genuine RAND inside SPARQL for
  every model (Poledna's stochastic shocks included).  An in-graph hash of
  the cell coordinates works too, but needs ~40 lines of hand-rolled
  modular-squaring arithmetic (maplib rejects ``SUBSTR`` on a variable in a
  CONSTRUCT, so ``MD5(id)`` can't be parsed there) — not worth it over one
  ``polars_random`` column while the UDF path is imminent.
* **One-to-one matching.**  Each mover must land on a *distinct* empty
  cell.  This is the same wall ``rules`` hits with "no search-and-matching"
  — and hitting it twice, in two unrelated model families, says the limit
  is structural rather than Poledna-specific: SPARQL has no sequential
  activation and no assignment operator.  ``RELOCATE`` works around it
  with a **rank join**: movers are ranked by their RNG draw, vacant cells
  by theirs, and rank *k* is paired with rank *k*.  Both orderings are
  random, so the matching is a uniform random permutation — statistically
  what AMBER's ``random.choice`` over ``empty_spots`` produces, obtained
  declaratively and in one batch.  Ranking is a correlated
  ``COUNT(<= mine)`` subquery, i.e. quadratic in the number of movers; fine
  at 10^3 agents, not at 10^6.

Deliberate semantic differences versus the AMBER reference:

* **Simultaneous, not sequential, activation.**  AMBER moves unhappy
  agents one at a time, so a cell vacated early in a sweep can be occupied
  later in the same sweep.  ``RELOCATE`` is one batch upsert: the target
  pool is the set of cells vacant at the *start* of the tick, and cells
  freed by this tick's movers only become available next tick.
* **No move cap.**  AMBER caps a sweep at 50 moves; every unhappy agent
  that finds a partner moves here.
* **No convergence stop.**  ``RDFSimulator`` runs exactly ``n_periods``;
  AMBER sets ``simulation_finished`` when nobody is unhappy.  With
  ``fit_iter`` the caller can ``break`` on the extracted state, which is
  where that check belongs anyway (see the demo).
* **share_similar is pre-move.**  ``HAPPINESS`` runs before ``RELOCATE``,
  matching AMBER's ``update()``-then-``step()`` order, so the segregation
  index yielded for tick *t* describes the configuration the movers were
  reacting to.

Everything is xsd:double — coordinates included.  SPARQL basic graph
patterns join on RDF *terms*, not values, so an ``xsd:integer`` produced by
``?x + ?dx`` would fail to match an ``xsd:long`` stored by maplib.  Keeping
one numeric type sidesteps the whole class of bug (and the decimal-literal
instability ``rules`` already warns about).
"""

from string import Template

import polars as pl

from skabm.rules import _PREFIXES, EX_NS

# ---------------------------------------------------------------------------
# Initialization rules — Model.insert, once, right after mapping
# ---------------------------------------------------------------------------

# Moore neighbourhood as an explicit relation.  The eight offsets are a
# VALUES block, so `?c2 def:x ?nx ; def:y ?ny` is an equality join (hash,
# O(8N)) rather than the O(N^2) coordinate FILTER a naive encoding invites.
# Off-grid neighbours simply match no cell: the grid is bounded, not toroidal,
# like the AMBER reference.  Note this never touches cell *ids* — unlike
# rules.FIRM_OWNERSHIP, which parses "firm_<j>" out of the IRI — so the id
# convention stays free.
GRID_NEIGHBORHOOD = Template(
    _PREFIXES
    + """
    CONSTRUCT { ?c def:neighbor ?c2 }
    WHERE {
        VALUES (?dx ?dy) {
            (-1e0 -1e0) (-1e0 0e0) (-1e0 1e0)
            ( 0e0 -1e0)            ( 0e0 1e0)
            ( 1e0 -1e0) ( 1e0 0e0) ( 1e0 1e0)
        }
        ?c a ex:Cell ; def:x ?x ; def:y ?y .
        BIND(?x + ?dx AS ?nx)
        BIND(?y + ?dy AS ?ny)
        ?c2 a ex:Cell ; def:x ?nx ; def:y ?ny .
    }
    """
)

# Settlement: the population is *derived*, not passed.  Where Poledna's
# FIRM_OWNERSHIP hands users the `owns` relation for free, here the whole
# Person population is free — the caller passes only the grid (`Cell=`), and
# each cell is independently occupied with probability $density (AMBER's
# `p['density']`), spawning one Person located on it, in one of $n_groups
# groups drawn uniformly.  Occupancy is thus stochastic: the count is
# ~$density*|cells|, not AMBER's exact int(density*size^2) — the difference
# is one Bernoulli draw per cell.
#
# All randomness is the cell's own def:rng stream (seeded, integer-valued),
# advanced three Park-Miller steps here so the occupancy draw, the group
# draw, and the seed handed to the spawned Person are decorrelated; the LCG
# modulo (2^31-1) is exact in doubles for the same reason as LCG_TICK.
#
# This is the one point where the grid asymmetry with Poledna bites: SPARQL
# CONSTRUCT can only bind over rows that exist, so it can populate a passed
# grid but cannot *generate* one (a density<1 world has more cells than
# agents).  The lattice therefore stays a passed population; only the agents
# on it are free.  Users override by passing their own `Person=` population —
# FILTER NOT EXISTS makes SETTLE no-op the moment any Person already exists,
# exactly as FIRM_OWNERSHIP only fills firms without a data-defined owner.
SETTLE = Template(
    _PREFIXES
    + """
    CONSTRUCT {
        ?p a ex:Person . ?p def:location ?c . ?p def:group ?g . ?p def:rng ?seed
    }
    WHERE {
        FILTER NOT EXISTS { ?existing a ex:Person }
        ?c a ex:Cell ; def:rng ?r .
        BIND(16807e0 * ?r AS ?x1)
        BIND(?x1 - 2147483647e0 * FLOOR(?x1 / 2147483647e0) AS ?d1)
        FILTER(?d1 / 2147483647e0 < $density)
        BIND(16807e0 * ?d1 AS ?x2)
        BIND(?x2 - 2147483647e0 * FLOOR(?x2 / 2147483647e0) AS ?d2)
        # ?u2 is its own BIND on purpose: maplib does not left-associate the
        # chain `?d2 / m * n_groups`, evaluating it as `?d2 / (m * n_groups)`,
        # which floors to 0 for every agent (one group, segregation trivially
        # 1.0).  Parenthesising via an intermediate is the safe form.
        # xsd:double casts are load-bearing too: FLOOR / the modulo yield
        # integer-valued results maplib would store as Int64, colliding with
        # the xsd:double def:rng column the cells already carry.
        BIND(?d2 / 2147483647e0 AS ?u2)
        BIND(xsd:double(FLOOR(?u2 * $n_groups)) AS ?g)
        BIND(16807e0 * ?d2 AS ?x3)
        BIND(xsd:double(?x3 - 2147483647e0 * FLOOR(?x3 / 2147483647e0)) AS ?seed)
        BIND(IRI(CONCAT(\""""
    + EX_NS
    + """settler_", STRAFTER(STR(?c), "#"))) AS ?p)
    }
    """
)


# ---------------------------------------------------------------------------
# Update rules — Model.update, every tick (upsert pattern)
# ---------------------------------------------------------------------------

# share_similar = |same-group occupied neighbours| / |occupied neighbours|.
# The aggregate lives in a subselect grouped by ?p, joined back through
# OPTIONAL so that an agent with no occupied neighbour at all still gets a
# value: a GROUP BY drops empty groups entirely, it does not yield 0.
HAPPINESS = Template(
    _PREFIXES
    + """
    DELETE { ?p def:share_similar ?s0 }
    INSERT { ?p def:share_similar ?s1 }
    WHERE {
        ?p a ex:Person .
        OPTIONAL { ?p def:share_similar ?s0 }
        OPTIONAL {
            { SELECT ?p (SUM(?same) AS ?n_similar) (COUNT(?n) AS ?n_total)
              WHERE {
                  ?p a ex:Person ; def:group ?g ; def:location ?c .
                  ?c def:neighbor ?cn .
                  ?n def:location ?cn ; def:group ?gn .
                  BIND(IF(?gn = ?g, 1e0, 0e0) AS ?same)
              } GROUP BY ?p }
        }
        BIND(IF(BOUND(?n_total), ?n_similar / ?n_total, 0e0) AS ?s1)
    }
    """
)

# Park-Miller minimal standard LCG, r <- 16807 r mod (2^31 - 1), in doubles:
# 16807 * (2^31-1) ~ 3.6e13 < 2^53, so every intermediate is exact **provided
# the seed is an integer-valued double in [1, 2^31-2]** — a continuous seed
# turns this into a float map whose entropy erodes with rounding, so the
# calibration layer must `.floor()` its draws.  Anchored on `def:rng` alone
# rather than on a class, so it advances persons and cells (and any future
# agent kind) in one pass.
LCG_TICK = Template(
    _PREFIXES
    + """
    DELETE { ?a def:rng ?r0 }
    INSERT { ?a def:rng ?r1 }
    WHERE {
        ?a def:rng ?r0 .
        BIND(16807e0 * ?r0 AS ?x)
        BIND(?x - 2147483647e0 * FLOOR(?x / 2147483647e0) AS ?r1)
    }
    """
)

# The move, as a rank join.  Movers (share_similar < $want_similar) are
# ranked by their RNG draw; vacant cells (no inbound def:location) by theirs.
# Both subselects project ?rank, so the natural join pairs the k-th mover
# with the k-th cell — a uniform random matching, since both orders are
# random.  Rank is COUNT(peers with rng <= mine); the `<=` (rather than `<`,
# which would be the natural rank) is what keeps the agent's own row in its
# own group, so the minimum never silently vanishes.  The off-by-one is
# identical on both sides and therefore cancels in the join.
#
# If movers outnumber vacant cells the surplus high ranks find no partner and
# stay put — which agent is squeezed out is random, since the ranking is.
RELOCATE = Template(
    _PREFIXES
    + """
    DELETE { ?p def:location ?c0 }
    INSERT { ?p def:location ?c1 }
    WHERE {
        { SELECT ?p (COUNT(?q) AS ?rank) WHERE {
              ?p a ex:Person ; def:rng ?rp ; def:share_similar ?sp .
              FILTER(?sp < $want_similar)
              ?q a ex:Person ; def:rng ?rq ; def:share_similar ?sq .
              FILTER(?sq < $want_similar)
              FILTER(?rq <= ?rp)
          } GROUP BY ?p }
        { SELECT ?c1 (COUNT(?d) AS ?rank) WHERE {
              ?c1 a ex:Cell ; def:rng ?rc .
              FILTER NOT EXISTS { ?u def:location ?c1 }
              ?d a ex:Cell ; def:rng ?rd .
              FILTER NOT EXISTS { ?v def:location ?d }
              FILTER(?rd <= ?rc)
          } GROUP BY ?c1 }
        ?p def:location ?c0 .
    }
    """
)


# ---------------------------------------------------------------------------
# State extract — one row per person, no aggregation
# ---------------------------------------------------------------------------


def state_extract(model) -> pl.DataFrame:
    """Per-person state: group, current coordinates, share of similar
    neighbours.  The segregation index and the unhappy count are plain polars
    expressions on the caller's side (see the demo) — same contract as
    ``rules.state_extract``."""
    return model.query(
        _PREFIXES
        + """
    SELECT ?agent ?group ?x ?y ?share_similar
    WHERE {
        ?agent a ex:Person ;
               def:group ?group ;
               def:share_similar ?share_similar ;
               def:location ?c .
        ?c def:x ?x ; def:y ?y .
    }
    """
    )


# ---------------------------------------------------------------------------
# Swapping the spatial structure — GeoSPARQL points instead of a lattice
# ---------------------------------------------------------------------------
#
# The whole spatial model is captured by ONE relation: `def:neighbor`.
# HAPPINESS, RELOCATE, SETTLE and LCG_TICK never mention coordinates — they
# route through `?c def:neighbor ?cn` and occupancy (`def:location`).  So a
# different space is a different `def:neighbor` rule and nothing else: the
# lattice's 8 integer offsets become "cells within radius R of each other",
# and continuous, irregular locations (real places, not a grid) drop in.
#
# maplib reality check (0.20.19): it *stores* GeoSPARQL geometry — a
# `"POINT(x y)"^^geo:wktLiteral` round-trips and keeps its datatype — but
# every `geof:` function (`geof:distance`, `geof:sfWithin`, `geof:buffer`, ...)
# is "not implemented yet".  So the distance test can't be the one-liner it
# should be; it's hand-rolled here by parsing x/y out of the WKT string and
# comparing squared Euclidean distance.  The moment maplib ships `geof:`, the
# WHERE body collapses to `FILTER(geof:distance(?g1, ?g2, uom:metre) < $radius)`
# and the parsing BINDs delete — the seam does not move.
#
# Cost: unlike GRID_NEIGHBORHOOD's O(8N) VALUES join this is an O(N^2) pair
# scan (no spatial index without `geof:`), fine for hundreds of locations,
# not for a country of them.  The geometry is carried as a plain string under
# `def:geometry` because `map_df`/`map_default` type every column by its
# polars dtype (String -> xsd:string); a fully conformant graph would tag the
# literal `geo:wktLiteral` and link it via `geo:asWKT`, a mapping detail
# orthogonal to the dynamics.
GEO_NEIGHBORHOOD = Template(
    _PREFIXES
    + """
    CONSTRUCT { ?c def:neighbor ?c2 }
    WHERE {
        ?c  a ex:Cell ; def:geometry ?w1 .
        ?c2 a ex:Cell ; def:geometry ?w2 .
        FILTER(?c != ?c2)
        BIND(STRBEFORE(STRAFTER(?w1, "("), ")") AS ?xy1)
        BIND(STRBEFORE(STRAFTER(?w2, "("), ")") AS ?xy2)
        BIND(xsd:double(STRBEFORE(?xy1, " ")) - xsd:double(STRBEFORE(?xy2, " ")) AS ?dx)
        BIND(xsd:double(STRAFTER(?xy1, " ")) - xsd:double(STRAFTER(?xy2, " ")) AS ?dy)
        FILTER(?dx * ?dx + ?dy * ?dy <= $radius * $radius)
    }
    """
)


def geo_state_extract(model) -> pl.DataFrame:
    """Geo counterpart of ``state_extract``: identical shape (one row per
    person, group / x / y / share_similar), so any visualization or summary
    written against the grid extract works unchanged — coordinates are just
    parsed out of the WKT point instead of read from ``def:x``/``def:y``."""
    return model.query(
        _PREFIXES
        + """
    SELECT ?agent ?group ?x ?y ?share_similar
    WHERE {
        ?agent a ex:Person ;
               def:group ?group ;
               def:share_similar ?share_similar ;
               def:location ?c .
        ?c def:geometry ?w .
        BIND(STRBEFORE(STRAFTER(?w, "("), ")") AS ?xy)
        BIND(xsd:double(STRBEFORE(?xy, " ")) AS ?x)
        BIND(xsd:double(STRAFTER(?xy, " ")) AS ?y)
    }
    """
    )


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

SCHELLING_PARAMS = {
    "want_similar": 0.375,  # 3 of 8 Moore neighbours, the classic threshold
    "density": 0.8,  # AMBER's p['density']: per-cell occupancy probability
    "n_groups": 2.0,  # AMBER's p['n_groups']
}

# GRID_NEIGHBORHOOD wires the topology; SETTLE derives the Person population
# onto the passed grid (unless the caller supplies their own).  Seeds ride in
# on the cells' `rng` column (see the module docstring's Randomness note).
SCHELLING_INIT_RULES = (GRID_NEIGHBORHOOD, SETTLE)

# The GeoSPARQL variant: identical except the neighbourhood rule.  Same SETTLE,
# same update rules, same RDFSimulator — proof that the spatial structure is a
# single swappable seam.
SCHELLING_GEO_INIT_RULES = (GEO_NEIGHBORHOOD, SETTLE)

SCHELLING_GEO_PARAMS = {**SCHELLING_PARAMS, "radius": 1.7}  # neighbour cutoff

SCHELLING_UPDATE_RULES = (
    HAPPINESS,  # AMBER: Person.update_happiness, in Model.update
    LCG_TICK,  # draw this tick's randomness
    RELOCATE,  # AMBER: Person.find_new_home, in Model.step
)
