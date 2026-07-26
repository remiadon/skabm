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

* **Randomness.**  Schelling is meaningless without it, and the graph gets
  it straight from SPARQL: the ``pr:uniform`` UDF (polars-random behind
  ``rules.register_polars_random``, maplib >= 0.20.26) is called in-rule via
  ``BIND(pr:uniform(0e0, 1e0) AS ?u)``.  ``SETTLE`` draws occupancy and group
  that way; the per-tick ``DRAW`` rule materialises a fresh ``def:draw`` per
  agent for the move.  No seed column, no in-graph PRNG — just
  ``RDFSimulator(random_seed=...)`` to make a run reproducible.
* **One-to-one matching.**  Each mover must land on a *distinct* empty
  cell.  This is the same wall ``rules`` hits with "no search-and-matching"
  — and hitting it twice, in two unrelated model families, says the limit
  is structural rather than Poledna-specific: SPARQL has no sequential
  activation and no assignment operator.  ``RELOCATE`` works around it
  with a **rank join**: movers are ranked by their ``def:draw``, vacant
  cells by theirs, and rank *k* is paired with rank *k*.  Both orderings are
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
# groups drawn uniformly.  Both draws come straight from the pr:uniform UDF,
# so occupancy is genuinely stochastic: the count is ~$density*|cells|, not
# AMBER's exact int(density*size^2) — one Bernoulli draw per cell.
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
        ?p a ex:Person . ?p def:location ?c . ?p def:group ?g
    }
    WHERE {
        FILTER NOT EXISTS { ?existing a ex:Person }
        ?c a ex:Cell .
        BIND(pr:uniform(0e0, 1e0) AS ?u_occ)
        FILTER(?u_occ < $density)
        # xsd:double cast: FLOOR yields an integer-valued result maplib would
        # otherwise store as Int64, an unwelcome type for a group label.
        BIND(pr:uniform(0e0, 1e0) AS ?u_grp)
        BIND(xsd:double(FLOOR(?u_grp * $n_groups)) AS ?g)
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

# Fresh per-agent random draw for this tick, straight from the pr:uniform UDF.
# RELOCATE ranks movers and vacant cells by this value, so it must be
# *materialised* (a stored triple stable across RELOCATE's self-join) rather
# than drawn inline — an inline BIND would re-draw on each side of the join.
# Anchored on `?a a ?cls` so it refreshes every agent kind (cells + persons)
# in one pass.
DRAW = Template(
    _PREFIXES
    + """
    DELETE { ?a def:draw ?d0 }
    INSERT { ?a def:draw ?d1 }
    WHERE {
        ?a a ?cls .
        OPTIONAL { ?a def:draw ?d0 }
        BIND(pr:uniform(0e0, 1e0) AS ?d1)
    }
    """
)

# The move, as a rank join.  Movers (share_similar < $want_similar) are
# ranked by this tick's def:draw; vacant cells (no inbound def:location) by
# theirs.  Both subselects project ?rank, so the natural join pairs the k-th
# mover with the k-th cell — a uniform random matching, since both orders are
# random.  Rank is COUNT(peers with draw <= mine); the `<=` (rather than `<`,
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
              ?p a ex:Person ; def:draw ?rp ; def:share_similar ?sp .
              FILTER(?sp < $want_similar)
              ?q a ex:Person ; def:draw ?rq ; def:share_similar ?sq .
              FILTER(?sq < $want_similar)
              FILTER(?rq <= ?rp)
          } GROUP BY ?p }
        { SELECT ?c1 (COUNT(?d) AS ?rank) WHERE {
              ?c1 a ex:Cell ; def:draw ?rc .
              FILTER NOT EXISTS { ?u def:location ?c1 }
              ?d a ex:Cell ; def:draw ?rd .
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
# HAPPINESS, RELOCATE, SETTLE and DRAW never mention coordinates — they
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
# onto the passed grid (unless the caller supplies their own).
SCHELLING_INIT_RULES = (GRID_NEIGHBORHOOD, SETTLE)

# The GeoSPARQL variant: identical except the neighbourhood rule.  Same SETTLE,
# same update rules, same RDFSimulator — proof that the spatial structure is a
# single swappable seam.
SCHELLING_GEO_INIT_RULES = (GEO_NEIGHBORHOOD, SETTLE)

SCHELLING_GEO_PARAMS = {**SCHELLING_PARAMS, "radius": 1.7}  # neighbour cutoff

SCHELLING_UPDATE_RULES = (
    HAPPINESS,  # AMBER: Person.update_happiness, in Model.update
    DRAW,  # this tick's random draw per agent (pr:uniform UDF)
    RELOCATE,  # AMBER: Person.find_new_home, in Model.step
)
