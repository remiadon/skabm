"""
Traffic behaviour templates — commuters choosing a mode and a route, day after day, on a
road network whose congestion is the sum of their choices.

A fourth model family on the same machinery.  The agents are ``Commuter``s, a sample of
real workers each standing for ``def:weight`` of them.  What they choose between are
``Route``s — one path for one origin-destination pair (OD) by one mode, its road
``Link``s attached as ``def:via`` — grouped into ``Option``s, one per OD x mode.  An
``Area`` is a polygon a policy talks about ("Grand Bayonne"), tied at init to the links it
touches by GeoSPARQL (``geof:sfIntersects``, from ``rules.register_geosparql``).

**One tick is one working day's morning peak** — the day-to-day process of Horowitz
(1984) and Cascetta (1989), MATSim's co-evolutionary loop in spirit (Horni, Nagel &
Axhausen 2016).  Choices use today's network (closures are known) but yesterday's
congestion (expectations are learned); what a commuter *experiences* is today's.  The
rule order is that event order: time the routes, update the choice probabilities,
choose, load the network, record what happened.

**Mode choice is pivot-point** (incremental logit: Koppelman 1983; Daly, Fox & Tsang
2005).  Each Option carries ``share0``, the OD's observed mode share (the census), and
the model only ever moves it by the change in that mode's logsum since the base::

    share_m  ∝  share0_m * (S_m / S0_m) ** mu,     S_m = sum over its routes of exp(-theta * T_r)

which is nested logit with routes nested in modes (Ben-Akiva & Lerman 1985) and ``mu``
the nest scale ratio.  Base shares are reproduced exactly, and errors in *absolute*
times — connector lengths, bus waits, parking — cancel; only changes move anyone.  The
base is whatever the model sees while ``$anchor`` is 1: warm up with it on, then switch it
off and intervene.  The price: a mode nobody on an OD uses today stays unused.

**Route generation is not a rule.**  SPARQL has no shortest path, so ``routes`` (scipy)
builds the choice set and the graph only ever chooses *within* it.  A closure is a graph
edit (``pedestrianize``) that makes routes infeasible — their commuters must replan the
next morning — and new routes are generated against the edited graph and mapped in; the
stable ``rank`` makes re-adding a known route a no-op.  ``def:via`` being many-valued, the
IR-derived ``sim.extract()`` repeats a route once per link it uses: query the classes you
need directly (``notebooks/bayonne/world.py`` does) or pass ``state_extract=``.

**What it leaves out**, deliberately: departure-time choice, destination change and trip
suppression (the other channels of "traffic evaporation", Cairns, Hass-Klau & Goodwin
1998 — here traffic can only switch route or mode), spillback (BPR is a static
volume-delay function, so a queue never blocks the link upstream), and car ownership
(commuters on an OD are exchangeable, so the census gives no one a reason to keep a car).
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from string import Template

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from skabm.rules import (
    _PREFIXES,
    register_geosparql,
    register_math,
    register_polars_random,
)

TRAFFIC_UDFS = (register_polars_random, register_math, register_geosparql)

# Route and Option ``def:mode`` codes.  Bike and walk routes carry a constant time and
# no links: no scenario here changes them, and under a pivot a constant only has to be
# the same before and after.
MODES = {"car": 0.0, "bus": 1.0, "bike": 2.0, "walk": 3.0}

_SOURCE = "Horowitz (1984) TR-B 18(1); Cascetta (1989) TR-B 23(1)"

# maplib gotcha: an OPTIONAL value defaulted with IF(BOUND(?x), ?x, 0e0) becomes a
# struct column when *no* row binds it (day one, before any link was loaded), and
# SUM over it panics.  COALESCE(?x, 0e0) stays a double, so every default below is one.


# ---------------------------------------------------------------------------
# area_membership — which links each Area touches (init, GeoSPARQL)
# ---------------------------------------------------------------------------
# Materialised once, so neither the per-tick accounting nor a closure pays for
# geometry again.  A link crossing the boundary (a bridge into the zone) is in.

area_membership = Template(
    _PREFIXES
    + """
CONSTRUCT { ?l def:area ?a }
WHERE {
    ?a a ex:Area ; def:geometry ?zone .
    ?l a ex:Link ; def:geometry ?g .
    FILTER(geof:sfIntersects(?g, ?zone))
}
"""
)
area_membership.metadata = {
    "@id": "area_membership",
    "@type": "Behaviour",
    "agentClass": "ex:Link",
    "source": "OGC GeoSPARQL 1.1, geof:sfIntersects",
}

# ---------------------------------------------------------------------------
# route_time — every route timed on today's network (update)
# ---------------------------------------------------------------------------
# Yesterday's link times (free flow before the first load), today's closures.  A
# car route is ``open`` only if every link on it admits cars; buses run on every
# link of their own routes, on bus lanes at free-flow speed, elsewhere stuck in
# the same congestion as cars — so a closure that empties a street speeds up the
# buses still using it.  Placeholder: ``bus_factor``.

route_time = Template(
    _PREFIXES
    + """
DELETE { ?r def:time ?t0 . ?r def:open ?o0 }
INSERT { ?r def:time ?t1 . ?r def:open ?o1 }
WHERE {
    { SELECT ?r (SUM(?tau) AS ?sum) (MIN(?ok) AS ?o1)
      WHERE {
          ?r a ex:Route ; def:mode ?m ; def:via ?l .
          ?l def:t0 ?free ; def:car ?car ; def:busway ?bw .
          OPTIONAL { ?l def:time ?loaded }
          BIND(COALESCE(?loaded, ?free) AS ?lt)
          BIND(IF(?m = 0e0, ?lt, $bus_factor * IF(?bw > 0e0, ?free, ?lt)) AS ?tau)
          BIND(IF(?m = 0e0, ?car, 1e0) AS ?ok)
      } GROUP BY ?r }
    ?r def:extra ?x .
    OPTIONAL { ?r def:time ?t0 }
    OPTIONAL { ?r def:open ?o0 }
    BIND(?x + ?sum AS ?t1)
}
"""
)
route_time.metadata = {
    "@id": "route_time",
    "@type": "Behaviour",
    "agentClass": "ex:Route",
    "source": _SOURCE,
}

# ---------------------------------------------------------------------------
# option_sum — each nest's logsum argument S_m (update)
# ---------------------------------------------------------------------------
# S_m = sum over the option's open routes of exp(-theta T_r).  While ``$anchor``
# is 1 the base S0_m follows it, so the pivot below is the identity and the
# census shares hold exactly; set it to 0 and S0_m freezes at the base.
# Placeholders: ``theta``, ``anchor``.

option_sum = Template(
    _PREFIXES
    + """
DELETE { ?o def:s ?s0 . ?o def:s0 ?b0 }
INSERT { ?o def:s ?s1 . ?o def:s0 ?b1 }
WHERE {
    { SELECT ?o (SUM(?e) AS ?s1)
      WHERE {
          ?r a ex:Route ; def:option ?o ; def:time ?t .
          OPTIONAL { ?r def:open ?open }
          BIND(COALESCE(?open, 1e0) * math:exp(0e0 - $theta * ?t) AS ?e)
      } GROUP BY ?o }
    OPTIONAL { ?o def:s ?s0 }
    OPTIONAL { ?o def:s0 ?b0 }
    BIND(IF($anchor > 0e0, ?s1, COALESCE(?b0, ?s1)) AS ?b1)
}
"""
)
option_sum.metadata = {
    "@id": "option_sum",
    "@type": "Behaviour",
    "agentClass": "ex:Option",
    "source": "Ben-Akiva & Lerman (1985), Discrete Choice Analysis, ch. 10",
}

# ---------------------------------------------------------------------------
# option_share — pivot-point mode shares (update)
# ---------------------------------------------------------------------------
# w_m = share0_m (S_m / S0_m)^mu, normalised over the OD.  A mode whose routes
# are all closed has S_m = 0 and drops out; an OD left with nothing keeps its
# census shares rather than dividing by zero.  Placeholder: ``mu``.

option_share = Template(
    _PREFIXES
    + """
DELETE { ?o def:share ?p0 }
INSERT { ?o def:share ?p1 }
WHERE {
    { SELECT ?od (SUM(?w) AS ?total)
      WHERE {
          ?k a ex:Option ; def:od ?od ; def:share0 ?q ; def:s ?s ; def:s0 ?b .
          BIND(IF(?s > 0e0 && ?b > 0e0, ?q * math:exp($mu * math:log(?s / ?b)), 0e0) AS ?w)
      } GROUP BY ?od }
    ?o a ex:Option ; def:od ?od ; def:share0 ?q1 ; def:s ?s1 ; def:s0 ?b1 .
    OPTIONAL { ?o def:share ?p0 }
    BIND(IF(?s1 > 0e0 && ?b1 > 0e0, ?q1 * math:exp($mu * math:log(?s1 / ?b1)), 0e0) AS ?w1)
    BIND(IF(?total > 0e0, ?w1 / ?total, ?q1) AS ?p1)
}
"""
)
option_share.metadata = {
    "@id": "option_share",
    "@type": "Behaviour",
    "agentClass": "ex:Option",
    "source": "Koppelman (1983) J. Transp. Eng. 109(4); Daly, Fox & Tsang (2005)",
}

# ---------------------------------------------------------------------------
# route_prob — choice probability and cumulative interval per route (update)
# ---------------------------------------------------------------------------
# p_r = share_m * exp(-theta T_r) / S_m, and cum_r = sum of p over the OD's
# routes ranked at or below r: the upper end of r's slice of [0, 1), which is
# what ``choose`` draws against.  ``rank`` is any total order within an OD.

route_prob = Template(
    _PREFIXES
    + """
DELETE { ?r def:prob ?p0 . ?r def:cum ?c0 }
INSERT { ?r def:prob ?p1 . ?r def:cum ?c1 }
WHERE {
    { SELECT ?r (SUM(?pk) AS ?c1)
      WHERE {
          ?r a ex:Route ; def:od ?od ; def:rank ?rank .
          ?k a ex:Route ; def:od ?od ; def:rank ?rk ; def:time ?tk ; def:option ?ok .
          FILTER(?rk <= ?rank)
          ?ok def:share ?shk ; def:s ?sk .
          OPTIONAL { ?k def:open ?openk }
          BIND(IF(?sk > 0e0,
                  (?shk * COALESCE(?openk, 1e0)) * (math:exp(0e0 - $theta * ?tk) / ?sk),
                  0e0) AS ?pk)
      } GROUP BY ?r }
    ?r def:time ?t ; def:option ?o .
    ?o def:share ?sh ; def:s ?s .
    OPTIONAL { ?r def:open ?open }
    OPTIONAL { ?r def:prob ?p0 }
    OPTIONAL { ?r def:cum ?c0 }
    BIND(IF(?s > 0e0,
            (?sh * COALESCE(?open, 1e0)) * (math:exp(0e0 - $theta * ?t) / ?s),
            0e0) AS ?p1)
}
"""
)
route_prob.metadata = {
    "@id": "route_prob",
    "@type": "Behaviour",
    "agentClass": "ex:Route",
    "source": "Daganzo & Sheffi (1977) Transp. Sci. 11(3)",
}

# ---------------------------------------------------------------------------
# choose — who replans, and onto which route (update)
# ---------------------------------------------------------------------------
# A commuter replans when its draw falls below ``$replan`` (inertia: most people
# drive yesterday's route), or unconditionally when its route closed overnight.
# The one draw serves twice — rescaled to [0, 1) on whichever side of $replan it
# fell, it picks the route whose slice (cum - prob, cum] contains it.  Taking the
# smallest qualifying ``cum`` is what makes the pick unique even when two slices
# meet at a floating-point seam.  Placeholder: ``replan``.

choose = Template(
    _PREFIXES
    + """
DELETE { ?c def:route ?r0 }
INSERT { ?c def:route ?r1 }
WHERE {
    { SELECT ?c (MIN(?hi) AS ?h)
      WHERE {
          ?c a ex:Commuter ; def:u ?u ; def:od ?od ; def:route ?now .
          OPTIONAL { ?now def:open ?still }
          FILTER(?u < $replan || COALESCE(?still, 1e0) = 0e0)
          BIND(IF(?u < $replan, ?u / $replan, (?u - $replan) / (1e0 - $replan)) AS ?v)
          ?r a ex:Route ; def:od ?od ; def:cum ?hi ; def:prob ?p .
          FILTER(?p > 1e-9 && ?v < ?hi)
      } GROUP BY ?c }
    ?c def:od ?od ; def:route ?r0 .
    ?r1 a ex:Route ; def:od ?od ; def:cum ?h ; def:prob ?p1 .
    FILTER(?p1 > 1e-9)
}
"""
)
choose.metadata = {
    "@id": "choose",
    "@type": "Behaviour",
    "agentClass": "ex:Commuter",
    "source": "Horni, Nagel & Axhausen (2016), The Multi-Agent Transport Simulation MATSim",
}

# ---------------------------------------------------------------------------
# link_load — flows and congested times (update)
# ---------------------------------------------------------------------------
# flow = (commuters driving through) x peak_factor x demand, in vehicles per
# peak hour, and the BPR volume-delay function t = t0 (1 + alpha (v/c)^beta).
# ``peak_factor`` turns one daily commuter into peak-hour vehicles — the share
# of commutes in the peak hour, times the uplift for everything that is not a
# commute (school runs, deliveries, through traffic) — and is the knob a count
# calibrates.  ``demand`` is the period of the year: 1 on a school-term weekday.
# Placeholders: ``peak_factor``, ``demand``, ``bpr_alpha``, ``bpr_beta``.

link_load = Template(
    _PREFIXES
    + """
DELETE { ?l def:flow ?f0 . ?l def:time ?t0 }
INSERT { ?l def:flow ?f1 . ?l def:time ?t1 }
WHERE {
    ?l a ex:Link ; def:t0 ?free ; def:capacity ?cap .
    OPTIONAL { ?l def:flow ?f0 }
    OPTIONAL { ?l def:time ?t0 }
    OPTIONAL {
        { SELECT ?l (SUM(?w) AS ?n)
          WHERE {
              ?c a ex:Commuter ; def:weight ?w ; def:route ?r .
              ?r def:mode 0e0 ; def:via ?l .
          } GROUP BY ?l }
    }
    BIND(COALESCE(?n, 0e0) * ($peak_factor * $demand) AS ?f1)
    BIND(?free * (1e0 + $bpr_alpha * math:exp($bpr_beta * math:log(?f1 / ?cap))) AS ?t1)
}
"""
)
link_load.metadata = {
    "@id": "link_load",
    "@type": "Behaviour",
    "agentClass": "ex:Link",
    "source": "Bureau of Public Roads (1964), Traffic Assignment Manual",
}

# ---------------------------------------------------------------------------
# commuter_state — the day as experienced, and tomorrow's draw (update)
# ---------------------------------------------------------------------------
# Travel time on today's loaded network, and one 0/1 indicator per mode so the
# mode shares are plain means — observables the IR derives with nothing
# declared.  The draw is materialised here, not drawn inside ``choose``: an
# inline BIND would re-draw on every candidate route of the join.

commuter_state = Template(
    _PREFIXES
    + """
DELETE { ?c def:time ?t0 . ?c def:car ?a0 . ?c def:bus ?b0 .
         ?c def:bike ?k0 . ?c def:walk ?w0 . ?c def:u ?u0 }
INSERT { ?c def:time ?t1 . ?c def:car ?a1 . ?c def:bus ?b1 .
         ?c def:bike ?k1 . ?c def:walk ?w1 . ?c def:u ?u1 }
WHERE {
    ?c a ex:Commuter ; def:route ?r .
    ?r def:mode ?m ; def:extra ?x .
    OPTIONAL {
        { SELECT ?c (SUM(?tau) AS ?sum)
          WHERE {
              ?c a ex:Commuter ; def:route ?r .
              ?r def:mode ?m ; def:via ?l .
              ?l def:time ?lt ; def:t0 ?free ; def:busway ?bw .
              BIND(IF(?m = 0e0, ?lt, $bus_factor * IF(?bw > 0e0, ?free, ?lt)) AS ?tau)
          } GROUP BY ?c }
    }
    OPTIONAL { ?c def:time ?t0 }
    OPTIONAL { ?c def:car ?a0 }
    OPTIONAL { ?c def:bus ?b0 }
    OPTIONAL { ?c def:bike ?k0 }
    OPTIONAL { ?c def:walk ?w0 }
    OPTIONAL { ?c def:u ?u0 }
    BIND(?x + COALESCE(?sum, 0e0) AS ?t1)
    BIND(IF(?m = 0e0, 1e0, 0e0) AS ?a1)
    BIND(IF(?m = 1e0, 1e0, 0e0) AS ?b1)
    BIND(IF(?m = 2e0, 1e0, 0e0) AS ?k1)
    BIND(IF(?m = 3e0, 1e0, 0e0) AS ?w1)
    BIND(pr:uniform(0e0, 1e0) AS ?u1)
}
"""
)
commuter_state.metadata = {
    "@id": "commuter_state",
    "@type": "Behaviour",
    "agentClass": "ex:Commuter",
    "source": _SOURCE,
}

# ---------------------------------------------------------------------------
# area_traffic — vehicle-km per peak hour inside each Area (update)
# ---------------------------------------------------------------------------
# The exposure an old town's foundations feel: every car on every link that
# touches the area, weighted by the link's length.  Buses are not counted.

area_traffic = Template(
    _PREFIXES
    + """
DELETE { ?a def:vkt ?v0 }
INSERT { ?a def:vkt ?v1 }
WHERE {
    ?a a ex:Area .
    OPTIONAL { ?a def:vkt ?v0 }
    OPTIONAL {
        { SELECT ?a (SUM(?f * ?len) AS ?metres)
          WHERE { ?l def:area ?a ; def:flow ?f ; def:length ?len } GROUP BY ?a }
    }
    BIND(COALESCE(?metres, 0e0) / 1e3 AS ?v1)
}
"""
)
area_traffic.metadata = {
    "@id": "area_traffic",
    "@type": "Behaviour",
    "agentClass": "ex:Area",
    "source": "skabm",
}


# ---------------------------------------------------------------------------
# Composition and defaults
# ---------------------------------------------------------------------------

# fmt: off
traffic_params = {
    "replan":      0.1,      # share of commuters reconsidering each day (MATSim's usual)
    "theta":       1 / 300,  # route-choice scale, per second: 5 minutes is a factor e
    "mu":          0.3,      # mode-level scale over route-level, <= 1 (nested logit)
    "bpr_alpha":   0.15,     # BPR (1964)
    "bpr_beta":    4.0,      # BPR (1964)
    "bus_factor":  1.6,      # bus in-vehicle time over car time: stops and dwell
    "peak_factor": 0.8,      # peak-hour vehicles per daily car commuter -- calibrate
    "demand":      1.0,      # period of the year; 1 = school-term weekday
    "anchor":      1.0,      # 1 while warming up the base, 0 to pivot against it
}
# fmt: on

TRAFFIC_INIT_RULES = (area_membership,)
TRAFFIC_UPDATE_RULES = (
    route_time,  # today's network, yesterday's congestion
    option_sum,  # nest logsums
    option_share,  # pivot-point mode shares
    route_prob,  # route probabilities and cumulative slices
    choose,  # inertia, forced replanning, one draw
    link_load,  # flows, BPR
    commuter_state,  # the day as experienced
    area_traffic,  # vehicle-km per area
)


def pedestrianize(areas: Sequence[str] = (), streets: Sequence[str] = ()) -> str:
    """The SPARQL UPDATE that closes streets to cars, as a string for ``model_.update``.

    A closure is named either by ``Area`` — every link the polygon touches, from
    ``area_membership`` — or by street name, which is how a corridor is closed
    without a polygon around it.  Buses, bikes and walkers are untouched: their
    routes never read ``def:car``.
    """
    quoted = [
        " ".join(f'"{name}"' for name in names) or '""' for names in (areas, streets)
    ]
    return (
        _PREFIXES
        + f"""
DELETE {{ ?l def:car ?c0 }}
INSERT {{ ?l def:car 0e0 }}
WHERE {{
    ?l a ex:Link ; def:car ?c0 .
    FILTER(?c0 > 0e0)
    {{ ?l def:area ?a . ?a def:name ?area . VALUES ?area {{ {quoted[0]} }} }}
    UNION
    {{ ?l def:name ?street . VALUES ?street {{ {quoted[1]} }} }}
}}
"""
    )


def routes(
    links: pl.DataFrame,
    nodes: pl.DataFrame,
    zones: pl.DataFrame,
    ods: pl.DataFrame,
    mode: float,
    cost: str = "t0",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """One shortest path per OD, driven (or ridden) between the zones' nearest nodes.

    *links* are the ones *mode* may use — ``id``, ``src``, ``dst`` and the *cost*
    column; *nodes* are ``id``, ``x``, ``y`` in metres; *zones* are ``id``, ``x``,
    ``y`` and ``speed``, the metres per second at which a zone reaches the
    network; *ods* are ``od``, ``origin``, ``destination`` (zone ids).

    A zone joins the network at the nearest node of its largest strongly
    connected component, and the connector at both ends is ``extra`` seconds.
    That one rule is what makes a pedestrianised centre reachable by car: a zone
    left inside it walks out to the nearest street cars still use.

    Returns ``(routes, via)``.  ``routes`` has ``id``, ``od``, ``mode``, ``rank``,
    ``extra`` and ``time``, the last filled only for a path of no links (both
    ends on one node).  ``rank`` is a CRC of mode and path, so a path found again
    gets the same id and rank, and re-mapping it adds nothing.  ``via`` is the
    long ``(id, via)`` frame ``templates.via_template`` maps.

    ponytail: one Dijkstra per distinct origin, 256 origins' predecessors in
    memory at a time (256 x nodes x 4 bytes); a bidirectional or A* search per
    OD would be the next step if OD pairs, not origins, became the bottleneck.
    """
    index = nodes.select("id").with_row_index("i")
    edges = (
        links.join(index.rename({"id": "src", "i": "s"}), on="src")
        .join(index.rename({"id": "dst", "i": "d"}), on="dst")
        .sort(cost)
        .unique(["s", "d"], keep="first", maintain_order=True)  # cheapest parallel
    )
    s, d = edges["s"].to_numpy(), edges["d"].to_numpy()
    weight = np.maximum(edges[cost].to_numpy(), 0.1)  # an explicit zero is no edge
    graph = csr_matrix((weight, (s, d)), shape=(nodes.height, nodes.height))
    link_of = dict(zip(zip(s.tolist(), d.tolist()), edges["id"].to_list()))

    _, component = connected_components(graph, directed=True, connection="strong")
    reachable = np.flatnonzero(component == np.bincount(component).argmax())
    xy = nodes.select("x", "y").to_numpy()
    gap, nearest = cKDTree(xy[reachable]).query(zones.select("x", "y").to_numpy())
    access = zones.select(
        "id", node=pl.Series(reachable[nearest]), walk=pl.Series(gap) / zones["speed"]
    )
    trips = ods.join(
        access.rename({"id": "origin", "node": "o", "walk": "wo"}), on="origin"
    ).join(
        access.rename({"id": "destination", "node": "d", "walk": "wd"}),
        on="destination",
    )

    name = nodes["id"].to_list()
    found, via = [], []
    batches = trips.with_columns(batch=pl.col("o").rank("dense") // 256)
    for chunk in batches.partition_by("batch"):  # 256 origins at a time: bounded memory
        origins = chunk["o"].unique().to_numpy()
        _, predecessors = dijkstra(graph, indices=origins, return_predecessors=True)
        row = {o: i for i, o in enumerate(origins.tolist())}
        for od, o, d, extra in chunk.select(
            "od", "o", "d", pl.col("wo") + pl.col("wd")
        ).iter_rows():
            before, path, end = predecessors[row[o]], [], d
            while d != o:
                path.append(link_of[(before[d], d)])
                d = before[d]
            path.reverse()
            # the end nodes too: an empty path is still a different route from a new node
            key = f"{mode}:{name[o]}:{name[end]}:{','.join(path)}"
            rank = zlib.crc32(key.encode())
            rid = f"{od}_m{int(mode)}_{rank:08x}"
            found.append((rid, od, float(rank), extra, None if path else extra))
            via.extend((rid, link) for link in path)
    return (
        pl.DataFrame(
            found,
            schema={
                "id": pl.String,
                "od": pl.String,
                "rank": pl.Float64,
                "extra": pl.Float64,
                "time": pl.Float64,
            },
            orient="row",
        ).with_columns(mode=pl.lit(mode)),
        pl.DataFrame(via, schema={"id": pl.String, "via": pl.String}, orient="row"),
    )
