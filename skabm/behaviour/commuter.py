"""Commuters, the routes and modes they choose among, and the road links they load.

Day-to-day route choice (Horowitz 1984; Cascetta 1989), pivot-point nested-logit mode
shares (Koppelman 1983), daily replanning (Horni, Nagel & Axhausen 2016), BPR link
delays (Bureau of Public Roads 1964), and the traffic an area's links carry.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence

import numpy as np
import polars as pl
import sympy as sp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from sympy.stats import Uniform

from skabm.dsl import (
    Agents,
    coalesce,
    pick,
    running_sum,
    sum_over,
    total_by,
)
from skabm.sparql import _PREFIXES

# def:mode codes; bike and walk routes have no links, their time is their extra
MODES = {"car": 0.0, "bus": 1.0, "bike": 2.0, "walk": 3.0}

Route, Option, Commuter = Agents("Route"), Agents("Option"), Agents("Commuter")
Link, Area = Agents("Link"), Agents("Area")
bus_factor, theta, anchor, mu, replan = sp.symbols("bus_factor theta anchor mu replan")
peak_factor, demand, bpr_alpha, bpr_beta = sp.symbols(
    "peak_factor demand bpr_alpha bpr_beta"
)
PARAMETERS = {
    bus_factor: 1.6,  # unsourced: bus over car in-vehicle time, stops and dwell
    theta: 1 / 300,  # unsourced: route-choice scale per second, 5 minutes is a factor e
    anchor: 1.0,  # scenario knob: 1 while warming up the base, 0 to pivot against it
    mu: 0.3,  # unsourced: mode over route scale, <= 1 for a nested logit
    replan: 0.1,  # share reconsidering each day, Horni, Nagel & Axhausen (2016)
    peak_factor: 0.8,  # unsourced: peak-hour vehicles per daily car commuter, calibrate
    demand: 1.0,  # scenario knob: period of the year, 1 = school-term weekday
    bpr_alpha: 0.15,  # Bureau of Public Roads (1964)
    bpr_beta: 4.0,  # Bureau of Public Roads (1964)
}


def _leg(route):
    """Seconds on each link of *route* (``Route`` or ``Commuter.route``): cars in
    yesterday's congestion (free flow before any load), buses slower by bus_factor,
    at free flow on a bus lane."""
    loaded = coalesce(route.via.time, route.via.t0)
    bus = sp.Piecewise((route.via.t0, route.via.busway > 0), (loaded, True))
    return sp.Piecewise((loaded, sp.Eq(route.mode, 0)), (bus_factor * bus, True))


# Horowitz (1984) TR-B 18(1); Cascetta (1989) TR-B 23(1)
route_time = {
    Route.time: Route.extra + sum_over(Route.via, _leg(Route)),
    Route.open: sp.Piecewise(
        (0, sp.Eq(Route.mode, 0) & (sum_over(Route.via, 1 - Route.via.car) > 0)),
        (1, True),
    ),
}
# Ben-Akiva & Lerman (1985), Discrete Choice Analysis, ch. 10
option_sum = {
    Option.s: sum_over(Route.option, Route.open * sp.exp(-theta * Route.time))
}
# Koppelman (1983) J. Transp. Eng. 109(4)
option_base = {
    Option.s0: sp.Piecewise(
        (Option.s, anchor > 0), (coalesce(Option.s0, Option.s), True)
    )
}
weight = sp.Piecewise(
    (Option.share0 * (Option.s / Option.s0) ** mu, (Option.s > 0) & (Option.s0 > 0)),
    (0, True),
)
# Koppelman (1983) J. Transp. Eng. 109(4); Daly, Fox & Tsang (2005)
option_share = {
    Option.share: sp.Piecewise(
        (weight / total_by(Option.od, weight), total_by(Option.od, weight) > 0),
        (Option.share0, True),
    )
}
prob = sp.Piecewise(
    (
        Route.option.share * Route.open * sp.exp(-theta * Route.time) / Route.option.s,
        Route.option.s > 0,
    ),
    (0, True),
)
# Daganzo & Sheffi (1977) Transp. Sci. 11(3)
route_prob = {Route.prob: prob}
route_cum = {Route.cum: running_sum(Route.rank, Route.prob, Route.od)}
draw = sp.Piecewise(
    (Commuter.u / replan, Commuter.u < replan),
    ((Commuter.u - replan) / (1 - replan), True),
)
# Horni, Nagel & Axhausen (2016), The Multi-Agent Transport Simulation MATSim
route_choice = {
    Commuter.route: coalesce(
        pick(
            Route,
            sp.Eq(Route.od, Commuter.od)
            & (Route.prob > 1e-9)
            & (draw < Route.cum)
            & ((Commuter.u < replan) | sp.Eq(Commuter.route.open, 0)),
            Route.cum,
        ),
        Commuter.route,
    )
}
route_load = {Route.load: sum_over(Commuter.route, Commuter.weight)}
# Bureau of Public Roads (1964), Traffic Assignment Manual
link_load = {
    Link.flow: sum_over(
        Route.via, sp.Piecewise((Route.load, sp.Eq(Route.mode, 0)), (0, True))
    )
    * peak_factor
    * demand
}
bpr_delay = {
    Link.time: Link.t0 * (1 + bpr_alpha * (Link.flow / Link.capacity) ** bpr_beta)
}
mode = Commuter.route.mode
commuter_state = {
    Commuter.time: Commuter.route.extra
    + sum_over(Commuter.route.via, _leg(Commuter.route)),
    **{
        getattr(Commuter, name): sp.Piecewise((1, sp.Eq(mode, code)), (0, True))
        for name, code in MODES.items()
    },
    Commuter.u: Uniform("u", 0, 1),
}
# skabm: the exposure an old town's foundations feel
area_traffic = {Area.vkt: sum_over(Link.area, Link.flow * Link.length) / 1000}


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
    long ``(id, via)`` frame ``template.via`` maps.

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
