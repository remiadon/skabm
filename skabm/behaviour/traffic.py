"""Commuters choosing a mode and a route each morning on a congestible road network,
Horowitz (1984) and Cascetta (1989), ``bayonne_rules``: twelve rules, in this order.

1. A route's time and open update together, in one rule. Its time becomes its extra plus
the sum, over the road links it goes via, of the link's leg time: for a car route (mode
0), the link's time if it has one, else its t0; for any other route, the parameter
bus_factor times the link's t0 on a bus lane (busway above 0), and bus_factor times (the
link's time if it has one, else its t0) elsewhere. In the same rule its open becomes 0
for a car route (mode 0) whose sum over its links of (1 - car) is above 0, and 1 for any
other route.

2. An option's s becomes the sum, over the routes whose option is this one, of their
open times exp(-the parameter theta times their time).

3. Then an option's s0 becomes its s when the parameter anchor is above 0; otherwise it
keeps its s0 if it has one, else its s.

4. An option's share becomes its weight divided by the total weight over the options
with the same od, when that total is above 0, and otherwise its share0, where its weight
is share0 times (s divided by s0) to the power the parameter mu when s and s0 are both
above 0, else 0.

5. A route's prob becomes, when its option's s is above 0, its option's share times its
open times exp(-the parameter theta times its time) divided by its option's s, and
otherwise 0.

6. Then a route's cum becomes the running sum of prob, in order of rank, over the routes
with the same od.

7. A commuter replans when u is below the parameter replan or when their route's open is
0. Their draw is u divided by replan when u is below replan, else (u - replan) divided
by (1 - replan). A commuter who replans takes the route with the same od whose prob is
above 1e-9 and whose cum is above the draw, picking the least cum. A commuter with no
such route, and anyone who does not replan, keeps their route.

8. A route's load becomes the sum of weight over the commuters whose route is this one.

9. A link's flow becomes the sum, over the routes going via this link, of their load for
a car route (mode 0) and 0 for any other, times the parameter peak_factor times the
parameter demand.

10. Then a link's time becomes its t0 times (1 + the parameter bpr_alpha times (its flow
divided by its capacity) to the power the parameter bpr_beta).

11. A commuter's time, car, bus, bike, walk and u update together, in one rule. Their
time becomes their route's extra plus the sum, over the road links their route goes via,
of the link's leg time: for a car route (mode 0), the link's time if it has one, else
its t0; for any other route, the parameter bus_factor times the link's t0 on a bus lane
(busway above 0), and bus_factor times (the link's time if it has one, else its t0)
elsewhere. In the same rule their car, bus, bike and walk become 1 when their route's
mode is 0, 1, 2 and 3 respectively, else 0, and their u becomes a fresh uniform draw
between 0 and 1.

12. An area's vkt becomes the sum, over the links in this area, of flow times length,
divided by 1000.
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
from skabm.sparql import (
    _PREFIXES,
    register_math,
    register_polars_random,
)

# Routes come from routes() (scipy shortest paths), not from a rule: the graph only
# chooses within the set it is given, and pedestrianize() edits it.
TRAFFIC_UDFS = (register_polars_random, register_math)

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
bayonne_route_time = {
    Route.time: Route.extra + sum_over(Route.via, _leg(Route)),
    Route.open: sp.Piecewise(
        (0, sp.Eq(Route.mode, 0) & (sum_over(Route.via, 1 - Route.via.car) > 0)),
        (1, True),
    ),
}
# Ben-Akiva & Lerman (1985), Discrete Choice Analysis, ch. 10
bayonne_option_sum = {
    Option.s: sum_over(Route.option, Route.open * sp.exp(-theta * Route.time))
}
# Koppelman (1983) J. Transp. Eng. 109(4)
bayonne_option_base = {
    Option.s0: sp.Piecewise(
        (Option.s, anchor > 0), (coalesce(Option.s0, Option.s), True)
    )
}
weight = sp.Piecewise(
    (Option.share0 * (Option.s / Option.s0) ** mu, (Option.s > 0) & (Option.s0 > 0)),
    (0, True),
)
# Koppelman (1983) J. Transp. Eng. 109(4); Daly, Fox & Tsang (2005)
bayonne_option_share = {
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
bayonne_route_prob = {Route.prob: prob}
bayonne_route_cum = {Route.cum: running_sum(Route.rank, Route.prob, Route.od)}
draw = sp.Piecewise(
    (Commuter.u / replan, Commuter.u < replan),
    ((Commuter.u - replan) / (1 - replan), True),
)
# Horni, Nagel & Axhausen (2016), The Multi-Agent Transport Simulation MATSim
bayonne_choose = {
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
bayonne_route_load = {Route.load: sum_over(Commuter.route, Commuter.weight)}
# Bureau of Public Roads (1964), Traffic Assignment Manual
bayonne_link_load = {
    Link.flow: sum_over(
        Route.via, sp.Piecewise((Route.load, sp.Eq(Route.mode, 0)), (0, True))
    )
    * peak_factor
    * demand
}
bayonne_link_time = {
    Link.time: Link.t0 * (1 + bpr_alpha * (Link.flow / Link.capacity) ** bpr_beta)
}
mode = Commuter.route.mode
bayonne_commuter_state = {
    Commuter.time: Commuter.route.extra
    + sum_over(Commuter.route.via, _leg(Commuter.route)),
    **{
        getattr(Commuter, name): sp.Piecewise((1, sp.Eq(mode, code)), (0, True))
        for name, code in MODES.items()
    },
    Commuter.u: Uniform("u", 0, 1),
}
# skabm: the exposure an old town's foundations feel
bayonne_area_traffic = {Area.vkt: sum_over(Link.area, Link.flow * Link.length) / 1000}

bayonne_rules = [
    bayonne_route_time,  # today's network, yesterday's congestion
    bayonne_option_sum,
    bayonne_option_base,
    bayonne_option_share,
    bayonne_route_prob,
    bayonne_route_cum,
    bayonne_choose,
    bayonne_route_load,
    bayonne_link_load,
    bayonne_link_time,
    bayonne_commuter_state,  # the day as experienced
    bayonne_area_traffic,
]


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


def area_membership(links: pl.DataFrame, areas: pl.DataFrame) -> pl.DataFrame:
    """``(id, area)``: every area each link's geometry intersects (a vertex inside the
    area's outer ring, or a segment crossing it), for ``template.links("area")``.

    Geometries are WKT, a link's a ``LINESTRING`` or ``POINT``, an area's a ``POLYGON``
    whose holes are ignored; coordinates are planar, lon/lat is fine at city scale.

    ponytail: one Python iteration per link and area, ~20k a second: fine once, at
    load; vectorise if it ever runs per tick.
    """
    numbers = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
    paths = links["geometry"].str.extract_all(numbers).cast(pl.List(pl.Float64))
    found = []
    for area, polygon in areas.select("id", "geometry").iter_rows():
        ring = polygon.split("((")[1].split(")")[0].replace(",", " ").split()
        a = np.array(ring, dtype=float).reshape(-1, 2)
        b = np.roll(a, -1, axis=0)  # the ring's edges run a -> b
        for link, xy in zip(links["id"], paths):
            p = np.asarray(xy).reshape(-1, 2)
            if _ray_parity(p, a, b).any() or _crossing(p[:-1], p[1:], a, b):
                found.append((link, area))
    return pl.DataFrame(found, schema=["id", "area"], orient="row")


def _ray_parity(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Even-odd test: is each point of *p* inside the ring with edges a -> b?"""
    y = p[:, 1:2]
    straddles = (a[:, 1] > y) != (b[:, 1] > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cut = a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
    return (straddles & (p[:, 0:1] < x_cut)).sum(axis=1) % 2 == 1


def _crossing(p: np.ndarray, q: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    """Does any segment p -> q properly cross any edge a -> b?"""

    def side(u, v, w):
        return np.sign(
            (v[..., 0] - u[..., 0]) * (w[..., 1] - u[..., 1])
            - (v[..., 1] - u[..., 1]) * (w[..., 0] - u[..., 0])
        )

    p, q = p[:, None], q[:, None]
    return bool(
        (
            (side(p, q, a) * side(p, q, b) < 0) & (side(a, b, p) * side(a, b, q) < 0)
        ).any()
    )
