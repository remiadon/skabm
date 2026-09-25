"""Commuter route and mode choice on the RDFSimulator machinery — skabm.behaviour.traffic.

A toy town with the shape of the Bayonne question: West and East joined by a centre
street through an ``Area``, with a slower, narrower bypass around it.

Covered: the GeoSPARQL predicates, park-and-walk access to a closed area, the pivot
holding census shares while anchored, closing by area and by street name, and the
closure itself — forced replanning on day one, re-routing against the edited graph,
the centre emptied of cars, and a mode shift that is partial because the bypass exists.
"""

import polars as pl
import pytest
from maplib import Model

from skabm.behaviour.traffic import (
    MODES,
    RULES,
    TRAFFIC_UDFS,
    area_membership,
    pedestrianize,
    routes,
)
from skabm.ottr import (
    area_template,
    commuter_template,
    link_template,
    links_template,
    option_template,
    route_template,
    via_template,
)
from skabm.simulation import RDFSimulator
from skabm.sparql import _PREFIXES

#        W ---- A ====== C ====== B ---- E        centre street, 50 km/h
#                \     [Area]    /
#                 D ----------- F                 bypass; D-F is 30 km/h, one lane
NODES = pl.DataFrame(
    {
        "id": ["W", "A", "C", "B", "E", "D", "F"],
        "x": [0.0, 1000.0, 2000.0, 3000.0, 4000.0, 1300.0, 2700.0],
        "y": [0.0, 0.0, 0.0, 0.0, 0.0, -1000.0, -1000.0],
    }
)
ROADS = [  # src, dst, metres, km/h, veh/h, car, busway, name
    ("W", "A", 1000, 50, 2000, 1, 0, "Avenue Ouest"),
    ("A", "C", 1000, 50, 900, 1, 0, "Rue du Centre"),
    ("C", "B", 1000, 50, 900, 1, 0, "Rue du Centre"),
    ("B", "E", 1000, 50, 2000, 1, 0, "Avenue Est"),
    ("A", "D", 1044, 70, 1500, 1, 0, "Boulevard Sud"),
    ("D", "F", 1400, 30, 600, 1, 0, "Boulevard Sud"),
    ("F", "B", 1044, 70, 1500, 1, 0, "Boulevard Sud"),
]
XY = dict(zip(NODES["id"], NODES.select("x", "y").rows()))
LINKS = pl.DataFrame(
    [
        (
            f"{a}{b}",
            a,
            b,
            float(m),
            m / (kmh / 3.6),
            float(cap),
            float(car),
            float(bw),
            name,
            f"LINESTRING({XY[a][0]} {XY[a][1]}, {XY[b][0]} {XY[b][1]})",
        )
        for a0, b0, m, kmh, cap, car, bw, name in ROADS
        for a, b in ((a0, b0), (b0, a0))
    ],
    schema=[
        "id",
        "src",
        "dst",
        "length",
        "t0",
        "capacity",
        "car",
        "busway",
        "name",
        "geometry",
    ],
    orient="row",
)
CENTRE = "POLYGON((1500 -300, 2500 -300, 2500 300, 1500 300, 1500 -300))"
ZONES = pl.DataFrame(
    {
        "id": ["west", "east", "centre"],
        "x": [-100.0, 4100.0, 2000.0],
        "y": [0.0, 0.0, 50.0],
        "speed": 1.3,
    }
)
ODS = pl.DataFrame(
    {
        "od": ["west_east", "west_centre"],
        "origin": "west",
        "destination": ["east", "centre"],
    }
)
SHARES = {"car": 0.8, "bus": 0.2}
PARAMS = {"peak_factor": 1.5}  # few agents, a real jam


def car_links(model: Model) -> pl.DataFrame:
    """The network as the graph says it is now — the single source of truth."""
    state = model.query(
        _PREFIXES
        + "SELECT ?l ?car ?time WHERE { ?l a ex:Link ; def:car ?car ; def:time ?time }"
    ).select(id=pl.col("l").str.extract(r"#(\w+)>"), car="car", time="time")
    return LINKS.drop("car").join(state, on="id").filter(pl.col("car") > 0)


def choice_set(
    links: pl.DataFrame, cost: str = "t0"
) -> tuple[pl.DataFrame, pl.DataFrame]:
    car, car_via = routes(links, NODES, ZONES, ODS, MODES["car"], cost)
    bus, bus_via = routes(LINKS, NODES, ZONES, ODS, MODES["bus"])
    found = pl.concat([car, bus]).with_columns(
        option=pl.format("{}_{}", "od", pl.col("mode").cast(pl.Int8)),
        time=pl.col("time"),
    )
    return found, pl.concat([car_via, bus_via])


def town(n: int = 60) -> tuple[Model, pl.DataFrame]:
    found, via = choice_set(LINKS.filter(pl.col("car") > 0))
    options = pl.DataFrame(
        [
            (f"{od}_{int(MODES[m])}", od, MODES[m], s)
            for od in ODS["od"]
            for m, s in SHARES.items()
        ],
        schema=["id", "od", "mode", "share0"],
        orient="row",
    )
    # everyone starts on the car route; the first days sort out who takes the bus
    commuters = (
        ODS.select("od")
        .join(found.filter(pl.col("mode") == 0), on="od")
        .select("od", route="id")
        .join(pl.DataFrame({"k": range(n)}), how="cross")
        .select(
            id=pl.format("c_{}_{}", "od", "k"),
            od="od",
            route="route",
            weight=pl.lit(10.0),
        )
    )
    world = Model()
    world.map(link_template, LINKS.with_iri("src", "dst"))
    areas = pl.DataFrame({"id": ["centre"], "name": ["Centre"], "geometry": [CENTRE]})
    world.map(area_template, areas.with_iri())
    world.map(links_template("area"), area_membership(LINKS, areas).with_iri("area"))
    world.map(option_template, options.with_iri())
    world.map(route_template, found.with_iri("option"))
    world.map(via_template, via.with_iri("via"))
    world.map(commuter_template, commuters.with_iri("route"))
    return world, found


def simulator(n_periods: int, anchor: float = 1.0) -> RDFSimulator:
    return RDFSimulator(
        rules=RULES,
        params={**PARAMS, "anchor": anchor},
        udfs=TRAFFIC_UDFS,
        n_periods=n_periods,
        random_seed=0,
    )


def query(model: Model, body: str) -> pl.DataFrame:
    return model.query(_PREFIXES + body)


def test_a_link_belongs_to_the_areas_it_touches():
    square = "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))"
    cases = {
        "POINT(5 5)": True,
        "POINT(15 5)": False,
        "LINESTRING(1 1, 2 2, 3 1)": True,
        "LINESTRING(5 5, 15 5)": True,  # leaves the square
        "LINESTRING(-5 5, 15 5)": True,  # both ends out, crosses through
        "LINESTRING(11 11, 20 20)": False,
    }
    links = pl.DataFrame({"id": list(cases), "geometry": list(cases)})
    inside = area_membership(links, pl.DataFrame({"id": ["sq"], "geometry": [square]}))
    assert set(inside["id"]) == {g for g, hit in cases.items() if hit}


def test_a_closed_area_is_reached_on_foot():
    # without the centre street, the centre zone's nearest car node is A or B,
    # 1 km away: the route drives there and walks the rest
    open_ = LINKS.filter(pl.col("car") > 0, ~pl.col("name").eq("Rue du Centre"))
    found, via = routes(open_, NODES, ZONES, ODS, MODES["car"])
    to_centre = found.filter(pl.col("od") == "west_centre").row(0, named=True)
    assert to_centre["extra"] == pytest.approx(
        (100 + 1000 + 50 / 1000 * 0) / 1.3, rel=0.01
    )
    assert set(via.filter(pl.col("id") == to_centre["id"])["via"]) == {"WA"}
    # a path found again is the same route: same id, same rank
    assert routes(open_, NODES, ZONES, ODS, MODES["car"])[0].equals(found)


def test_census_shares_hold_while_anchored():
    world, _ = town()
    sim = simulator(n_periods=12).fit(world)
    shares = query(
        sim.model_,
        "SELECT ?q ?p WHERE { ?o a ex:Option ; def:share0 ?q ; def:share ?p }",
    )
    assert shares["p"].to_list() == pytest.approx(shares["q"].to_list())
    # and the agents drift to them: all started by car
    row = query(
        sim.model_, "SELECT (AVG(?b) AS ?bus) WHERE { ?c a ex:Commuter ; def:bus ?b }"
    )
    assert 0.05 < row["bus"][0] < 0.3


def test_closing_the_centre():
    world, _ = town()
    sim = simulator(n_periods=15)
    base = pl.DataFrame(sim.fit_iter(world))
    assert base["sig__SUM__Area__vkt"][-1] > 0  # cars cross the centre today

    # the policy, by area — and the same four links by street name
    sim.model_.update(pedestrianize(areas=["Centre"]))
    closed = query(sim.model_, "SELECT ?l WHERE { ?l a ex:Link ; def:car 0e0 }")[
        "l"
    ].str.extract(r"#(\w+)>")
    assert set(closed) == {"AC", "CA", "CB", "BC"}
    by_street, _ = town()
    RDFSimulator(
        rules=RULES,
        params=PARAMS,
        udfs=TRAFFIC_UDFS,
        n_periods=0,
    ).fit(by_street)
    by_street.update(pedestrianize(streets=["Rue du Centre"]))
    assert set(
        query(by_street, "SELECT ?l WHERE { ?l a ex:Link ; def:car 0e0 }")[
            "l"
        ].str.extract(r"#(\w+)>")
    ) == set(closed)

    # new routes against the graph as edited, on today's congested times
    found, via = choice_set(car_links(sim.model_), cost="time")
    sim.model_.map(route_template, found.with_iri("option"))
    sim.model_.map(via_template, via.with_iri("via"))

    sim.set_params(warm_start=True, n_periods=10, params={**PARAMS, "anchor": 0.0})
    after = pl.DataFrame(sim.fit_iter())

    # day one: nobody drives a closed street, nobody lost or doubled
    first = query(
        sim.model_,
        "SELECT (COUNT(?c) AS ?n) WHERE { ?c a ex:Commuter ; def:route ?r . ?r def:open 0e0 }",
    )
    assert first["n"][0] == 0
    assert after["sig__SUM__Area__vkt"][0] == 0
    routes_per_commuter = query(
        sim.model_,
        "SELECT ?c (COUNT(?r) AS ?n) WHERE { ?c a ex:Commuter ; def:route ?r } GROUP BY ?c",
    )
    assert routes_per_commuter.height == 120 and set(routes_per_commuter["n"]) == {1}

    # some drivers moved to the bus, most found the bypass; everyone's trip got longer
    car_before, car_after = (
        base["sig__AVG__Commuter__car"][-1],
        after["sig__AVG__Commuter__car"][-1],
    )
    assert 0.4 < car_after < car_before
    assert after["sig__AVG__Commuter__time"][-1] > base["sig__AVG__Commuter__time"][-1]
