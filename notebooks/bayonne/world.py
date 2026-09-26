"""The Bayonne world: OSM roads, 2022 census commuting flows, and the populations
``skabm.behaviour.traffic`` expects.

Dataset-specific, like the CSV handling in ``labour_automation.ipynb``: none of it
belongs in the library and the test suite never touches it.  Every download is cached
under ``notebooks/data/bayonne``; delete a file there to refetch it.

Sources, all open and queried through APIs:

- roads: OpenStreetMap through Overpass (ODbL);
- communes, populations, outlines: geo.api.gouv.fr;
- home-to-work flows by commune pair and mode, census 2022: INSEE, as republished by the
  "Tableau de bord des mobilités durables" on data.gouv.fr (tabular API).
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path

import httpx
import numpy as np
import polars as pl
from maplib import Model
from matplotlib.path import Path as Polygon
from sklearn.cluster import KMeans

from skabm.behaviour.traffic import (
    MODES,
    TRAFFIC_UDFS,
    RULES,
    area_membership,
    pedestrianize,
    routes,
)
from skabm import template
from skabm.simulation import RDFSimulator
from skabm.sparql import _PREFIXES

CACHE = Path(__file__).resolve().parent.parent / "data" / "bayonne"
# Every street from Biarritz to Tarnos, the coast to the A63/A64 junction; and the
# main roads of the whole commuting basin, Hendaye to Dax, the coast to Saint-Palais,
# so the inland arrives by the A64, the D932 and the D936 rather than cross-country.
SOUTH, WEST, NORTH, EAST = 43.425, -1.600, 43.565, -1.385
REGION = (43.15, -1.80, 43.80, -0.95)
FARTHEST = 80_000.0  # metres, crow-fly: a census "usual workplace" farther is rarely daily
LON0, LAT0 = -1.475, 43.49  # local metric projection origin
FLOWS = "39df0b9c-dd76-44ce-8a15-8c00d8a211c1"  # data.gouv.fr resource: flows by mode

# Drawn on OSM, not taken from it (OSM has the quarters as points only).  Grand
# Bayonne: west of the Nive, south of the Adour, north of the ramparts' road, east of
# Allées Paulmy, which stays outside it.
# The Nive bridges cross the east edge, so closing the area closes them.
AREAS = {
    # east edge: mid-river through the Mayou, Marengo, Pannecau and Génie bridges
    "Grand Bayonne": "POLYGON((-1.4781 43.4945, -1.4762 43.4941, -1.4742 43.4936, "
    "-1.4741 43.4923, -1.4744 43.4910, -1.4746 43.4897, -1.4749 43.4879, "
    "-1.4762 43.4876, -1.4801 43.4884, -1.4798 43.4914, -1.4791 43.4931, "
    "-1.4781 43.4945))",
    "Petit Bayonne": "POLYGON((-1.4742 43.4936, -1.4728 43.4935, -1.4700 43.4922, "
    "-1.4686 43.4910, -1.4690 43.4878, -1.4722 43.4868, -1.4749 43.4879, "
    "-1.4746 43.4897, -1.4744 43.4910, -1.4741 43.4923, -1.4742 43.4936))",
}

_MAJOR = "motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link"
_HIGHWAYS = _MAJOR + "|tertiary|tertiary_link|unclassified|residential|living_street|busway|service|pedestrian"
# km/h and veh/h per lane when OSM says nothing (French defaults; MATSim's OSM reader
# for capacities).
_DEFAULTS = {
    "motorway": (110, 2000), "trunk": (90, 1800), "primary": (50, 1500),
    "secondary": (50, 1000), "tertiary": (50, 800), "unclassified": (40, 600),
    "residential": (30, 500), "living_street": (20, 300), "pedestrian": (10, 300),
    # only bus-designated service roads survive _access: bus lanes, and a lane if reopened
    "busway": (30, 1000), "service": (30, 1000),
}
_MAXSPEED = {"FR:urban": 50, "FR:rural": 80, "FR:zone30": 30, "FR:motorway": 130, "walk": 6}
# Census "mode_transport" to model mode; working from home is no trip.
_MODE = {"Voiture": "car", "Deux-roues motorisé": "car", "Transports en commun": "bus",
         "Vélo": "bike", "Marche": "walk"}


def _cached(name: str, fetch) -> object:
    path = CACHE / name
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fetch()))
    return json.loads(path.read_text())


def _get(url: str, **params) -> object:
    response = httpx.get(url, params=params, timeout=180,
                         headers={"User-Agent": "skabm-research (github.com/remiadon/skabm)"})
    response.raise_for_status()
    return response.json()


def metres(lon, lat):
    """Local equirectangular projection around Bayonne — metres east and north."""
    return ((lon - LON0) * 111_320 * math.cos(math.radians(LAT0)), (lat - LAT0) * 110_574)


# ---------------------------------------------------------------------------
# Roads
# ---------------------------------------------------------------------------


def osm() -> dict:
    query = f"""[out:json][timeout:170];
    (way["highway"~"^({_HIGHWAYS})$"]({SOUTH},{WEST},{NORTH},{EAST});
     way["highway"~"^({_MAJOR})$"]{REGION};);
    out body; >; out skel qt;"""
    return _cached("osm.json", lambda: _get("https://overpass-api.de/api/interpreter", data=query))


def _speed(tags: dict, kind: str) -> float:
    raw = tags.get("maxspeed", "")
    if raw in _MAXSPEED:
        return _MAXSPEED[raw]
    number = raw.split()[0] if raw else ""
    if number.isdigit():
        return float(number) * (1.609 if "mph" in raw else 1)
    return _DEFAULTS[kind][0]


def _access(tags: dict, kind: str) -> tuple[bool, bool, bool]:
    """(cars may use it, buses may, it is a bus lane or bus-only)."""
    motor = tags.get("motor_vehicle", tags.get("motorcar", tags.get("vehicle")))
    closed = kind in ("busway", "pedestrian") or tags.get("access") in ("no", "private")
    car = motor in ("yes", "destination", "designated") if (closed or motor) else True
    if kind == "service" and "designated" not in (tags.get("bus"), tags.get("psv")):
        return False, False, False  # driveways and car parks are not the network
    bus = car or tags.get("bus") in ("yes", "designated") or tags.get("psv") in ("yes", "designated")
    lane = "designated" in tags.get("bus:lanes", "") or tags.get("busway") in ("lane", "opposite_lane")
    return car, bus, (bus and not car) or lane


def network() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Directed links between intersections, and the nodes they join (metres)."""
    data = osm()["elements"]
    where = {e["id"]: (e["lon"], e["lat"]) for e in data if e["type"] == "node"}
    ways = [e for e in data if e["type"] == "way"]
    seen: dict = {}
    for way in ways:
        for i, node in enumerate(way["nodes"]):
            seen[node] = seen.get(node, 0) + (2 if i in (0, len(way["nodes"]) - 1) else 1)
    rows = []
    for way in ways:
        tags = way["tags"]
        kind = tags["highway"].removesuffix("_link")
        car, bus, busway = _access(tags, kind)
        if not bus:
            continue
        oneway = tags.get("oneway") in ("yes", "true", "1") or (
            kind == "motorway" or tags.get("junction") == "roundabout"
        ) and tags.get("oneway") != "no"
        reverse = tags.get("oneway") == "-1"
        lanes = float(tags.get("lanes", "0").split(";")[0] or 0) or (2 if kind == "motorway" else 1)
        lanes = lanes if oneway or reverse else max(1.0, lanes / 2)
        speed, per_lane = _speed(tags, kind), _DEFAULTS[kind][1]
        cut = [0] + [i for i, n in enumerate(way["nodes"][1:-1], 1) if seen[n] > 1] + [len(way["nodes"]) - 1]
        for a, b in zip(cut, cut[1:]):
            chain = way["nodes"][a : b + 1]
            xy = [metres(*where[n]) for n in chain]
            length = sum(math.dist(p, q) for p, q in zip(xy, xy[1:]))
            line = ", ".join(f"{where[n][0]:.6f} {where[n][1]:.6f}" for n in chain)
            for src, dst, points in (
                [] if reverse else [(chain[0], chain[-1], line)]
            ) + ([] if oneway else [(chain[-1], chain[0], ", ".join(reversed(line.split(", "))))]):
                rows.append((f"w{way['id']}_{a}_{src}", str(src), str(dst), max(length, 1.0),
                             max(length, 1.0) / (speed / 3.6), lanes * per_lane, float(car),
                             float(busway), tags.get("name"), kind, f"LINESTRING({points})"))
    links = pl.DataFrame(
        rows, orient="row",
        schema=["id", "src", "dst", "length", "t0", "capacity", "car", "busway", "name", "kind", "geometry"],
    ).unique("id", maintain_order=True)
    used = pl.concat([links["src"], links["dst"]]).unique(maintain_order=True)
    nodes = pl.DataFrame(
        [(str(n), *where[n], *metres(*where[n])) for n in map(int, used)],
        schema=["id", "lon", "lat", "x", "y"], orient="row",
    )
    return links, nodes


# ---------------------------------------------------------------------------
# People: communes, flows, zones
# ---------------------------------------------------------------------------


def communes() -> pl.DataFrame:
    """Every commune of the Pyrénées-Atlantiques and the Landes: code, name, population, centre."""
    found = []
    for department in ("64", "40"):
        found += _cached(
            f"communes_{department}.json",
            lambda d=department: _get("https://geo.api.gouv.fr/communes", codeDepartement=d,
                                      fields="code,nom,population,centre"),
        )
    frame = pl.DataFrame(
        [(c["code"], c["nom"], float(c.get("population") or 0), *c["centre"]["coordinates"]) for c in found],
        schema=["code", "name", "population", "lon", "lat"], orient="row",
    )
    x, y = metres(frame["lon"], frame["lat"])
    inside = frame["lon"].is_between(WEST, EAST) & frame["lat"].is_between(SOUTH, NORTH)
    return frame.with_columns(x=x, y=y, internal=inside)


def outline(code: str) -> np.ndarray:
    shape = _cached(f"contour_{code}.json", lambda: _get(
        f"https://geo.api.gouv.fr/communes/{code}", fields="contour", format="json"))["contour"]
    parts = [shape["coordinates"]] if shape["type"] == "Polygon" else shape["coordinates"]
    return np.array(max((part[0] for part in parts), key=len))  # outer ring of the largest part


def flows(internal: list[str]) -> pl.DataFrame:
    """Workers by (home, work, mode), every pair with an internal end."""
    def fetch():
        rows = []
        url = f"https://tabular-api.data.gouv.fr/api/resources/{FLOWS}/data/"
        for side in ("code_com_1__in", "code_com_2__in"):
            page, total = 1, 1
            while (page - 1) * 200 < total:  # the API offers a `next` past the last page
                body = _get(url, **{side: ",".join(internal), "page_size": 200, "page": page})
                rows += body["data"]
                page, total = page + 1, body["meta"]["total"]
        return rows
    raw = pl.DataFrame(_cached("flows.json", fetch))
    return (
        raw.unique("__id")
        .filter(pl.col("mode_transport").is_in(list(_MODE)))
        .select(home="code_com_1", work="code_com_2",
                mode=pl.col("mode_transport").replace_strict(_MODE), workers="valeur")
        .group_by("home", "work", "mode").agg(pl.col("workers").sum())
        .sort("home", "work", "mode")  # the sample is drawn row by row: fix the rows' order
    )


def zones(towns: pl.DataFrame, nodes: pl.DataFrame, per_zone: float = 1500.0) -> pl.DataFrame:
    """Sub-zones of the internal communes; every other commune is one zone at its centre.

    An internal commune is cut into ~one zone per ``per_zone`` inhabitants by k-means
    over its street nodes (few inhabitants per zone keeps any one street from
    loading a whole neighbourhood's trips), each zone weighted by its node count — street density
    standing in for where people live and work — and reached on foot.  An external
    commune is driven from, at ``DRIVE``, to its nearest road of the regional network.

    ponytail: residents and jobs share one weight; POIs or SIRENE establishments
    would separate them when destination choice starts to matter.
    """
    points = nodes.select("x", "y").to_numpy()
    parts, taken = [], np.zeros(len(points), dtype=bool)
    for code, population in towns.filter("internal").select("code", "population").iter_rows():
        ring = np.array([metres(lon, lat) for lon, lat in outline(code)])
        mine = Polygon(ring).contains_points(points) & ~taken
        taken |= mine
        if mine.sum() == 0:
            continue
        k = int(min(max(round(population / per_zone), 1), 40, mine.sum()))
        fit = KMeans(k, n_init=4, random_state=0).fit(points[mine])
        weight = np.bincount(fit.labels_, minlength=k)
        parts.append(pl.DataFrame({
            "id": [f"z{code}_{i}" for i in range(k)], "code": code,
            "x": fit.cluster_centers_[:, 0], "y": fit.cluster_centers_[:, 1],
            "weight": weight / weight.sum(), "speed": WALK,
        }))
    parts.append(towns.filter(~pl.col("internal")).select(
        id=pl.format("z{}", "code"), code="code", x="x", y="y", weight=pl.lit(1.0), speed=pl.lit(DRIVE),
    ))
    return pl.concat(parts)


def commuters(od: pl.DataFrame, zone: pl.DataFrame, sample: float, seed: int = 0) -> pl.DataFrame:
    """A *sample* of the workers as agents, each placed in a zone of its home and work communes.

    Counts are rounded stochastically (a flow of 3.4 workers at 10% is one agent with
    probability 0.34), so small flows survive in expectation; the zone within a
    commune is drawn by weight.
    """
    rng = np.random.default_rng(seed)
    expected = od["workers"].to_numpy() * sample
    count = np.floor(expected + rng.random(len(expected))).astype(int)
    agents = od.with_columns(n=pl.Series(count)).filter(pl.col("n") > 0).select(
        "home", "work", "mode", pl.int_ranges("n").alias("k")
    ).explode("k")

    def place(codes: pl.Series) -> list[str]:
        out = []
        members = {c: g for (c,), g in zone.group_by("code")}
        for code in codes:
            options = members.get(code)
            out.append(None if options is None else rng.choice(options["id"].to_numpy(), p=options["weight"].to_numpy()))
        return out

    at = zone.select("id", "x", "y")
    return (
        agents.with_columns(origin=pl.Series(place(agents["home"])),
                            destination=pl.Series(place(agents["work"])))
        .drop_nulls()
        .join(at.rename({"id": "origin", "x": "xo", "y": "yo"}), on="origin")
        .join(at.rename({"id": "destination", "x": "xd", "y": "yd"}), on="destination")
        .filter(((pl.col("xo") - pl.col("xd")) ** 2 + (pl.col("yo") - pl.col("yd")) ** 2).sqrt() < FARTHEST)
        .select("home", "work", "mode", "origin", "destination",
                od=pl.format("{}_{}", "origin", "destination"), weight=pl.lit(1 / sample),
                id=pl.format("c{}", pl.int_range(pl.len())))
    )


# ---------------------------------------------------------------------------
# The world
# ---------------------------------------------------------------------------

WALK, BIKE, DRIVE = 1.3, 4.2, 11.0  # m/s: walking, cycling, driving local roads
REACH = {"walk": 4_000.0, "bike": 15_000.0}  # metres crow-fly between zones: beyond, a census artefact
BUS_WAIT = 480.0  # seconds: half a typical headway


def choice_set(links, nodes, zone, ods, modes: pl.DataFrame, cost: str = "t0"):
    """Routes and via for every OD x mode in *modes* (``od``, ``mode``); car on the links
    that admit cars, bus on the links that admit buses, bike and walk crow-fly."""
    found, via = [], []
    for mode, allowed in (("car", pl.col("car") > 0), ("bus", pl.lit(True))):
        wanted = ods.join(modes.filter(pl.col("mode") == mode), on="od", how="semi")
        if wanted.height:
            r, v = routes(links.filter(allowed), nodes, zone, wanted, MODES[mode], cost)
            found.append(r.with_columns(extra=pl.col("extra") + (BUS_WAIT if mode == "bus" else 0.0)))
            via.append(v)
    ends = ods.join(zone.select(origin="id", xo="x", yo="y"), on="origin").join(
        zone.select(destination="id", xd="x", yd="y"), on="destination")
    crow = ((pl.col("xo") - pl.col("xd")) ** 2 + (pl.col("yo") - pl.col("yd")) ** 2).sqrt().clip(500) * 1.3
    for mode, speed in (("bike", BIKE), ("walk", WALK)):
        wanted = ends.join(modes.filter(pl.col("mode") == mode), on="od", how="semi")
        found.append(wanted.select(
            id=pl.format("{}_m{}", "od", pl.lit(int(MODES[mode]))), od="od",
            rank=pl.lit(float(MODES[mode])), extra=crow / speed, time=crow / speed,
            mode=pl.lit(MODES[mode]),
        ))
    routed = pl.concat(found, how="vertical_relaxed").select(
        "id", "od", "rank", "mode", "extra", "time",
        option=pl.format("{}_{}", "od", pl.col("mode").cast(pl.Int8)))
    return routed, pl.concat(via)


def frames(sample: float = 0.1, seed: int = 0) -> dict:
    """Everything the world is made of, as polars frames — cacheable, unlike a Model."""
    links, nodes = network()
    towns = communes()
    od = flows(towns.filter("internal")["code"].to_list())
    zone = zones(towns, nodes)
    people = commuters(od, zone, sample, seed)
    ods = people.select("od", "origin", "destination", "home", "work").unique(maintain_order=True)
    # the pivot's base is the census share of the commune pair, not of the few agents
    # a zone pair happens to draw: every mode the pair uses is open to every agent on it —
    # within reach.  The census has Capbreton residents walking to Bayonne; the pivot
    # would keep that 25 km walk open forever, so out-of-reach modes go, and so do the
    # agents drawn onto them.
    at = zone.select("id", "x", "y")
    reach = ods.join(at.rename({"id": "origin", "x": "xo", "y": "yo"}), on="origin").join(
        at.rename({"id": "destination", "x": "xd", "y": "yd"}), on="destination"
    ).select("od", "home", "work", metres=((pl.col("xo") - pl.col("xd")) ** 2 + (pl.col("yo") - pl.col("yd")) ** 2).sqrt())
    census = (
        reach.join(od, on=["home", "work"])
        .filter(pl.col("metres") <= pl.col("mode").replace_strict(REACH, default=float("inf")))
        .select("od", "mode", share0=pl.col("workers") / pl.col("workers").sum().over("od"))
    )
    people = people.join(census.select("od", "mode"), on=["od", "mode"], how="semi")
    found, via = choice_set(links, nodes, zone, ods, census.select("od", "mode"))
    # everyone starts on a route of the mode the census gives them
    first = found.sort("rank").unique(["od", "mode"], keep="first")
    code = pl.col("mode").replace_strict(MODES)
    agents = people.with_columns(mode=code).join(first.select("od", "mode", route="id"), on=["od", "mode"])
    options = census.with_columns(mode=code).with_columns(
        id=pl.format("{}_{}", "od", pl.col("mode").cast(pl.Int8)))
    return {"links": links, "nodes": nodes, "zones": zone, "ods": ods, "census": census,
            "commuters": agents, "options": options, "routes": found, "via": via,
            "communes": towns, "flows": od}


def world(parts: dict) -> Model:
    """A fresh maplib Model of *parts* — a fit advances the one it is given."""
    model = Model()  # every frame in id order: the draws follow the rows, so a seed is a seed
    model.map(template.link, parts["links"].drop("kind").with_columns(time=pl.col("t0")).sort("id").with_iri("src", "dst"))
    areas = pl.DataFrame({"id": ["grand", "petit"], "name": list(AREAS), "geometry": list(AREAS.values())})
    model.map(template.area, areas.with_iri())
    links = parts["links"].sort("id")
    model.map(template.links("area"), area_membership(links, areas).sort("id", "area").with_iri("area"))
    model.map(template.option, parts["options"].sort("id").with_iri())
    model.map(template.route, parts["routes"].sort("id").with_iri("option"))
    model.map(template.via, parts["via"].sort("id", "via").with_iri("via"))
    model.map(template.commuter, parts["commuters"].select("id", "od", "route", "weight").sort("id").with_iri("route"))
    return model


def reroute(model: Model, parts: dict) -> int:
    """Car routes against the network as the graph has it now, on today's times.

    The column-generation step: after an intervention (or congestion) the choice set
    gains each OD's current shortest path, mapped straight into the live model — a
    path already known maps to the same route and adds nothing.  Returns how many
    routes the model has now.
    """
    state = model.query(
        _PREFIXES + "SELECT ?l ?car ?time WHERE { ?l a ex:Link ; def:car ?car ; def:time ?time }"
    ).select(id=pl.col("l").str.slice(len("<http://example.net/skabm#")).str.strip_suffix(">"),
             car="car", time="time")
    links = parts["links"].drop("car").join(state, on="id")
    cars = parts["census"].filter(pl.col("mode") == "car").select("od", "mode")
    found, via = choice_set(links, parts["nodes"], parts["zones"], parts["ods"], cars, cost="time")
    model.map(template.route, found.sort("id").with_iri("option"))
    model.map(template.via, via.sort("id", "via").with_iri("via"))
    return model.query(_PREFIXES + "SELECT (COUNT(?r) AS ?n) WHERE { ?r a ex:Route }")["n"][0]


# ---------------------------------------------------------------------------
# Scenarios and the protocol
# ---------------------------------------------------------------------------

# Areas closed to cars, streets (re)opened to them.
# What a scenario closes to cars: (areas, streets).  The corridor is the through route
# across the old town, Saint-Esprit to Allées Marines; Pont Mayou is the Nive crossing
# between the two places the corridor names, so it goes with them.
CORRIDOR = ("Place du Réduit", "Pont Mayou", "Quai Amiral Lespes", "Place de la Liberté",
            "Rue Bernède", "Place Général de Gaulle", "Avenue Léon Bonnat")
SCENARIOS = {
    "Today": ((), ()),
    "Old-town axis closed to cars": ((), CORRIDOR),
    "Fêtes perimeter: Grand + Petit Bayonne car-free": (("Grand Bayonne", "Petit Bayonne"), ()),
}
# Commuters on the road relative to a school-term weekday, i.e. `demand`.  Assumptions
# standing in for counts, not measurements: the knob a seasonal count calibrates.
PERIODS = {
    "School-term weekday": 1.0,
    "School holidays": 0.85,
    "July (fêtes de Bayonne)": 0.85,
    "August": 0.7,
}
PARAMS = {"peak_factor": 0.7}
# Cars on the road in each half hour, as a share of the 8 o'clock peak the rules settle.
# An assumption, and the first thing a permanent counter's own hourly profile replaces.
MORNING = {"06:00": 0.22, "06:30": 0.35, "07:00": 0.55, "07:30": 0.78, "08:00": 1.0}


WARMUP, DAYS, EVERY = 15, 15, 5  # days settling the base, days lived after it, days between re-routes


def simulate(parts: dict, scenario: str, period: str, params: dict | None = None,
             seed: int = 0, warmup: int = WARMUP, days: int = DAYS, every: int = EVERY):
    """Warm up on the census, then live *days* under *scenario* and *period*.

    Yields ``(phase, frame, sim)`` per day, *frame* every agent after it.  The warm-up anchors every OD's mode
    shares on the census at school-term demand while routes equilibrate, re-routing
    on congested times every *every* days; then the anchor is dropped, demand
    becomes the period's, the scenario's streets close and open, and the choice set
    is re-routed once more — for "Today" too, so a scenario differs from it by the
    policy alone.  The draws line up across scenarios (one per commuter per day, one
    seed), so two runs are paired: common random numbers.
    """
    params = {**PARAMS, **(params or {})}
    areas, streets = SCENARIOS[scenario]
    model = world(parts)
    sim = RDFSimulator(rules=RULES,
                       params={**params, "anchor": 1.0, "demand": 1.0}, udfs=TRAFFIC_UDFS,
                       n_periods=every, random_seed=seed)
    for segment in range(warmup // every):
        for frame in sim.fit_iter(model if segment == 0 else None):
            yield "warm-up", frame, sim
        reroute(sim.model_, parts)
        sim.set_params(warm_start=True, random_seed=None)  # one draw stream, not restarts
    unknown = set(streets) - set(parts["links"]["name"].drop_nulls())
    if unknown:
        raise ValueError(f"no link is named {sorted(unknown)}: nothing would close")
    if areas or streets:
        sim.model_.update(pedestrianize(areas, streets))
    reroute(sim.model_, parts)
    sim.set_params(n_periods=days, params={**params, "anchor": 0.0, "demand": PERIODS[period]})
    for frame in sim.fit_iter():
        yield scenario, frame, sim


def run_once(scenario: str, period: str, sample: float, knobs: tuple) -> dict:
    """A whole run, in and out as plain frames: every settled peak, and its link flows.

    Takes only picklable arguments and returns only picklable frames, because the app
    calls it in a subprocess — maplib models made in one of Streamlit's script threads
    and dropped in the next segfault the server (exit 139), and nothing of the graph
    survives this call.
    """
    parts = frames(sample=sample)
    rows, flows = [], {}
    for phase, frame, sim in simulate(parts, scenario, period, dict(knobs)):
        t = frame["t"][0]
        commuters = frame.filter(pl.col("class") == "Commuter")
        day = commuters.select(pl.col("time", "car", "bus").mean()).row(0, named=True)
        rows.append({"day": t, "phase": phase, **day, "vkt": area_state(sim)["Grand Bayonne"]})
        flows["before" if phase == "warm-up" else t] = link_state(sim).select("id", "flow")
    return {"days": pl.DataFrame(rows), "flows": flows}


def paths(links: pl.DataFrame) -> pl.Series:
    """Each link's WKT as a list of [lon, lat] — what a map layer draws."""
    return links["geometry"].str.extract(r"\((.*)\)").str.split(", ").list.eval(
        pl.element().str.split(" ").cast(pl.List(pl.Float64)))


def area_state(sim) -> dict:
    """Vehicle-km per peak hour inside each Area, by name."""
    rows = sim.model_.query(_PREFIXES + "SELECT ?name ?vkt WHERE { ?a a ex:Area ; def:name ?name ; def:vkt ?vkt }")
    return dict(rows.iter_rows())


def link_state(sim) -> pl.DataFrame:
    """Every link's flow, time and access now, joined back to its name and geometry."""
    return sim.model_.query(
        _PREFIXES + "SELECT ?l ?flow ?time ?car WHERE { ?l a ex:Link ; def:flow ?flow ; def:time ?time ; def:car ?car }"
    ).select(id=pl.col("l").str.slice(len("<http://example.net/skabm#")).str.strip_suffix(">"),
             flow="flow", time="time", car="car")


def at_hour(flows: pl.DataFrame, links: pl.DataFrame, share: float, params: dict | None = None) -> pl.DataFrame:
    """The settled peak scaled down to one half hour, and the delay that follows from it.

    A replay, not a simulation: the rules settle the peak, and the hours before it are
    that loading scaled by ``MORNING`` — so the flows fall linearly and the delay does
    not, which is the jam forming.  No queue carries over from one half hour to the next.

    ponytail: the BPR of ``traffic.link_time`` written once more, in polars, because a
    replay does not re-run the rules.  Keep the two in step.
    """
    knobs = {**PARAMS, **(params or {})}
    return (
        flows.join(links.select("id", "capacity", "t0"), on="id")
        .with_columns(flow=pl.col("flow") * share)
        .with_columns(ratio=pl.col("flow") / pl.col("capacity"))
        .with_columns(time=pl.col("t0") * (1 + knobs["bpr_alpha"] * pl.col("ratio") ** knobs["bpr_beta"]))
    )


if __name__ == "__main__" and len(sys.argv) > 1:
    # `python world.py <scenario> <period> <sample> <knobs json> <out.pkl>` — one run, as
    # its own process.  That is how the app asks for one: a maplib model made in one of
    # Streamlit's script threads and dropped in the next takes the server down (exit 139).
    scenario, period, sample, knobs, out = sys.argv[1:6]
    with open(out, "wb") as handle:
        pickle.dump(run_once(scenario, period, float(sample), tuple(json.loads(knobs).items())), handle)

elif __name__ == "__main__":  # python world.py — the replay must agree with the rules it replays
    parts = frames(sample=0.05)
    for _, _, sim in simulate(parts, "Today", "School-term weekday", warmup=5, days=0, every=5):
        pass
    graph = link_state(sim)
    replay = at_hour(graph.select("id", "flow"), parts["links"], 1.0)
    gap = graph.join(replay.select("id", replayed="time"), on="id").select(
        (pl.col("time") - pl.col("replayed")).abs().max()
    ).item()
    assert gap < 1e-6, f"at_hour drifted from traffic.link_time by {gap} s"
    print(f"at_hour matches the rule's BPR on {graph.height:,} links (worst gap {gap:.2e} s)")
