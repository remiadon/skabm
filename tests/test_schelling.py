"""Schelling segregation on the RDFSimulator machinery — a tiny suite.

The model lives in skabm/behaviour/schelling.py: the 2D grid is a `Cell` population,
the Moore neighbourhood and the Person population are data (`grid_neighbors`,
`settle`), and `RULES` advances the world.  There is no Eurostat access and no
calibration layer here.

Randomness comes from `settle(seed=)` and the `pr:uniform` SPARQL UDF, which
`RDFSimulator(random_seed=)` pins, so a whole run is reproducible to the last digit.

Covered: settlement, one-person-per-cell (the rank-matched move), the core
segregation result, reproducibility, and the swap to points in continuous space.
"""

import polars as pl
import polars_random as pr
from maplib import Model
from worlds import world

from skabm.behaviour.schelling import (
    PARAMETERS,
    RULES,
    geo_neighbors,
    geo_state_extract,
    grid_neighbors,
    settle,
    state_extract,
    want_similar,
)
from skabm.ottr import links_template
from skabm.simulation import RDFSimulator
from skabm.sparql import _PREFIXES


def grid(size: int) -> pl.DataFrame:
    """A size x size Cell population: float coordinates and an id."""
    return (
        pl.DataFrame({"x": range(size)})
        .join(pl.DataFrame({"y": range(size)}), how="cross")
        .with_columns(
            pl.format("cell_{}_{}", pl.col("x"), pl.col("y")).alias("id"),
            pl.col("x").cast(pl.Float64),
            pl.col("y").cast(pl.Float64),
        )
    )


def town(cells: pl.DataFrame, neighbors: pl.DataFrame, seed: int = 0) -> Model:
    """The cells, their neighbour links and the people settled on them."""
    model = world(links=("location",), Cell=cells, Person=settle(cells, seed=seed))
    model.map(links_template("neighbor"), neighbors.with_iri("neighbor"))
    return model


def simulator(n_periods: int, seed: int = 0, extract=state_extract) -> RDFSimulator:
    return RDFSimulator(
        rules=RULES,
        n_periods=n_periods,
        state_extract=extract,
        random_seed=seed,
    )


def test_settle_derives_population():
    # At density 1.0 every cell is occupied (the draw is always < 1), so the
    # count is exactly the cell count, one settler per cell, and both groups appear.
    people = settle(grid(6), density=1.0)
    assert people.height == 36
    assert people["location"].n_unique() == 36
    assert set(people["group"]) == {0.0, 1.0}
    # and the Moore neighbourhood of a bounded 6 x 6 grid has 4*3 + 16*5 + 16*8 links
    assert grid_neighbors(grid(6)).height == 4 * 3 + 16 * 5 + 16 * 8


def test_one_person_per_cell_invariant():
    # RELOCATE's rank-join must never place two movers on the same cell.  After
    # several ticks, occupied cells still equal people — matching is one-to-one.
    sim = simulator(8).fit(town(grid(12), grid_neighbors(grid(12))))
    occ = sim.model_.query(
        _PREFIXES + "SELECT ?p ?c WHERE { ?p a ex:Person ; def:location ?c }"
    )
    assert occ["c"].n_unique() == occ.height


def test_segregation_rises_and_converges():
    # The result Schelling is famous for: mild same-group preference drives the
    # share of similar neighbours up until nobody is unhappy.
    want = PARAMETERS[want_similar]  # 3 of 8 Moore neighbours, Schelling (1971)
    sim = simulator(30)
    seg, unhappy = [], []
    for row in sim.fit_iter(town(grid(12), grid_neighbors(grid(12)))):
        seg.append(row["sig__AVG__Person__share_similar"])
        unhappy.append((sim.extract()["share_similar"] < want).sum())
    assert seg[-1] > seg[0] + 0.2  # substantial rise from the mixed start
    assert min(unhappy) == 0  # reaches a configuration with nobody unhappy


def test_reproducible_under_random_seed():
    # settle(seed=) and random_seed pin the whole trajectory; another seed changes it.
    def seg_path(seed: int) -> list[float]:
        world = town(grid(10), grid_neighbors(grid(10)), seed)
        return [
            round(row["sig__AVG__Person__share_similar"], 6)
            for row in simulator(10, seed).fit_iter(world)
        ]

    assert seg_path(0) == seg_path(0)
    assert seg_path(0) != seg_path(1)


def test_geo_variant_plugs_in():
    # Swapping the lattice for points in continuous space changes only the
    # neighbour links (geo_neighbors) and the coordinate column: settle, the update
    # rules and RDFSimulator are reused verbatim, and segregation still rises.
    pr.set_random_seed(0)  # reproducible point positions
    cells = (
        pl.select(
            x=pr.uniform(0.0, 15.0, size=300),
            y=pr.uniform(0.0, 15.0, size=300),
        )
        .with_columns(
            pl.format("cell_{}", pl.int_range(pl.len())).alias("id"),
            pl.format("POINT({} {})", pl.col("x"), pl.col("y")).alias("geometry"),
        )
        .drop("x", "y")
    )
    sim = simulator(15, extract=geo_state_extract)
    seg = [
        row["sig__AVG__Person__share_similar"]
        for row in sim.fit_iter(town(cells, geo_neighbors(cells)))
    ]
    neighbours = sim.model_.query(
        _PREFIXES + "SELECT (COUNT(*) AS ?n) WHERE { ?c def:neighbor ?c2 }"
    )["n"][0]
    assert neighbours > 0  # geo_neighbors wired a topology from geometry
    assert seg[-1] > seg[0]  # and the same dynamics segregate on it
