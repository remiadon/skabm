"""Schelling segregation on the RDFSimulator machinery — a tiny suite.

The model lives in skabm/schelling.py: the 2D grid is a `Cell` population,
the Moore neighbourhood a CONSTRUCT, the people are *derived* by the SETTLE
rule from `$density`, and HAPPINESS / DRAW / RELOCATE advance the world.
There is no Eurostat access and no calibration layer here — the `Cell` frame
is purely spatial (x/y + id), no random column.

Randomness comes from the `pr:uniform` SPARQL UDF
(`rules.register_polars_random`, maplib >= 0.20.26); `RDFSimulator(random_seed=)`
pins it, so a whole run is reproducible to the last digit — see
``test_reproducible_under_random_seed``.

Covered: population derivation + user override, one-person-per-cell (the
rank-join matching), the core segregation result, reproducibility, and the
GeoSPARQL spatial-structure swap.
"""

import polars as pl
import polars_random as pr

from skabm.rules import _PREFIXES
from skabm.schelling import (
    SCHELLING_GEO_INIT_RULES,
    SCHELLING_GEO_PARAMS,
    SCHELLING_INIT_RULES,
    SCHELLING_PARAMS,
    SCHELLING_UPDATE_RULES,
    geo_state_extract,
    state_extract,
)
from skabm.simulation import RDFSimulator


def grid(size: int) -> pl.DataFrame:
    """A size x size Cell population: just float coordinates and an id — the
    randomness lives in the pr:uniform UDF, not in the data."""
    return (
        pl.DataFrame({"x": range(size)})
        .join(pl.DataFrame({"y": range(size)}), how="cross")
        .with_columns(
            pl.format("cell_{}_{}", pl.col("x"), pl.col("y")).alias("id"),
            pl.col("x").cast(pl.Float64),
            pl.col("y").cast(pl.Float64),
        )
    )


def test_settle_derives_population():
    # Passing only Cell, SETTLE creates the people.  At density 1.0 every cell
    # is occupied (the occupancy draw is always < 1), so the count is exactly
    # the cell count — no statistics needed.  Each settler sits on a distinct
    # cell, and both groups appear.
    params = dict(SCHELLING_PARAMS, density=1.0)
    sim = RDFSimulator(
        init_rules=SCHELLING_INIT_RULES,
        update_rules=SCHELLING_UPDATE_RULES,
        params=params,
        n_periods=0,
        state_extract=state_extract,
        random_seed=0,
    ).fit(Cell=grid(6))
    people = sim.model_.query(
        _PREFIXES
        + "SELECT ?p ?g ?c WHERE { ?p a ex:Person ; def:group ?g ; def:location ?c }"
    )
    assert people.height == 36  # 6 x 6, all cells settled
    assert people["c"].n_unique() == 36  # one settler per cell
    assert set(people["g"].to_list()) == {0.0, 1.0}  # both of n_groups=2


def test_settle_yields_to_user_population():
    # The SETTLE gate (FILTER NOT EXISTS a ex:Person) mirrors FIRM_OWNERSHIP:
    # a caller-supplied Person population turns settlement off entirely.  Such
    # persons need no rng column — DRAW assigns def:draw each tick.
    persons = pl.DataFrame(
        {
            "id": ["alice", "bob"],
            "group": [0.0, 1.0],
            "location": ["cell_0_0", "cell_1_1"],
        }
    )
    sim = RDFSimulator(
        init_rules=SCHELLING_INIT_RULES,
        update_rules=SCHELLING_UPDATE_RULES,
        params=SCHELLING_PARAMS,
        n_periods=1,
        state_extract=state_extract,
        random_seed=0,
    ).fit(Cell=grid(6), Person=persons)
    n = sim.model_.query(
        _PREFIXES + "SELECT (COUNT(?p) AS ?n) WHERE { ?p a ex:Person }"
    )["n"][0]
    assert n == 2  # SETTLE produced nobody


def test_one_person_per_cell_invariant():
    # RELOCATE's rank-join must never place two movers on the same cell.  After
    # several ticks, occupied cells still equal people — matching is one-to-one.
    sim = RDFSimulator(
        init_rules=SCHELLING_INIT_RULES,
        update_rules=SCHELLING_UPDATE_RULES,
        params=SCHELLING_PARAMS,
        n_periods=8,
        state_extract=state_extract,
        random_seed=0,
    ).fit(Cell=grid(12))
    occ = sim.model_.query(
        _PREFIXES + "SELECT ?p ?c WHERE { ?p a ex:Person ; def:location ?c }"
    )
    assert occ["c"].n_unique() == occ.height


def test_segregation_rises_and_converges():
    # The result Schelling is famous for: mild same-group preference drives the
    # share of similar neighbours up until nobody is unhappy.
    want = SCHELLING_PARAMS["want_similar"]
    sim = RDFSimulator(
        init_rules=SCHELLING_INIT_RULES,
        update_rules=SCHELLING_UPDATE_RULES,
        params=SCHELLING_PARAMS,
        n_periods=30,
        state_extract=state_extract,
        random_seed=0,
    )
    seg, unhappy = [], []
    for s in sim.fit_iter(Cell=grid(12)):
        seg.append(s["share_similar"].mean())
        unhappy.append((s["share_similar"] < want).sum())
    assert seg[-1] > seg[0] + 0.2  # substantial rise from the mixed start
    assert min(unhappy) == 0  # reaches a configuration with nobody unhappy


def test_reproducible_under_random_seed():
    # The pr:uniform UDF draws the reproducible size=N Series form, so a fixed
    # random_seed pins the whole trajectory; a different seed changes it.
    def seg_path(seed: int) -> list[float]:
        sim = RDFSimulator(
            init_rules=SCHELLING_INIT_RULES,
            update_rules=SCHELLING_UPDATE_RULES,
            params=SCHELLING_PARAMS,
            n_periods=10,
            state_extract=state_extract,
            random_seed=seed,
        )
        return [
            round(s["share_similar"].mean(), 6) for s in sim.fit_iter(Cell=grid(10))
        ]

    assert seg_path(0) == seg_path(0)
    assert seg_path(0) != seg_path(1)


def test_geo_variant_plugs_in():
    # Swapping the lattice for continuous GeoSPARQL points changes only the
    # neighbourhood rule (GEO_NEIGHBORHOOD) and the coordinate column: SETTLE,
    # the update rules and RDFSimulator are reused verbatim, and segregation
    # still rises.  (maplib has no geof: functions, so the distance test is
    # hand-rolled from the WKT string — see skabm/schelling.py.)
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
    sim = RDFSimulator(
        init_rules=SCHELLING_GEO_INIT_RULES,
        update_rules=SCHELLING_UPDATE_RULES,
        params=SCHELLING_GEO_PARAMS,
        n_periods=15,
        state_extract=geo_state_extract,
        random_seed=0,
    )
    seg = [s["share_similar"].mean() for s in sim.fit_iter(Cell=cells)]
    neighbours = sim.model_.query(
        _PREFIXES + "SELECT (COUNT(*) AS ?n) WHERE { ?c def:neighbor ?c2 }"
    )["n"][0]
    assert neighbours > 0  # GEO_NEIGHBORHOOD wired a topology from geometry
    assert seg[-1] > seg[0]  # and the same dynamics segregate on it
