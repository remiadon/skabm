"""Schelling on GeoSPARQL points instead of the in-house lattice.

The experiment: keep everything, change only the space.  The caller passes a
`Cell` population whose locations are irregular continuous points carried as
WKT geometry (`"POINT(x y)"`), not a regular integer grid.  The neighbourhood
becomes "within radius R" instead of the 8 Moore offsets.

Everything else is byte-for-byte the grid model: `SCHELLING_UPDATE_RULES`
(HAPPINESS / LCG_TICK / RELOCATE) is imported unchanged, `SETTLE` derives the
people unchanged, and the same `RDFSimulator` runs it.  The only differences
from `schelling_maplib_demo.py` are the init tuple (GEO_NEIGHBORHOOD replaces
GRID_NEIGHBORHOOD), the coordinate column (a WKT string instead of x/y), and
the geo state extract.  That is the entire cost of swapping spatial structures
— the abstraction seam is the `def:neighbor` relation.

maplib note: geometry is stored and queried, but maplib 0.20.19 implements no
`geof:` functions, so GEO_NEIGHBORHOOD parses the WKT and does the distance
test in arithmetic (see skabm/schelling.py).  With `geof:distance` it would be
a one-line FILTER; nothing else here would change.
"""

from time import time

import polars as pl
import polars_random as pr

from skabm.schelling import (
    SCHELLING_GEO_INIT_RULES,
    SCHELLING_GEO_PARAMS,
    SCHELLING_UPDATE_RULES,
    geo_state_extract,
)
from skabm.simulation import RDFSimulator

N_CELLS = 600  # irregular locations scattered in a continuous 20x20 extent
EXTENT = 20.0

pr.set_random_seed(0)  # one global seed governs every draw below (polars-random >=0.5)

# --- Cells: irregular points, as WKT geometry -------------------------------
# Real-world analogue: actual addresses/parcels, not a lattice.  x/y are drawn
# continuously and formatted into a WKT POINT; `rng` seeds each location's
# Park-Miller stream (SETTLE places a settler, RELOCATE shuffles vacancies).
cells = (
    pl.select(
        x=pr.uniform(0.0, EXTENT, size=N_CELLS),
        y=pr.uniform(0.0, EXTENT, size=N_CELLS),
        rng=pr.uniform(1.0, 2147483646.0, size=N_CELLS).floor(),
    )
    .with_columns(
        pl.format("cell_{}", pl.int_range(pl.len())).alias("id"),
        pl.format("POINT({} {})", pl.col("x"), pl.col("y")).alias("geometry"),
    )
    .drop("x", "y")
)  # geometry is the sole spatial column, as in a GeoSPARQL graph

# --- Simulation: same simulator, geo init rules + geo extract ----------------
sim = RDFSimulator(
    init_rules=SCHELLING_GEO_INIT_RULES,
    update_rules=SCHELLING_UPDATE_RULES,
    params=SCHELLING_GEO_PARAMS,
    n_periods=40,
    state_extract=geo_state_extract,
)

t0 = time()
for t, state in enumerate(sim.fit_iter(Cell=cells)):
    summary = state.select(
        pl.lit(t).alias("t"),
        pl.col("share_similar").mean().round(3).alias("segregation"),
        (pl.col("share_similar") < SCHELLING_GEO_PARAMS["want_similar"])
        .sum()
        .alias("unhappy"),
        pl.len().alias("people"),
    )
    print(summary)
    if summary["unhappy"][0] == 0:
        print(f"converged at t={t}")
        break
t1 = time()
print(f"{t + 1} ticks in {t1 - t0:.2f}s ({(t1 - t0) / (t + 1):.3f} s/tick)")
