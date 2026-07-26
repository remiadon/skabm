"""Schelling segregation on the Poledna machinery, unchanged.

The point of this demo is what it does *not* contain: no space module, no
scheduler, no agent classes, no visitation loop — and now no agent
population either.  The caller declares only the **grid** (a `Cell`
population); the people are derived in SPARQL by `schelling.SETTLE`, which
occupies each cell with probability `$density` and assigns a random group,
exactly as Poledna's `FIRM_OWNERSHIP` hands out firm ownership for free.
Everything else — the Moore neighbourhood, happiness, the move — is a
`string.Template` rule in `skabm/schelling.py`, run by the same
`RDFSimulator.fit_iter` that drives the macro ABM.

Compare with the AMBER reference (examples/segregation_model.py there):
`Model.grid`, `Model.empty_spots`, `get_neighbors`, `Person.find_new_home`,
the placement loop in `setup`, and the sequential sweep all collapse into
declarative graph patterns.
"""

from time import time

import polars as pl

from skabm.schelling import (
    SCHELLING_INIT_RULES,
    SCHELLING_PARAMS,
    SCHELLING_UPDATE_RULES,
    state_extract,
)
from skabm.simulation import RDFSimulator

SIZE = 50  # grid side; 2500 cells, ~2000 settled at density 0.8

# --- Cells: the grid, and the only population passed -------------------------
# Purely spatial: coordinates and an id, nothing random.  x/y are Float64
# because SPARQL BGPs join on RDF terms, so the xsd:double produced by
# `?x + ?dx` in GRID_NEIGHBORHOOD must meet stored xsd:doubles.
cells = (
    pl.DataFrame({"x": range(SIZE)})
    .join(pl.DataFrame({"y": range(SIZE)}), how="cross")
    .with_columns(
        pl.format("cell_{}_{}", pl.col("x"), pl.col("y")).alias("id"),
        pl.col("x").cast(pl.Float64),
        pl.col("y").cast(pl.Float64),
    )
)

# --- Simulation --------------------------------------------------------------
# density and n_groups live in SCHELLING_PARAMS, consumed by SETTLE; random_seed
# makes the pr:uniform draws reproducible.  Only the rule tuples, the params,
# and the state extract differ from the Poledna demo's RDFSimulator call.
sim = RDFSimulator(
    init_rules=SCHELLING_INIT_RULES,
    update_rules=SCHELLING_UPDATE_RULES,
    params=SCHELLING_PARAMS,
    n_periods=40,
    state_extract=state_extract,
    random_seed=0,
)

t0 = time()
for t, state in enumerate(sim.fit_iter(Cell=cells)):
    summary = state.select(
        pl.lit(t).alias("t"),
        # AMBER's get_segregation()
        pl.col("share_similar").mean().round(3).alias("segregation"),
        (pl.col("share_similar") < SCHELLING_PARAMS["want_similar"])
        .sum()
        .alias("unhappy"),
    )
    print(summary)
    # RDFSimulator has no convergence stop — it runs exactly n_periods.  The
    # generator is where that belongs anyway: the caller owns the criterion.
    if summary["unhappy"][0] == 0:
        print(f"converged at t={t}")
        break
t1 = time()
print(f"{t + 1} ticks in {t1 - t0:.2f}s ({(t1 - t0) / (t + 1):.3f} s/tick)")
