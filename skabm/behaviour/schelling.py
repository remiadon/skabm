"""Schelling's segregation model: people living on the cells of a grid. Five rules, in this
order.

1. A cell's occupied, resident and draw update together, in one rule. Its occupied
becomes the number of people whose location is this cell, whatever their group: the sum,
over those people, of 1. Its resident becomes the sum of group over the people whose
location is this cell. Its draw becomes a fresh uniform draw between 0 and 1.

2. A person's share_similar and draw update together, in one rule. Their share_similar
becomes the sum, over the cells neighbouring their location, of 1 for a cell whose
occupied is greater than 0 and whose resident equals the person's group, and 0 for any
other cell, divided by the larger of 1 and the sum of occupied over the cells
neighbouring their location. Their draw becomes a fresh uniform draw between 0 and 1.

3. A cell's rank becomes the running sum over all cells, in order of draw, of 1 for a
vacant cell, whose occupied equals 0, and 0 for any other cell.

4. A person's rank becomes the running sum over all people, in order of draw, of 1 for a
person whose share_similar is below the parameter want_similar, and 0 for any other
person.

5. A person's location becomes the cell they pick, or stays their location when they
pick none. The condition for picking a cell is that the cell's occupied equals 0, the
cell's rank equals the person's rank, and the person's share_similar is below the
parameter want_similar; among such cells the order is the cell's rank.
"""

import numpy as np
import polars as pl
import sympy as sp
from scipy.spatial import cKDTree
from sympy.stats import Uniform

from skabm.dsl import Agents, coalesce, pick, running_sum, sum_over
from skabm.sparql import _PREFIXES

# Schelling (1971), J. Math. Sociology 1(2); AMBER segregation demo
want_similar = sp.Symbol("want_similar")
PARAMETERS = {
    want_similar: 0.375,  # 3 of 8 Moore neighbours, Schelling (1971)
}

Cell, Person = Agents("Cell"), Agents("Person")

OCCUPANCY = {
    Cell.occupied: sum_over(Person.location, 1),
    Cell.resident: sum_over(Person.location, Person.group),
    Cell.draw: Uniform("u", 0, 1),
}
around = Person.location.neighbor
HAPPINESS = {
    Person.share_similar: sum_over(
        around,
        sp.Piecewise(
            (1, sp.Eq(around.resident, Person.group) & (around.occupied > 0)),
            (0, True),
        ),
    )
    / sp.Max(sum_over(around, around.occupied), 1),
    Person.draw: Uniform("u", 0, 1),
}
CELL_RANK = {
    Cell.rank: running_sum(
        Cell.draw, sp.Piecewise((1, sp.Eq(Cell.occupied, 0)), (0, True))
    )
}
unhappy = Person.share_similar < want_similar
PERSON_RANK = {
    Person.rank: running_sum(Person.draw, sp.Piecewise((1, unhappy), (0, True)))
}
RELOCATE = {
    Person.location: coalesce(
        pick(
            Cell,
            sp.Eq(Cell.rank, Person.rank) & (sp.Eq(Cell.occupied, 0)) & unhappy,
            Cell.rank,
        ),
        Person.location,
    )
}


def state_extract(model) -> pl.DataFrame:
    """One row per person: group, cell coordinates, share of similar neighbours."""
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


RULES = [OCCUPANCY, HAPPINESS, CELL_RANK, PERSON_RANK, RELOCATE]


def settle(
    cells: pl.DataFrame,
    density: float = 0.8,  # AMBER's p['density']: per-cell occupancy probability
    n_groups: int = 2,  # AMBER's p['n_groups']
    seed: int = 0,
) -> pl.DataFrame:
    """The Person population: each cell settled with probability *density* by one person
    (``settler_<cell>``), of a group drawn uniformly among *n_groups*."""
    rng = np.random.default_rng(seed)
    settled = rng.random(cells.height) < density
    group = np.floor(rng.random(cells.height) * n_groups)
    return pl.DataFrame(
        {
            "id": ("settler_" + cells["id"]).filter(pl.Series(settled)),
            "group": group[settled],
            "location": cells["id"].filter(pl.Series(settled)),
        }
    )


def grid_neighbors(cells: pl.DataFrame) -> pl.DataFrame:
    """``(id, neighbor)``: the up to eight cells one step away in x, y or both, a bounded
    Moore neighbourhood, for ``template.links("neighbor")``."""
    steps = (
        pl.DataFrame({"dx": [-1.0, 0.0, 1.0]})
        .join(pl.DataFrame({"dy": [-1.0, 0.0, 1.0]}), how="cross")
        .filter((pl.col("dx") != 0) | (pl.col("dy") != 0))
    )
    return (
        cells.select("id", "x", "y")
        .join(steps, how="cross")
        .select("id", x=pl.col("x") + pl.col("dx"), y=pl.col("y") + pl.col("dy"))
        .join(cells.select(neighbor="id", x="x", y="y"), on=["x", "y"])
        .select("id", "neighbor")
    )


def geo_neighbors(
    cells: pl.DataFrame,
    radius: float = 1.7,  # scenario knob: neighbour cutoff, in the points' units
) -> pl.DataFrame:
    """``(id, neighbor)``: the other cells whose ``geometry`` point lies within *radius*."""
    xy = cells["geometry"].str.extract_groups(r"\(([-\d.e]+) ([-\d.e]+)\)")
    points = np.column_stack([xy.struct[0].cast(float), xy.struct[1].cast(float)])
    pairs = np.array(sorted(cKDTree(points).query_pairs(radius)), dtype=int).reshape(
        -1, 2
    )
    ids = cells["id"].to_numpy()
    both = np.vstack([pairs, pairs[:, ::-1]])
    return pl.DataFrame({"id": ids[both[:, 0]], "neighbor": ids[both[:, 1]]})
