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

from string import Template

import polars as pl
import sympy as sp
from sympy.stats import Uniform

from skabm.dsl import Agents, coalesce, pick, running_sum, sum_over
from skabm.sparql import _PREFIXES, EX_NS

# Schelling (1971), J. Math. Sociology 1(2); AMBER segregation demo
density, n_groups, radius = sp.symbols("density n_groups radius")
want_similar = sp.symbols("want_similar")
PARAMETERS = {
    density: 0.8,  # AMBER's p['density']: per-cell occupancy probability
    n_groups: 2.0,  # AMBER's p['n_groups']
    radius: 1.7,  # scenario knob: neighbour cutoff, in the points' units
    want_similar: 0.375,  # 3 of 8 Moore neighbours, Schelling (1971)
}

# coordinates are doubles: a join on ?x + ?dx matches terms, not values
# The space is one relation, def:neighbor: GRID_NEIGHBORHOOD builds it on a lattice,
# GEO_NEIGHBORHOOD from WKT points (no geof:distance in maplib yet, so the distance
# is parsed out of the strings).
# A cell's neighbours are the up to eight cells one step away in x, y or both: a
# bounded Moore neighbourhood.
GRID_NEIGHBORHOOD = Template(
    _PREFIXES
    + """
    CONSTRUCT { ?c def:neighbor ?c2 }
    WHERE {
        VALUES (?dx ?dy) {
            (-1e0 -1e0) (-1e0 0e0) (-1e0 1e0)
            ( 0e0 -1e0)            ( 0e0 1e0)
            ( 1e0 -1e0) ( 1e0 0e0) ( 1e0 1e0)
        }
        ?c a ex:Cell ; def:x ?x ; def:y ?y .
        BIND(?x + ?dx AS ?nx)
        BIND(?y + ?dy AS ?ny)
        ?c2 a ex:Cell ; def:x ?nx ; def:y ?ny .
    }
    """
)
# Unless people already exist, each cell is settled with probability density by one
# person, of a group drawn uniformly among n_groups.
SETTLE = Template(
    _PREFIXES
    + """
    CONSTRUCT {
        ?p a ex:Person . ?p def:location ?c . ?p def:group ?g
    }
    WHERE {
        FILTER NOT EXISTS { ?existing a ex:Person }
        ?c a ex:Cell .
        BIND(pr:uniform(0e0, 1e0) AS ?u_occ)
        FILTER(?u_occ < $density)
        BIND(pr:uniform(0e0, 1e0) AS ?u_grp)
        BIND(xsd:double(FLOOR(?u_grp * $n_groups)) AS ?g)
        BIND(IRI(CONCAT(\""""
    + EX_NS
    + """settler_", STRAFTER(STR(?c), "#"))) AS ?p)
    }
    """
)
# A cell's neighbours are the other cells whose point lies within radius of its own.
GEO_NEIGHBORHOOD = Template(
    _PREFIXES
    + """
    CONSTRUCT { ?c def:neighbor ?c2 }
    WHERE {
        ?c  a ex:Cell ; def:geometry ?w1 .
        ?c2 a ex:Cell ; def:geometry ?w2 .
        FILTER(?c != ?c2)
        BIND(STRBEFORE(STRAFTER(?w1, "("), ")") AS ?xy1)
        BIND(STRBEFORE(STRAFTER(?w2, "("), ")") AS ?xy2)
        BIND(xsd:double(STRBEFORE(?xy1, " ")) - xsd:double(STRBEFORE(?xy2, " ")) AS ?dx)
        BIND(xsd:double(STRAFTER(?xy1, " ")) - xsd:double(STRAFTER(?xy2, " ")) AS ?dy)
        FILTER(?dx * ?dx + ?dy * ?dy <= $radius * $radius)
    }
    """
)
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


def geo_state_extract(model) -> pl.DataFrame:
    """``state_extract`` with the coordinates parsed out of the cell's WKT point."""
    return model.query(
        _PREFIXES
        + """
    SELECT ?agent ?group ?x ?y ?share_similar
    WHERE {
        ?agent a ex:Person ;
               def:group ?group ;
               def:share_similar ?share_similar ;
               def:location ?c .
        ?c def:geometry ?w .
        BIND(STRBEFORE(STRAFTER(?w, "("), ")") AS ?xy)
        BIND(xsd:double(STRBEFORE(?xy, " ")) AS ?x)
        BIND(xsd:double(STRAFTER(?xy, " ")) AS ?y)
    }
    """
    )


SCHELLING_INIT_RULES = (GRID_NEIGHBORHOOD, SETTLE)
SCHELLING_GEO_INIT_RULES = (GEO_NEIGHBORHOOD, SETTLE)
SCHELLING_UPDATE_RULES = (OCCUPANCY, HAPPINESS, CELL_RANK, PERSON_RANK, RELOCATE)
