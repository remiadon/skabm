"""Where agents sit and who their neighbours are: ``(id, neighbor)`` frames for
``template.links("neighbor")`` on a lattice or among points, and the areas a line or
point falls in, for ``template.links("area")``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.spatial import cKDTree


def moore_neighbors(agents: pl.DataFrame, torus: bool = False) -> pl.DataFrame:
    """``(id, neighbor)``: the up to eight *agents* one step away in ``x``, ``y`` or both,
    a Moore neighbourhood, bounded by the lattice's edges or, if *torus*, wrapped around
    them."""
    size = agents["x"].max() + 1
    steps = (
        pl.DataFrame({"dx": [-1.0, 0.0, 1.0]})
        .join(pl.DataFrame({"dy": [-1.0, 0.0, 1.0]}), how="cross")
        .filter((pl.col("dx") != 0) | (pl.col("dy") != 0))
    )

    def step(axis: str) -> pl.Expr:
        moved = pl.col(axis) + pl.col(f"d{axis}")
        return moved % size if torus else moved

    return (
        agents.select("id", "x", "y")
        .join(steps, how="cross")
        .select("id", x=step("x"), y=step("y"))
        .join(agents.select(neighbor="id", x="x", y="y"), on=["x", "y"])
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
