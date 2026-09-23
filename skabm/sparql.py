"""
Infrastructure for SPARQL-based ABM simulation.

Provides the constants, rendering, mapping, and random-UDF registration that
the behaviour templates (``skabm.behaviour.*``) and the simulator
(``skabm.simulation.RDFSimulator``) depend on.

SPARQL rule *logic* lives in ``skabm.behaviour`` (firm.py, household.py,
macro.py).  This module carries only the plumbing: namespaces, ``render()``,
``dbl()`` and the UDF registrars ``register_polars_random()``, ``register_math()``
and ``register_geosparql()``.  Mapping is ``skabm.ottr``'s.

There is no ``state_extract`` here any more: what a model's per-agent frame
should contain is derivable from the rules themselves, and ``skabm.ir`` derives
it (``ModelIR.extract``).
"""

from __future__ import annotations

from string import Template

import numpy as np
import polars as pl
import polars_random as pr
from maplib import xsd

EX_NS = "http://example.net/skabm#"
DEF_NS = "urn:maplib_default:"
PR_NS = "urn:pr:"  # polars-random UDFs, registered by register_polars_random
MATH_NS = "urn:math:"  # transcendental UDFs, registered by register_math
GEOF_NS = "http://www.opengis.net/def/function/geosparql/"  # register_geosparql

_PREFIXES = (
    f"PREFIX ex:<{EX_NS}>\n"
    f"PREFIX def:<{DEF_NS}>\n"
    f"PREFIX pr:<{PR_NS}>\n"
    f"PREFIX math:<{MATH_NS}>\n"
    f"PREFIX geof:<{GEOF_NS}>\n"
    "PREFIX xsd:<http://www.w3.org/2001/XMLSchema#>\n"
)


def register_polars_random(model) -> None:
    """Expose polars-random draws to SPARQL as UDFs (maplib >= 0.20.26).

    Registers ``pr:uniform(low, high)`` and ``pr:normal(mean, std)`` — callable
    in any rule via ``BIND(pr:uniform(0e0, 1e0) AS ?u)`` — giving the graph the
    seedable RAND plain SPARQL lacks.  Each UDF receives a DataFrame with one
    column per argument (``"0"``, ``"1"``) and returns the ``out`` Series.

    The draw uses polars-random's ``size=`` Series form (the only one that
    honours ``pr.set_random_seed``; the Series-argument form returns a
    non-reproducible expression) and affine-transforms it per row, so bounds
    may vary by row and a run is reproducible whenever the caller fixes the
    seed (see ``RDFSimulator(random_seed=...)``).
    """

    def _uniform(df: pl.DataFrame) -> pl.Series:
        u = pr.uniform(0.0, 1.0, size=len(df))
        return (df["0"] + (df["1"] - df["0"]) * u).alias("out")

    def _normal(df: pl.DataFrame) -> pl.Series:
        z = pr.normal(0.0, 1.0, size=len(df))
        return (df["0"] + df["1"] * z).alias("out")

    model.add_udf(PR_NS + "uniform", _uniform, xsd.double, [xsd.double, xsd.double])
    model.add_udf(PR_NS + "normal", _normal, xsd.double, [xsd.double, xsd.double])


def register_math(model) -> None:
    """Expose the transcendental functions SPARQL lacks as UDFs.

    Registers ``math:exp(x)`` and ``math:log(x)`` — callable in any rule via
    ``BIND(math:exp(?x) AS ?y)``.  Plain SPARQL has no ``EXP``/``LOG`` (only
    the four arithmetic operators plus ``ABS``/``ROUND``/``FLOOR``/``CEIL``),
    which is what forces log-level laws of motion into linear growth factors
    in the Poledna rule set.  Models whose *specification* is transcendental —
    the urn-ball matching function and the S-curve technology shock of
    del Rio-Chanona et al. (2021) — need the real thing, so it is registered
    here on the same ``Model.add_udf`` seam as the random draws.

    Pass alongside ``register_polars_random`` via
    ``RDFSimulator(udfs=(register_polars_random, register_math))``.
    """

    def _exp(df: pl.DataFrame) -> pl.Series:
        return df["0"].exp().alias("out")

    def _log(df: pl.DataFrame) -> pl.Series:
        return df["0"].log().alias("out")

    model.add_udf(MATH_NS + "exp", _exp, xsd.double, [xsd.double])
    model.add_udf(MATH_NS + "log", _log, xsd.double, [xsd.double])


def register_geosparql(model) -> None:
    """Expose GeoSPARQL's ``geof:sfIntersects`` and ``geof:sfWithin`` as UDFs.

    maplib stores WKT but implements no ``geof:`` function (0.20.29 answers
    "Custom function not found ... define a function using m.add_udf()"), so
    they arrive on the same seam as ``pr:`` and ``math:`` — under their
    *standard* IRIs, which keeps a rule portable to any GeoSPARQL store::

        FILTER(geof:sfIntersects(?road, ?zone))

    Scope: the first argument a ``POINT`` or ``LINESTRING``, the second a
    ``POLYGON`` whose outer ring is used (holes are ignored), coordinates
    planar — lon/lat is fine at city scale.  Intersects means a vertex inside
    or a segment crossing the ring; within means every vertex inside and no
    crossing.  A vertex exactly on the boundary is decided by floating point.

    ponytail: one Python iteration per row, ~20k geometries a second — an
    init-rule cost.  Vectorise across rows if a rule calls it every tick.
    """

    def _topology(df: pl.DataFrame) -> tuple[np.ndarray, ...]:
        any_in, all_in, crosses = (np.zeros(len(df), dtype=bool) for _ in range(3))
        numbers = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
        frame = df.with_row_index("row").with_columns(
            pl.col("0").str.extract_all(numbers).cast(pl.List(pl.Float64))
        )
        for (polygon,), rows in frame.group_by("1"):
            outer = polygon.split("((")[1].split(")")[0].replace(",", " ").split()
            a = np.array(outer, dtype=float).reshape(-1, 2)
            b = np.roll(a, -1, axis=0)  # the ring's edges run a -> b
            for row, xy in zip(rows["row"], rows["0"]):
                p = np.asarray(xy).reshape(-1, 2)
                hit = _ray_parity(p, a, b)
                any_in[row], all_in[row] = hit.any(), hit.all()
                crosses[row] = _crossing(p[:-1], p[1:], a, b)
        return any_in, all_in, crosses

    def _intersects(df: pl.DataFrame) -> pl.Series:
        any_in, _, crosses = _topology(df)
        return pl.Series("out", any_in | crosses)

    def _within(df: pl.DataFrame) -> pl.Series:
        _, all_in, crosses = _topology(df)
        return pl.Series("out", all_in & ~crosses)

    for name, udf in (("sfIntersects", _intersects), ("sfWithin", _within)):
        model.add_udf(GEOF_NS + name, udf, xsd.boolean, [xsd.string, xsd.string])


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


# maplib SPARQL gotcha, worth knowing before writing any rule: arithmetic
# operators of equal precedence associate to the *right*, against the SPARQL
# grammar.  ``?a - ?b + ?c`` evaluates as ``?a - (?b + ?c)`` and ``?a / ?b * ?c``
# as ``?a / (?b * ?c)``; both are silently wrong, with no error and no warning.
# Chains that start with ``+`` or ``*`` happen to survive (``a + (b - c)`` and
# ``a * (b / c)`` are algebraically what you meant), which is why the Poledna
# and Schelling rules are unaffected — but a chain led by ``-`` or ``/`` is a
# live bug.  Bracket every mixed chain explicitly.


def dbl(x: float) -> str:
    """Format a Python float as a SPARQL xsd:double literal."""
    return f"{x:.6e}"


def render(rule: Template | str, params: dict) -> str:
    """Substitute a rule Template's $-placeholders with xsd:double literals.

    *params* is laid over the rule's own ``default`` (``behaviour.DefaultTemplate``).
    Numeric values go through ``dbl`` so decimal literals can never leak into
    the SPARQL; plain-string rules pass through unchanged.  A placeholder with
    neither a default nor a param raises ``KeyError`` naming it.
    """
    if isinstance(rule, Template):
        params = {**getattr(rule, "default", {}), **params}
        missing = sorted(set(rule.get_identifiers()) - set(params))
        if missing:
            raise KeyError(
                f"rule needs parameters {missing}: pass them in params= "
                "(a Template lists its own with .get_identifiers())"
            )
        return rule.substitute(
            {k: dbl(v) if isinstance(v, (int, float)) else v for k, v in params.items()}
        )
    return rule
