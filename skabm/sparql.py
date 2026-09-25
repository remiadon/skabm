"""SPARQL plumbing for the simulator: namespaces and the UDF registrars
``register_polars_random()`` and ``register_math()``.  Rules are ``skabm.dsl`` dicts,
compiled by ``dsl.sparql``; mapping is ``skabm.template``'s.
"""

from __future__ import annotations

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


# maplib associates equal-precedence operators to the right (``?a - ?b + ?c`` is
# ``?a - (?b + ?c)``), silently; ``dsl.sparql`` brackets every operation for it.
