"""
Infrastructure for SPARQL-based ABM simulation.

Provides the constants, rendering, mapping, and random-UDF registration that
the behaviour templates (``skabm.behaviour.*``) and the simulator
(``skabm.simulation.RDFSimulator``) depend on.

SPARQL rule *logic* lives in ``skabm.behaviour`` (firm.py, household.py,
macro.py).  This module carries only the plumbing: namespaces, ``render()``,
``map_df()``, ``register_polars_random()``, ``state_extract()`` (Poledna
variant), and ``dbl()``.
"""

from __future__ import annotations

from string import Template

import polars as pl
import polars_random as pr
from maplib import xsd

EX_NS = "http://example.net/skabm#"
DEF_NS = "urn:maplib_default:"
PR_NS = "urn:pr:"  # polars-random UDFs, registered by register_polars_random

_PREFIXES = (
    f"PREFIX ex:<{EX_NS}>\n"
    f"PREFIX def:<{DEF_NS}>\n"
    f"PREFIX pr:<{PR_NS}>\n"
    "PREFIX xsd:<http://www.w3.org/2001/XMLSchema#>\n"
)

_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


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


def dbl(x: float) -> str:
    """Format a Python float as a SPARQL xsd:double literal."""
    return f"{x:.6e}"


def render(rule: "Template | str", params: dict) -> str:
    """Substitute a rule Template's $-placeholders with xsd:double literals.

    Numeric parameter values go through ``dbl`` so decimal literals can
    never leak into the SPARQL; plain-string rules pass through unchanged.
    Missing placeholders raise ``KeyError`` (loudly, at fit time).
    """
    if isinstance(rule, Template):
        return rule.substitute(
            {k: dbl(v) if isinstance(v, (int, float)) else v for k, v in params.items()}
        )
    return rule


def map_df(model, df: pl.DataFrame, kind: str) -> None:
    """Map an agent population into the model in one call.

    Users pass bare local names everywhere (``"firm_0"``): the ``id``
    column is prefixed with ``EX_NS``, and a *link* column — one whose bare
    values name agents already mapped into the model — is prefixed to the
    same nodes, so ``employer="firm_0"`` resolves to the firm.  Which
    columns are links is read from the graph, not declared: a value is a
    reference iff an agent with that id already exists.  (Map referenced
    populations first; e.g. firms before households.)

    ``map_default`` generates the template from the DataFrame schema (every
    non-``id`` column becomes a ``def:`` predicate; IRI-valued and nullable
    columns are detected) **and applies it to ``df`` in the same call** —\n    its return value is the template document for inspection only, and a
    follow-up ``model.map`` would map the rows a second time.

    The generated template emits no class triple, and the class is not
    derivable from the schema (two populations may share identical
    columns), so every agent is tagged ``rdf:type ex:<kind>`` here — that
    is what lets rules anchor on ``?hh a ex:Household``.
    """
    df = df.with_columns(pl.format(EX_NS + "{}", pl.col("id")).alias("id"))
    known = {s.strip("<>") for s in model.query("SELECT ?s WHERE { ?s a ?c }")["s"]}
    for c in df.columns:
        if c == "id" or df.schema[c] != pl.String:
            continue
        vals = {EX_NS + v for v in df[c].drop_nulls().to_list()}
        if vals and vals <= known:
            df = df.with_columns((pl.lit(EX_NS) + pl.col(c)).alias(c))
    model.map_default(df, primary_key_column="id")
    model.map_triples(
        df.select(subject="id").with_columns(object=pl.lit(EX_NS + kind)),
        predicate=_RDF_TYPE,
    )


def state_extract(model) -> pl.DataFrame:
    """Per-agent state as a sparse wide frame — no aggregation in SPARQL.

    One row per firm (price, output, tech_share), household (wealth), and
    central bank (policy_rate); the other columns are null.  Summary logic
    (GDP, price level, ...) belongs in polars expressions on the caller's
    side.

    This is the Poledna (2023) column set.  Other model families pass their
    own extract function (e.g. ``skabm.schelling.state_extract``).
    """
    return model.query(
        _PREFIXES
        + """
    SELECT ?agent ?price ?output ?tech_share ?wealth ?policy_rate
    WHERE {
        { ?agent a ex:Firm ; def:price ?price ; def:output ?output ;
                 def:tech_share ?tech_share }
        UNION { ?agent a ex:Household ; def:wealth ?wealth }
        UNION { ?agent a ex:CentralBank ; def:policy_rate ?policy_rate }
    }
    """
    )
