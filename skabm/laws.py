"""Non-relational laws of motion as plain polars expressions.

These are the *physics* of the model: single-agent laws that read and write
an agent's own scalar attributes and never touch topology.  They are the
polars counterpart of the non-relational SPARQL update rules in
``skabm.rules`` (``FIRM_PRODUCTION``, ``FIRM_PRICING``, ``HOUSEHOLD_UPDATE``,
``GOVERNMENT_CONSUMPTION``, ``TAYLOR_RULE``) and are meant to be applied to a
per-kind state frame with ``.with_columns(...)``::

    firms = firms.with_columns(firm_production(p), firm_pricing(p))

Each function returns a polars expression (or a list of them, for a
multi-attribute upsert) *aliased to the attribute it writes* — the update is
in place, so column names are stable for lineage and re-mapping.

Division of labour (see the architecture discussion): laws of motion are
library-shipped and non-relational; the *relational* rules — edge creation,
dynamic topology, and recursive contagion — stay in the graph as SPARQL /
``infer`` (``skabm.rules``), because that is what maplib buys us.  A law of
motion may only read and write node attributes; it must never write an edge.

Stochastic laws draw AR(1) innovations with ``polars_random``.  The bare
expression form (no ``size=``) is used deliberately: in polars-random 0.5.0
it advances the global RNG per draw *and* reproduces across runs once
``pr.set_random_seed`` has been called — which ``RDFSimulator`` already does
at the start of each fit when ``random_seed`` is set.  With the default
``*_sigma = 0`` the innovation is identically zero and the laws are
deterministic (the draw is still taken, so RNG consumption matches the SPARQL
``pr:normal`` reference exactly).

Parameters are read straight from a params dict ``p`` (``POLEDNA_PARAMS``
merged with user overrides) — no ``xsd:double`` formatting, no ``$``
substitution: the numbers are just Python floats here.
"""

from __future__ import annotations

import polars as pl
import polars_random as pr


def firm_production(p: dict) -> pl.Expr:
    """Supply choice (eq. 5) capped by labour capacity (eq. 12, Leontief).

    ``output <- min(output * (1 + growth_e + eps), alpha * size)`` with the
    AR(1) innovation ``eps ~ N(0, growth_sigma)`` (eq. 6).  Reads
    ``output, alpha, size``.
    """
    desired = pl.col("output") * (1 + p["growth_e"] + pr.normal(0.0, p["growth_sigma"]))
    capacity = pl.col("alpha") * pl.col("size")
    return pl.min_horizontal(desired, capacity).alias("output")


def firm_pricing(p: dict) -> pl.Expr:
    """Cost-push price setting (eq. 8), expected-inflation passthrough.

    ``price <- price * (1 + inflation_e + eps)`` with ``eps ~ N(0,
    inflation_sigma)``.  Reads ``price``.
    """
    return (
        pl.col("price") * (1 + p["inflation_e"] + pr.normal(0.0, p["inflation_sigma"]))
    ).alias("price")


def household_update(p: dict) -> pl.Expr:
    """Consumption (eq. 40) out of income; savings absorb the rest (eq. 50).

    ``wealth <- wealth + income - psi * income / (1 + vat_rate)``.  Reads
    ``wealth, psi, income`` — ``income`` is produced upstream by the
    relational income rule, so run that first in the tick.
    """
    consumption = pl.col("psi") * pl.col("income") / (1 + p["vat_rate"])
    return (pl.col("wealth") + pl.col("income") - consumption).alias("wealth")


def government_consumption(p: dict) -> pl.Expr:
    """Government consumption AR(1) drift (eq. 51).

    ``budget <- budget * (1 + gov_growth + eps)`` with ``eps ~ N(0,
    gov_growth_sigma)``.  Reads ``budget``.
    """
    return (
        pl.col("budget") * (1 + p["gov_growth"] + pr.normal(0.0, p["gov_growth_sigma"]))
    ).alias("budget")


def taylor_rule(p: dict, output_sum: float, price_mean: float) -> list[pl.Expr]:
    """Generalized Taylor rule (eq. 69), euro-area terms dropped.

    A single central-bank agent reacting to economy-wide aggregates, so the
    two cross-firm reductions are passed in as scalars rather than read from
    the CB frame: ``output_sum = firms['output'].sum()`` and ``price_mean =
    firms['price'].mean()``.  Realized growth/inflation are measured against
    the lagged aggregates stored on the CB node, refreshed by the same upsert.

    Returns three expressions (rate + the two lagged aggregates) for one
    ``.with_columns(*taylor_rule(p, y, pm))``.  Reads ``policy_rate,
    prev_output, prev_price``.
    """
    growth = output_sum / pl.col("prev_output") - 1
    inflation = price_mean / pl.col("prev_price") - 1
    r_raw = p["rho"] * pl.col("policy_rate") + (1 - p["rho"]) * (
        p["r_star"]
        + p["pi_star"]
        + p["xi_pi"] * (inflation - p["pi_star"])
        + p["xi_gamma"] * growth
    )
    return [
        pl.max_horizontal(r_raw, 0.0).alias("policy_rate"),
        pl.lit(output_sum).alias("prev_output"),
        pl.lit(price_mean).alias("prev_price"),
    ]
