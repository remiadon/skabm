"""Households: what they earn and what they spend.

- Income from a wage, else a dividend on the firm owned, else a benefit; consumption from
  a propensity to consume out of expected income: Poledna et al. (2023), European
  Economic Review 151, 104306, eqs. 40, 49, 50.
"""

from __future__ import annotations

import polars as pl
import sympy as sp

from skabm.behaviour import defaults
from skabm.behaviour.learning import expect
from skabm.dsl import Agents, coalesce, mean

Household, Firm = Agents("Household"), Agents("Firm")
dividend_ratio, benefit_replacement = sp.symbols("dividend_ratio benefit_replacement")
vat_rate = sp.Symbol("vat_rate")

PARAMETERS = {
    benefit_replacement: 0.3586,  # θ^UB, Poledna et al. (2023) Table 2
}

# Poledna et al. (2023) eq. 49
earned = coalesce(
    Household.employer.w_bar,
    dividend_ratio * sp.Max(Household.owns.profit, 0),
    benefit_replacement * mean(Firm.w_bar),
)
income_by_status = {Household.income: earned}
# Poledna et al. (2023) eqs. 40, 50
spent = (
    Household.psi
    * Household.income
    * (1 + expect("SUM", "Household", "income"))
    / (1 + vat_rate)
)
propensity_consumption = {Household.wealth: Household.wealth + Household.income - spent}


TOTAL_DEPOSITS = (
    222_933.2e6  # D^H, Poledna et al. (2023) Table 2; rescale to the population
)


def ownership(
    households: pl.DataFrame,
    firms: pl.DataFrame,
    ratio: float = 0.03,  # unsourced: investor share of households, Poledna et al. (2023) §3.2
) -> pl.DataFrame:
    """*households* with ``owns`` filled in: the firm in row j owned by the household in
    row floor(j / ratio), unless the data already names the firm's owner or gives that
    household a firm.

    ponytail: one firm per household (``owns`` is a single link); with ratio > 1 the
    later firms of a household stay unowned.
    """
    if "owns" not in households.columns:
        households = households.with_columns(owns=pl.lit(None, pl.String))
    taken = set(households["owns"].drop_nulls())
    rows = households.height
    pairs = {}
    for j, owned in enumerate(firms["id"]):
        i = int(j // ratio)
        if owned not in taken and i < rows:
            pairs.setdefault(i, owned)
    assigned = pl.Series([pairs.get(i) for i in range(rows)], dtype=pl.String)
    return households.with_columns(owns=pl.coalesce("owns", assigned))


def opening_balance(
    households: pl.DataFrame, firms: pl.DataFrame, **params
) -> pl.DataFrame:
    """*households* with a starting income and wealth wherever the data has none.

    Income is Poledna et al. (2023) eq. 49 on the opening firms, as ``income_by_status`` computes it each step: the
    employer's wage, else the dividend on the owned firm's profit, else the benefit.
    Wealth is ``total_deposits`` shared in proportion to income (Section 5.2).  *params*
    are the simulator's, by name; the rest come from ``behaviour.defaults()``.
    """
    p = {"total_deposits": TOTAL_DEPOSITS, **defaults()}
    p |= params
    frame = households
    for column in ("employer", "owns", "income", "wealth"):
        if column not in frame.columns:
            frame = frame.with_columns(
                pl.lit(
                    None, pl.Float64 if column in ("income", "wealth") else pl.String
                ).alias(column)
            )

    def column(name):  # a firm field the data may not carry, missing like a gap
        return pl.col(name) if name in firms.columns else pl.lit(None, pl.Float64)

    frame = frame.join(
        firms.select("id", _wage=column("w_bar")).rename({"id": "employer"}),
        on="employer",
        how="left",
    ).join(
        firms.select("id", _profit=column("profit")).rename({"id": "owns"}),
        on="owns",
        how="left",
    )
    paid = pl.when(pl.col("_profit").is_not_null()).then(
        p["dividend_ratio"] * pl.max_horizontal("_profit", 0.0)
    )
    benefit = p["benefit_replacement"] * firms.select(column("w_bar").mean()).item()
    frame = frame.with_columns(
        income=pl.coalesce("income", "_wage", paid, pl.lit(benefit))
    ).with_columns(
        wealth=pl.coalesce(
            "wealth", p["total_deposits"] * pl.col("income") / pl.col("income").sum()
        )
    )
    added = [c for c in ("employer", "owns") if c not in households.columns]
    return frame.drop("_wage", "_profit", *added)
