"""Households, Poledna et al. (2023), ``poledna_rules``: two rules, run each step, in
this order.

1. A household's income becomes the first that exists of its employer's w_bar; the
parameter dividend_ratio times the larger of 0 and the profit of the firm it owns; the
parameter benefit_replacement times the mean over firms of their w_bar.

2. Its wealth becomes its wealth plus its income minus psi times its income times (1 +
the expected growth of the total over households of their income), divided by (1 + the
parameter vat_rate).
"""

from __future__ import annotations

import polars as pl
import sympy as sp

from skabm.behaviour.learning import expect
from skabm.dsl import Agents, coalesce, mean

Household, Firm = Agents("Household"), Agents("Firm")
dividend_ratio, benefit_replacement = sp.symbols("dividend_ratio benefit_replacement")
vat_rate = sp.Symbol("vat_rate")

PARAMETERS = {
    dividend_ratio: 0.7768,  # θ^DIV, Poledna et al. (2023) Table 2
    benefit_replacement: 0.3586,  # θ^UB, Poledna et al. (2023) Table 2
    vat_rate: 0.1529,  # τ^VAT, Poledna et al. (2023) Table 2
}

# Poledna et al. (2023) eq. 49
income = coalesce(
    Household.employer.w_bar,
    dividend_ratio * sp.Max(Household.owns.profit, 0),
    benefit_replacement * mean(Firm.w_bar),
)
poledna_income = {Household.income: income}
# Poledna et al. (2023) eqs. 40, 50
spent = (
    Household.psi
    * Household.income
    * (1 + expect("SUM", "Household", "income"))
    / (1 + vat_rate)
)
poledna_consume = {Household.wealth: Household.wealth + Household.income - spent}

poledna_rules = [poledna_income, poledna_consume]

TOTAL_DEPOSITS = (
    222_933.2e6  # D^H, Poledna et al. (2023) Table 2; rescale to the population
)


def initial(households: pl.DataFrame, firms: pl.DataFrame, **params) -> pl.DataFrame:
    """*households* with a starting income and wealth wherever the data has none.

    Income is eq. 49 on the opening firms, the rule ``poledna_income`` computes each
    step: the employer's wage, else the dividend on the owned firm's profit, else the
    benefit.  Wealth is ``total_deposits`` shared in proportion to income (Section 5.2).
    *params* are the simulator's, by name; the rest come from ``PARAMETERS``.
    """
    p = {"total_deposits": TOTAL_DEPOSITS, **{str(k): v for k, v in PARAMETERS.items()}}
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
    dividend = pl.when(pl.col("_profit").is_not_null()).then(
        p["dividend_ratio"] * pl.max_horizontal("_profit", 0.0)
    )
    benefit = p["benefit_replacement"] * firms.select(column("w_bar").mean()).item()
    frame = frame.with_columns(
        income=pl.coalesce("income", "_wage", dividend, pl.lit(benefit))
    ).with_columns(
        wealth=pl.coalesce(
            "wealth", p["total_deposits"] * pl.col("income") / pl.col("income").sum()
        )
    )
    added = [c for c in ("employer", "owns") if c not in households.columns]
    return frame.drop("_wage", "_profit", *added)
