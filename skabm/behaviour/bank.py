"""Banks, a skabm extension of Poledna et al. (2023). Four rules, run each step, in this
order.

1. A bank's distressed becomes 1 when its capital_ratio is below the parameter
distress_threshold, or when (the total over households of their flees_amount) plus (the
total over firms of their flees_amount) is above the parameter flee_amount_threshold;
otherwise 0.

2. A household's flees_amount becomes its wealth when the bank it holds at is distressed
(distressed above 0), and 0 otherwise.

3. A firm's flees_amount becomes its liquidity when the bank it holds at is distressed
(distressed above 0), and 0 otherwise.

4. A bank's capital_ratio becomes its capital_ratio minus (the sum of flees_amount over
the households holding at it, plus the sum of flees_amount over the firms holding at
it) divided by (its leverage times the parameter bank_asset_scale).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import sympy as sp

from skabm.dsl import Agents, sum_over, total

Bank, Household, Firm = Agents("Bank"), Agents("Household"), Agents("Firm")
bank_asset_scale, distress_threshold, flee_amount_threshold = sp.symbols(
    "bank_asset_scale distress_threshold flee_amount_threshold"
)

# a skabm extension, not in Poledna et al. (2023): no published calibration
PARAMETERS = {
    bank_asset_scale: 1e3,  # unsourced: deposit base per unit of leverage; scale with the population
    distress_threshold: 0.03,  # ζ, the Basel III minimum capital ratio
    flee_amount_threshold: 10.0,  # unsourced: a total flight that destabilises every bank
}

# Contagion, one step at a time: last step's flight distresses the banks this step.
fled = total(Household.flees_amount) + total(Firm.flees_amount)  # last step's flight
bank_distress = {
    Bank.distressed: sp.Piecewise(
        (1, (Bank.capital_ratio < distress_threshold) | (fled > flee_amount_threshold)),
        (0, True),
    )
}
household_flight = {
    Household.flees_amount: sp.Piecewise(
        (Household.wealth, Household.holds_at.distressed > 0), (0, True)
    )
}
firm_flight = {
    Firm.flees_amount: sp.Piecewise(
        (Firm.liquidity, Firm.holds_at.distressed > 0), (0, True)
    )
}
outflow = sum_over(Household.holds_at, Household.flees_amount) + sum_over(
    Firm.holds_at, Firm.flees_amount
)
bank_capital = {
    Bank.capital_ratio: Bank.capital_ratio
    - outflow / (Bank.leverage * bank_asset_scale)
}

RULES = [bank_distress, household_flight, firm_flight, bank_capital]


def depositors(
    agents: pl.DataFrame, banks: pl.DataFrame, seed: int = 0
) -> pl.DataFrame:
    """*agents* (households or firms) with ``holds_at``: one bank each, drawn with
    probability proportional to the bank's ``deposit_share``."""
    share = banks["deposit_share"].to_numpy()
    drawn = np.random.default_rng(seed).choice(
        banks["id"].to_numpy(), size=agents.height, p=share / share.sum()
    )
    return agents.with_columns(holds_at=pl.Series(drawn, dtype=pl.String))
