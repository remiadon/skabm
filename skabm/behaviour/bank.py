"""Banks and their depositors: a bank below a capital ratio, or caught in enough flight
elsewhere, is distressed; its households and firms pull their deposits, which erodes its
capital the next step, a cascade one step per tick. A skabm extension with no published
calibration.
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
distress = {
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
capital = {
    Bank.capital_ratio: Bank.capital_ratio
    - outflow / (Bank.leverage * bank_asset_scale)
}


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
