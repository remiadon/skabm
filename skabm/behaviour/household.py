"""Households, Poledna et al. (2023). Five rules, in this order.

1. At the start, a household keeps its income if it has one: its income becomes the
first that exists of its own income; its employer's w_bar; the parameter dividend_ratio
times the larger of 0 and the profit of the firm it owns; the parameter
benefit_replacement times the mean over firms of their w_bar.

2. Each step, its income becomes the first that exists of its employer's w_bar; the
parameter dividend_ratio times the larger of 0 and the profit of the firm it owns; the
parameter benefit_replacement times the mean over firms of their w_bar.

3. At the start, a household keeps its wealth if it has one: its wealth becomes the
first that exists of its own wealth; the parameter total_deposits times its income
divided by the total over households of their income.

4. Each step, its wealth becomes its wealth plus its income minus psi times its income
times (1 + the expected growth of the total over households of their income), divided by
(1 + the parameter vat_rate).

5. Alternatively, with kinked consumption, its wealth becomes its wealth plus its income
minus (the parameter subsistence plus the parameter psi_2 times the larger of 0 and (its
income minus subsistence times (1 + vat_rate))).
"""

from __future__ import annotations

import sympy as sp

from skabm.behaviour.learning import expect
from skabm.dsl import Agents, coalesce, mean, total

Household, Firm = Agents("Household"), Agents("Firm")
dividend_ratio, benefit_replacement = sp.symbols("dividend_ratio benefit_replacement")
total_deposits, vat_rate = sp.symbols("total_deposits vat_rate")
subsistence, psi_2 = sp.symbols("subsistence psi_2")

PARAMETERS = {  # subsistence and psi_2 are unpublished: no default
    dividend_ratio: 0.7768,  # θ^DIV, Poledna et al. (2023) Table 2
    benefit_replacement: 0.3586,  # θ^UB, Poledna et al. (2023) Table 2
    total_deposits: 222_933.2e6,  # D^H, Poledna et al. (2023) Table 2; rescale to the population
    vat_rate: 0.1529,  # τ^VAT, Poledna et al. (2023) Table 2
}

# Poledna et al. (2023) eq. 49
income = coalesce(
    Household.employer.w_bar,
    dividend_ratio * sp.Max(Household.owns.profit, 0),
    benefit_replacement * mean(Firm.w_bar),
)
household_income_init = {Household.income: coalesce(Household.income, income)}
household_income = {Household.income: income}
# Poledna et al. (2023) Section 5.2
household_wealth_init = {
    Household.wealth: coalesce(
        Household.wealth, total_deposits * Household.income / total(Household.income)
    )
}
# Poledna et al. (2023) eqs. 40, 50
spent = (
    Household.psi
    * Household.income
    * (1 + expect("SUM", "Household", "income"))
    / (1 + vat_rate)
)
satisificing_consume = {Household.wealth: Household.wealth + Household.income - spent}
# unsourced: CANVAS-style kinked consumption
spent_kinked = subsistence + psi_2 * sp.Max(
    Household.income - subsistence * (1 + vat_rate), 0
)
kinked_consume = {Household.wealth: Household.wealth + Household.income - spent_kinked}
