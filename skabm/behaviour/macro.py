"""Government, central bank and regions.

- Government spending growing by an AR(1) drift; a Taylor rule on realised growth and
  inflation: Poledna et al. (2023), European Economic Review 151, 104306, eqs. 51, 69.
- A Taylor rule on SAC forecasts: CANVAS, Hommes et al. (2025), J. Econ. Dyn. Control
  172, 104986, eq. 7. Its coefficients are re-estimated each quarter on the model's own
  history and never reported, so boc_rho, boc_r_star, boc_xi_pi and boc_xi_gamma stay
  required.
- A region's day: the deaths reported among its people the day before, and its
  lockdown and mandates: Pangallo et al. (2024), Nat. Hum. Behav. 8, 264-275, Methods
  ("Coupling", "Timeline").
"""

from __future__ import annotations

import sympy as sp
from sympy.stats import Normal

from skabm.behaviour.learning import expect
from skabm.behaviour.person import REPORTED
from skabm.dsl import Agents, mean, total

Government, CentralBank = Agents("Government"), Agents("CentralBank")
Firm, Region, Person = Agents("Firm"), Agents("Region"), Agents("Person")
gov_growth, gov_growth_sigma = sp.symbols("gov_growth gov_growth_sigma")
rho, r_star, pi_star, xi_pi, xi_gamma = sp.symbols("rho r_star pi_star xi_pi xi_gamma")
boc_rho, boc_r_star, boc_xi_pi, boc_xi_gamma = sp.symbols(
    "boc_rho boc_r_star boc_xi_pi boc_xi_gamma"
)
lockdown_start, lockdown_end = sp.symbols("lockdown_start lockdown_end")

PARAMETERS = {
    gov_growth: 0.005,  # unsourced: eq. 51 drift
    gov_growth_sigma: 0.0,  # scenario knob: AR(1) innovation std, 0 = deterministic
    rho: 0.9263,  # policy-rate smoothing, Poledna et al. (2023) Table 2
    r_star: -0.0034,  # real equilibrium rate, Poledna et al. (2023) Table 2
    # inflation target, Poledna et al. (2023) Table 2 and Hommes et al. (2025) Table 4
    pi_star: 0.005,
    xi_pi: 0.3214,  # inflation-gap weight, Poledna et al. (2023) Table 2
    xi_gamma: 1.2994,  # growth-gap weight, Poledna et al. (2023) Table 2
    # 16 March 2020 in days after 12 February, Pangallo et al. (2024) Methods, Timeline
    lockdown_start: 33.0,
    lockdown_end: 93.0,  # 15 May 2020, Pangallo et al. (2024) Methods, Timeline
}

# Poledna et al. (2023) eq. 51, linearised
government_spending = {
    Government.budget: Government.budget
    * (1 + gov_growth + Normal("eps", 0, gov_growth_sigma))
}
total_output, mean_price = total(Firm.output), mean(Firm.price)
growth = total_output / CentralBank.prev_output - 1
inflation = mean_price / CentralBank.prev_price - 1
taylor = rho * CentralBank.policy_rate + (1 - rho) * (
    r_star + pi_star + xi_pi * (inflation - pi_star) + xi_gamma * growth
)
# Poledna et al. (2023) eq. 69, euro-area terms dropped
taylor_rule = {
    CentralBank.policy_rate: sp.Max(taylor, 0),
    CentralBank.prev_output: total_output,
    CentralBank.prev_price: mean_price,
}

# Hommes et al. (2025) eq. 7, on the SAC forecasts (eq. 20), with no lower bound
# (footnote 30)
gap = boc_xi_pi * (expect("AVG", "Firm", "price") - pi_star) + boc_xi_gamma * expect(
    "SUM", "Firm", "output"
)
forecast_taylor_rule = {
    CentralBank.policy_rate: boc_rho * CentralBank.policy_rate
    + (1 - boc_rho) * (boc_r_star + pi_star + gap)
}

# Pangallo et al. (2024) Methods: the region counts its people's reported deaths,
# yesterday's D_{t-1}; the lockdown holds from lockdown_start to lockdown_end, and
# school closures and work from home from lockdown_start on
reported = total(sp.Piecewise((1, sp.Eq(Person.state, REPORTED)), (0, True)))
today = Region.t + 1
region_day = {
    Region.t: today,
    Region.reported: reported,
    Region.deaths: reported - Region.reported,
    Region.lockdown: sp.Piecewise(
        (1, (today >= lockdown_start) & (today < lockdown_end)), (0, True)
    ),
    Region.mandates: sp.Piecewise((1, today >= lockdown_start), (0, True)),
}
