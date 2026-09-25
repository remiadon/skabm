"""Government and central bank, Poledna et al. (2023).

Each step, the government's budget is multiplied by (1 + the parameter gov_growth + a
normal shock with mean 0 and standard deviation the parameter gov_growth_sigma).

Then the central bank updates its policy_rate, prev_output and prev_price together, in
one rule. Its policy_rate becomes the larger of 0 and the parameter rho times its
policy_rate plus (1 - rho) times (the parameter r_star + the parameter pi_star + the
parameter xi_pi times ((the mean (average) over firms of their price) / prev_price - 1 -
pi_star) + the parameter xi_gamma times ((the total over firms of their output) /
prev_output - 1)). Its prev_output becomes the total over firms of their output, and its
prev_price the mean (average) over firms of their price.
"""

from __future__ import annotations

import sympy as sp
from sympy.stats import Normal

from skabm.dsl import Agents, mean, total

Government, CentralBank, Firm = (
    Agents("Government"),
    Agents("CentralBank"),
    Agents("Firm"),
)
gov_growth, gov_growth_sigma = sp.symbols("gov_growth gov_growth_sigma")
rho, r_star, pi_star, xi_pi, xi_gamma = sp.symbols("rho r_star pi_star xi_pi xi_gamma")

PARAMETERS = {
    gov_growth: 0.005,  # unsourced: eq. 51 drift
    gov_growth_sigma: 0.0,  # scenario knob: AR(1) innovation std, 0 = deterministic
    rho: 0.9263,  # policy-rate smoothing, Poledna et al. (2023) Table 2
    r_star: -0.0034,  # real equilibrium rate, Poledna et al. (2023) Table 2
    pi_star: 0.005,  # inflation target, Poledna et al. (2023) Table 2
    xi_pi: 0.3214,  # inflation-gap weight, Poledna et al. (2023) Table 2
    xi_gamma: 1.2994,  # growth-gap weight, Poledna et al. (2023) Table 2
}

# Poledna et al. (2023) eq. 51, linearised
government_spend = {
    Government.budget: Government.budget
    * (1 + gov_growth + Normal("eps", 0, gov_growth_sigma))
}
output, price = total(Firm.output), mean(Firm.price)
growth = output / CentralBank.prev_output - 1
inflation = price / CentralBank.prev_price - 1
taylor = rho * CentralBank.policy_rate + (1 - rho) * (
    r_star + pi_star + xi_pi * (inflation - pi_star) + xi_gamma * growth
)
# Poledna et al. (2023) eq. 69, euro-area terms dropped
centralbank_rate = {
    CentralBank.policy_rate: sp.Max(taylor, 0),
    CentralBank.prev_output: output,
    CentralBank.prev_price: price,
}

RULES = [government_spend, centralbank_rate]
