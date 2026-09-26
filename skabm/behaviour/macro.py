"""Government and central bank, by source: Poledna et al. (2023) and CANVAS, Hommes et
al. (2025).

Poledna et al. (2023), ``poledna_rules``.

Each step, the government's budget is multiplied by (1 + the parameter gov_growth + a
normal shock with mean 0 and standard deviation the parameter gov_growth_sigma).

Then the central bank updates its policy_rate, prev_output and prev_price together, in
one rule. Its policy_rate becomes the larger of 0 and the parameter rho times its
policy_rate plus (1 - rho) times (the parameter r_star + the parameter pi_star + the
parameter xi_pi times ((the mean (average) over firms of their price) / prev_price - 1 -
pi_star) + the parameter xi_gamma times ((the total over firms of their output) /
prev_output - 1)). Its prev_output becomes the total over firms of their output, and its
prev_price the mean (average) over firms of their price.

CANVAS, Hommes, He, Poledna, Siqueira & Zhang (2025), ``canvas_rules``.

The central bank's policy_rate becomes the parameter boc_rho times its policy_rate plus
(1 - boc_rho) times the sum of four terms: the parameter boc_r_star; the parameter
pi_star; the parameter boc_xi_pi times the gap between the expected growth of the mean
over firms of their price and pi_star; and the parameter boc_xi_gamma times the expected
growth of the total over firms of their output.
"""

from __future__ import annotations

import sympy as sp
from sympy.stats import Normal

from skabm.behaviour.learning import expect
from skabm.dsl import Agents, mean, total

Government, CentralBank, Firm = (
    Agents("Government"),
    Agents("CentralBank"),
    Agents("Firm"),
)
gov_growth, gov_growth_sigma = sp.symbols("gov_growth gov_growth_sigma")
rho, r_star, pi_star, xi_pi, xi_gamma = sp.symbols("rho r_star pi_star xi_pi xi_gamma")
boc_rho, boc_r_star, boc_xi_pi, boc_xi_gamma = sp.symbols(
    "boc_rho boc_r_star boc_xi_pi boc_xi_gamma"
)

# CANVAS's central bank re-estimates boc_rho, boc_r_star, boc_xi_pi and boc_xi_gamma
# every quarter on the model's own history (eq. 7), and the paper reports none: no
# default.
PARAMETERS = {
    gov_growth: 0.005,  # unsourced: eq. 51 drift
    gov_growth_sigma: 0.0,  # scenario knob: AR(1) innovation std, 0 = deterministic
    rho: 0.9263,  # policy-rate smoothing, Poledna et al. (2023) Table 2
    r_star: -0.0034,  # real equilibrium rate, Poledna et al. (2023) Table 2
    # inflation target, Poledna et al. (2023) Table 2 and Hommes et al. (2025) Table 4
    pi_star: 0.005,
    xi_pi: 0.3214,  # inflation-gap weight, Poledna et al. (2023) Table 2
    xi_gamma: 1.2994,  # growth-gap weight, Poledna et al. (2023) Table 2
}

# Poledna et al. (2023) eq. 51, linearised
poledna_spend = {
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
poledna_rate = {
    CentralBank.policy_rate: sp.Max(taylor, 0),
    CentralBank.prev_output: output,
    CentralBank.prev_price: price,
}

poledna_rules = [poledna_spend, poledna_rate]

# Hommes et al. (2025) eq. 7, on the SAC forecasts (eq. 20), with no lower bound
# (footnote 30)
expected_growth = expect("SUM", "Firm", "output")
expected_inflation = expect("AVG", "Firm", "price")
gap = boc_xi_pi * (expected_inflation - pi_star) + boc_xi_gamma * expected_growth
canvas_rate = {
    CentralBank.policy_rate: boc_rho * CentralBank.policy_rate
    + (1 - boc_rho) * (boc_r_star + pi_star + gap)
}

canvas_rules = [canvas_rate]
