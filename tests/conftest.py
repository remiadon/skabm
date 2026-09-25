"""Shared fixtures."""

import pytest


@pytest.fixture
def poledna_params() -> dict:
    """Every placeholder the default Poledna rule set reads, as the paper gives it.

    Poledna et al. (2023), European Economic Review 151, 104306, Table 2 (Austria,
    2010:Q4) unless marked ``unsourced``.  The rules carry these as their own
    defaults; this is the independent copy they are checked against.  A fresh dict
    per test: override freely.
    """
    # fmt: off
    return {
        "dividend_ratio":       0.7768,       # θ^DIV
        "benefit_replacement":  0.3586,       # θ^UB
        "vat_rate":             0.1529,       # τ^VAT
        "total_deposits":       222_933.2e6,  # D^H
        "rho":                  0.9263,       # Taylor-rule smoothing
        "r_star":              -0.0034,       # r*
        "pi_star":              0.005,        # π*
        "xi_pi":                0.3214,       # ξ^π
        "xi_gamma":             1.2994,       # ξ^γ
        "gov_growth":           0.005,        # unsourced: eq. 51 drift
        "growth_sigma":         0.0,          # unsourced: 0 = deterministic
        "inflation_sigma":      0.0,          # unsourced: 0 = deterministic
        "gov_growth_sigma":     0.0,          # unsourced: 0 = deterministic
    }
    # fmt: on
