"""
Behavioural parameters — canonical numeric defaults from Poledna et al. (2023),
Table 2 (Austria 2010:Q4).

These values are merged over any user-supplied ``params`` dict at fit time, so
overriding one number (``params={"vat_rate": 0.2}``) never means re-writing a
rule template.  All values are plain Python floats; ``render()`` formats them
as ``xsd:double`` literals.
"""

from __future__ import annotations

# fmt: off
poledna_params = {
    "firm_ownership_ratio": 0.03,   # investor share of households
    "dividend_ratio":        0.7768, # theta^DIV
    "benefit_replacement":   0.3586, # theta^UB
    "vat_rate":              0.1529, # tau^VAT
    "total_deposits":        222_933.2e6,  # D^H, rescale for demos
    # Expected growth and inflation are no longer parameters: the default rule
    # set learns them from the model's own history (behaviour.learning), so
    # eq. 6/9 expectations are estimated, not assumed.  gov_growth stays a
    # parameter because eq. 51 is an exogenous process, not an expectation.
    "gov_growth":            0.005,  # government consumption drift
    # AR(1) innovation std devs — default 0 => deterministic drifts
    "growth_sigma":          0.0,    # firm output growth shock
    "inflation_sigma":       0.0,    # firm price shock
    "gov_growth_sigma":      0.0,    # government consumption shock
    # Taylor rule (Poledna Table 2)
    "rho":       0.9263,   # policy-rate smoothing
    "r_star":    -0.0034,  # real equilibrium rate
    "pi_star":   0.005,    # inflation target
    "xi_pi":     0.3214,   # inflation gap weight
    "xi_gamma":  1.2994,   # growth gap weight
    # Interbank contagion parameters (skabm extension — not in Poledna 2023)
    "distress_threshold":     0.03,  # capital ratio below which a bank is distressed
    "flee_amount_threshold":  10.0,  # amount (wealth/liquidity) above which a bank loses enough deposits to become distressed
}
# fmt: on
