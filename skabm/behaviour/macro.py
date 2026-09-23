"""
Macro policy behaviour templates — government spending and Taylor-rule central bank.

All templates anchor on their respective classes (``ex:Government``,
``ex:CentralBank``).  Templates are ``string.Template`` objects with
``$placeholder`` references.
"""

from __future__ import annotations

from skabm.behaviour import DefaultTemplate
from skabm.sparql import _PREFIXES

# ---------------------------------------------------------------------------
# government_spend — AR(1) consumption process (update)
# ---------------------------------------------------------------------------
# Poledna eq. 51 (linearized): budget(t+1) = budget(t) * (1 + gov_growth + eps).
# Placeholder: ``gov_growth``, ``gov_growth_sigma``.

government_spend = DefaultTemplate(
    _PREFIXES
    + """
DELETE { ?j def:budget ?b0 }
INSERT { ?j def:budget ?b1 }
WHERE {
    ?j a ex:Government ;
        def:budget ?b0 .
    BIND(?b0 * (1e0 + $gov_growth + pr:normal(0e0, $gov_growth_sigma)) AS ?b1)
}
""",
    {
        "gov_growth": 0.005,  # unsourced: eq. 51 drift
        "gov_growth_sigma": 0.0,  # AR(1) innovation std; 0 = deterministic
    },
)

# ---------------------------------------------------------------------------
# centralbank_rate — generalized Taylor rule (update)
# ---------------------------------------------------------------------------
# Poledna eq. 69 (euro-area terms dropped):
# r_raw = rho * r(t-1) + (1-rho) * (r_star + pi_star
#            + xi_pi * (pi(t) - pi_star) + xi_gamma * gamma(t))
# where gamma(t) = Y(t)/Y(t-1) - 1, pi(t) = P(t)/P(t-1) - 1.
# Floored at 0.  Placeholders: ``rho``, ``r_star``, ``pi_star``, ``xi_pi``, ``xi_gamma``.

centralbank_rate = DefaultTemplate(
    _PREFIXES
    + """
DELETE { ?cb def:policy_rate ?r0 .
         ?cb def:prev_output ?py .
         ?cb def:prev_price ?pp }
INSERT { ?cb def:policy_rate ?r1 .
         ?cb def:prev_output ?y_now .
         ?cb def:prev_price ?p_now }
WHERE {
    { SELECT (SUM(?y_f) AS ?y_now) (AVG(?p_f) AS ?p_now)
      WHERE { ?f a ex:Firm ; def:output ?y_f ; def:price ?p_f } }
    ?cb a ex:CentralBank ;
        def:policy_rate ?r0 ;
        def:prev_output ?py ;
        def:prev_price ?pp .
    BIND(?y_now / ?py - 1e0 AS ?growth)
    BIND(?p_now / ?pp - 1e0 AS ?inflation)
    BIND($rho * ?r0
         + (1e0 - $rho) * ($r_star + $pi_star
            + $xi_pi * (?inflation - $pi_star)
            + $xi_gamma * ?growth) AS ?r_raw)
    BIND(IF(?r_raw > 0e0, ?r_raw, 0e0) AS ?r1)
}
""",
    {
        "rho": 0.9263,  # policy-rate smoothing, Poledna et al. (2023) Table 2
        "r_star": -0.0034,  # real equilibrium rate, Poledna et al. (2023) Table 2
        "pi_star": 0.005,  # inflation target, Poledna et al. (2023) Table 2
        "xi_pi": 0.3214,  # inflation-gap weight, Poledna et al. (2023) Table 2
        "xi_gamma": 1.2994,  # growth-gap weight, Poledna et al. (2023) Table 2
    },
)
