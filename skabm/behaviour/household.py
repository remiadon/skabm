"""
Household behaviour templates — income, wealth, and consumption.

All templates anchor on ``ex:Household`` and read/write ``def:wealth``,
``def:income``, ``def:psi``.

Templates are ``string.Template`` objects with ``$placeholder`` references.
"""

from __future__ import annotations

from string import Template

from skabm.behaviour.learning import expect
from skabm.rules import _PREFIXES

# ---------------------------------------------------------------------------
# household_income_init — initial income by activity status (init CONSTRUCT)
# ---------------------------------------------------------------------------
# Poledna eq. 49.  Income priority: wage > dividend > unemployment benefit.
# Run once after mapping.  Placeholders: ``dividend_ratio``, ``benefit_replacement``.
#
# The ``FILTER NOT EXISTS`` is load-bearing, for the same reason it is in
# ``firm_ownership``: an init rule is a CONSTRUCT applied through ``insert``,
# which *adds* triples.  Without the guard, a population that already carries an
# ``income`` column ends up with two ``def:income`` values on every household —
# silently, permanently, and double-counted by every aggregate and extract
# afterwards.  Initial conditions fill in only what the data left undefined.

household_income_init = Template(
    _PREFIXES
    + """
CONSTRUCT { ?hh def:income ?income }
WHERE {
    { SELECT (AVG(?any_w) AS ?w_avg) WHERE { ?any_f def:w_bar ?any_w } }
    ?hh a ex:Household .
    FILTER NOT EXISTS { ?hh def:income ?given }
    OPTIONAL { ?hh def:employer ?f . ?f def:w_bar ?w . }
    OPTIONAL { ?hh def:owns ?g . ?g def:profit ?p . }
    BIND(
        IF(BOUND(?w), ?w,
        IF(BOUND(?p), $dividend_ratio * IF(?p > 0e0, ?p, 0e0),
        $benefit_replacement * ?w_avg)) AS ?income)
}
"""
)

# ---------------------------------------------------------------------------
# household_income — per-tick income refresh (update)
# ---------------------------------------------------------------------------
# Same logic as ``household_income_init`` but as an upsert, so dividends track
# the owned firm's evolving profit.  Placeholders: ``dividend_ratio``,
# ``benefit_replacement``.

household_income = Template(
    _PREFIXES
    + """
DELETE { ?hh def:income ?i0 }
INSERT { ?hh def:income ?i1 }
WHERE {
    { SELECT (AVG(?any_w) AS ?w_avg) WHERE { ?any_f def:w_bar ?any_w } }
    ?hh a ex:Household .
    OPTIONAL { ?hh def:income ?i0 }
    OPTIONAL { ?hh def:employer ?f . ?f def:w_bar ?w . }
    OPTIONAL { ?hh def:owns ?g . ?g def:profit ?p . }
    BIND(
        IF(BOUND(?w), ?w,
        IF(BOUND(?p), $dividend_ratio * IF(?p > 0e0, ?p, 0e0),
        $benefit_replacement * ?w_avg)) AS ?i1)
}
"""
)

# ---------------------------------------------------------------------------
# household_wealth_init — initial wealth proportional to initial income (init)
# ---------------------------------------------------------------------------
# Poledna Section 5.2: D_h(0) = total_deposits * Y_h(0) / sum Y_h(0).
# Run once after ``household_income_init``.  Placeholder: ``total_deposits``.
# Guarded like ``household_income_init`` above — see the note there.

household_wealth_init = Template(
    _PREFIXES
    + """
CONSTRUCT { ?hh def:wealth ?wealth }
WHERE {
    { SELECT (SUM(?any_i) AS ?total) WHERE { ?any_hh def:income ?any_i } }
    ?hh def:income ?income .
    FILTER NOT EXISTS { ?hh def:wealth ?given }
    BIND($total_deposits * ?income / ?total AS ?wealth)
}
"""
)

# ---------------------------------------------------------------------------
# satisificing_consume — fixed fraction psi of income (update)
# ---------------------------------------------------------------------------
# Poledna eq. 40 + 50.  C = psi * expected_income / (1 + vat_rate); savings
# absorb the rest.  Placeholder: ``vat_rate``.  ``psi`` is an agent attribute
# (def:psi), not a parameter.
#
# Eq. 40 budgets out of *expected* income, not realized income: households
# smooth consumption against where they think their income is heading.  ?ig_e is
# the SAC-learned growth of realized SUM(def:income) over ex:Household — the
# same learning machinery firms use for output and prices — so expected income
# is this quarter's income carried forward one step.

satisificing_consume = Template(
    _PREFIXES
    + """
DELETE { ?hh def:wealth ?w0 }
INSERT { ?hh def:wealth ?w1 }
WHERE {
    ?hh a ex:Household ;
        def:wealth ?w0 ;
        def:psi ?psi ;
        def:income ?inc ."""
    + expect("SUM", "Household", "income", out="ig_e")
    + """
    BIND(?inc * (1e0 + ?ig_e) AS ?exp_income)
    BIND(?psi * ?exp_income / (1e0 + $vat_rate) AS ?consumption)
    BIND(?w0 + (?inc - ?consumption) AS ?w1)
}
"""
)

# ---------------------------------------------------------------------------
# kinked_consume — subsistence floor + higher propensity above it (update)
# ---------------------------------------------------------------------------
# CANVAS-style: households spend subsistence C0 first, then psi_2 on income
# above the VAT-adjusted subsistence.  Placeholders: ``subsistence``,
# ``vat_rate``, ``psi_2``.

kinked_consume = Template(
    _PREFIXES
    + """
DELETE { ?hh def:wealth ?w0 }
INSERT { ?hh def:wealth ?w1 }
WHERE {
    ?hh a ex:Household ;
        def:wealth ?w0 ;
        def:income ?inc .
    BIND($subsistence AS ?C0)
    BIND(?C0 * (1e0 + $vat_rate) AS ?subs_real)
    BIND(IF(?inc > ?subs_real, ?inc - ?subs_real, 0e0) AS ?above)
    BIND(?C0 + $psi_2 * ?above AS ?consumption)
    BIND(?w0 + ?inc - ?consumption AS ?w1)
}
"""
)
