"""
Firm behaviour templates.

Anchored on ``ex:Firm`` and read/write ``def:output``, ``def:price``,
``def:alpha``, ``def:size``, ``def:margin``, ``def:liquidity``, ``def:profit``.

All templates are ``string.Template`` objects with ``$placeholder`` references.
All SPARQL strings use triple-quoted formatting — no ``\n`` concatenation.
"""

from __future__ import annotations

from string import Template

from skabm.rules import EX_NS, _PREFIXES

# ---------------------------------------------------------------------------
# firm_ownership — assign firm owners to investor households (init CONSTRUCT)
# ---------------------------------------------------------------------------
# Poledna Section 3.2: a fraction ``$firm_ownership_ratio`` of households own
# firms.  Firm j is owned by household floor(j / ratio).  Deterministic; the
# ``pr:uniform`` UDF could randomise it.  Run once after mapping.
# Placeholder: ``firm_ownership_ratio``.

firm_ownership = Template(
    _PREFIXES
    + f"""
CONSTRUCT {{ ?owner def:owns ?f }}
WHERE {{
    ?f a ex:Firm .
    FILTER NOT EXISTS {{ ?anyone def:owns ?f }}
    BIND(xsd:integer(STRAFTER(STR(?f), "#firm_")) AS ?j)
    BIND(xsd:integer(FLOOR(?j / $firm_ownership_ratio)) AS ?i)
    BIND(IRI(CONCAT("{EX_NS}hh_", STR(?i))) AS ?owner)
}}
"""
)
firm_ownership.metadata = {
    "@id": "firm_ownership",
    "@type": "Behaviour",
    "agentClass": "ex:Firm",
    "source": "Poledna et al. (2023), European Economic Review 151, 104306, Section 3.2",
}

# ---------------------------------------------------------------------------
# firm_produce — supply choice with capacity cap (update)
# ---------------------------------------------------------------------------
# Y_i(t+1) = min(Y_i(t) * (1 + growth_e + eps), alpha * size)
# Poledna eq. 5 + 12.  AR(1) innovation eps ~ N(0, growth_sigma) via UDF.
# Placeholder: ``growth_e``, ``growth_sigma``.

firm_produce = Template(
    _PREFIXES
    + """
DELETE { ?f def:output ?y0 }
INSERT { ?f def:output ?y1 }
WHERE {
    ?f a ex:Firm ;
        def:output ?y0 ;
        def:alpha ?alpha ;
        def:size ?n .
    BIND(?y0 * (1e0 + $growth_e + pr:normal(0e0, $growth_sigma)) AS ?y_desired)
    BIND(?alpha * ?n AS ?y_capacity)
    BIND(IF(?y_desired < ?y_capacity, ?y_desired, ?y_capacity) AS ?y1)
}
"""
)
firm_produce.metadata = {
    "@id": "firm_produce",
    "@type": "Behaviour",
    "agentClass": "ex:Firm",
    "source": "Poledna et al. (2023), European Economic Review 151, 104306, eq. 5 + 12",
}

# ---------------------------------------------------------------------------
# firm_price — cost-push price setting (update)
# ---------------------------------------------------------------------------
# P_i(t+1) = P_i(t) * (1 + inflation_e + eps)
# Poledna eq. 8.  AR(1) innovation eps ~ N(0, inflation_sigma) via UDF.
# Placeholder: ``inflation_e``, ``inflation_sigma``.

firm_price = Template(
    _PREFIXES
    + """
DELETE { ?f def:price ?p0 }
INSERT { ?f def:price ?p1 }
WHERE {
    ?f a ex:Firm ;
        def:price ?p0 .
    BIND(?p0 * (1e0 + $inflation_e + pr:normal(0e0, $inflation_sigma)) AS ?p1)
}
"""
)
firm_price.metadata = {
    "@id": "firm_price",
    "@type": "Behaviour",
    "agentClass": "ex:Firm",
    "source": "Poledna et al. (2023), European Economic Review 151, 104306, eq. 8",
}

# ---------------------------------------------------------------------------
# firm_sales — goods-market allocation (update)
# ---------------------------------------------------------------------------
# Total nominal demand allocated to firms proportionally to supply share,
# capped by supply.  profit = margin * revenue; liquidity accumulates profit.
# Placeholder: ``vat_rate``.

firm_sales = Template(
    _PREFIXES
    + """
DELETE { ?f def:profit ?pi0 . ?f def:liquidity ?d0 }
INSERT { ?f def:profit ?pi1 . ?f def:liquidity ?d1 }
WHERE {
    { SELECT (SUM(?psi_h * ?i_h) AS ?c_hh)
      WHERE { ?h a ex:Household ; def:psi ?psi_h ; def:income ?i_h } }
    { SELECT (SUM(?b_j) AS ?c_gov) WHERE { ?j a ex:Government ; def:budget ?b_j } }
    { SELECT (SUM(?d_l) AS ?c_row) WHERE { ?l a ex:ForeignFirm ; def:demand_size ?d_l } }
    { SELECT (SUM(?p_g * ?y_g) AS ?supply)
      WHERE { ?g a ex:Firm ; def:price ?p_g ; def:output ?y_g } }
    ?f a ex:Firm ;
        def:price ?p ;
        def:output ?y ;
        def:margin ?mrg ;
        def:profit ?pi0 ;
        def:liquidity ?d0 .
    BIND(IF(BOUND(?c_hh), ?c_hh / (1e0 + $vat_rate), 0e0)
         + IF(BOUND(?c_gov), ?c_gov, 0e0)
         + IF(BOUND(?c_row), ?c_row, 0e0) AS ?demand_total)
    BIND(?demand_total * (?p * ?y) / ?supply AS ?demand_f)
    BIND(IF(?demand_f < ?p * ?y, ?demand_f, ?p * ?y) AS ?revenue)
    BIND(?mrg * ?revenue AS ?pi1)
    BIND(?d0 + ?pi1 AS ?d1)
}
"""
)
firm_sales.metadata = {
    "@id": "firm_sales",
    "@type": "Behaviour",
    "agentClass": "ex:Firm",
    "source": "Poledna et al. (2023), European Economic Review 151, 104306, eqs. 1 + 2 + 27 + 31",
}

# ---------------------------------------------------------------------------
# firm_labor — hire workers when output exceeds current capacity (update)
# ---------------------------------------------------------------------------
# Poledna eq. 9 + 11.  Proportional hiring: dL / L = (desired_output - current)/
# current, capped by available workers.  Placeholder: none (reads firm predicates).

firm_labor = Template(
    _PREFIXES
    + """
DELETE { ?f def:size ?n0 }
INSERT { ?f def:size ?n1 }
WHERE {
    ?f a ex:Firm ;
        def:size ?n0 ;
        def:alpha ?alpha ;
        def:output ?y .
    BIND(?y / ?alpha AS ?n_desired)
    BIND(?n0 + IF(?n_desired > ?n0, (?n_desired - ?n0) * 0.1e0, 0e0) AS ?n1)
}
"""
)
firm_labor.metadata = {
    "@id": "firm_labor",
    "@type": "Behaviour",
    "agentClass": "ex:Firm",
    "source": "Poledna et al. (2023), European Economic Review 151, 104306, eq. 9 + 11",
}

# ---------------------------------------------------------------------------
# firm_entry — logit-based firm entry (update)
# ---------------------------------------------------------------------------
# Poledna eq. 13: new firms enter with probability proportional to log-profit.
# Placeholder: ``entry_barrier``, ``entry_sigma``.  One new firm per tick max
# (proxy for the paper's continuous-time entry rate).

firm_entry = Template(
    _PREFIXES
    + """
DELETE { ?f def:output ?y0 . ?f def:price ?p0 }
INSERT { ?f def:output ?y0 . ?f def:price ?p0 . ?f a ex:Firm }
WHERE {
    { SELECT (SUM(?p_f * ?y_f) AS ?total_revenue)
      WHERE { ?f a ex:Firm ; def:price ?p_f ; def:output ?y_f } }
    BIND($entry_barrier + pr:normal(0e0, $entry_sigma) AS ?threshold)
    FILTER(?total_revenue > ?threshold)
    BIND(IRI(CONCAT("{EX_NS}firm_", STR(xsd:integer(FLOOR(?total_revenue))))) AS ?f)
    BIND(0e0 AS ?y0)
    BIND(1e0 AS ?p0)
}
LIMIT 1
"""
)
firm_entry.metadata = {
    "@id": "firm_entry",
    "@type": "Behaviour",
    "agentClass": "ex:Firm",
    "source": "Poledna et al. (2023), European Economic Review 151, 104306, eq. 13",
}

# ---------------------------------------------------------------------------
# firm_dividends — shell out profits to owners (update)
# ---------------------------------------------------------------------------
# Poledna eq. 14: dividend = dividend_ratio * max(0, profit).
# Placeholder: ``dividend_ratio``.

firm_dividends = Template(
    _PREFIXES
    + """
DELETE { ?f def:profit ?pi0 . ?f def:dividend ?div0 }
INSERT { ?f def:profit ?pi1 . ?f def:dividend ?div1 }
WHERE {
    ?f a ex:Firm ;
        def:profit ?pi0 .
    OPTIONAL { ?f def:dividend ?div0 }
    BIND($dividend_ratio * IF(?pi0 > 0e0, ?pi0, 0e0) AS ?div1)
    BIND(?pi0 - ?div1 AS ?pi1)
}
"""
)
firm_dividends.metadata = {
    "@id": "firm_dividends",
    "@type": "Behaviour",
    "agentClass": "ex:Firm",
    "source": "Poledna et al. (2023), European Economic Review 151, 104306, eq. 14",
}
