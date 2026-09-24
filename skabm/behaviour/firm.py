"""Firms, Poledna et al. (2023). Six rules, in this order.

1. A firm's output becomes the smaller of alpha times its size and its output multiplied
by (1 + the expected growth of the total over firms of their output + a normal shock
with mean 0 and standard deviation the parameter growth_sigma).

2. A firm's price is multiplied by (1 + the expected growth of the mean over firms of
their price + a normal shock with mean 0 and standard deviation the parameter
inflation_sigma).

3. A firm's profit becomes its margin times the smaller of two amounts: its price times
its output; and total demand times its price times its output divided by the total over
firms of their price times their output. Total demand is the sum of three totals: the
total over households of their psi times their income, divided by (1 + the parameter
vat_rate); the total over governments of their budget; and the total over foreign firms
of their demand_size.

4. Then a firm's liquidity becomes its liquidity plus its profit.

5. A firm's size becomes its size plus the larger of 0 and 0.1 times (its output divided
by alpha, minus its size): it hires towards the workforce its output needs and never
fires.

6. A firm's dividend and profit update together, in one rule. Its dividend becomes the
parameter dividend_ratio times the larger of 0 and its profit. Its profit becomes its
profit minus dividend_ratio times the larger of 0 and its profit.
"""

from __future__ import annotations

from string import Template

import sympy as sp
from sympy.stats import Normal

from skabm.behaviour.learning import expect
from skabm.dsl import Agents, total
from skabm.sparql import _PREFIXES, EX_NS

Firm, Household = Agents("Firm"), Agents("Household")
Government, ForeignFirm = Agents("Government"), Agents("ForeignFirm")
growth_sigma, inflation_sigma = sp.symbols("growth_sigma inflation_sigma")
vat_rate, dividend_ratio = sp.symbols("vat_rate dividend_ratio")
firm_ownership_ratio = sp.Symbol("firm_ownership_ratio")

PARAMETERS = {  # entry_barrier and entry_sigma are unpublished: no default
    growth_sigma: 0.0,  # scenario knob: AR(1) innovation std, 0 = deterministic
    inflation_sigma: 0.0,  # scenario knob: AR(1) innovation std, 0 = deterministic
    vat_rate: 0.1529,  # τ^VAT, Poledna et al. (2023) Table 2
    dividend_ratio: 0.7768,  # θ^DIV, Poledna et al. (2023) Table 2
    firm_ownership_ratio: 0.03,  # unsourced: investor share of households (§3.2)
}

# Poledna et al. (2023) Section 3.2: firm firm_<j> with no owner yet is owned by
# household hh_<floor(j / firm_ownership_ratio)>.
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
# Poledna et al. (2023) eqs. 5, 12
firm_produce = {
    Firm.output: sp.Min(
        Firm.output
        * (1 + expect("SUM", "Firm", "output") + Normal("eps", 0, growth_sigma)),
        Firm.alpha * Firm.size,
    )
}
# Poledna et al. (2023) eq. 8
firm_price = {
    Firm.price: Firm.price
    * (1 + expect("AVG", "Firm", "price") + Normal("eps", 0, inflation_sigma))
}
# Poledna et al. (2023) eqs. 1, 2, 27
demand = (
    total(Household.psi * Household.income) / (1 + vat_rate)
    + total(Government.budget)
    + total(ForeignFirm.demand_size)
)
supplied = Firm.price * Firm.output
firm_sales = {
    Firm.profit: Firm.margin * sp.Min(demand * supplied / total(supplied), supplied)
}
# Poledna et al. (2023) eq. 31
firm_liquidity = {Firm.liquidity: Firm.liquidity + Firm.profit}
# Poledna et al. (2023) eqs. 9, 11
firm_labor = {
    Firm.size: Firm.size + sp.Max(0, (Firm.output / Firm.alpha - Firm.size) * 0.1)
}
# Poledna et al. (2023) eq. 13: at most one firm enters per step, with output 0 and
# price 1, when total revenue exceeds entry_barrier plus a normal shock of standard
# deviation entry_sigma.
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
# Poledna et al. (2023) eq. 14
dividend = dividend_ratio * sp.Max(Firm.profit, 0)
firm_dividends = {Firm.dividend: dividend, Firm.profit: Firm.profit - dividend}
