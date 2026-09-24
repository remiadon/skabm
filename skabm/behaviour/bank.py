"""Banks, a skabm extension of Poledna et al. (2023).

A bank's capital_ratio becomes its capital_ratio minus (the sum of flees_amount over the
households holding at it, plus the sum of flees_amount over the firms holding at it)
divided by (its leverage times the parameter bank_asset_scale).
"""

from __future__ import annotations

from string import Template

import sympy as sp

from skabm.dsl import Agents, sum_over
from skabm.sparql import _PREFIXES, EX_NS

Bank, Household, Firm = Agents("Bank"), Agents("Household"), Agents("Firm")
bank_asset_scale = sp.Symbol("bank_asset_scale")
distress_threshold, flee_amount_threshold = sp.symbols(
    "distress_threshold flee_amount_threshold"
)

# a skabm extension, not in Poledna et al. (2023): no published calibration
PARAMETERS = {
    bank_asset_scale: 1e3,  # unsourced: deposit base per unit of leverage; scale with the population
    distress_threshold: 0.03,  # ζ, the Basel III minimum capital ratio
    flee_amount_threshold: 10.0,  # unsourced: outflow that destabilises the receiving bank
}

# Every household with wealth and every firm with liquidity holds deposits at each bank
# with probability the bank's deposit_share.  maplib rejects a line break between a
# CONSTRUCT's closing } and WHERE.
bank_depositors = Template(
    _PREFIXES
    + """CONSTRUCT { ?agent def:holds_at ?bank } WHERE {
    { ?agent a ex:Household ; def:wealth ?w }
    UNION
    { ?agent a ex:Firm ; def:liquidity ?l }
    ?bank a ex:Bank ; def:deposit_share ?ds .
    FILTER(pr:uniform(0e0, 1e0) < ?ds)
}"""
)
outflow = sum_over(Household.holds_at, Household.flees_amount) + sum_over(
    Firm.holds_at, Firm.flees_amount
)
bank_capital = {
    Bank.capital_ratio: Bank.capital_ratio
    - outflow / (Bank.leverage * bank_asset_scale)
}
_HEAD = f"PREFIX ex: <{EX_NS}>PREFIX def: <urn:maplib_default:>"
# Contagion, through RDFSimulator(infer=...), which needs maplib's licensed reasoning
# add-on: a bank below distress_threshold is distressed; a household or firm holding
# at a distressed bank flees to every bank not distressed, with all its wealth or
# liquidity; a bank receiving a flight above flee_amount_threshold is distressed.
interbank_contagion = [
    Template(
        _HEAD
        + "CONSTRUCT { ?b ex:is_distressed true } WHERE { ?b a ex:Bank ; def:capital_ratio ?cr . FILTER(?cr < $distress_threshold) . }"
    ),
    Template(
        _HEAD
        + "CONSTRUCT { ?agent def:flees_to ?safe ; def:flees_amount ?amount } WHERE { { ?agent a ex:Household ; def:holds_at ?distressed ; def:wealth ?amount } UNION { ?agent a ex:Firm ; def:holds_at ?distressed ; def:liquidity ?amount } ?distressed ex:is_distressed true . ?safe a ex:Bank . FILTER NOT EXISTS { ?safe ex:is_distressed true } . }"
    ),
    Template(
        _HEAD
        + "CONSTRUCT { ?b ex:is_distressed true } WHERE { ?other def:flees_to ?b ; def:flees_amount ?amount . ?b a ex:Bank ; def:capital_ratio ?cr . FILTER(?amount > $flee_amount_threshold) . }"
    ),
]
