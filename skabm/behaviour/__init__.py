"""
Behavioural rule registry — SPARQL templates for economic ABM.

Submodules group templates by economic function.  Import explicitly from the
submodule you need (no package-level re-exports):

    from skabm.behaviour.firm import firm_produce, firm_price, firm_sales
    from skabm.behaviour.household import satisificing_consume, kinked_consume
    from skabm.behaviour.macro import government_spend, centralbank_rate
    from skabm.behaviour.params import poledna_params

Templates are ``string.Template`` objects with ``$placeholder`` references;
numeric values are supplied at composition time via ``render(rule, params)``
from ``skabm.rules``, or directly to ``RDFSimulator(params=...)``.

The canonical Poledna (2023) parameter set lives in ``behaviour.params``.
The canonical rule composition used by ``RDFSimulator`` defaults is defined in
``skabm.simulation`` (DEFAULT_INIT_RULES, DEFAULT_UPDATE_RULES).
"""

# no imports at package level — users import from submodules directly
