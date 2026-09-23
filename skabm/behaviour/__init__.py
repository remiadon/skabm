"""
Behavioural rule registry — SPARQL templates for economic ABM.

Submodules group templates by economic function.  Import explicitly from the
submodule you need (no package-level re-exports):

    from skabm.behaviour.firm import firm_produce, firm_price, firm_sales
    from skabm.behaviour.household import satisificing_consume, kinked_consume
    from skabm.behaviour.macro import government_spend, centralbank_rate

Every rule is a ``DefaultTemplate``: SPARQL with ``$placeholder`` references,
carrying the published value of each placeholder next to its citation.  Importing
a rule is enough to see what it reads and what it assumes,

    firm_sales.get_identifiers()    # ['vat_rate']
    firm_sales.default              # {'vat_rate': 0.1529}  — τ^VAT, Poledna Table 2

and ``RDFSimulator(params=...)`` (or ``skabm.sparql.render``) overrides any of
them.  A placeholder with no published value has no default and must be passed.
The canonical rule composition used by ``RDFSimulator`` defaults is defined in
``skabm.simulation`` (DEFAULT_INIT_RULES, DEFAULT_UPDATE_RULES).
"""

from string import Template

from skabm.sparql import dbl


class DefaultTemplate(Template):
    """A ``string.Template`` carrying default values for its placeholders.

    ``substitute`` and ``safe_substitute`` lay the caller's mapping over
    ``default``, and format every number with ``dbl``: a bare ``0.7768`` would
    be an ``xsd:decimal`` in SPARQL, and the rules compute in doubles.
    """

    def __init__(self, template: str, default: dict | None = None):
        super().__init__(template)
        self.default = dict(default or {})

    def _merged(self, mapping, kws) -> dict:
        values = {**self.default, **(mapping or {}), **kws}
        return {
            k: dbl(v) if isinstance(v, (int, float)) else v for k, v in values.items()
        }

    def substitute(self, mapping=None, /, **kws):
        return super().substitute(self._merged(mapping, kws))

    def safe_substitute(self, mapping=None, /, **kws):
        return super().safe_substitute(self._merged(mapping, kws))
