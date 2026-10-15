"""The graph a test simulates, built the way a user would: template, ``with_iri``, ``Model.map``."""

from maplib import Model, RDFType

from skabm.behaviour import firm, household, macro
from skabm.template import LINK, TEMPLATES, _agent

# Poledna et al. (2023)'s quarter in the paper's order, the economy engine tests run
POLEDNA = [
    firm.adaptive_output,  # supply choice, eqs. 5, 12
    firm.adaptive_price,  # price setting, eq. 8
    household.income_by_status,  # income refresh, eq. 49
    household.propensity_consumption,  # consumption and savings, eqs. 40, 50
    firm.market_share_profit,  # goods market, eqs. 1-2, 27
    firm.retained_profit,  # eq. 31
    macro.government_spending,  # AR(1), eq. 51
    macro.taylor_rule,  # Taylor rule, eq. 69
]


def world(links: tuple = (), **populations) -> Model:
    """A fresh graph per call — a cold fit advances the one it is given.

    A class without a shipped template gets one declaring exactly the frame's
    columns; *links* names those whose values are ids.
    """
    model = Model()
    for klass, df in populations.items():
        template = TEMPLATES.get(klass) or _agent(
            klass,
            columns={
                c: LINK if c in links else df.schema[c] for c in df.columns if c != "id"
            },
        )
        declared = [
            p.variable.name
            for p in template.parameters
            if p.rdf_type == RDFType.IRI and p.variable.name in df.columns
        ]
        model.map(template, df.with_iri(*(c for c in declared if c != "id")))
    return model
