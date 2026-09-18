"""The graph a test simulates, built the way a user would: template, ``with_iri``, ``Model.map``."""

from maplib import Model, RDFType

from skabm.templates import LINK, TEMPLATES, agent_template


def world(links: tuple = (), **populations) -> Model:
    """A fresh graph per call — a cold fit advances the one it is given.

    A class without a shipped template gets one declaring exactly the frame's
    columns; *links* names those whose values are ids.
    """
    model = Model()
    for klass, df in populations.items():
        template = TEMPLATES.get(klass) or agent_template(
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
