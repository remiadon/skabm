"""
Agent templates: the shape a population has to have to be a graph.

The point of this module is the templates, and the way to use one is maplib's
own ``Model.map``::

    from maplib import Model
    from skabm.ottr import conform, firm_template

    Model().map(firm_template, conform(firms, firm_template))

which raises now, naming the column, instead of running twelve ticks of
nothing.  ``map_default`` could not: a template generated from the frame's own
schema accepts anything, so a missing column became a missing predicate, a
typo became a predicate no rule reads, and an ``Int64`` became a basic graph
pattern that joins with nothing.

Required parameters are what the rules read but never write
(``ir.ModelIR.structure``) — nothing in the model can produce them.  Declared
ones are cast by ``conform`` to the type the rules join on, which is the
failure worth the most.  Undeclared columns are refused, so ``extend`` widens a
template with what a frame actually carries when you want the rest mapped too.
"""

from __future__ import annotations

import polars as pl
from maplib import (
    IRI,
    Model,
    Parameter,
    Prefix,
    RDFType,
    Template,
    Triple,
    Variable,
    xsd,
)

from skabm.rules import DEF_NS, EX_NS

EX = Prefix(EX_NS)
DEF = Prefix(DEF_NS)
RDF_TYPE = IRI("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")

LINK = "link"

_XSD = {
    pl.Boolean: xsd.boolean,
    pl.Date: xsd.date,
    pl.Datetime: xsd.dateTime,
    pl.Duration: xsd.duration,
    pl.Float32: xsd.float,
    pl.Float64: xsd.double,
    pl.Int8: xsd.byte,
    pl.Int16: xsd.short,
    pl.Int32: xsd.int_,
    pl.Int64: xsd.long,
    pl.String: xsd.string,
    pl.UInt8: xsd.unsignedByte,
    pl.UInt16: xsd.unsignedShort,
    pl.UInt32: xsd.unsignedInt,
    pl.UInt64: xsd.unsignedLong,
}


def _rdf_type(dtype) -> RDFType:
    """An IRI for a link, else the column's XSD datatype."""
    if dtype is LINK:
        return RDFType.IRI
    base = dtype.base_type() if hasattr(dtype, "base_type") else dtype
    return RDFType.Literal(_XSD.get(base, xsd.string))


def _polars_type(rdf_type: RDFType):
    """The polars dtype a parameter's values are cast to, or None."""
    if rdf_type == RDFType.IRI:
        return pl.String
    for dtype, iri in _XSD.items():
        if rdf_type == RDFType.Literal(iri):
            return dtype
    return None


def agent_template(
    klass: str, quantities: str = "", columns: dict | None = None, required: tuple = ()
) -> Template:
    """One row of a population to one agent of *klass*.

    *quantities* is a whitespace-separated list of ``xsd:double`` columns, the
    common case; *columns* maps any other name to its polars dtype, or to
    ``LINK`` when its values name other agents.  ``required`` names become
    non-optional parameters, which is what makes a frame lacking them raise.
    """
    columns = dict.fromkeys(quantities.split(), pl.Float64) | (columns or {})
    parameters = [Parameter(Variable("id"), rdf_type=RDFType.IRI)]
    instances = [Triple(Variable("id"), RDF_TYPE, EX.suf(klass))]
    for name, dtype in columns.items():
        parameters.append(
            Parameter(
                Variable(name),
                optional=name not in required,
                rdf_type=_rdf_type(dtype),
            )
        )
        instances.append(Triple(Variable("id"), DEF.suf(name), Variable(name)))
    return Template(EX.suf(klass), parameters, instances)


firm_template = agent_template(
    "Firm",
    "alpha margin size w_bar output price liquidity profit dividend",
    required=("alpha", "margin", "size"),
)

household_template = agent_template(
    "Household",
    "psi income wealth",
    {"employer": LINK, "owns": LINK},
    required=("psi",),
)

government_template = agent_template(
    "Government", "budget tax_rate", {"purchase_sector": pl.String}
)

central_bank_template = agent_template(
    "CentralBank", "policy_rate inflation_target prev_output prev_price"
)

foreign_firm_template = agent_template(
    "ForeignFirm",
    "demand_size",
    {"source_industry": pl.String},
    required=("demand_size",),
)

bank_template = agent_template(
    "Bank",
    "deposit_share leverage capital_ratio",
    required=("deposit_share", "leverage"),
)

cell_template = agent_template("Cell", "x y", {"geometry": pl.String})

occupation_template = agent_template(
    "Occupation",
    "demand_init demand_final target_demand employment unemployment vacancies "
    "separations openings applications app_norm job_finding "
    "u_spell1 u_spell2 u_spell3 ltu",
    required=("demand_init", "demand_final"),
)

edge_template = agent_template(
    "Edge", "weight flow", {"src": LINK, "dst": LINK}, ("src", "dst", "weight")
)

clock_template = agent_template("Clock", "t")

TEMPLATES = {
    template.iri.iri[len(EX_NS) :]: template
    for template in (
        firm_template,
        household_template,
        government_template,
        central_bank_template,
        foreign_firm_template,
        bank_template,
        cell_template,
        occupation_template,
        edge_template,
        clock_template,
    )
}


def extend(template: Template, df: pl.DataFrame, links: tuple = ()) -> Template:
    """*template*, widened with the columns *df* carries and it does not declare.

    Undeclared columns are optional by definition and typed from the frame,
    except those in *links*, whose values name other agents.
    """
    declared = {p.variable.name for p in template.parameters}
    extra = {c: df.schema[c] for c in df.columns if c not in declared}
    if not extra:
        return template
    parameters = list(template.parameters)
    instances = list(template.instances)
    for name, dtype in extra.items():
        parameters.append(
            Parameter(
                Variable(name),
                optional=True,
                rdf_type=_rdf_type(LINK if name in links else dtype),
            )
        )
        instances.append(Triple(Variable("id"), DEF.suf(name), Variable(name)))
    return Template(template.iri, parameters, instances)


def conform(df: pl.DataFrame, template: Template, links: tuple = ()) -> pl.DataFrame:
    """*df* cast to what *template* declares, with its IRI columns prefixed.

    The cast is the point: basic graph patterns join on RDF terms, so an
    ``xsd:long`` never meets an ``xsd:double`` and the rule reading it matches
    nothing.  A column that cannot be cast raises here instead.
    """
    iris = {"id", *links}
    casts = []
    for parameter in template.parameters:
        name = parameter.variable.name
        if name not in df.columns:
            continue
        dtype = _polars_type(parameter.rdf_type)
        if parameter.rdf_type == RDFType.IRI:
            iris.add(name)
        elif dtype is not None and df.schema[name] != dtype:
            casts.append(pl.col(name).cast(dtype))
    return df.with_columns(
        *casts,
        *(
            pl.when(pl.col(c).is_not_null())
            .then(pl.format(EX_NS + "{}", pl.col(c)))
            .alias(c)
            for c in sorted(iris)
            if c in df.columns
        ),
    )


def graph_links(known: set, df: pl.DataFrame, declared: set) -> tuple:
    """Undeclared string columns whose values all name agents in *known*.

    Needed only where no template speaks: a declared link is a link whatever
    the data happens to hold.
    """
    found = []
    for column in df.columns:
        if column == "id" or column in declared or df.schema[column] != pl.String:
            continue
        values = {EX_NS + v for v in df[column].drop_nulls().to_list()}
        if values and values <= known:
            found.append(column)
    return tuple(found)


def map_populations(model: Model, populations: dict) -> None:
    """Map every population into *model*, resolving references across all of them.

    What ``RDFSimulator.fit`` does with a mapping rather than a model; map by hand
    when you want the contract to fail before a simulator exists.  An undeclared
    reference is a value naming an agent, and the candidates are every id in
    this call plus every agent already in the graph — resolving against the
    union is what stops which class went first from deciding what is an edge.
    """
    known = {s.strip("<>") for s in model.query("SELECT ?s WHERE { ?s a ?c }")["s"]}
    known |= {
        EX_NS + str(value)
        for df in populations.values()
        for value in df["id"].to_list()
    }
    for klass, df in populations.items():
        template = TEMPLATES.get(klass) or agent_template(klass)
        declared = {p.variable.name for p in template.parameters}
        links = graph_links(known, df, declared)
        model.map(extend(template, df, links), conform(df, template, links))
