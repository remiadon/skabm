"""
Agent templates: the shape a population has to have to be a graph.

The point of this module is the templates, and the way to use one is maplib's
own ``Model.map``::

    from maplib import Model
    from skabm.ottr import firm_template   # also registers DataFrame.with_iri

    world = Model()
    world.map(firm_template, firms.with_iri())

which raises now, naming the column, instead of running twelve ticks of
nothing.  ``map_default`` could not: a template generated from the frame's own
schema accepts anything, so a missing column became a missing predicate, a
typo became a predicate no rule reads, and an ``Int64`` became a basic graph
pattern that joins with nothing.

Required parameters are what the rules read but never write
(``ir.ModelIR.structure``) — nothing in the model can produce them.  Quantities
are ``xsd:double``, which maplib fills from any float width; an integer column
is refused at map time rather than joining with nothing at run time.  Undeclared
columns are refused too: drop them, or declare them with ``agent_template``.
"""

from __future__ import annotations

import polars as pl
from maplib import (
    IRI,
    Parameter,
    Prefix,
    RDFType,
    Template,
    Triple,
    Variable,
    xsd,
)

from skabm.sparql import DEF_NS, EX_NS

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
    "alpha margin size w_bar output price liquidity profit dividend delta tech_share",
    {"industry": pl.String},
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

# Traffic (skabm.behaviour.traffic).  A Route's links are many per route, so they
# are not a column: `via_template` maps a long (id, via) frame onto routes that
# `route_template` already typed.
link_template = agent_template(
    "Link",
    "length t0 capacity car busway flow time",
    {"src": LINK, "dst": LINK, "name": pl.String, "geometry": pl.String},
    required=("src", "dst", "length", "t0", "capacity", "car", "busway"),
)
route_template = agent_template(
    "Route",
    "mode rank extra time open prob cum",
    {"od": pl.String, "option": LINK},
    required=("mode", "rank", "extra", "od", "option"),
)
option_template = agent_template(
    "Option", "mode share0 s s0 share", {"od": pl.String}, ("mode", "share0", "od")
)
commuter_template = agent_template(
    "Commuter",
    "weight u time car bus bike walk",
    # a commuter is somebody's household member going to somebody's firm: the link is
    # optional, so a transport-only world leaves it null, and a world that also runs
    # the economic rules has one graph rather than two vocabularies for one person.
    {"od": pl.String, "route": LINK, "household": LINK},
    required=("weight", "od", "route"),
)
area_template = agent_template(
    "Area", "vkt", {"name": pl.String, "geometry": pl.String}, ("name", "geometry")
)
via_template = Template(
    EX.suf("via"),
    [
        Parameter(Variable("id"), rdf_type=RDFType.IRI),
        Parameter(Variable("via"), rdf_type=RDFType.IRI),
    ],
    [Triple(Variable("id"), DEF.suf("via"), Variable("via"))],
)

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
        link_template,
        route_template,
        option_template,
        commuter_template,
        area_template,
    )
}


@pl.api.register_dataframe_namespace("with_iri")
class WithIri:
    """``df.with_iri(*links)``: the frame ``Model.map`` takes, ids and links as ``ex:`` IRIs.

    A frame without an ``id`` gets one from its row position, as ``with_row_index``
    would; *prefix* keeps two such classes apart, since ``ex:0`` is one agent
    whichever frame minted it.  Nulls in a link stay null.
    """

    def __init__(self, df: pl.DataFrame):
        self._df = df

    def __call__(self, *links: str, prefix: str = "") -> pl.DataFrame:
        df = self._df
        if "id" not in df.columns:
            df = df.with_row_index("id").with_columns(
                pl.format(prefix + "{}", "id").alias("id")
            )
        return df.with_columns(
            (EX_NS + pl.col("id", *links).cast(pl.String)).name.keep()
        )
