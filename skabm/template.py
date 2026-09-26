"""
Agent templates: the shape a population has to have to be a graph.

The point of this module is the templates, and the way to use one is maplib's
own ``Model.map``::

    from maplib import Model
    from skabm.template import firm   # also registers DataFrame.with_iri

    world = Model()
    world.map(firm, firms.with_iri())

which raises now, naming the column, instead of running twelve ticks of
nothing.  ``map_default`` could not: a template generated from the frame's own
schema accepts anything, so a missing column became a missing predicate, a
typo became a predicate no rule reads, and an ``Int64`` became a basic graph
pattern that joins with nothing.

Required parameters are what the rules read but never write: nothing in the
model can produce them.  Quantities
are ``xsd:double``, which maplib fills from any float width; an integer column
is refused at map time rather than joining with nothing at run time.  Undeclared
columns are refused too: drop them, or declare them with ``agent``.
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


class Link(str):
    """A link column's dtype: ``Link("Firm")`` names firms; ``LINK`` names anything.

    ``many=True`` for a link an agent holds several of (a route's ``via`` links).
    """

    def __new__(cls, target: str = "", many: bool = False):
        link = super().__new__(cls, target)
        link.many = many
        return link


LINK = Link()

# What each field *is*, per class: ``{class: {field: (dtype | Link, description)}}``.
# The OTTR template knows names and types; this also knows where a link lands and
# what a quantity means, which is what a reader, and ``skabm.dsl``, need.  Filled by
# ``agent``, so an ad-hoc class is described too.
SCHEMA: dict[str, dict[str, tuple]] = {}

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
    if isinstance(dtype, Link):
        return RDFType.IRI
    base = dtype.base_type() if hasattr(dtype, "base_type") else dtype
    return RDFType.Literal(_XSD.get(base, xsd.string))


def agent(
    klass: str,
    quantities: str = "",
    columns: dict | None = None,
    required: tuple = (),
    doc: dict | None = None,
) -> Template:
    """One row of a population to one agent of *klass*.

    *quantities* is a whitespace-separated list of ``xsd:double`` columns, the
    common case; *columns* maps any other name to its polars dtype, or to a
    ``Link`` when its values name other agents (``Link("Firm")``, or ``LINK`` when
    any class will do).  ``required`` names become non-optional parameters, which
    is what makes a frame lacking them raise.  *doc* says what each column means;
    with the dtypes it is recorded in ``SCHEMA``.
    """
    columns = dict.fromkeys(quantities.split(), pl.Float64) | (columns or {})
    SCHEMA[klass] = {
        name: (dtype, (doc or {}).get(name, "")) for name, dtype in columns.items()
    }
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


firm = agent(
    "Firm",
    "alpha margin size w_bar output price liquidity profit dividend delta tech_share "
    "flees_amount supply demand sales inventory gamma_d pi_d pi_c",
    {"industry": pl.String, "holds_at": Link("Bank"), "sector": Link("Sector")},
    required=("alpha", "margin", "size"),
    doc={
        "alpha": "labour productivity: output per employee (Eurostat IO table)",
        "margin": "share of revenue kept as profit",
        "size": "number of employees",
        "w_bar": "average annual wage per employee (Eurostat IO table)",
        "output": "real output this quarter, Y_i",
        "price": "price of the firm's good, P_i",
        "liquidity": "cash held; profits accumulate here",
        "profit": "profit this quarter, before dividends",
        "dividend": "share of profit paid to the owner this quarter",
        "delta": "depreciation as a share of output (Eurostat IO table)",
        "tech_share": "intermediate inputs as a share of output (Eurostat IO table)",
        "industry": "NACE industry code",
        "holds_at": "the bank the firm keeps deposits at (bank.depositors)",
        "flees_amount": "deposits pulled from its bank this step, if distressed",
        "supply": "goods on offer this step: its output plus the inventory carried in, Q^o",
        "demand": "real demand for the firm's good this step, Q^d",
        "sales": "goods sold this step, Q",
        "inventory": "unsold goods carried into the next step, S",
        "gamma_d": "growth of its quantity chosen from last step's market, gamma^d",
        "pi_d": "growth of its price chosen from last step's market (demand-pull), pi^d",
        "pi_c": "growth of its unit costs from last step's prices (cost-push), pi^c",
        "sector": "the sector (industry) whose good the firm produces",
    },
)

# CANVAS's production network (skabm.behaviour.canvas): a sector is its good.
sector = agent(
    "Sector",
    "price_index b_hh b_cf final_demand purchases demand price_weight output "
    "labour_cost material_cost capital_cost",
    required=("b_hh", "b_cf", "final_demand"),
    doc={
        "price_index": "producer price index of its good: its firms' prices weighted by "
        "their sales, P_g",
        "b_hh": "weight of its good in the households' basket (the CPI), b^HH_g",
        "b_cf": "weight of its good in capital formation (the capital price index), b^CF_g",
        "final_demand": "nominal final demand for its good this step: households, "
        "government, exports and investment",
        "purchases": "intermediate inputs its firms buy this step, real",
        "demand": "nominal demand for its good this step, final and intermediate",
        "price_weight": "sum over its firms of exp(-2 x price), which a firm's chance of "
        "being picked for its price is divided by",
        "output": "total output of its firms this step",
        "labour_cost": "the consumer price index over its good's price_index, minus 1: "
        "what wages indexed to consumer prices add to its unit costs",
        "material_cost": "the price of its intermediate inputs (each supplier's "
        "price_index weighted by its input share) over its good's price_index, minus 1",
        "capital_cost": "the capital price index over its good's price_index, minus 1",
    },
)
sector_input = agent(
    "Input",
    "share",
    {"buyer": Link("Sector"), "supplier": Link("Sector")},
    required=("share", "buyer", "supplier"),
    doc={
        "share": "share of the supplier's good in the buyer's intermediate inputs, a_sg",
        "buyer": "the sector using the good as an input, s",
        "supplier": "the sector producing the good, g",
    },
)

household = agent(
    "Household",
    "psi income wealth flees_amount",
    {"employer": Link("Firm"), "owns": Link("Firm"), "holds_at": Link("Bank")},
    required=("psi",),
    doc={
        "psi": "propensity to consume out of expected income",
        "income": "income this quarter: a wage, dividends, or unemployment benefit",
        "wealth": "deposits; saving adds to it, consumption draws on it",
        "employer": "the firm this household works for; empty if not employed",
        "owns": "the firm this household owns and receives dividends from",
        "holds_at": "the bank the household keeps deposits at (bank.depositors)",
        "flees_amount": "deposits pulled from its bank this step, if distressed",
    },
)

government = agent(
    "Government",
    "budget tax_rate",
    {"purchase_sector": pl.String},
    doc={
        "budget": "government consumption this quarter",
        "tax_rate": "tax rate (carried; no shipped rule reads it)",
        "purchase_sector": "industry the government buys from (carried)",
    },
)

central_bank = agent(
    "CentralBank",
    "policy_rate inflation_target prev_output prev_price",
    doc={
        "policy_rate": "nominal policy rate set by the Taylor rule",
        "inflation_target": "inflation target (carried; the rule reads pi_star)",
        "prev_output": "total firm output last quarter, kept for the growth gap",
        "prev_price": "mean firm price last quarter, kept for inflation",
    },
)

foreign_firm = agent(
    "ForeignFirm",
    "demand_size",
    {"source_industry": pl.String},
    required=("demand_size",),
    doc={
        "demand_size": "export demand from the rest of the world this quarter",
        "source_industry": "industry the demand is for (carried)",
    },
)

bank = agent(
    "Bank",
    "deposit_share leverage capital_ratio distressed",
    required=("deposit_share", "leverage"),
    doc={
        "deposit_share": "probability that a depositor banks here",
        "leverage": "assets over capital",
        "capital_ratio": "capital over assets; distressed below a threshold",
        "distressed": "1 when the bank is distressed, else 0",
    },
)

cell = agent(
    "Cell",
    "x y draw occupied resident rank",
    {"geometry": pl.String, "neighbor": Link("Cell", many=True)},
    doc={
        "x": "column on the grid",
        "y": "row on the grid",
        "geometry": "WKT point, for irregular spaces",
        "neighbor": "the cells next to this one (schelling.grid_neighbors, links)",
        "draw": "uniform draw ranking the vacant cells this step",
        "occupied": "people living in the cell",
        "resident": "the group of the person living in the cell",
        "rank": "position among the vacant cells, by draw",
    },
)

person = agent(
    "Person",
    "group draw share_similar rank",
    {"location": Link("Cell")},
    doc={
        "group": "the group the person belongs to",
        "location": "the cell the person lives in",
        "draw": "uniform draw ranking the people who move this step",
        "share_similar": "share of the neighbours in the same group",
        "rank": "position among the people who move, by draw",
    },
)

occupation = agent(
    "Occupation",
    "demand_init demand_final target_demand employment unemployment vacancies "
    "separations openings applications app_norm job_finding "
    "u_spell1 u_spell2 u_spell3 ltu",
    required=("demand_init", "demand_final"),
    doc={
        "demand_init": "labour demand (employed + vacancies) before any shock",
        "demand_final": "labour demand after the automation shock, d-dagger",
        "target_demand": "labour demand the occupation is moving towards this step",
        "employment": "workers employed in the occupation",
        "unemployment": "unemployed workers whose last job was in the occupation",
        "vacancies": "open vacancies",
        "separations": "workers who lost their job this step",
        "openings": "vacancies opened this step",
        "applications": "applications received this step",
        "app_norm": "sum over reachable occupations of vacancies x transition weight",
        "job_finding": "share of the occupation's unemployed hired this step",
        "u_spell1": "unemployed for one step",
        "u_spell2": "unemployed for two steps",
        "u_spell3": "unemployed for three steps",
        "ltu": "long-term unemployed: more than three steps (27 weeks)",
    },
)

edge = agent(
    "Edge",
    "weight flow",
    {"src": Link("Occupation"), "dst": Link("Occupation")},
    ("src", "dst", "weight"),
    doc={
        "weight": "probability that a job search from src targets dst",
        "flow": "unemployed workers of src hired into dst this step",
        "src": "occupation the job seeker last worked in",
        "dst": "occupation the job seeker applies to",
    },
)

clock = agent("Clock", "t", doc={"t": "time steps elapsed"})

# Traffic (skabm.behaviour.traffic).  A Route's links are many per route, so they
# are not a column: `via` maps a long (id, via) frame onto routes that
# `route` already typed.
link = agent(
    "Link",
    "length t0 capacity car busway flow time",
    {
        "src": LINK,
        "dst": LINK,
        "name": pl.String,
        "geometry": pl.String,
        "area": Link("Area", many=True),
    },
    required=("src", "dst", "length", "t0", "capacity", "car", "busway"),
    doc={
        "length": "metres",
        "t0": "free-flow travel time, seconds",
        "capacity": "vehicles per hour",
        "car": "1 if cars may use the street, 0 once closed to them",
        "busway": "1 if buses run at free-flow speed on a bus lane",
        "flow": "cars per peak hour this morning",
        "time": "congested travel time this morning, seconds (BPR)",
        "src": "road-network node the link starts at",
        "dst": "road-network node the link ends at",
        "name": "street name",
        "geometry": "WKT line",
        "area": "the areas the link touches (traffic.area_membership, template.links)",
    },
)
route = agent(
    "Route",
    "mode rank extra time open prob cum load",
    {"od": pl.String, "option": Link("Option"), "via": Link("Link", many=True)},
    required=("mode", "rank", "extra", "od", "option"),
    doc={
        "mode": "0 car, 1 bus, 2 bike, 3 walk",
        "rank": "a stable order of the routes of one origin-destination pair",
        "extra": "seconds spent off the road network (walking to and from it)",
        "time": "travel time on today's network, seconds",
        "open": "1 if every link on it admits its mode",
        "prob": "probability a replanning commuter picks this route",
        "cum": "probability summed over its OD's routes ranked up to this one",
        "od": "origin-destination pair",
        "option": "the OD x mode option this route belongs to",
        "via": "the road links the route drives, in no order (template.via)",
        "load": "commuters taking the route today",
    },
)
option = agent(
    "Option",
    "mode share0 s s0 share",
    {"od": pl.String},
    ("mode", "share0", "od"),
    doc={
        "mode": "0 car, 1 bus, 2 bike, 3 walk",
        "share0": "observed (census) share of the OD's commuters using this mode",
        "s": "sum over its open routes of exp(-theta x time)",
        "s0": "s in the base period",
        "share": "share of the OD's commuters using this mode now",
        "od": "origin-destination pair",
    },
)
commuter = agent(
    "Commuter",
    "weight u time car bus bike walk",
    # a commuter is somebody's household member going to somebody's firm: the link is
    # optional, so a transport-only world leaves it null, and a world that also runs
    # the economic rules has one graph rather than two vocabularies for one person.
    {"od": pl.String, "route": Link("Route"), "household": Link("Household")},
    required=("weight", "od", "route"),
    doc={
        "weight": "real commuters this sampled one stands for",
        "u": "uniform draw deciding tomorrow's replanning",
        "time": "travel time today, seconds",
        "car": "1 if today's mode is car",
        "bus": "1 if today's mode is bus",
        "bike": "1 if today's mode is bike",
        "walk": "1 if today's mode is walking",
        "od": "origin-destination pair",
        "route": "the route taken",
        "household": "the household this commuter belongs to",
    },
)
area = agent(
    "Area",
    "vkt",
    {"name": pl.String, "geometry": pl.String},
    ("name", "geometry"),
    doc={
        "vkt": "car kilometres per peak hour on the links touching the area",
        "name": "name a policy uses for it",
        "geometry": "WKT polygon",
    },
)


def links(field: str) -> Template:
    """A link an agent holds several of, as a long frame: one ``(id, field)`` row each."""
    return Template(
        EX.suf(field),
        [
            Parameter(Variable("id"), rdf_type=RDFType.IRI),
            Parameter(Variable(field), rdf_type=RDFType.IRI),
        ],
        [Triple(Variable("id"), DEF.suf(field), Variable(field))],
    )


via = links("via")

TEMPLATES = {
    template.iri.iri[len(EX_NS) :]: template
    for template in (
        firm,
        sector,
        sector_input,
        household,
        government,
        central_bank,
        foreign_firm,
        bank,
        cell,
        person,
        occupation,
        edge,
        clock,
        link,
        route,
        option,
        commuter,
        area,
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
