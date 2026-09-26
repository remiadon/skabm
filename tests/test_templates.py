"""Agent templates (``skabm.template``) — the three failures ``map_default``
could not catch, the one thing it did that still has to work, and a drift tripwire:
the templates are hand-written, so only a test keeps them in step with the rules.
"""

import importlib
import pkgutil

import polars as pl
import pytest
from maplib import Model

import skabm.behaviour
from skabm import template
from skabm.dsl import _Rule, is_rule
from skabm.sparql import EX_NS
from skabm.template import LINK, SCHEMA, TEMPLATES, Link

FIRMS = pl.DataFrame(
    {
        "id": ["firm_0", "firm_1"],
        "alpha": [10.0, 12.0],
        "margin": [0.1, 0.2],
        "size": [3.0, 4.0],
    }
).with_iri()


def predicates(model) -> set:
    return {
        p.strip("<>").rsplit(":", 1)[-1]
        for p in model.query("SELECT DISTINCT ?p WHERE { ?s ?p ?o }")["p"]
    }


def test_the_manual_path_is_maplib_and_it_names_the_missing_column():
    """The workflow the templates exist for: check a frame before simulating."""
    model = Model()
    model.map(template.firm, FIRMS)
    assert model.query(f"PREFIX ex:<{EX_NS}> SELECT ?f WHERE {{ ?f a ex:Firm }}").height
    with pytest.raises(Exception, match="alpha"):
        model.map(template.firm, FIRMS.drop("alpha"))


def test_a_misspelt_column_is_rejected_rather_than_becoming_a_predicate():
    """``map_default`` would have made ``marign`` a predicate nothing reads."""
    with pytest.raises(Exception, match="margin"):
        Model().map(template.firm, FIRMS.rename({"margin": "marign"}))


def test_any_float_width_joins_and_an_integer_is_refused_by_name():
    """The silent failure the types exist for: a term that cannot join.

    ``xsd:double`` takes every float width, so no cast is owed for those; an
    ``xsd:long`` would never equal the doubles the rules write, so maplib
    refuses it at map time, naming the column.
    """
    model = Model()
    model.map(template.firm, FIRMS.with_columns(pl.col("size").cast(pl.Float32)))
    joined = model.query(
        f"PREFIX def:<urn:maplib_default:> PREFIX ex:<{EX_NS}> "
        "SELECT ?f WHERE { ?f a ex:Firm ; def:size ?s . FILTER(?s = 3e0) }"
    )
    assert joined.height == 1

    with pytest.raises(Exception, match="size"):
        Model().map(template.firm, FIRMS.with_columns(pl.col("size").cast(pl.Int64)))


def test_with_iri_mints_ids_by_position_when_a_frame_has_none():
    minted = pl.DataFrame({"x": [1.0, 2.0]}).with_iri(prefix="cell_")
    assert minted["id"].to_list() == [EX_NS + "cell_0", EX_NS + "cell_1"]
    # an existing id is prefixed, never replaced; an Enum or int id is fine too
    kept = pl.DataFrame({"id": [7], "link": pl.Series(["a"], dtype=pl.Enum(["a"]))})
    assert kept.with_iri("link").row(0) == (EX_NS + "7", EX_NS + "a")


def test_a_declared_link_needs_no_mapping_order():
    """``employer`` is declared, so it is an edge however late the firms go in."""
    households = pl.DataFrame(
        {"id": ["hh_0", "hh_1"], "psi": [0.9, 0.9], "employer": ["firm_0", None]}
    ).with_iri("employer")
    model = Model()
    model.map(template.household, households)
    model.map(template.firm, FIRMS)
    edge = model.query(
        f"PREFIX def:<urn:maplib_default:> PREFIX ex:<{EX_NS}> "
        "SELECT ?f WHERE { ?h def:employer ?f . ?f a ex:Firm }"
    )
    assert edge.height == 1  # the null employer stayed null, not an IRI


def test_an_undeclared_link_is_named_not_guessed():
    """A class with no shipped template declares its own; a column of ids
    becomes an edge because the caller says so."""
    auditors = pl.DataFrame({"id": ["a_0"], "audits": ["firm_1"]})
    model = Model()
    model.map(template.firm, FIRMS)
    model.map(
        template.agent("Auditor", columns={"audits": LINK}),
        auditors.with_iri("audits"),
    )
    edge = model.query(
        f"PREFIX def:<urn:maplib_default:> PREFIX ex:<{EX_NS}> "
        "SELECT ?f WHERE { ?a a ex:Auditor ; def:audits ?f . ?f a ex:Firm }"
    )
    assert edge.height == 1


def test_every_template_field_says_what_it_is():
    """What a reader, and an agent writing rules, is told about a class.

    A description per field, and a link's target a class that exists.
    """
    blank = {k: [f for f, (_, doc) in SCHEMA[k].items() if not doc] for k in TEMPLATES}
    assert {k: v for k, v in blank.items() if v} == {}
    targets = {
        str(kind)
        for klass in TEMPLATES
        for kind, _ in SCHEMA[klass].values()
        if isinstance(kind, Link) and kind
    }
    assert targets <= set(TEMPLATES)


def test_every_cited_value_is_read_and_has_one_value():
    """A module's PARAMETERS gives only what its rules read, and a name shared by
    several modules (``vat_rate``, ``dividend_ratio``) has one value everywhere."""
    for info in pkgutil.iter_modules(skabm.behaviour.__path__):
        module = importlib.import_module(f"skabm.behaviour.{info.name}")
        rules = [
            rule
            for value in vars(module).values()
            for rule in (value if isinstance(value, list) else [value])
            if is_rule(rule)
        ]
        cited = {str(k) for k in getattr(module, "PARAMETERS", {})}
        assert cited <= set().union(*(_Rule(r).parameters() for r in rules)), info.name
    assert skabm.behaviour.defaults()  # raises on a name with two values
