"""Agent templates (``skabm.ottr``) — the three failures ``map_default``
could not catch, the one thing it did that still has to work, and a drift tripwire:
the templates are hand-written, so only a test keeps them in step with the rules.
"""

import importlib
import pkgutil
from string import Template

import polars as pl
import pytest
from maplib import Model

import skabm.behaviour
from skabm.behaviour import DefaultTemplate
from skabm.behaviour.labour import LABOUR_UPDATE_RULES
from skabm.behaviour.schelling import (
    SCHELLING_GEO_INIT_RULES,
    SCHELLING_INIT_RULES,
    SCHELLING_UPDATE_RULES,
)
from skabm.ir import analyse, rule_name
from skabm.ottr import (
    LINK,
    TEMPLATES,
    agent_template,
    firm_template,
    household_template,
)
from skabm.simulation import DEFAULT_INIT_RULES, DEFAULT_UPDATE_RULES
from skabm.sparql import EX_NS, render

FIRMS = pl.DataFrame(
    {
        "id": ["firm_0", "firm_1"],
        "alpha": [10.0, 12.0],
        "margin": [0.1, 0.2],
        "size": [3.0, 4.0],
    }
).with_iri()

RULE_SETS = {
    "poledna": (*DEFAULT_INIT_RULES, *DEFAULT_UPDATE_RULES),
    "schelling": (*SCHELLING_INIT_RULES, *SCHELLING_UPDATE_RULES),
    "schelling_geo": (*SCHELLING_GEO_INIT_RULES, *SCHELLING_UPDATE_RULES),
    "labour": LABOUR_UPDATE_RULES,
}


def predicates(model) -> set:
    return {
        p.strip("<>").rsplit(":", 1)[-1]
        for p in model.query("SELECT DISTINCT ?p WHERE { ?s ?p ?o }")["p"]
    }


def test_the_manual_path_is_maplib_and_it_names_the_missing_column():
    """The workflow the templates exist for: check a frame before simulating."""
    model = Model()
    model.map(firm_template, FIRMS)
    assert model.query(f"PREFIX ex:<{EX_NS}> SELECT ?f WHERE {{ ?f a ex:Firm }}").height
    with pytest.raises(Exception, match="alpha"):
        model.map(firm_template, FIRMS.drop("alpha"))


def test_a_misspelt_column_is_rejected_rather_than_becoming_a_predicate():
    """``map_default`` would have made ``marign`` a predicate nothing reads."""
    with pytest.raises(Exception, match="margin"):
        Model().map(firm_template, FIRMS.rename({"margin": "marign"}))


def test_any_float_width_joins_and_an_integer_is_refused_by_name():
    """The silent failure the types exist for: a term that cannot join.

    ``xsd:double`` takes every float width, so no cast is owed for those; an
    ``xsd:long`` would never equal the doubles the rules write, so maplib
    refuses it at map time, naming the column.
    """
    model = Model()
    model.map(firm_template, FIRMS.with_columns(pl.col("size").cast(pl.Float32)))
    joined = model.query(
        f"PREFIX def:<urn:maplib_default:> PREFIX ex:<{EX_NS}> "
        "SELECT ?f WHERE { ?f a ex:Firm ; def:size ?s . FILTER(?s = 3e0) }"
    )
    assert joined.height == 1

    with pytest.raises(Exception, match="size"):
        Model().map(firm_template, FIRMS.with_columns(pl.col("size").cast(pl.Int64)))


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
    model.map(household_template, households)
    model.map(firm_template, FIRMS)
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
    model.map(firm_template, FIRMS)
    model.map(
        agent_template("Auditor", columns={"audits": LINK}),
        auditors.with_iri("audits"),
    )
    edge = model.query(
        f"PREFIX def:<urn:maplib_default:> PREFIX ex:<{EX_NS}> "
        "SELECT ?f WHERE { ?a a ex:Auditor ; def:audits ?f . ?f a ex:Firm }"
    )
    assert edge.height == 1


@pytest.mark.parametrize("name", sorted(RULE_SETS))
def test_every_predicate_the_rules_cannot_produce_is_declared(name):
    """Structure is what no rule writes, so only a population can supply it."""
    rules = RULE_SETS[name]
    ir = analyse((rule_name(rule, i), render(rule, {})) for i, rule in enumerate(rules))
    for klass, predicate in ir.structure:
        assert klass in TEMPLATES, f"{name}: no template for {klass}"
        declared = {p.variable.name for p in TEMPLATES[klass].parameters}
        assert predicate in declared, f"{name}: {klass} template lacks {predicate}"


def behaviour_rules() -> dict[str, Template]:
    """Every rule template shipped in ``skabm.behaviour``, by qualified name."""
    rules = {}
    for info in pkgutil.iter_modules(skabm.behaviour.__path__):
        module = importlib.import_module(f"skabm.behaviour.{info.name}")
        for name, value in vars(module).items():
            for i, rule in enumerate(value if isinstance(value, list) else [value]):
                if isinstance(rule, Template):
                    rules[f"{info.name}.{name}[{i}]"] = rule
    return rules


def test_behaviour_rules_carry_consistent_defaults():
    """A rule is a DefaultTemplate, and a placeholder means one number everywhere.

    Defaults are written on each rule next to their citation, so a value shared
    by several rules (``vat_rate``, ``theta``) is written more than once; this is
    what keeps the copies from drifting apart.
    """
    rules = behaviour_rules()
    plain = [
        name for name, rule in rules.items() if not isinstance(rule, DefaultTemplate)
    ]
    assert not plain, f"plain string.Template rules: {plain}"

    seen: dict[str, set] = {}
    for rule in rules.values():
        assert set(rule.default) <= set(rule.get_identifiers()), rule.default
        for key, value in rule.default.items():
            seen.setdefault(key, set()).add(value)
    assert {k: v for k, v in seen.items() if len(v) > 1} == {}


def test_default_template_substitutes_its_defaults_as_doubles():
    rule = DefaultTemplate("FILTER(?x < $a && ?y < $b)", {"a": 0.5})
    assert rule.substitute(b=2) == "FILTER(?x < 5.000000e-01 && ?y < 2.000000e+00)"
    assert rule.substitute({"a": 1.0, "b": 2.0}).startswith("FILTER(?x < 1.000000e+00")
    assert rule.safe_substitute() == "FILTER(?x < 5.000000e-01 && ?y < $b)"
    with pytest.raises(KeyError, match="b"):
        rule.substitute()
