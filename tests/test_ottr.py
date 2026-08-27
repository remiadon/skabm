"""Agent templates (``skabm.ottr``) — the three failures ``map_default`` could
not catch, the one thing it did that still has to work, and a drift tripwire:
the templates are hand-written, so only a test keeps them in step with the rules.
"""

import polars as pl
import pytest
from maplib import Model

from skabm.behaviour.labour import LABOUR_UPDATE_RULES, labour_params
from skabm.behaviour.params import poledna_params
from skabm.ir import analyse, rule_name
from skabm.ottr import TEMPLATES, conform, firm_template, map_populations
from skabm.rules import EX_NS, render
from skabm.schelling import (
    SCHELLING_GEO_INIT_RULES,
    SCHELLING_GEO_PARAMS,
    SCHELLING_INIT_RULES,
    SCHELLING_PARAMS,
    SCHELLING_UPDATE_RULES,
)
from skabm.simulation import DEFAULT_INIT_RULES, DEFAULT_UPDATE_RULES

FIRMS = pl.DataFrame(
    {
        "id": ["firm_0", "firm_1"],
        "alpha": [10.0, 12.0],
        "margin": [0.1, 0.2],
        "size": [3.0, 4.0],
    }
)

RULE_SETS = {
    "poledna": ((*DEFAULT_INIT_RULES, *DEFAULT_UPDATE_RULES), poledna_params),
    "schelling": (
        (*SCHELLING_INIT_RULES, *SCHELLING_UPDATE_RULES),
        SCHELLING_PARAMS,
    ),
    "schelling_geo": (
        (*SCHELLING_GEO_INIT_RULES, *SCHELLING_UPDATE_RULES),
        SCHELLING_GEO_PARAMS,
    ),
    "labour": (LABOUR_UPDATE_RULES, labour_params),
}


def predicates(model) -> set:
    return {
        p.strip("<>").rsplit(":", 1)[-1]
        for p in model.query("SELECT DISTINCT ?p WHERE { ?s ?p ?o }")["p"]
    }


def test_the_manual_path_is_maplib_and_it_names_the_missing_column():
    """The workflow the templates exist for: check a frame before simulating."""
    model = Model()
    model.map(firm_template, conform(FIRMS, firm_template))
    assert model.query(f"PREFIX ex:<{EX_NS}> SELECT ?f WHERE {{ ?f a ex:Firm }}").height
    with pytest.raises(Exception, match="alpha"):
        model.map(firm_template, conform(FIRMS.drop("alpha"), firm_template))


def test_a_misspelt_column_is_rejected_rather_than_becoming_a_predicate():
    """``map_default`` would have made ``marign`` a predicate nothing reads."""
    typo = FIRMS.rename({"margin": "marign"})
    with pytest.raises(Exception, match="(?i)margin"):
        map_populations(Model(), {"Firm": typo})


def test_a_declared_column_is_cast_to_the_type_the_rules_join_on():
    """The silent failure the templates exist for: a term that cannot join."""
    model = Model()
    map_populations(model, {"Firm": FIRMS.with_columns(pl.col("size").cast(pl.Int64))})
    joined = model.query(
        f"PREFIX def:<urn:maplib_default:> PREFIX ex:<{EX_NS}> "
        "SELECT ?f WHERE { ?f a ex:Firm ; def:size ?s . FILTER(?s > 2.5e0) }"
    )
    assert joined.height == 2

    with pytest.raises(Exception, match="(?i)conversion|cast|invalid"):
        map_populations(Model(), {"Firm": FIRMS.with_columns(size=pl.lit("large"))})


def test_columns_no_rule_mentions_are_carried_not_refused():
    model = Model()
    map_populations(model, {"Firm": FIRMS.with_columns(tech_share=pl.lit(0.4))})
    assert "tech_share" in predicates(model)


def test_a_declared_link_needs_no_mapping_order():
    """``employer`` is declared, so it is an edge however late the firms go in."""
    households = pl.DataFrame({"id": ["hh_0"], "psi": [0.9], "employer": ["firm_0"]})
    model = Model()
    map_populations(model, {"Household": households})
    map_populations(model, {"Firm": FIRMS})
    edge = model.query(
        f"PREFIX def:<urn:maplib_default:> PREFIX ex:<{EX_NS}> "
        "SELECT ?f WHERE { ?h def:employer ?f . ?f a ex:Firm }"
    )
    assert edge.height == 1


def test_an_undeclared_link_still_resolves_off_the_graph():
    """A class with no shipped template gets one from its schema, as before."""
    model = Model()
    map_populations(model, {"Firm": FIRMS})
    map_populations(
        model, {"Auditor": pl.DataFrame({"id": ["a_0"], "audits": ["firm_1"]})}
    )
    edge = model.query(
        f"PREFIX def:<urn:maplib_default:> PREFIX ex:<{EX_NS}> "
        "SELECT ?f WHERE { ?a a ex:Auditor ; def:audits ?f . ?f a ex:Firm }"
    )
    assert edge.height == 1


def test_reference_resolution_does_not_depend_on_mapping_order():
    """Which class went first used to decide what became an edge.

    An undeclared string column is a link iff its values name agents, and the
    candidates are now every id in the call rather than only the ones already
    mapped — so both orders give the same graph.
    """
    auditors = pl.DataFrame({"id": ["a_0"], "audits": ["firm_1"]})
    audits = (
        f"PREFIX def:<urn:maplib_default:> PREFIX ex:<{EX_NS}> "
        "SELECT ?a WHERE { ?a def:audits ?f . ?f a ex:Firm }"
    )
    firms_first, auditors_first = Model(), Model()
    map_populations(firms_first, {"Firm": FIRMS, "Auditor": auditors})
    map_populations(auditors_first, {"Auditor": auditors, "Firm": FIRMS})
    assert firms_first.query(audits).height == 1
    assert auditors_first.query(audits).height == 1


@pytest.mark.parametrize("name", sorted(RULE_SETS))
def test_every_predicate_the_rules_cannot_produce_is_declared(name):
    """Structure is what no rule writes, so only a population can supply it."""
    rules, params = RULE_SETS[name]
    ir = analyse(
        (rule_name(rule, i), render(rule, params)) for i, rule in enumerate(rules)
    )
    for klass, predicate in ir.structure:
        assert klass in TEMPLATES, f"{name}: no template for {klass}"
        declared = {p.variable.name for p in TEMPLATES[klass].parameters}
        assert predicate in declared, f"{name}: {klass} template lacks {predicate}"
