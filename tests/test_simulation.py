"""RDFSimulator: SPARQL update rules advance a small in-memory model.

No Eurostat access — populations are tiny hand-written DataFrames passed
directly to fit()/fit_iter() as keyword arguments (auto-generated
templates, predicates = column names).  Rules are the default
string.Template objects; test-specific numbers come in through the params
dict, merged over rules.POLEDNA_PARAMS.  Covered mechanics: iteration
yielding per-agent state, cold-fit rebuild semantics, warm_start
continuation, upsert semantics (no duplicate state triples), the paper's
capacity cap, income by activity status, unreferenced-population warning,
and sklearn get_params/clone compatibility.
"""

import polars as pl
import pytest
from sklearn.base import clone

from skabm.rules import DEF_NS, FIRM_OWNERSHIP, FIRM_PRODUCTION, HOUSEHOLD_INCOME
from skabm.simulation import RDFSimulator

_PREFIX = f"PREFIX def:<{DEF_NS}>"


def local(iri: str) -> str:
    """Local name of a returned IRI ('<...#hh_0>' -> 'hh_0'), so tests never
    spell out the model namespace."""
    return iri.rsplit("#", 1)[-1].rstrip(">")


FIRMS = pl.DataFrame(
    {
        "id": ["firm_0", "firm_1"],  # bare local names throughout
        "size": [10, 20],
        "alpha": [100.0, 200.0],
        "w_bar": [30.0, 40.0],
        "tech_share": [0.4, 0.5],
        "output": [900.0, 3600.0],  # below labor capacity alpha * size
        "price": [1.0, 1.0],
        "profit": [50.0, -10.0],
        "margin": [0.1, 0.05],
        "liquidity": [10.0, 10.0],
    }
)
HOUSEHOLDS = pl.DataFrame(
    {
        "id": [f"hh_{i}" for i in range(3)],  # all bare local names
        "employer": ["firm_0", None, None],  # worker; resolves to the firm
        "owns": [None, "firm_1", None],  # investor; hh_2 unemployed
        "psi": [0.9, 0.9, 0.9],
    }
)
CENTRAL_BANK = pl.DataFrame(
    {
        "id": ["cb"],
        "policy_rate": [0.01],
        "inflation_target": [0.005],
        "prev_output": [4500.0],
        "prev_price": [1.0],
    }
)

# overrides merged over rules.POLEDNA_PARAMS at fit time
PARAMS = {
    "dividend_ratio": 0.8,
    "benefit_replacement": 0.4,
    "total_deposits": 3000.0,
    "vat_rate": 0.15,
    "growth_e": 0.01,
    "rho": 0.9,
    "r_star": 0.0,
    "xi_pi": 0.5,
    "xi_gamma": 0.5,
}


def gdp(state: pl.DataFrame) -> float:
    return state.select(
        (pl.col("price") * pl.col("output") * (1 - pl.col("tech_share"))).sum()
    ).item()


def test_fit_iter_yields_state_per_tick():
    sim = RDFSimulator(params=PARAMS, n_periods=4)

    gdp_path = [
        gdp(state)
        for state in sim.fit_iter(
            Firm=FIRMS, Household=HOUSEHOLDS, CentralBank=CENTRAL_BANK
        )
    ]
    assert len(gdp_path) == 4
    # below the capacity cap, GDP = sum P*Y*(1 - tech_share) grows every tick
    assert gdp_path == sorted(gdp_path) and gdp_path[0] < gdp_path[-1]

    # a cold refit rebuilds the world from scratch: same trajectory again
    gdp_path_2 = [
        gdp(state)
        for state in sim.fit_iter(
            Firm=FIRMS, Household=HOUSEHOLDS, CentralBank=CENTRAL_BANK
        )
    ]
    assert gdp_path_2 == gdp_path


def test_warm_start_continues_the_world():
    sim = RDFSimulator(params=PARAMS, n_periods=2).fit(
        Firm=FIRMS, Household=HOUSEHOLDS, CentralBank=CENTRAL_BANK
    )
    gdp_after_cold = gdp(
        sim.model_.query(
            f"{_PREFIX} SELECT ?price ?output ?tech_share WHERE "
            "{ ?f def:price ?price ; def:output ?output ; def:tech_share ?tech_share }"
        )
    )

    warm = RDFSimulator(params=PARAMS, n_periods=2, warm_start=True)
    warm.model_ = sim.model_
    states = list(warm.fit_iter())
    assert len(states) == 2
    assert gdp(states[-1]) > gdp_after_cold  # advanced beyond the cold fit

    with pytest.raises(ValueError, match="warm_start"):
        warm.fit(Firm=FIRMS)


def test_default_params_present():
    # no params given: the published Table 2 values run out of the box
    sim = RDFSimulator(n_periods=1).fit(
        Firm=FIRMS, Household=HOUSEHOLDS, CentralBank=CENTRAL_BANK
    )
    wealth = sim.model_.query(f"{_PREFIX} SELECT ?h ?w WHERE {{ ?h def:wealth ?w }}")
    assert wealth.height == 3


def test_default_rules_scoped_to_passed_kinds():
    # only firms: default rules anchored on absent classes (Household,
    # CentralBank) match nothing and no-op — SPARQL pattern matching scopes
    # the full default rule set to the kinds actually mapped
    sim = RDFSimulator(params=PARAMS, n_periods=2).fit(Firm=FIRMS)

    outputs = sim.model_.query(f"{_PREFIX} SELECT ?f ?y WHERE {{ ?f def:output ?y }}")
    assert outputs.height == 2
    assert (outputs["y"] > FIRMS["output"]).all()  # production rule did run
    incomes = sim.model_.query(f"{_PREFIX} SELECT ?h ?i WHERE {{ ?h def:income ?i }}")
    assert incomes.height == 0  # no household rules without Household=


def test_unreferenced_population_warns():
    sim = RDFSimulator(params=PARAMS, n_periods=1)
    ghosts = pl.DataFrame({"id": ["ghost_0"], "x": [1.0]})
    with pytest.warns(UserWarning, match="Ghost"):
        sim.fit(
            Firm=FIRMS, Household=HOUSEHOLDS, CentralBank=CENTRAL_BANK, Ghost=ghosts
        )


def test_upserts_do_not_duplicate_state():
    sim = RDFSimulator(params=PARAMS, n_periods=3).fit(
        Firm=FIRMS, Household=HOUSEHOLDS, CentralBank=CENTRAL_BANK
    )

    m = sim.model_
    # exactly one state triple per agent after repeated upserts
    assert m.query(f"{_PREFIX} SELECT ?h ?i WHERE {{ ?h def:income ?i }}").height == 3
    assert m.query(f"{_PREFIX} SELECT ?h ?w WHERE {{ ?h def:wealth ?w }}").height == 3
    assert m.query(f"{_PREFIX} SELECT ?f ?y WHERE {{ ?f def:output ?y }}").height == 2
    assert m.query(f"{_PREFIX} SELECT ?f ?p WHERE {{ ?f def:price ?p }}").height == 2


def test_production_respects_labor_capacity():
    sim = RDFSimulator(
        init_rules=[],
        update_rules=[FIRM_PRODUCTION],
        params={"growth_e": 0.5},
        n_periods=10,
    ).fit(Firm=FIRMS)

    outputs = sim.model_.query(
        f"{_PREFIX} SELECT ?f ?y ?alpha ?n "
        "WHERE { ?f def:output ?y ; def:alpha ?alpha ; def:size ?n }"
    )
    assert (outputs["y"] <= outputs["alpha"] * outputs["n"] + 1e-9).all()


def test_income_by_activity_status():
    sim = RDFSimulator(params=PARAMS, n_periods=1).fit(
        Firm=FIRMS, Household=HOUSEHOLDS, CentralBank=CENTRAL_BANK
    )

    income = {
        local(row["h"]): row["i"]
        for row in sim.model_.query(
            f"{_PREFIX} SELECT ?h ?i WHERE {{ ?h def:income ?i }}"
        ).to_dicts()
    }
    assert income["hh_0"] == 30.0  # worker: employer's wage
    assert income["hh_1"] >= 0.0  # investor: dividend clipped at 0
    # unemployed: benefit_replacement * average wage = 0.4 * 35
    assert income["hh_2"] == 0.4 * 35.0


# FIRM_OWNERSHIP references households via the CONCAT'd full IRI, not an
# ex:Household anchor, so the inert-population heuristic can't see it; the
# household population is not actually inert (it receives the owns triples).
@pytest.mark.filterwarnings("ignore:population 'Household'")
def test_firm_ownership_assigned_in_graph():
    # households carry NO owns column — FIRM_OWNERSHIP assigns it in-graph.
    # ratio = n_firms / n_households puts one distinct owner on each firm.
    households = pl.DataFrame({"id": [f"hh_{i}" for i in range(6)], "psi": [0.9] * 6})
    sim = RDFSimulator(
        init_rules=[FIRM_OWNERSHIP],
        update_rules=[],
        params={"firm_ownership_ratio": 2 / 6},
        n_periods=0,
    ).fit(Firm=FIRMS, Household=households)

    owns = {
        (local(r["h"]), local(r["f"]))
        for r in sim.model_.query(
            f"{_PREFIX} SELECT ?h ?f WHERE {{ ?h def:owns ?f }}"
        ).to_dicts()
    }
    # firm j -> household floor(j / ratio): firm_0 -> hh_0, firm_1 -> hh_3
    assert owns == {("hh_0", "firm_0"), ("hh_3", "firm_1")}


@pytest.mark.filterwarnings("ignore:population 'Household'")
def test_firm_ownership_preserves_data_defined_owner():
    # a data-defined owner (firm_1 owned by hh_5) must survive FILTER NOT EXISTS
    households = pl.DataFrame(
        {
            "id": [f"hh_{i}" for i in range(6)],
            "owns": [None, None, None, None, None, "firm_1"],  # bare reference
            "psi": [0.9] * 6,
        }
    )
    sim = RDFSimulator(
        init_rules=[FIRM_OWNERSHIP],
        update_rules=[],
        params={"firm_ownership_ratio": 2 / 6},
        n_periods=0,
    ).fit(Firm=FIRMS, Household=households)

    owners = {
        (local(r["h"]), local(r["f"]))
        for r in sim.model_.query(
            f"{_PREFIX} SELECT ?h ?f WHERE {{ ?h def:owns ?f }}"
        ).to_dicts()
    }
    assert ("hh_5", "firm_1") in owners  # data-defined owner survived
    assert sum(f == "firm_1" for _, f in owners) == 1  # not overwritten


def test_class_free_rule_survives_kind_filter():
    # HOUSEHOLD_WEALTH names no ex:Class (it reads derived def:income); the
    # fit-time filter must keep it, else household wealth never materializes
    sim = RDFSimulator(params=PARAMS, n_periods=1).fit(Firm=FIRMS, Household=HOUSEHOLDS)
    wealth = sim.model_.query(f"{_PREFIX} SELECT ?h ?w WHERE {{ ?h def:wealth ?w }}")
    assert wealth.height == 3
    assert (wealth["w"] > 0).any()


@pytest.mark.filterwarnings("ignore:population 'Firm'")
def test_bare_link_resolves_from_model():
    # both id and the employer link are bare local names; map_df recognizes
    # "firm_0" as a reference (a firm with that id was mapped first) and
    # prefixes it, so the income rule traverses employer -> firm.
    firms = pl.DataFrame({"id": ["firm_0"], "w_bar": [30.0]})
    households = pl.DataFrame({"id": ["hh_0"], "employer": ["firm_0"], "psi": [0.9]})
    sim = RDFSimulator(init_rules=[HOUSEHOLD_INCOME], update_rules=[], n_periods=0).fit(
        Firm=firms, Household=households
    )
    income = sim.model_.query(f"{_PREFIX} SELECT ?i WHERE {{ ?h def:income ?i }}")
    assert income["i"].to_list() == [30.0]


def test_get_params_and_clone():
    sim = RDFSimulator(params=PARAMS, n_periods=7)
    params = sim.get_params()
    assert set(params) == {
        "init_rules",
        "update_rules",
        "params",
        "n_periods",
        "warm_start",
    }
    assert params["n_periods"] == 7
    assert params["params"]["total_deposits"] == 3000.0

    # clone deep-copies params; Template lacks __eq__, so compare the text
    fresh = clone(sim)
    fresh_params = fresh.get_params()
    assert fresh_params["params"] == PARAMS and fresh_params["n_periods"] == 7
    assert [t.template for t in fresh_params["update_rules"]] == [
        t.template for t in params["update_rules"]
    ]
    assert fresh.model_.query("SELECT ?s WHERE { ?s ?p ?o }").height == 0
