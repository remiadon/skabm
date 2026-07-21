"""RDFSimulator: SPARQL update rules advance a small in-memory model.

No Eurostat access — populations are tiny hand-written DataFrames passed
directly to fit()/fit_iter() as keyword arguments (auto-generated
templates, predicates = column names).  Covered mechanics: iteration
yielding per-agent state, cold-fit rebuild semantics, warm_start
continuation, upsert semantics (no duplicate state triples), the paper's
capacity cap, income by activity status, unreferenced-population warning,
and sklearn get_params/clone compatibility.
"""

import polars as pl
import pytest
from sklearn.base import clone

from skabm.rules import (
    DEF_NS,
    EX_NS,
    firm_pricing,
    firm_production,
    household_income,
    household_income_update,
    household_update,
    household_wealth,
    taylor_rule,
)
from skabm.simulation import RDFSimulator

_PREFIX = f"PREFIX def:<{DEF_NS}>"

FIRMS = pl.DataFrame(
    {
        "id": [EX_NS + "firm_0", EX_NS + "firm_1"],
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
        "id": [EX_NS + f"hh_{i}" for i in range(3)],
        "employer": [EX_NS + "firm_0", None, None],  # worker
        "owns": [None, EX_NS + "firm_1", None],  # investor; hh_2 unemployed
        "psi": [0.9, 0.9, 0.9],
    }
)
CENTRAL_BANK = pl.DataFrame(
    {
        "id": [EX_NS + "cb"],
        "policy_rate": [0.01],
        "inflation_target": [0.005],
        "prev_output": [4500.0],
        "prev_price": [1.0],
    }
)

INIT_RULES = [household_income(0.8, 0.4), household_wealth(3000.0)]
UPDATE_RULES = [
    firm_production(0.01),
    firm_pricing(0.005),
    household_income_update(0.8, 0.4),
    household_update(0.15),
    taylor_rule(rho=0.9, r_star=0.0, pi_star=0.005, xi_pi=0.5, xi_gamma=0.5),
]


def gdp(state: pl.DataFrame) -> float:
    return state.select(
        (pl.col("price") * pl.col("output") * (1 - pl.col("tech_share"))).sum()
    ).item()


def test_fit_iter_yields_state_per_tick():
    sim = RDFSimulator(init_rules=INIT_RULES, update_rules=UPDATE_RULES, n_periods=4)

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
    sim = RDFSimulator(
        init_rules=INIT_RULES, update_rules=UPDATE_RULES, n_periods=2
    ).fit(Firm=FIRMS, Household=HOUSEHOLDS, CentralBank=CENTRAL_BANK)
    gdp_after_cold = gdp(
        sim.model_.query(
            f"{_PREFIX} SELECT ?price ?output ?tech_share WHERE "
            "{ ?f def:price ?price ; def:output ?output ; def:tech_share ?tech_share }"
        )
    )

    warm = RDFSimulator(update_rules=UPDATE_RULES, n_periods=2, warm_start=True)
    warm.model_ = sim.model_
    states = list(warm.fit_iter())
    assert len(states) == 2
    assert gdp(states[-1]) > gdp_after_cold  # advanced beyond the cold fit

    with pytest.raises(ValueError, match="warm_start"):
        warm.fit(Firm=FIRMS)


def test_default_rules_present():
    # no rules given: the Poledna defaults run out of the box
    sim = RDFSimulator(n_periods=1).fit(
        Firm=FIRMS, Household=HOUSEHOLDS, CentralBank=CENTRAL_BANK
    )
    wealth = sim.model_.query(f"{_PREFIX} SELECT ?h ?w WHERE {{ ?h def:wealth ?w }}")
    assert wealth.height == 3


def test_unreferenced_population_warns():
    sim = RDFSimulator(init_rules=INIT_RULES, update_rules=UPDATE_RULES, n_periods=1)
    ghosts = pl.DataFrame({"id": [EX_NS + "ghost_0"], "x": [1.0]})
    with pytest.warns(UserWarning, match="Ghost"):
        sim.fit(
            Firm=FIRMS, Household=HOUSEHOLDS, CentralBank=CENTRAL_BANK, Ghost=ghosts
        )


def test_upserts_do_not_duplicate_state():
    sim = RDFSimulator(
        init_rules=INIT_RULES, update_rules=UPDATE_RULES, n_periods=3
    ).fit(Firm=FIRMS, Household=HOUSEHOLDS, CentralBank=CENTRAL_BANK)

    m = sim.model_
    # exactly one state triple per agent after repeated upserts
    assert m.query(f"{_PREFIX} SELECT ?h ?i WHERE {{ ?h def:income ?i }}").height == 3
    assert m.query(f"{_PREFIX} SELECT ?h ?w WHERE {{ ?h def:wealth ?w }}").height == 3
    assert m.query(f"{_PREFIX} SELECT ?f ?y WHERE {{ ?f def:output ?y }}").height == 2
    assert m.query(f"{_PREFIX} SELECT ?f ?p WHERE {{ ?f def:price ?p }}").height == 2


def test_production_respects_labor_capacity():
    sim = RDFSimulator(
        init_rules=[], update_rules=[firm_production(0.5)], n_periods=10
    ).fit(Firm=FIRMS)

    outputs = sim.model_.query(
        f"{_PREFIX} SELECT ?f ?y ?alpha ?n "
        "WHERE { ?f def:output ?y ; def:alpha ?alpha ; def:size ?n }"
    )
    assert (outputs["y"] <= outputs["alpha"] * outputs["n"] + 1e-9).all()


def test_income_by_activity_status():
    sim = RDFSimulator(
        init_rules=INIT_RULES, update_rules=UPDATE_RULES, n_periods=1
    ).fit(Firm=FIRMS, Household=HOUSEHOLDS, CentralBank=CENTRAL_BANK)

    income = {
        row["h"]: row["i"]
        for row in sim.model_.query(
            f"{_PREFIX} SELECT ?h ?i WHERE {{ ?h def:income ?i }}"
        ).to_dicts()
    }
    assert income[f"<{EX_NS}hh_0>"] == 30.0  # worker: employer's wage
    assert income[f"<{EX_NS}hh_1>"] >= 0.0  # investor: dividend clipped at 0
    # unemployed: benefit_replacement * average wage = 0.4 * 35
    assert income[f"<{EX_NS}hh_2>"] == 0.4 * 35.0


def test_get_params_and_clone():
    sim = RDFSimulator(init_rules=INIT_RULES, update_rules=UPDATE_RULES, n_periods=7)
    params = sim.get_params()
    assert set(params) == {"init_rules", "update_rules", "n_periods", "warm_start"}
    assert params["n_periods"] == 7

    fresh = clone(sim)
    assert fresh.get_params() == params
    assert fresh.model_.query("SELECT ?s WHERE { ?s ?p ?o }").height == 0
