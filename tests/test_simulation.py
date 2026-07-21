"""RDFSimulator: SPARQL update rules advance a small in-memory model.

No Eurostat access — populations are tiny hand-written DataFrames lifted
with map_df (auto-generated templates, predicates = column names); the
point is the simulation mechanics: iteration yielding per-agent state,
upsert semantics (no duplicate state triples), the paper's capacity cap,
and income by activity status.
"""

import polars as pl
from maplib import Model

from skabm.rules import (
    DEF_NS,
    EX_NS,
    map_df,
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


def build_model() -> Model:
    firms = pl.DataFrame(
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
    households = pl.DataFrame(
        {
            "id": [EX_NS + f"hh_{i}" for i in range(3)],
            "employer": [EX_NS + "firm_0", None, None],  # worker
            "owns": [None, EX_NS + "firm_1", None],  # investor; hh_2 unemployed
            "psi": [0.9, 0.9, 0.9],
        }
    )
    central_bank = pl.DataFrame(
        {
            "id": [EX_NS + "cb"],
            "policy_rate": [0.01],
            "inflation_target": [0.005],
            "prev_output": [4500.0],
            "prev_price": [1.0],
        }
    )
    m = Model()
    map_df(m, firms, "Firm")
    map_df(m, households, "Household")
    map_df(m, central_bank, "CentralBank")
    m.insert(household_income(0.8, 0.4), transient=True)
    m.insert(household_wealth(3000.0))
    return m


RULES = [
    firm_production(0.01),
    firm_pricing(0.005),
    household_income_update(0.8, 0.4),
    household_update(0.15),
    taylor_rule(rho=0.9, r_star=0.0, pi_star=0.005, xi_pi=0.5, xi_gamma=0.5),
]


def test_iteration_yields_state_per_tick():
    sim = RDFSimulator(update_rules=RULES, n_periods=4).fit(build_model())

    gdp_path = [
        state.select(
            (pl.col("price") * pl.col("output") * (1 - pl.col("tech_share"))).sum()
        ).item()
        for state in sim
    ]
    assert len(gdp_path) == 4
    # below the capacity cap, GDP = sum P*Y*(1 - tech_share) grows every tick
    assert gdp_path == sorted(gdp_path) and gdp_path[0] < gdp_path[-1]

    # one more pass continues from the evolved state
    assert next(iter(sim)).select(pl.col("policy_rate").max()).item() >= 0


def test_upserts_do_not_duplicate_state():
    sim = RDFSimulator(update_rules=RULES, n_periods=3).fit(build_model())
    for _ in sim:
        pass

    m = sim.model_
    # exactly one state triple per agent after repeated upserts
    assert m.query(f"{_PREFIX} SELECT ?h ?i WHERE {{ ?h def:income ?i }}").height == 3
    assert m.query(f"{_PREFIX} SELECT ?h ?w WHERE {{ ?h def:wealth ?w }}").height == 3
    assert m.query(f"{_PREFIX} SELECT ?f ?y WHERE {{ ?f def:output ?y }}").height == 2
    assert m.query(f"{_PREFIX} SELECT ?f ?p WHERE {{ ?f def:price ?p }}").height == 2


def test_production_respects_labor_capacity():
    sim = RDFSimulator(update_rules=[firm_production(0.5)], n_periods=10)
    for _ in sim.fit(build_model()):
        pass

    outputs = sim.model_.query(
        f"{_PREFIX} SELECT ?f ?y ?alpha ?n "
        "WHERE { ?f def:output ?y ; def:alpha ?alpha ; def:size ?n }"
    )
    assert (outputs["y"] <= outputs["alpha"] * outputs["n"] + 1e-9).all()


def test_income_by_activity_status():
    sim = RDFSimulator(update_rules=RULES, n_periods=1).fit(build_model())
    for _ in sim:
        pass

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
