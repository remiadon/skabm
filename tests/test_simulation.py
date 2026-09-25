"""RDFSimulator: SPARQL update rules advance a small in-memory model.

No Eurostat access — populations are tiny hand-written DataFrames mapped
into a fresh maplib Model per fit (``worlds.world``: shipped templates,
predicates = column names).  Rules are the default
rule dicts; test-specific numbers come in through the params
dict, laid over the ``poledna_params`` fixture.  Covered: iteration yielding
per-agent state, cold-fit rebuild semantics, warm_start continuation,
upsert semantics, the paper's capacity cap, income by activity status,
unreferenced-population warning, and sklearn get_params/clone compatibility.
"""

import polars as pl
import pytest
import sympy as sp
from maplib import Model
from sklearn.base import clone
from worlds import world

from skabm import template
from skabm.behaviour.firm import Firm, firm_produce, ownership
from skabm.behaviour.household import household_income, initial
from skabm.simulation import RDFSimulator
from skabm.sparql import DEF_NS

_PREFIX = f"PREFIX def:<{DEF_NS}> PREFIX ex:<http://example.net/skabm#>"


def local(iri: str) -> str:
    """Local name of a returned IRI ('<...#hh_0>' -> 'hh_0'), so tests never
    spell out the model namespace."""
    return iri.rsplit("#", 1)[-1].rstrip(">")


FIRMS = pl.DataFrame(
    {
        "id": ["firm_0", "firm_1"],  # bare local names throughout
        "size": [10.0, 20.0],
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
RAW_HOUSEHOLDS = pl.DataFrame(
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

# laid over the poledna_params fixture by the ``params`` fixture below
OVERRIDES = {
    "dividend_ratio": 0.8,
    "benefit_replacement": 0.4,
    "total_deposits": 3000.0,
    "vat_rate": 0.15,
    "rho": 0.9,
    "r_star": 0.0,
    "xi_pi": 0.5,
    "xi_gamma": 0.5,
}
HOUSEHOLDS = initial(
    RAW_HOUSEHOLDS, FIRMS, **OVERRIDES
)  # the opening income and wealth


# Expectations are learned, not parameterised, so nothing makes output move on
# its own: with a cold (empty) history the first forecast is zero, the realized
# growth it then learns from is zero, and the model sits at a steady state.
# Shocks are what give the AR(1) something to estimate, so tests that need a
# non-flat trajectory use these params together with a fixed random_seed.


@pytest.fixture
def params(poledna_params):
    return {**poledna_params, **OVERRIDES}


@pytest.fixture
def shocked(params):
    return {**params, "growth_sigma": 0.02}


def gdp(state: pl.DataFrame) -> float:
    return state.select(
        (pl.col("price") * pl.col("output") * (1 - pl.col("tech_share"))).sum()
    ).item()


def path(sim) -> list:
    """GDP per tick, off the frame — a product no class-level aggregate spans."""
    return [
        gdp(sim.extract())
        for _ in sim.fit_iter(
            world(Firm=FIRMS, Household=HOUSEHOLDS, CentralBank=CENTRAL_BANK)
        )
    ]


def test_fit_iter_yields_measurements_per_tick(shocked):
    sim = RDFSimulator(params=shocked, n_periods=4, random_seed=11)

    run = pl.DataFrame(
        sim.fit_iter(world(Firm=FIRMS, Household=HOUSEHOLDS, CentralBank=CENTRAL_BANK))
    )
    # one row per tick, one column per observable plus t, and no hole in it
    assert run.height == 4
    assert run["t"].to_list() == [1, 2, 3, 4]
    assert set(run.columns) == {"t"} | {o.signal for o in sim.observables_}
    assert run["sig__SUM__Firm__output"].null_count() == 0
    # shocks move output, so the path is not flat
    assert run["sig__SUM__Firm__output"].n_unique() == 4

    # a cold refit rebuilds the world from scratch: same trajectory again
    gdp_path = path(sim)
    assert path(sim) == gdp_path
    assert len(set(gdp_path)) == 4


def test_warm_start_continues_the_world(shocked):
    sim = RDFSimulator(params=shocked, n_periods=2, random_seed=5).fit(
        world(Firm=FIRMS, Household=HOUSEHOLDS, CentralBank=CENTRAL_BANK)
    )
    gdp_after_cold = gdp(
        sim.model_.query(
            f"{_PREFIX} SELECT ?price ?output ?tech_share WHERE "
            "{ ?f def:price ?price ; def:output ?output ; def:tech_share ?tech_share }"
        )
    )

    warm = RDFSimulator(params=shocked, n_periods=2, warm_start=True, random_seed=5)
    warm.model_ = sim.model_
    rows = list(warm.fit_iter())
    assert [r["t"] for r in rows] == [1, 2]
    assert gdp(warm.extract()) != gdp_after_cold  # advanced beyond the cold fit

    with pytest.raises(TypeError, match="maplib Model"):
        warm.fit({"Firm": FIRMS})  # populations are mapped by the caller, not here


def test_fit_takes_a_model_someone_else_built(params):
    """The world is an argument, not something the simulator has to construct.

    Anything reachable through maplib is therefore reachable here: a graph
    assembled by other code, deserialized, or intervened on before the first
    tick.  It is advanced in place, so the caller keeps the handle.
    """
    world = Model()
    world.map(template.firm, FIRMS.with_iri())
    world.map(template.household, HOUSEHOLDS.with_iri("employer", "owns"))
    world.update(
        f"{_PREFIX} DELETE {{ ?f def:price ?p }} INSERT {{ ?f def:price 3e0 }} "
        "WHERE { ?f a ex:Firm ; def:price ?p }"
    )
    sim = RDFSimulator(params=params, n_periods=2, random_seed=3)
    run = pl.DataFrame(sim.fit_iter(world))

    assert sim.model_ is world  # advanced in place, not copied
    assert run.height == 2
    # the init rules still ran on it, and the intervention was the opening state
    assert world.query(f"{_PREFIX} SELECT ?w WHERE {{ ?h def:wealth ?w }}").height == 3
    assert run["sig__AVG__Firm__price"][0] > 2.0


def test_warm_start_takes_the_model_to_continue(params):
    """The replacement for assigning ``model_`` and flipping the flag."""
    cold = RDFSimulator(params=params, n_periods=2, random_seed=3).fit(
        world(Firm=FIRMS, Household=HOUSEHOLDS)
    )
    owners = cold.model_.query(f"{_PREFIX} SELECT ?h WHERE {{ ?h def:owns ?f }}").height

    warm = RDFSimulator(params=params, n_periods=2, warm_start=True, random_seed=3)
    rows = list(warm.fit_iter(cold.model_))

    assert warm.model_ is cold.model_ and len(rows) == 2
    # init rules did not run again: ownership was assigned once, not twice
    assert (
        warm.model_.query(f"{_PREFIX} SELECT ?h WHERE {{ ?h def:owns ?f }}").height
        == owners
    )


def test_a_placeholder_without_a_published_value_is_named():
    # a parameter with no published value has no default: the caller is told
    # rather than handed a number nobody chose
    entry = {Firm.output: Firm.output + sp.Symbol("entry_barrier")}
    with pytest.raises(KeyError, match="entry_barrier"):
        RDFSimulator(rules=(entry,), n_periods=1).fit(world(Firm=FIRMS))


def test_default_rules_scoped_to_passed_kinds(shocked):
    # only firms: default rules anchored on absent classes (Household,
    # CentralBank) match nothing and no-op — SPARQL pattern matching scopes
    # the full default rule set to the kinds actually mapped
    sim = RDFSimulator(params=shocked, n_periods=2, random_seed=5).fit(
        world(Firm=FIRMS)
    )

    outputs = sim.model_.query(f"{_PREFIX} SELECT ?f ?y WHERE {{ ?f def:output ?y }}")
    assert outputs.height == 2
    assert (outputs["y"] != FIRMS["output"]).all()  # production rule did run
    incomes = sim.model_.query(f"{_PREFIX} SELECT ?h ?i WHERE {{ ?h def:income ?i }}")
    assert incomes.height == 0  # no household rules without Household=


def test_unreferenced_population_warns(params):
    sim = RDFSimulator(params=params, n_periods=1)
    ghosts = pl.DataFrame({"id": ["ghost_0"], "x": [1.0]})
    with pytest.warns(UserWarning, match="Ghost"):
        sim.fit(
            world(
                Firm=FIRMS, Household=HOUSEHOLDS, CentralBank=CENTRAL_BANK, Ghost=ghosts
            )
        )


def test_upserts_do_not_duplicate_state(params):
    sim = RDFSimulator(params=params, n_periods=3).fit(
        world(Firm=FIRMS, Household=HOUSEHOLDS, CentralBank=CENTRAL_BANK)
    )

    m = sim.model_
    # exactly one state triple per agent after repeated upserts
    assert m.query(f"{_PREFIX} SELECT ?h ?i WHERE {{ ?h def:income ?i }}").height == 3
    assert m.query(f"{_PREFIX} SELECT ?h ?w WHERE {{ ?h def:wealth ?w }}").height == 3
    assert m.query(f"{_PREFIX} SELECT ?f ?y WHERE {{ ?f def:output ?y }}").height == 2
    assert m.query(f"{_PREFIX} SELECT ?f ?p WHERE {{ ?f def:price ?p }}").height == 2


def test_production_respects_labor_capacity():
    sim = RDFSimulator(
        rules=[firm_produce],
        params={"growth_sigma": 0.0},
        n_periods=10,
    ).fit(world(Firm=FIRMS))

    outputs = sim.model_.query(
        f"{_PREFIX} SELECT ?f ?y ?alpha ?n "
        "WHERE { ?f def:output ?y ; def:alpha ?alpha ; def:size ?n }"
    )
    assert (outputs["y"] <= outputs["alpha"] * outputs["n"] + 1e-9).all()


def test_income_by_activity_status(poledna_params):
    # Poledna eq. 49 with Table 2's own θ^UB and θ^DIV
    sim = RDFSimulator(params=poledna_params, n_periods=1).fit(
        world(Firm=FIRMS, Household=HOUSEHOLDS, CentralBank=CENTRAL_BANK)
    )

    income = {
        local(row["h"]): row["i"]
        for row in sim.model_.query(
            f"{_PREFIX} SELECT ?h ?i WHERE {{ ?h def:income ?i }}"
        ).to_dicts()
    }
    assert income["hh_0"] == 30.0  # worker: employer's wage
    assert income["hh_1"] == 0.0  # investor in a loss-maker: θ^DIV * max(0, profit)
    # unemployed: θ^UB * average wage
    assert income["hh_2"] == pytest.approx(poledna_params["benefit_replacement"] * 35.0)


def test_ownership_gives_each_firm_an_owner_unless_the_data_does():
    # firm j -> household floor(j / ratio): firm_0 -> hh_0, firm_1 -> hh_3
    households = pl.DataFrame({"id": [f"hh_{i}" for i in range(6)]})
    assert ownership(households, FIRMS, 2 / 6)["owns"].to_list() == [
        "firm_0",
        None,
        None,
        "firm_1",
        None,
        None,
    ]
    # a data-defined owner survives, and the firm is not handed out twice
    owned = households.with_columns(owns=pl.Series([None] * 5 + ["firm_1"]))
    assert ownership(owned, FIRMS, 2 / 6)["owns"].to_list() == [
        "firm_0",
        None,
        None,
        None,
        None,
        "firm_1",
    ]


@pytest.mark.filterwarnings("ignore:population 'Firm'")
def test_bare_link_resolves_from_model(params):
    # both id and the employer link are bare local names; map_population recognizes
    # "firm_0" as a reference (a firm with that id was mapped first) and
    # prefixes it, so the income rule traverses employer -> firm.
    firms = pl.DataFrame(
        {
            "id": ["firm_0"],
            "w_bar": [30.0],
            "alpha": [10.0],
            "margin": [0.1],
            "size": [1.0],
        }
    )
    households = pl.DataFrame({"id": ["hh_0"], "employer": ["firm_0"], "psi": [0.9]})
    sim = RDFSimulator(rules=[household_income], params=params, n_periods=1).fit(
        world(Firm=firms, Household=households)
    )
    income = sim.model_.query(f"{_PREFIX} SELECT ?i WHERE {{ ?h def:income ?i }}")
    assert income["i"].to_list() == [30.0]


def test_ar_shocks_opt_in_and_reproducible(params, shocked):
    # AR(1) innovations (pr:normal UDF) are opt-in: sigma defaults to 0 so the
    # default run is a deterministic drift, reproducible with no seed at all.
    def outputs(sim):
        return sorted(
            round(x, 4)
            for x in sim.model_.query(
                f"{_PREFIX} SELECT ?y WHERE {{ ?f def:output ?y }}"
            )["y"]
        )

    base = outputs(RDFSimulator(params=params, n_periods=3).fit(world(Firm=FIRMS)))
    assert base == outputs(
        RDFSimulator(params=params, n_periods=3).fit(world(Firm=FIRMS))
    )

    # growth_sigma > 0 turns FIRM_PRODUCTION stochastic; random_seed pins it.
    shocked = dict(params, growth_sigma=0.02)
    s1 = outputs(
        RDFSimulator(params=shocked, n_periods=3, random_seed=1).fit(world(Firm=FIRMS))
    )
    s2 = outputs(
        RDFSimulator(params=shocked, n_periods=3, random_seed=1).fit(world(Firm=FIRMS))
    )
    s3 = outputs(
        RDFSimulator(params=shocked, n_periods=3, random_seed=2).fit(world(Firm=FIRMS))
    )
    assert s1 != base  # shocks moved the path off the deterministic drift
    assert s1 == s2  # same seed -> identical run
    assert s1 != s3  # the seed matters


def test_get_params_and_clone(params):
    sim = RDFSimulator(params=params, n_periods=7)
    got = sim.get_params()
    assert set(got) == {
        "rules",
        "params",
        "n_periods",
        "warm_start",
        "state_extract",
        "track",
        "udfs",
        "random_seed",
        "learner",
        "duckdb_connection",
    }
    assert got["n_periods"] == 7
    assert got["params"]["total_deposits"] == 3000.0

    # clone deep-copies params; a rule dict compares by its expressions
    fresh = clone(sim)
    fresh_params = fresh.get_params()
    assert fresh_params["params"] == params and fresh_params["n_periods"] == 7
    assert list(fresh_params["rules"]) == list(got["rules"])
    assert fresh.model_.query("SELECT ?s WHERE { ?s ?p ?o }").height == 0


def test_inject_metadata_adds_triples():
    # _inject_metadata is the SPARQL-free provenance path; exercise it
    # directly so the insertion block is covered without a full cold fit.
    from maplib import Model

    from skabm.simulation import _inject_metadata

    model = Model()
    _inject_metadata(model, [firm_produce])
    triples = model.query("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")
    assert triples.height >= 3  # class-behaviour, type, source
    sources = [o for p, o in zip(triples["p"], triples["o"]) if "source" in p]
    assert "Poledna" in sources[0]  # the module's docstring, citation included
    pvals = {str(p).replace("<", "").replace(">", "") for p in triples["p"]}
    assert "http://example.net/skabm#behaviour" in pvals


def test_the_world_and_the_sidecar_stay_separate(params):
    """``model_`` is the world; ``meta_`` is what the simulator knows about the rules.

    Keeping provenance out means a plain ``?s ?p ?o`` over ``model_`` returns the
    world, so the simulation graph stays portable.
    """
    sim = RDFSimulator(params=params, n_periods=1).fit(
        world(Firm=FIRMS, Household=HOUSEHOLDS)
    )

    predicates = sim.model_.query(f"{_PREFIX} SELECT ?p WHERE {{ ?s ?p ?o }}")["p"]
    assert predicates.len() > 0
    # the rules' own provenance does not live in the world
    assert not any("behaviour" in p for p in predicates)

    side = sim.meta_.query(f"{_PREFIX} SELECT ?p WHERE {{ ?s ?p ?o }}")["p"]
    assert any("behaviour" in p for p in side)  # provenance
    # ... and no agent leaked into the sidecar
    assert sim.meta_.query(f"{_PREFIX} SELECT ?f WHERE {{ ?f a ex:Firm }}").height == 0


def test_the_ir_decides_what_the_frontend_shows(params):
    """The IR is machinery: it is never exposed, but it settles both frontends.

    ``extract()``'s columns and the yielded row's keys are both consequences of
    the state / structure partition, which is why nothing in either has to be
    named by hand.  Pinned through the public surface, not through ``_ir``.
    """
    sim = RDFSimulator(params=params, n_periods=1).fit(
        world(Firm=FIRMS, Household=HOUSEHOLDS)
    )
    assert not hasattr(sim, "ir_"), "the IR is machinery, not an interface"

    columns = set(sim.extract().columns)
    assert {"output", "price", "wealth"} <= columns  # state, written by the rules
    assert {"alpha", "size", "psi"} <= columns  # structure, carried by the data

    signals = {o.signal for o in sim.observables_}
    assert "sig__SUM__Firm__output" in signals  # state is measured
    assert not any("__alpha" in s for s in signals)  # structure never moves


def test_starting_values_fill_in_rather_than_pile_on():
    """``household.initial`` keeps what the data carries and fills in the rest: eq. 49
    for income, the deposits shared in proportion to it for wealth (Section 5.2)."""
    carried = RAW_HOUSEHOLDS.with_columns(income=pl.lit(3.0), wealth=pl.lit(7.0))
    kept = initial(carried, FIRMS, **OVERRIDES)
    assert (
        kept["income"].to_list() == [3.0] * 3 and kept["wealth"].to_list() == [7.0] * 3
    )
    filled = initial(RAW_HOUSEHOLDS, FIRMS, **OVERRIDES)
    # worker: the wage; investor in a loss-maker: nothing; unemployed: the benefit
    assert filled["income"].to_list() == pytest.approx([30.0, 0.0, 0.4 * 35.0])
    assert filled["wealth"].sum() == pytest.approx(OVERRIDES["total_deposits"])
    assert filled.columns == [*RAW_HOUSEHOLDS.columns, "income", "wealth"]
