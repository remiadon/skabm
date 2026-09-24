"""``skabm.dsl`` — one rule, two backends.

Engine tests: they pin that the SPARQL and JAX compilations of the same rule compute
the same thing, that the JAX one differentiates, and that a rule is checked against
the templates when it is compiled.  The labour market is the workload because it is
all per-agent arithmetic plus sums over a network; what the model itself must
reproduce is ``tests/test_labour.py``'s.
"""

import jax
import numpy as np
import polars as pl
import pytest
import sympy as sp
from maplib import Model
from sympy.stats import Exponential, Normal, Uniform
from test_labour import NEVER, clock, market, occupations, ring

from skabm.behaviour.firm import firm_entry
from skabm.behaviour.labour import LABOUR_UPDATE_RULES
from skabm.behaviour.labour import PARAMETERS as LABOUR
from skabm.behaviour.schelling import RELOCATE
from skabm.behaviour.traffic import area_traffic
from skabm.dsl import (
    Agents,
    DSLError,
    arrays,
    jax_tick,
    lag,
    mean,
    node,
    pick,
    sparql,
    sum_over,
    total,
)
from skabm.ottr import firm_template
from skabm.sparql import _PREFIXES, register_math, register_polars_random, render

jax.config.update("jax_enable_x64", True)

Occupation, Edge, Firm = Agents("Occupation"), Agents("Edge"), Agents("Firm")
N, TICKS = 8, 25
CALM = {**{str(k): v for k, v in LABOUR.items()}, "shock_start": NEVER}


def labour_state():
    occ = occupations(
        pl.DataFrame({"id": [f"occ_{i}" for i in range(N)], "employment": [1e3] * N})
    )
    frames = {"Occupation": occ, "Edge": ring(N), "Clock": clock()}
    return occ, arrays(frames, LABOUR_UPDATE_RULES)


def test_sparql_and_jax_agree_on_the_labour_market():
    sim, graph = market(n=N, n_periods=TICKS)
    sim.fit(graph)
    sparql = (
        sim.extract()
        .filter(pl.col("employment").is_not_null())
        .with_columns(id=pl.col("agent").str.extract(r"#(.*)>$"))
        .sort("id")
    )

    occ, state = labour_state()
    tick = jax.jit(jax_tick(LABOUR_UPDATE_RULES))
    for _ in range(TICKS):
        state = tick(state, CALM)

    order = np.argsort(occ["id"].to_numpy())
    for field in ("employment", "unemployment", "vacancies", "ltu", "applications"):
        got = np.asarray(state["Occupation"][field])[order]
        assert got == pytest.approx(sparql[field].to_numpy(), rel=1e-12), field


def test_a_gradient_flows_through_the_whole_run():
    """d(unemployment after 25 ticks)/d(gamma), through ``lax.scan``, as a finite
    difference of the same run would have it."""
    _, start = labour_state()
    tick = jax_tick(LABOUR_UPDATE_RULES)

    def unemployed(gamma):
        params = {**CALM, "gamma": gamma}
        state, _ = jax.lax.scan(
            lambda s, _: (tick(s, params), None), start, None, length=TICKS
        )
        return state["Occupation"]["unemployment"].sum()

    h = 1e-6
    numeric = (unemployed(0.16 + h) - unemployed(0.16 - h)) / (2 * h)
    assert jax.grad(unemployed)(0.16) == pytest.approx(numeric, rel=1e-6)


def test_rules_are_checked_when_compiled():
    with pytest.raises(AttributeError, match="did you mean 'employment'"):
        _ = Occupation.employmnet
    with pytest.raises(AttributeError, match="did you mean 'vacancies'"):
        _ = Edge.dst.vacancie
    with pytest.raises(DSLError, match="reach it by a link"):
        sparql({Edge.flow: Occupation.vacancies})  # whose vacancies? the edge has none
    with pytest.raises(DSLError, match="belongs to Edge"):
        sparql(
            {Occupation.vacancies: Edge.src.unemployment}
        )  # an edge's link, not ours
    with pytest.raises(DSLError, match="points to Occupation"):
        sparql({Firm.output: sum_over(Edge.dst, Edge.flow)})
    with pytest.raises(DSLError, match="one class"):
        sparql({Occupation.vacancies: 0, Edge.flow: 0})
    with pytest.raises(DSLError, match="exactly one class"):
        sparql({Occupation.vacancies: total(Occupation.vacancies * Edge.weight)})
    with pytest.raises(DSLError, match="own agents"):
        sparql({Occupation.vacancies: sum_over(Edge.dst, lag(Edge.flow))})
    with pytest.raises(DSLError, match="named node"):
        sparql({node("one", "vacancies"): sum_over(Edge.dst, Edge.flow)})
    with pytest.raises(DSLError, match="True"):
        sparql({Firm.output: sp.Piecewise((1, Firm.price > 1))})
    with pytest.raises(DSLError, match="no SPARQL form"):
        sparql({Firm.output: sp.sin(Firm.price)})
    with pytest.raises(DSLError, match="only Normal and Uniform"):
        sparql({Firm.output: Exponential("x", 1)})
    # a path reads two links on; a sum runs over one
    Commuter = Agents("Commuter")
    walked = sparql({Commuter.time: Commuter.route.option.share})
    assert "def:route" in walked and "def:option" in walked
    with pytest.raises(DSLError, match="one link"):
        sparql({Agents("Option").s: sum_over(Commuter.route.option, 1)})
    Route, Cell = Agents("Route"), Agents("Cell")
    # another class's field, read through the one link that reaches it
    Option, Person, Cell = Agents("Option"), Agents("Person"), Agents("Cell")
    assert sparql({Route.prob: Option.share}) == sparql(
        {Route.prob: Route.option.share}
    )
    near = Person.location.neighbor
    assert sparql({Person.rank: sum_over(Cell.neighbor, Cell.draw)}) == sparql(
        {Person.rank: sum_over(near, near.draw)}
    )
    # a far agent's field may be named bare inside a sum over the agent's own link
    far = sparql({Route.time: sum_over(Route.via, Agents("Link").t0)})
    assert far == sparql({Route.time: sum_over(Route.via, Route.via.t0)})
    with pytest.raises(DSLError, match="use sum_over"):
        sparql({Route.time: Route.via.t0})  # which of its links?
    with pytest.raises(DSLError, match="not for its own class"):
        sparql({Cell.rank: pick(Cell, Cell.draw > 0, Cell.draw)})


def test_jax_refuses_what_it_cannot_shape():
    with pytest.raises(TypeError, match="firm_entry"):
        jax_tick([firm_entry])  # adds agents: stays SPARQL
    with pytest.raises(DSLError, match="no JAX form"):
        jax_tick([RELOCATE])  # pick has no fixed-shape form yet
    with pytest.raises(DSLError, match="no JAX form"):
        jax_tick([area_traffic])  # a scatter over a many-valued link


def test_a_draw_is_bound_once_however_often_it_is_read():
    """Two fields written from one draw get the same number, in both backends."""
    eps = Normal("eps", 0, sp.Symbol("sigma"))
    rule = {Firm.output: eps, Firm.price: eps}
    assert sparql(rule, {"sigma": 1.0}).count("pr:normal") == 1
    share = {Firm.margin: Uniform("u", 0.2, 0.4)}
    assert "pr:uniform" in sparql(share)

    world = Model()
    register_polars_random(world)
    world.map(
        firm_template,
        pl.DataFrame(
            {"id": ["f"], "alpha": [1.0], "margin": [0.1], "size": [1.0]}
        ).with_iri(),
    )
    world.update(render(rule, {"sigma": 1.0}))
    row = world.query(
        _PREFIXES + "SELECT ?y ?p WHERE { ?f def:output ?y ; def:price ?p }"
    )
    assert row["y"][0] == row["p"][0]

    state = {"Firm": {"alpha": jax.numpy.ones(3)}}
    with pytest.raises(ValueError, match="key"):
        jax_tick([rule])(state)
    with pytest.raises(KeyError, match="needs parameter 'sigma'"):
        jax_tick([{Firm.output: Firm.alpha * sp.Symbol("sigma")}])(state)
    drawn = jax_tick([rule])(state, {"sigma": 1.0}, key=jax.random.key(0))["Firm"]
    assert (drawn["output"] == drawn["price"]).all()
    assert len(set(np.asarray(drawn["output"]).tolist())) == 3  # one draw per agent
    margin = jax_tick([share])(state, key=jax.random.key(1))["Firm"]["margin"]
    assert ((margin >= 0.2) & (margin <= 0.4)).all()


def test_every_node_prints_the_same_value_in_both_backends():
    """The printers, node by node, checked against each other on three firms."""
    x, y = Firm.output, Firm.price
    rule = {
        Firm.liquidity: sp.sqrt(x) + y**3 - 1 / x + sp.pi,
        Firm.profit: sp.Abs(y - x) + sp.log(x) * sp.exp(-y),
        Firm.dividend: sp.Piecewise(
            (x, sp.Or(x > 2, sp.Not(y < 1))), (y, sp.Eq(x, y)), (0, True)
        ),
        Firm.delta: sp.Max(x, y, 1.5) - sp.Min(x, 2 * y),
        Firm.tech_share: x / mean(Firm.output) + lag(x),
        Firm.w_bar: sp.Piecewise((1, sp.ITE(x > 2, y > 1, y < 1)), (0, True)),
    }
    firms = pl.DataFrame(
        {
            "id": ["a", "b", "c"],
            "output": [0.5, 2.0, 3.0],
            "price": [0.8, 2.0, 1.2],
            "alpha": [1.0] * 3,
            "margin": [0.1] * 3,
            "size": [1.0] * 3,
        }
    )
    world = Model()
    register_math(world)
    world.map(firm_template, firms.with_iri())
    world.update(render(rule, {}))
    written = [str(field).split(".")[1] for field in rule]
    sparql = world.query(
        _PREFIXES
        + f"SELECT ?f {' '.join(f'?{f}' for f in written)} WHERE {{ "
        + " ".join(f"?f def:{f} ?{f} ." for f in written)
        + " } ORDER BY ?f"
    )
    lowered = jax_tick([rule])(arrays({"Firm": firms}, [rule]))["Firm"]
    for field in written:
        assert np.asarray(lowered[field]) == pytest.approx(
            sparql[field].to_numpy(), rel=1e-12
        ), field


def test_the_poledna_quarter_runs_in_jax_too(poledna_params):
    """The default rule set, init rules and SAC learners included, stepped in JAX
    from the same frames, lands where the SPARQL simulator does."""
    from test_simulation import CENTRAL_BANK, FIRMS, HOUSEHOLDS, OVERRIDES
    from worlds import world

    from skabm.behaviour.household import household_income_init, household_wealth_init
    from skabm.behaviour.learning import sac
    from skabm.simulation import DEFAULT_UPDATE_RULES, RDFSimulator

    params, ticks = {**poledna_params, **OVERRIDES}, 6
    init = (household_income_init, household_wealth_init)
    sim = RDFSimulator(init_rules=init, params=params, n_periods=ticks)
    sim.fit(world(Firm=FIRMS, Household=HOUSEHOLDS, CentralBank=CENTRAL_BANK))
    sparql = sim.extract().with_columns(id=pl.col("agent").str.extract(r"#(.*)>$"))

    frames = {"Firm": FIRMS, "Household": HOUSEHOLDS, "CentralBank": CENTRAL_BANK}
    learners = [sac(*signal) for signal in sorted(sim._consumed)]
    state = arrays(frames, [*init, *DEFAULT_UPDATE_RULES, *learners])
    key = jax.random.key(0)  # the draws are there, scaled by a zero sigma
    learn, quarter = jax_tick(learners), jax_tick(DEFAULT_UPDATE_RULES)
    state = learn(jax_tick(init)(state, params), params)
    for _ in range(ticks):
        state = learn(quarter(state, params, key), params)

    for klass, fields in {
        "Firm": ("output", "price", "profit", "liquidity"),
        "Household": ("income", "wealth"),
        "CentralBank": ("policy_rate", "prev_output"),
    }.items():
        ids = frames[klass]["id"].to_list()
        agents = sparql.filter(pl.col("id").is_in(ids)).sort("id")
        order = np.argsort(frames[klass]["id"].to_numpy())
        for field in fields:
            got = np.asarray(state[klass][field])[order]
            assert got == pytest.approx(agents[field].to_numpy(), rel=1e-9), field
