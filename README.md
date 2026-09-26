# skabm

![coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)

**Agent-based models written as equations, or in plain English, and run on a knowledge graph.**

Say what an agent does and skabm turns it into a rule. Run the rules over a population
calibrated from public data, and get back every agent after every tick, as polars: any
macro quantity is one expression. Agents are nodes of an RDF graph ([maplib](https://github.com/DataTreehouse/maplib)),
relations are its edges, populations are [polars](https://pola.rs) DataFrames, and the API is
scikit-learn's. The reference model is Poledna, Miess, Hommes & Rabitsch (2023), *Economic
forecasting with an agent-based model* (EER 151): a six-sector economy calibrated 1:1 from
Eurostat.

## What skabm gives researchers

| | |
|---|---|
| **Behaviour as equations** | a rule is a dict of SymPy expressions, `{Firm.price: Firm.price * (1 + inflation)}`. The same rule runs on the knowledge graph and, differentiably, in JAX |
| **Rules from a paper** | the `skabm` skill takes a coding agent from a model section to cited, tested rules ([.claude/skills/skabm](.claude/skills/skabm/SKILL.md)) |
| **Observables in polars** | a run is every agent after every tick, one DataFrame, so any macro quantity, a Gini included, is a polars expression you write. Nothing is declared to the simulator |
| **Populations from public data** | Eurostat loaders, and calibrators that fit agent rows to known joint facts while every marginal survives exactly |
| **Citations built in** | each model module carries its published parameter values next to their table and equation |
| **Interventions as graph edits** | after a fit, the world is a graph: rewire an ownership edge, delete a bank, close a street, then keep simulating |
| **Learned expectations** | agents forecast with learning rules (SAC, Hommes & Zhu 2014) that are just more rules |

### A module knows its rules and its parameters

```python
from skabm.behaviour import defaults, firm, household, macro

household.RULES                        # income, then consumption, each step
firm.PARAMETERS[firm.vat_rate]         # 0.1529: τ^VAT, Poledna et al. (2023) Table 2
macro.PARAMETERS[macro.rho]            # 0.9263, the Taylor-rule smoothing
assert defaults()["vat_rate"] == 0.1529 and len(household.RULES) == 2
```

`RDFSimulator(params={"vat_rate": 0.2})` overrides a default by name. More in
[skabm/behaviour](skabm/behaviour/README.md).

### What a run yields

```python
import polars as pl
from maplib import Model

from skabm.simulation import RDFSimulator
from skabm import template  # + DataFrame.with_iri

firms = pl.DataFrame({"id": ["firm_0", "firm_1"], "output": [100.0, 120.0],
                      "price": [1.0, 1.1], "alpha": [10.0, 10.0], "size": [12.0, 13.0],
                      "margin": [0.1, 0.1], "liquidity": [50.0, 60.0], "profit": [1.0, 1.0]})
households = pl.DataFrame({"id": ["hh_0", "hh_1"], "wealth": [100.0, 200.0],
                          "income": [10.0, 12.0], "psi": [0.9, 0.9],
                          "employer": ["firm_0", "firm_1"]})

def economy() -> Model:
    world = Model()
    world.map(template.firm, firms.with_iri())
    world.map(template.household, households.with_iri("employer"))
    return world

sim = RDFSimulator(n_periods=8, random_seed=0)
run = pl.concat(sim.fit_iter(economy()))   # every agent after every tick: agent, class, t, ...

gdp = (pl.col("price") * pl.col("output")).sum()
run.group_by("t").agg(gdp=gdp, wealth=pl.col("wealth").sum())
```

A macro quantity is a polars expression, and so is one no aggregate spans, like a Gini.
`class` tells apart the agents that share a field:

```python continuation
def gini(column: str) -> pl.Expr:
    x = pl.col(column).drop_nulls().sort()
    n = x.len()
    return 2 * (pl.int_range(1, n + 1, dtype=pl.Int64) * x).sum() / (n * x.sum()) - (n + 1) / n

is_firm = pl.col("class") == "Firm"
run.group_by("t").agg(gini("wealth"), price_level=pl.col("price").filter(is_firm).mean())
```

### Getting data in

The world is maplib's own `Model`, and `Model.map` checks a population against its template
before anything simulates:

```python continuation
try:
    Model().map(template.firm, firms.drop("alpha").with_iri())
except Exception as error:
    print(error)          # Expected column alpha is missing
```

### One rule, two backends

Your own rule is one line. It runs on the graph, and the same dict gives you `jax.grad`:

```python continuation
import jax
import sympy as sp

from skabm.dsl import Agents, arrays, jax_tick

Firm, inflation = Agents("Firm"), sp.Symbol("inflation")
reprice = {Firm.price: Firm.price * (1 + inflation)}

sim = RDFSimulator(rules=[reprice], params={"inflation": 0.02}, n_periods=4)
*_, last = sim.fit_iter(economy())
last["price"]                                  # every price up 2% a tick, on the graph

tick = jax_tick([reprice])

def price_level(inflation):
    state = arrays({"Firm": firms}, [reprice])
    for _ in range(4):
        state = tick(state, {"inflation": inflation})
    return state["Firm"]["price"].mean()

jax.grad(price_level)(0.02)                    # 4 × 1.05 × 1.02³, the exact derivative
```

### From a paper to rules

A rule is short enough that a coding agent writes it from a paper's model section. The
`skabm` skill ([.claude/skills/skabm](.claude/skills/skabm/SKILL.md)) is the method: read the
source, fix the scope, map the equations to the templates' fields, cite every parameter,
then check the rules on the graph, in JAX and over a long run. `behaviour.canvas` came out
of it, from Hommes et al. (2025).

## Quickstart

A population calibrated from real Eurostat data, then simulated:

```python
import polars as pl
import polars_random as pr

from skabm.calibration import GeneticConstraintCalibration, make_dataset, weighted_enum
from maplib import Model

from skabm.behaviour.firm import ownership
from skabm.behaviour.household import initial
from skabm.datasets import build_firm_io_df
from skabm.simulation import RDFSimulator
from skabm import template

# 1. Real data: Austrian input-output table + business demography (Poledna §4.1).
io = build_firm_io_df("AT", 2010).drop_nulls(["n_firms", "alpha_s"])
by_industry = lambda v: pl.col("industry").replace_strict(io["industry"], v, return_dtype=pl.Float64)  # noqa: E731

# 2. Marginals: industry in proportion to real enterprise counts, size log-normal (§4.1.1).
firms = make_dataset(
    samplers={
        "industry": weighted_enum(pl.Enum(io["industry"].to_list()), io["n_firms"], seed=100),
        "size": pr.normal(3.0, 1.0, seed=101).exp().cast(pl.Int64).clip(1, None),
    },
    n_agents=300,
    seed=0,
)

# 3. Population calibration: industry and size were drawn independently, but Eurostat
#    reports persons per enterprise per industry — a joint fact no marginal encodes.
observed = by_industry((io["n_employed"] / io["n_firms"]).log())
shift = firms.select(pl.col("size").log().mean() - observed.mean()).item()
firms = pl.DataFrame(
    GeneticConstraintCalibration(
        constraints=[(pl.col("size").log().mean().over("industry"), observed + shift)],
        n_generations=600,
        seed=7,
    ).fit(firms).samplers_
).with_row_index("id")

# 4. IO coefficients are functions of industry, attached after calibration.
firms = firms.with_columns(
    pl.format("firm_{}", pl.col("id")).alias("id"), pl.col("industry").cast(pl.String),
    alpha=by_industry(io["alpha_s"]), w_bar=by_industry(io["w_bar_s"]),
    delta=by_industry(io["delta_s"]), tech_share=by_industry(io["tech_share_s"]),
    price=pl.lit(1.0), margin=pl.lit(0.2), liquidity=pl.lit(0.0),
    size=pl.col("size").cast(pl.Float64),           # a headcount, as the double the rules compute in
).with_columns(output=0.9 * pl.col("alpha") * pl.col("size"))

# 5. Households: census active/inactive shares, and an `employer` *link* column whose
#    values name firms — it becomes a graph edge at map time, not a join you maintain.
households = make_dataset(
    samplers={
        "status": weighted_enum(pl.Enum(["active", "inactive"]), [4_729_215, 4_130_385], seed=110),
        "employer": weighted_enum(pl.Enum(firms["id"].to_list()), firms["size"], seed=111),
    },
    n_agents=10_000,
    seed=1,
).with_columns(
    pl.format("hh_{}", pl.col("id")).alias("id"),
    psi=pl.lit(0.9394),                             # propensity to consume, Table 2
    employer=pl.when(pl.col("status") == "active")  # inactive households supply no labour
              .then(pl.col("employer").cast(pl.String)).otherwise(None),
).drop("status")

# 6. Map, then simulate. The default rule set is the full Poledna economy and
#    self-scopes to the classes in the graph, so a Firm + Household world executes
#    exactly those dynamics.
world = Model()
world.map(template.firm, firms.with_iri())
households = ownership(households, firms, ratio=300 / 10_000)  # who owns which firm, §3.2
households = initial(households, firms)  # the starting income and wealth, eq. 49 and §5.2
world.map(template.household, households.with_iri("employer", "owns"))
sim = RDFSimulator(n_periods=12, params={"growth_sigma": 0.02, "inflation_sigma": 0.01})
for frame in sim.fit_iter(world):
    # every agent after the tick: the price level and GDP are polars over it
    print(frame.select(price_level=pl.col("price").filter(pl.col("class") == "Firm").mean(),
                       gdp=(pl.col("price") * pl.col("output") * (1 - pl.col("tech_share"))).sum()))
```

Step 3 is what a marginal sampler cannot do, and both marginals survive it exactly
([why](skabm/calibration/README.md#populations)).

| Where to look next | |
|---|---|
| [notebooks/poledna](notebooks/poledna/poledna.ipynb) | the model in five chapters on one population: **0** calibration, **1** the quarter plus an intervention, **2** a banking layer whose distress spreads a step per quarter, **3** learned expectations as running sums on the signal nodes, **4** black-it parameter calibration read as an identification test |
| [notebooks/extraction](notebooks/extraction.ipynb) | what to *do* with derived state: Gini as one polars expression, an intervention that moves it, telemetry into JAX |
| [notebooks/labour_automation](notebooks/labour_automation.ipynb) | occupational mobility under an automation shock: a labour-flow network whose central quantity lives on the *edge* |
| [notebooks/schelling](notebooks/schelling.ipynb) | spatial segregation on the lattice; the continuous-geometry swap is covered by `tests/test_schelling.py` |
| [notebooks/bayonne](notebooks/bayonne/app.py) | closing Bayonne's old town to cars: 9,400 commuters from the 2022 census on 37,000 OpenStreetMap links, the closure as a graph edit, in a Streamlit map (`streamlit run notebooks/bayonne/app.py`) |

## Architecture

| Package | What it is |
|---|---|
| [skabm](skabm/README.md) | the engine: templates, the rule DSL and its SPARQL and JAX compilers, the simulator, what gets measured, rules from a description |
| [skabm.behaviour](skabm/behaviour/README.md) | the rule library: Poledna's economy and CANVAS's firms, banks, the labour market, learning, Schelling, traffic |
| [skabm.calibration](skabm/calibration/README.md) | populations fitted to public data, and parameters fitted to macro series with black-it |

## Limitations

The backend is pure SPARQL, deliberately: a stress test of how far a declarative rule
engine carries an economic ABM. Its walls, in short:
[details](skabm/README.md#limitations).

- **No random sequential matching**: markets clear simultaneously. Matching stated in
  closed form (urn-ball, one-to-one assignment) works.
- **The past is state**: a rule sees one time slice, so history is kept as lags and
  running sums.
- **Synchronous, staged activation**: all agents update at once, rules fire in order.

## Roadmap

| | |
|---|---|
| **The rest in JAX** | the relational operators (`total_by`, `running_sum`, `pick`) have no JAX form yet, so traffic and Schelling run on the graph only |
| **Rule-scoped validation** | a rule already names the fields it reads and writes; check them against the graph at fit time and emit SHACL shapes |
| **Sensitivity-ranked observables** | rank the series by paired-seed shocks, to order the panels and detect divergence during calibration |
| **An MCP server** | run a simulation on remote compute, or serve an environment (a grid, a street map), for a client that cannot run skabm itself |

## References

- Poledna, Miess, Hommes & Rabitsch (2023). Economic forecasting with an agent-based model.
  *European Economic Review* 151, 104306 · del Rio-Chanona, Mealy, Beguerisse-Díaz, Lafond &
  Farmer (2021). Occupational mobility and automation. *J. R. Soc. Interface* 18(174), 20200898.
- Hommes & Zhu (2014). Behavioral learning equilibria. *JET* 150 · Huberman & Glance (1993).
  Evolutionary games and computer simulations. *PNAS* 90(16) · Deville & Särndal (1992).
  Calibration estimators in survey sampling. *JASA* 87(418) · Fagiolo, Guerini, Lamperti,
  Moneta & Roventini (2019). Validation of ABMs. In *Computer Simulation Validation*.
- [black-it](https://github.com/bancaditalia/black-it) (ABM parameter calibration) ·
  [maplib](https://github.com/DataTreehouse/maplib) (knowledge graphs, OTTR, SPARQL) · [AMBER](https://github.com/a11to1n3/AMBER) (polars-based ABM).
