# skabm

![coverage](https://img.shields.io/badge/coverage-97%25-brightgreen)

**scikit-learn-style agent-based modeling on a knowledge graph.**

Populations are calibrated as [polars](https://pola.rs) DataFrames, lifted into an RDF
knowledge graph ([maplib](https://github.com/DataTreehouse/maplib)), and simulated by applying
SPARQL rules to it — behind a scikit-learn-shaped API. The reference implementation follows
Poledna, Miess, Hommes & Rabitsch (2023), *Economic forecasting with an agent-based model*
(EER 151): a six-sector economy calibrated 1:1 from Eurostat. Array-based frameworks (e.g.
[AMBER](https://github.com/a11to1n3/AMBER)) get performance right but flatten heterogeneous
populations and their typed relations into undocumented index arrays; skabm keeps the polars
performance and makes the structure explicit — agents are nodes, relations are triples.

## What skabm gives researchers

You bring behavioural rules and data. These come back without your writing them:

| | |
|---|---|
| **Populations from public data** | `skabm.datasets` pulls Eurostat IO tables and business demography; `calibration.population` fits agent *rows* to accounting identities and known margins by permutation, so every marginal survives exactly |
| **Behaviour as data** | a rule is a dict of SymPy expressions, `{Firm.output: ...}`, passed to `__init__`, so `get_params()`/`clone()` work and a rule set is serialisable, diffable and shareable. It compiles to SPARQL at fit time; each module's `PARAMETERS` holds its published numbers next to their citations, overridable through `params` |
| **Derived observables** | Mesa makes you declare `model_reporters`, Agents.jl `adata`, NetLogo a metric list — their rules are opaque host-language functions. skabm's are SymPy, so it reads the `(class, field, aggregate)` triples they imply off the expressions. Every binding constraint becomes a series too: the `min` in eq. 5 gives the share of firms pinned at `alpha * size`, the `max` in the Taylor rule the share of quarters at the zero lower bound |
| **One rule, two backends** | `skabm.dsl` writes a rule as SymPy over the template's fields: `{Occupation.separations: delta_u * Occupation.employment + ...}`, a link followed by name (`Edge.src.unemployment`), `sum_over` for a sum over incoming links, `total`/`mean`, `lag`, `Normal`/`Uniform`. It compiles to SPARQL for maplib and, through `sympy.lambdify`, to JAX for `jax.grad`. The labour market runs both ways to 1e-12, and a gradient through 25 ticks matches finite differences. Rules that add agents stay SPARQL |
| **Rules from a description** | `skabm.translate.rules(description, templates)` turns a researcher's account of what an agent does into rules. The OTTR templates are the context and the grammar: a model generating under it (llguidance, a local 7B by default) can only write the DSL over the fields those templates declare, so its answer runs as the Python it is |
| **A graph with memory** | a rule naming an `ex:sig__<agg>__<Class>__<predicate>` signal reads a learned expectation, and the simulator runs one `learner` rule per signal after every tick. SAC learning is a DSL rule over seven running sums on the signal node, so it needs no stored series; a moving average or an RL update is another `learner` function |
| **Fixed-point propagation** | `infer=` hands recursive CONSTRUCT rules to maplib's reasoner, so contagion, transitive closure and reachability propagate across the whole graph in one call rather than hand-tuned sub-tick passes |
| **First-class interventions** | `model_` is a regular maplib model post-fit: rewire an ownership edge, delete a bank, halve a sector's demand with `model_.update(...)`, then continue with `warm_start=True` |

### A module knows its parameters

Each module in `skabm.behaviour` holds the published values of the parameters its rules
read, next to their citations:

```python
from skabm.behaviour import defaults, firm, macro

firm.PARAMETERS[firm.vat_rate]         # 0.1529: τ^VAT, Poledna et al. (2023) Table 2
macro.PARAMETERS[macro.rho]            # 0.9263, the Taylor-rule smoothing
firm.firm_entry.get_identifiers()      # ['entry_barrier', 'entry_sigma']: SPARQL text
"entry_barrier" in defaults()          # False: no published calibration, pass it
assert defaults()["vat_rate"] == 0.1529 and "entry_sigma" not in defaults()
```

`RDFSimulator(params={"vat_rate": 0.2})` overrides a default by name, and a parameter with
no default raises at fit time, naming itself.

### What a run yields

```python
import polars as pl
from maplib import Model

from skabm.simulation import RDFSimulator
from skabm.ottr import firm_template, household_template  # + DataFrame.with_iri

firms = pl.DataFrame({"id": ["firm_0", "firm_1"], "output": [100.0, 120.0],
                      "price": [1.0, 1.1], "alpha": [10.0, 10.0], "size": [12.0, 13.0],
                      "margin": [0.1, 0.1], "liquidity": [50.0, 60.0], "profit": [1.0, 1.0]})
households = pl.DataFrame({"id": ["hh_0", "hh_1"], "wealth": [100.0, 200.0],
                          "income": [10.0, 12.0], "psi": [0.9, 0.9],
                          "employer": ["firm_0", "firm_1"]})

def economy() -> Model:
    world = Model()
    world.map(firm_template, firms.with_iri())
    world.map(household_template, households.with_iri("employer"))
    return world

sim = RDFSimulator(n_periods=8, random_seed=0)
run = pl.DataFrame(sim.fit_iter(economy()))

run.shape                    # 8 ticks x (t + every aggregate the rules imply)
run.columns[:3]              # sig__AVG__Firm__binds__output, sig__AVG__Firm__liquidity, ...
len(sim.extract().columns)   # 18 per-agent columns, none of them named by hand
```

A whole run is `pl.DataFrame(sim.fit_iter(world))`, one tick is the metrics dict a callback API
takes, nothing was declared, and nothing opened a database. A fit advances its model in place,
which is why `economy()` is a function: each run gets a fresh world. Each predicate is measured under
the aggregate the rules apply to it — `centralbank_rate` takes `AVG(?price)`, so the price
signal is a mean, not a sum. **Where the line is:** `SUM(def:output)` follows from the rules,
but calling it *GDP* is a modelling claim and a **distributional** statistic is not derivable
at all. Both stay with you, in polars over `sim.extract()` — itself derived from the rules.

```python continuation
def gini(column: str) -> pl.Expr:
    x = pl.col(column).drop_nulls().sort()
    n = x.len()
    return 2 * (pl.int_range(1, n + 1, dtype=pl.Int64) * x).sum() / (n * x.sum()) - (n + 1) / n

panel = pl.concat(
    sim.extract().with_columns(t=pl.lit(row["t"]))
    for row in sim.fit_iter(economy())
)
panel.lazy().group_by("t").agg(gini("wealth")).sort("t").collect()
```

### Getting data in

**The world is an argument.** `fit` takes one thing: a maplib `Model`, advanced in place — one
you mapped, a graph another system built, or one you deserialized and intervened on.
`skabm.ottr` is a registry of maplib `Template`s, one per agent class, and mapping a
population is maplib's own `Model.map` — which checks the frame before anything simulates:

```python continuation
try:
    Model().map(firm_template, firms.drop("alpha").with_iri())
except Exception as error:
    print(error)          # Expected column alpha is missing
```

Required parameters are what the rules read but never write: nothing produces them, so a
frame lacking one is a run of nothing. Quantities are `xsd:double`, which takes any float
width as is; an integer column is refused by name, because an `xsd:long` never equals the
doubles the rules write and would join with nothing, silently. Columns a template does not
declare are refused too: drop them, or declare your own class with `agent_template`.
Importing `skabm.ottr` registers `DataFrame.with_iri(*links)`, which turns `id` and the
named link columns into the IRIs the graph needs — and mints `id` from row position, like
`with_row_index`, when a frame has none (`prefix=` keeps two such classes apart).

```python continuation
world = economy()
world.update("""
PREFIX ex: <http://example.net/skabm#>
PREFIX def: <urn:maplib_default:>
DELETE { ?f def:price ?p } INSERT { ?f def:price 3e0 }
WHERE  { ?f a ex:Firm ; def:price ?p }
""")
RDFSimulator(n_periods=4).fit(world)      # your graph, ticked
```

### Three things are the frontend

**`fit_iter()`** for telemetry a calibrator or JAX takes as-is, **`extract()`** for statistics
over per-agent state, **`model_`** for interventions. The derived observables *are* the
summary statistics a method-of-moments loss consumes:

```python notest
model = simulator_model(economy, free=["growth_sigma"])    # a fresh world per candidate
model(theta, 120, seed)    # (120, D)
model.observables_         # the D column names, derived and name-sorted
```

Early stopping plugs into the same call, but skabm ships **no criterion** — `flax`'s
`EarlyStopping` already is that algorithm, so skabm supplies the seam and the padding only.

```python notest
model = simulator_model(economy, free=["growth_sigma"], stop=settled(patience=10))
model.ticks_               # 13 of 120: the run had settled
```

Padding carries the last row forward, so the loss stays finite and the candidate is judged on
its own numbers — 9.2s of candidates down to 1.0s here. `notebooks/extraction.ipynb` §4–5.

### Two models, and the JAX boundary

`model_` is the world: agents, plus the signal nodes their learned expectations live on, so
`?s ?p ?o` returns what a modeller expects. `meta_` is behaviour provenance. A rule written with
`skabm.dsl` has a differentiable tick: `dsl.jax_tick(rules)` steps `{class: {field: array}}`
(`dsl.arrays` builds it from the populations' frames, links as row indices) and drops into
`jax.lax.scan`. A rule that is still SPARQL text (Poledna, Schelling, traffic) reaches JAX the
way it reaches black-it, **through the telemetry**: the yielded rows are a numeric table that
`polars.DataFrame.to_jax` turns into arrays.

## Quickstart

A population calibrated from real Eurostat data, then simulated:

```python
import polars as pl
import polars_random as pr

from skabm.calibration import GeneticConstraintCalibration, make_dataset, weighted_enum
from maplib import Model

from skabm.datasets import build_firm_io_df
from skabm.simulation import RDFSimulator
from skabm.ottr import firm_template, household_template

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
world.map(firm_template, firms.with_iri())
world.map(household_template, households.with_iri("employer"))
sim = RDFSimulator(n_periods=12, params={"firm_ownership_ratio": 300 / 10_000,
                                         "growth_sigma": 0.02, "inflation_sigma": 0.01})
for row in sim.fit_iter(world):
    # the price level and household wealth are yielded; GDP is a per-agent
    # product taken before the sum, so it comes off the frame
    active = sim.extract().drop_nulls("output")
    print(row["sig__AVG__Firm__price"],
          (active["price"] * active["output"] * (1 - active["tech_share"])).sum())
```

Step 3 is what a marginal sampler cannot do. `size` and `industry` are drawn independently, so
their joint structure is whatever the draw produced; the calibrator permutes rows until
per-industry mean size matches Eurostat, and permutation leaves each column's multiset
untouched, so both marginals survive exactly — which is also why only *relative* sizes are
imposable.

| Where to look next | |
|---|---|
| [notebooks/poledna](notebooks/poledna/poledna.ipynb) | the model in five chapters on one population: **0** calibration, **1** the seven-rule quarter plus an intervention, **2** a banking layer propagating to a fixed point through `infer=`, **3** learned expectations as running sums on the signal nodes, **4** black-it parameter calibration read as an identification test. |
| [notebooks/extraction](notebooks/extraction.ipynb) | what to *do* with derived state: Gini as one polars expression, an intervention that moves it, telemetry into JAX |
| [notebooks/labour_automation](notebooks/labour_automation.ipynb) | occupational mobility under an automation shock — a labour-flow network whose central quantity lives on the *edge* |
| [notebooks/schelling](notebooks/schelling.ipynb) | spatial segregation on the lattice; the continuous-geometry swap is covered by `tests/test_schelling.py` |
| [notebooks/bayonne](notebooks/bayonne/app.py) | closing Bayonne's old town to cars: 9,400 commuters from the 2022 census on 37,000 OpenStreetMap links, day-to-day route and pivot-point mode choice, the closure as a graph edit — a Streamlit map, one working day per tick (`streamlit run notebooks/bayonne/app.py`) |

## Architecture

| Module | Role |
|---|---|
| `skabm.datasets` | Eurostat loaders (IO tables, business demography) |
| `skabm.calibration.population` | Population samplers + constraint calibrators (GA, Metropolis–Hastings), sklearn estimator API |
| `skabm.calibration.parameters` | Model calibration: `RDFSimulator` as a black-it `model(theta, N, seed)`, plus `noise_floor` |
| `skabm.ottr` | A registry of maplib `Template`s, one per agent class — the contract `Model.map` checks a population against — plus `SCHEMA`, what each field means and where each link lands, and `DataFrame.with_iri`, which gives a frame its IRIs |
| `skabm.dsl` | Rules as SymPy dicts: `Agents`, the graph operators, and the SPARQL and JAX compilers |
| `skabm.translate` | `rules(description, templates)`: DSL rules from a description, generated under the templates' grammar |
| `skabm.sparql` | Namespaces, `render` (param substitution) and the UDF registrars — random draws, `exp`/`log`, GeoSPARQL's `geof:sfIntersects`/`sfWithin` |
| `skabm.behaviour` | The rule library by function: `firm`, `household`, `macro`, `bank`, `labour`, `learning`, `traffic` |
| `skabm.history` | The observables the DSL rules imply, measuring them, and the DuckDB table that persists them when asked |
| `skabm.simulation` | `RDFSimulator`: `fit`/`fit_iter` over SPARQL update rules |

| Design decision, in sklearn vocabulary | |
|---|---|
| **X is the world** | a maplib `Model`, populations mapped into it with `skabm.ottr` and `Model.map`. Predicates are column names — the DataFrame schema *is* the graph schema — and an IRI-valued column (`employer`, `owns`) is a graph edge. Invariants that always hold ship as init rules (`firm_ownership`) rather than being re-encoded per dataset |
| **Rules are hyperparameters** | `string.Template` objects in `__init__`, so `get_params`/`clone` work. `init_rules` set initial conditions once after mapping; `update_rules` are the dynamics, upserted every tick in the paper's event order. The default set self-scopes: rules whose classes are all absent match nothing |
| **`model_` is the fitted artifact** | the graph where data and rules blend into one evolving world. Cold `fit` rebuilds it; `warm_start=True` continues it |
| **Structure vs state** | predicates no update rule touches (links, coefficients) are *structure*, written at fit and edited only by intervention; predicates the rules upsert are *state*, owned by the rules after t=0. Derived, not asserted: a `dsl.Rule` names what it writes, and `history.observables` measures exactly that |
| **Randomness is a UDF** | SPARQL has no `RAND`, so polars-random is registered as `pr:uniform`/`pr:normal` and called in-rule via `BIND(...)`, pinned by `RDFSimulator(random_seed=...)`. That is what lets the Schelling example derive its whole population in-graph with no seed column, and gives Poledna genuine AR(1) innovations |

**Two senses of "calibration".** `calibration.population` fits agent *rows* to accounting
identities and known margins (Poledna §4; Deville & Särndal 1992); `calibration.parameters`
fits *behavioural parameters* to macro series (Fagiolo et al. 2019), skabm supplying the
adapter and [black-it](https://github.com/bancaditalia/black-it) the search. The population is
calibrated once, outside the search loop, and held frozen while θ moves.

**Hooks, split by what they cost.** Per commit: `ruff`, a minimum-function-length check
(`tools/`), every ```python fence here, and `pytest --testmon` (~0.3s on an untouched tree).
Per push: the full suite with the coverage gate, plus `pytest --nbmake notebooks/`, so the
long-form docs get the guarantee the README does — minus `notebooks/poledna`, which is 3m24
and the only notebook hitting the network.

## Limitations

The backend is **pure SPARQL**, deliberately — a stress test of how far a declarative rule
engine carries an economic ABM, so the walls are documented on purpose.

| Wall | Why, and what is *not* walled |
|---|---|
| **No sequential search-and-matching** | the paper's goods, labour and credit markets are *random sequential*: consumers visit firms in random order until stocks run out. SPARQL expresses only the simultaneous approximation, so Poledna's employment links stay static. Matching *as such* is fine where a model states it in closed form: `behaviour.labour` does del Rio-Chanona et al. (2021) in five DSL rules, urn-ball function included, and reproduces the Beveridge curve; `schelling.RELOCATE` does one-to-one assignment with `pick` on matched ranks. The wall is agent-by-agent *ordering* |
| **The past is state, not a series** | a rule sees one time slice, so a behaviour that depends on history keeps what it needs as state: `dsl.lag` for a value one run ago, running sums for a statistic of the whole sample (SAC's mean and autocorrelation decompose exactly). A statistic over the unbounded past that does not decompose has no home |
| **Right-associative arithmetic** | in maplib's SPARQL, operators of equal precedence associate right, against the grammar: `?a - ?b + ?c` evaluates as `?a - (?b + ?c)`, silently. The `skabm.dsl` printer brackets every operation; a hand-written rule has to bracket every mixed chain, and a conservation law deserves a test (`test_labour_force_is_conserved` exists for this) |
| **Synchronous, staged activation** | and no scheduler, on purpose. A SPARQL UPDATE evaluates its WHERE against the pre-update graph, so all agents update simultaneously and rules fire in list order — exactly Mesa's `StagedActivation`, with `RandomActivation` structurally unreachable. Activation regime is a modelling assumption (Huberman & Glance 1993), and it only becomes an implementable choice with a numerical backend |
| **No `log`/`exp` in SPARQL** | a choice, not a limit: `rules.register_math` supplies `math:exp`/`math:log` on the same `add_udf` seam as the random draws, which is how `behaviour.labour` states its urn-ball function and S-curve verbatim. Which UDFs a rule set needs is the `udfs=` hyperparameter |
| **Reproducibility** | a *simulation* is reproducible via `random_seed=`; *calibration* is subtler, since `polars_random.set_random_seed` reaches only the `size=N` Series form and not the expression-over-frame form `make_dataset` uses, so a population is reproducible only when each sampler gets an explicit `seed=` |

## Roadmap

| | |
|---|---|
| **The rest in JAX** | Every rule that updates agents is a `skabm.dsl` rule, traffic and Schelling included. Their relational operators (`sum_over`, `total_by`, `running_sum`, `pick`) and scatters over many-valued links have no JAX form yet, and `jax_tick` refuses them by name. Rules that build topology or create agents (`firm_ownership`, `firm_entry`, `bank_depositors`, `SETTLE`, the neighbourhoods) stay SPARQL by design |
| **Every module from its docstring** | Each module in `skabm/behaviour` is a docstring specifying its rules, then the rules. `SKABM_LLM=1 pytest -k regenerates` has Qwen2.5-Coder-7B write them back from the docstring: `firm`, `household`, `macro`, `bank` and `labour` come back identical, in order; `schelling` and `traffic` do not yet (xfail, the reason on each), and `learning` has no rule of its own. A bigger model is the next thing to measure |
| **Rule-scoped validation** | a rule dict already names the fields it reads and writes; the step left is checking them against the graph at fit time and emitting SHACL shapes from the same sets |
| **Sensitivity-ranked observables** | 18 series is more than a figure needs. Ranking by paired-seed ε-shocks would order the panels and give calibration a divergence detector. Deliberately unbuilt: k+1 simulations per k parameters, and only valid once every draw in a tick is provably seeded |

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
