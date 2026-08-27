# skabm

![coverage](https://img.shields.io/badge/coverage-99%25-brightgreen)

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
| **Behaviour as data** | a rule is a SPARQL `string.Template` passed to `__init__`, so `get_params()`/`clone()` work and a rule set is serialisable, diffable and shareable — logic in the template, numbers in `params` |
| **Derived observables** | Mesa makes you declare `model_reporters`, Agents.jl `adata`, NetLogo a metric list — their rules are opaque host-language functions. skabm's are SPARQL, so it parses them through a real algebra and derives the `(class, predicate, aggregate)` triples they imply. Every binding constraint becomes a series too: the `min` in eq. 5 gives the share of firms pinned at `alpha * size`, the `max` in the Taylor rule the share of quarters at the zero lower bound |
| **A graph with memory** | a rule naming an `ex:sig__<agg>__<Class>__<predicate>` signal opens a DuckDB history; `history_rules` re-run over the accumulated series each tick and upsert their results back as triples. SAC learning is one `SELECT` plus one polars UDF; a moving average or an RL update is another, with no engine change |
| **Fixed-point propagation** | `infer=` hands recursive CONSTRUCT rules to maplib's reasoner, so contagion, transitive closure and reachability propagate across the whole graph in one call rather than hand-tuned sub-tick passes |
| **First-class interventions** | `model_` is a regular maplib model post-fit: rewire an ownership edge, delete a bank, halve a sector's demand with `model_.update(...)`, then continue with `warm_start=True` |

### What a run yields

```python
import polars as pl

from skabm.simulation import RDFSimulator

firms = pl.DataFrame({"id": ["firm_0", "firm_1"], "output": [100.0, 120.0],
                      "price": [1.0, 1.1], "alpha": [10.0, 10.0], "size": [12.0, 13.0],
                      "margin": [0.1, 0.1], "liquidity": [50.0, 60.0], "profit": [1.0, 1.0]})
households = pl.DataFrame({"id": ["hh_0", "hh_1"], "wealth": [100.0, 200.0],
                          "income": [10.0, 12.0], "psi": [0.9, 0.9],
                          "employer": ["firm_0", "firm_1"]})

sim = RDFSimulator(n_periods=8, random_seed=0)
run = pl.DataFrame(sim.fit_iter({"Firm": firms, "Household": households}))

run.shape                    # 8 ticks x (t + every aggregate the rules imply)
run.columns[:3]              # sig__AVG__Firm__binds__output, sig__AVG__Firm__liquidity, ...
len(sim.extract().columns)   # 18 per-agent columns, none of them named by hand
```

A whole run is `pl.DataFrame(sim.fit_iter(...))`, one tick is the metrics dict a callback API
takes, nothing was declared, and nothing opened a database. Each predicate is measured under
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
    for row in sim.fit_iter({"Firm": firms, "Household": households})
)
panel.lazy().group_by("t").agg(gini("wealth")).sort("t").collect()
```

### Getting data in

**Check the frames before you simulate.** Map a template yourself — maplib's own API:
```python continuation
from maplib import Model
from skabm.ottr import conform, firm_template, map_populations

Model().map(firm_template, conform(firms, firm_template))    # fine

try:
    Model().map(firm_template, conform(firms.drop("alpha"), firm_template))
except Exception as error:
    print(error)          # Expected column alpha is missing
```

Required parameters are what the rules read but never write: nothing produces them, so a
frame lacking one is a run of nothing. `conform` casts declared columns to the type the rules
join on — the failure worth the most, since an `Int64` where a rule wrote `xsd:double` joins
with nothing, silently.

**The world is an argument.** `fit` takes one thing: a maplib `Model`, advanced in place — a
graph another system built, or one you deserialized and intervened on. A `{class: DataFrame}`
mapping is put into a fresh one first and then does exactly the same:

```python continuation
world = Model()
map_populations(world, {"Firm": firms, "Household": households})
world.update("""
PREFIX ex: <http://example.net/skabm#>
PREFIX def: <urn:maplib_default:>
DELETE { ?f def:price ?p } INSERT { ?f def:price 3e0 }
WHERE  { ?f a ex:Firm ; def:price ?p }
""")
RDFSimulator(n_periods=4).fit(world)      # your graph, ticked
```

That one call maps every population together, so an undeclared reference resolves against
every id in it, not only the classes that went first: both key orders give the same graph.

### Three things are the frontend

**`fit_iter()`** for telemetry a calibrator or JAX takes as-is, **`extract()`** for statistics
over per-agent state, **`model_`** for interventions. The derived observables *are* the
summary statistics a method-of-moments loss consumes:

```python notest
model = simulator_model(populations, free=["growth_sigma"])
model(theta, 120, seed)    # (120, D)
model.observables_         # the D column names, derived and name-sorted
```

Early stopping plugs into the same call, but skabm ships **no criterion** — `flax`'s
`EarlyStopping` already is that algorithm, so skabm supplies the seam and the padding only.

```python notest
model = simulator_model(populations, free=["growth_sigma"], stop=settled(patience=10))
model.ticks_               # 13 of 120: the run had settled
```

Padding carries the last row forward, so the loss stays finite and the candidate is judged on
its own numbers — 9.2s of candidates down to 1.0s here. `notebooks/extraction.ipynb` §4–5.

### Two models, and the JAX boundary

`model_` is the world — agents and nothing else, so `?s ?p ?o` returns what a modeller expects.
`meta_` is the sidecar: rule IR, behaviour metadata, signal nodes, the virtualized history, any
UDF a history rule calls. The split is forced — registering a virtualization silently nulls
every graph-local aggregate on the model carrying it, and the behaviour rules are built out of
those. `meta_` is also the extension seam: a researcher's own forecasting module arrives as
`Model.add_udf`, named in SPARQL like any built-in. And skabm is JAX-compatible the way it is
black-it-compatible — **through the telemetry**, the yielded rows being a numeric table
`polars.DataFrame.to_jax` turns into arrays. There is no differentiable tick; every SPARQL
round trip breaks the trace.

## Quickstart

A population calibrated from real Eurostat data, then simulated:

```python
import polars as pl
import polars_random as pr

from skabm.calibration import GeneticConstraintCalibration, make_dataset, weighted_enum
from skabm.datasets import build_firm_io_df
from skabm.simulation import RDFSimulator

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

# 6. Simulate. The default rule set is the full Poledna economy and self-scopes to the
#    populations passed, so a Firm + Household run executes exactly those dynamics.
sim = RDFSimulator(n_periods=12, params={"firm_ownership_ratio": 300 / 10_000,
                                         "growth_sigma": 0.02, "inflation_sigma": 0.01})
for row in sim.fit_iter({"Firm": firms, "Household": households}):
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
| [notebooks/poledna](notebooks/poledna/poledna.ipynb) | the model in five chapters on one population: **0** calibration, **1** the seven-rule quarter plus an intervention, **2** a banking layer propagating to a fixed point through `infer=`, **3** learned expectations over a virtualized history, **4** black-it parameter calibration read as an identification test. |
| [notebooks/extraction](notebooks/extraction.ipynb) | what to *do* with derived state: Gini as one polars expression, an intervention that moves it, telemetry into JAX |
| [notebooks/labour_automation](notebooks/labour_automation.ipynb) | occupational mobility under an automation shock — a labour-flow network whose central quantity lives on the *edge* |
| [examples/schelling.py](examples/schelling.py) | spatial segregation on the lattice; the continuous-geometry swap is covered by `tests/test_schelling.py` |

## Architecture

| Module | Role |
|---|---|
| `skabm.datasets` | Eurostat loaders (IO tables, business demography) |
| `skabm.calibration.population` | Population samplers + constraint calibrators (GA, Metropolis–Hastings), sklearn estimator API |
| `skabm.calibration.parameters` | Model calibration: `RDFSimulator` as a black-it `model(theta, N, seed)`, plus `noise_floor` |
| `skabm.ottr` | One maplib `Template` per agent class: the contract a population is checked and cast against |
| `skabm.rules` | Namespaces, `render` (param substitution) and the UDF registrars |
| `skabm.behaviour` | The rule library by economic function: `firm`, `household`, `macro`, `bank`, `labour`, `learning` |
| `skabm.ir` | Rule IR: read/write sets off the SPARQL algebra, the state/structure partition, the per-class schema, the observables implied |
| `skabm.history` | Measuring the observables, and the DuckDB sidecar that persists and virtualizes them (chrontext) |
| `skabm.simulation` | `RDFSimulator`: `fit`/`fit_iter` over SPARQL update rules |

| Design decision, in sklearn vocabulary | |
|---|---|
| **X is the world** | a maplib `Model`, or a `{class: DataFrame}` mapping put into one. Predicates are column names — the DataFrame schema *is* the graph schema — and an IRI-valued column (`employer`, `owns`) is a graph edge. Invariants that always hold ship as init rules (`firm_ownership`) rather than being re-encoded per dataset |
| **Rules are hyperparameters** | `string.Template` objects in `__init__`, so `get_params`/`clone` work. `init_rules` set initial conditions once after mapping; `update_rules` are the dynamics, upserted every tick in the paper's event order. The default set self-scopes: rules whose classes are all absent match nothing |
| **`model_` is the fitted artifact** | the graph where data and rules blend into one evolving world. Cold `fit` rebuilds it; `warm_start=True` continues it |
| **Structure vs state** | predicates no update rule touches (links, coefficients) are *structure*, written at fit and edited only by intervention; predicates the rules upsert are *state*, owned by the rules after t=0. Derived, not asserted — `skabm.ir` reads it off the algebra, and it settles both the extract's columns and the observables |
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
| **No sequential search-and-matching** | the paper's goods, labour and credit markets are *random sequential*: consumers visit firms in random order until stocks run out. SPARQL expresses only the simultaneous approximation, so Poledna's employment links stay static. Matching *as such* is fine where a model states it in closed form: `behaviour.labour` does del Rio-Chanona et al. (2021) in seven rules, urn-ball function included, and reproduces the Beveridge curve; `schelling.RELOCATE` does one-to-one assignment as a rank join. The wall is agent-by-agent *ordering* |
| **Estimation on history is a database** | a SPARQL update rule sees one time slice, so the model's own past lives in DuckDB, virtualized back through chrontext. Two chrontext properties shape this and both fail *silently*: only `Model.query` federates (`update`/`insert` see virtualized data as empty, CONSTRUCT panics), and registering a virtualization makes every graph-local aggregate return `None`. `tests/test_history.py` pins both |
| **Right-associative arithmetic** | in maplib's SPARQL, operators of equal precedence associate right, against the grammar: `?a - ?b + ?c` evaluates as `?a - (?b + ?c)`, silently. Chains led by `+` or `*` survive algebraically, which is why the shipped rules are unaffected — bracket every mixed chain, and give conservation laws a test (`test_labour_force_is_conserved` exists for this) |
| **Synchronous, staged activation** | and no scheduler, on purpose. A SPARQL UPDATE evaluates its WHERE against the pre-update graph, so all agents update simultaneously and rules fire in list order — exactly Mesa's `StagedActivation`, with `RandomActivation` structurally unreachable. Activation regime is a modelling assumption (Huberman & Glance 1993), and it only becomes an implementable choice with a numerical backend |
| **No `log`/`exp` in SPARQL** | a choice, not a limit: `rules.register_math` supplies `math:exp`/`math:log` on the same `add_udf` seam as the random draws, which is how `behaviour.labour` states its urn-ball function and S-curve verbatim. Which UDFs a rule set needs is the `udfs=` hyperparameter |
| **Reproducibility** | a *simulation* is reproducible via `random_seed=`; *calibration* is subtler, since `polars_random.set_random_seed` reaches only the `size=N` Series form and not the expression-over-frame form `make_dataset` uses, so a population is reproducible only when each sampler gets an explicit `seed=` |

## Roadmap

| | |
|---|---|
| **Numerical backends** | compile the graph to polars frames (or `.to_jax()` for differentiable kernels), step in frame-land, re-map at observation points. The SPARQL path becomes the slow, semantically transparent reference the fast kernels are validated against — and the home of search-and-matching, activation regimes and gradient-based calibration |
| **Rule-scoped validation** | `skabm.ir` already knows the predicates each rule traverses; the step left is checking coverage against the graph at fit time and emitting SHACL shapes from the same read/write sets |
| **Sensitivity-ranked observables** | 19 series is more than a figure needs. Ranking by paired-seed ε-shocks would order the panels and give calibration a divergence detector. Deliberately unbuilt: k+1 simulations per k parameters, and only valid once every draw in a tick is provably seeded |

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
