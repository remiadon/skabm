# skabm

![coverage](https://img.shields.io/badge/coverage-97%25-brightgreen)

**scikit-learn-style agent-based modeling on a knowledge graph.**

skabm builds economic agent-based models (ABMs) from public data with as
little user code as possible. Agent populations are calibrated as
[polars](https://pola.rs) DataFrames, lifted into an RDF knowledge graph
([maplib](https://github.com/DataTreehouse/maplib)), and simulated by
applying SPARQL rules to that graph — all behind a scikit-learn-shaped API.

The reference implementation follows Poledna, Miess, Hommes & Rabitsch
(2023), *Economic forecasting with an agent-based model* (European Economic
Review 151): a full six-sector economy — firms, households, government,
banks, central bank, rest of world — calibrated 1:1 from Eurostat national
accounts, input–output tables, census and business demography data.

## Why a knowledge graph?

Traditional ABM frameworks struggle to express **heterogeneous agents**:
populations of different sizes, different attributes, and typed relations
between them (household → employer, household → owned firm, firm →
industry). Array-based frameworks (e.g. [AMBER](https://github.com/a11to1n3/AMBER))
get the performance story right but flatten this structure into
undocumented index arrays. skabm keeps the polars performance and makes the
structure explicit: agents are RDF nodes, relations are triples, and
behavior is SPARQL over those triples. What used to be a DataFrame `join`
becomes graph traversal; what used to be a hard-coded interaction matrix
becomes a queryable, editable, shareable ontology.

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
for state in sim.fit_iter(Firm=firms, Household=households):
    active = state.drop_nulls("output")
    print((active["price"] * active["output"] * (1 - active["tech_share"])).sum())
```

Step 3 is the part a marginal sampler cannot do. `size` and `industry` are drawn
independently, so their joint structure is whatever the draw happened to produce;
the calibrator permutes rows until per-industry mean size matches Eurostat, and
because permutation leaves each column's multiset untouched, both marginals survive
exactly. That invariance also fixes the `shift`: the global mean of `log(size)` cannot
move, so only *relative* sizes across industries are imposable — the absolute scale
stays the sampler's. Pass the calibrator only the columns its constraint mentions;
each free column is permuted independently, so a derived column handed to it (`alpha`
here) would be shuffled away from the industry it belongs to.

[notebooks/poledna/poledna.ipynb](notebooks/poledna/poledna.ipynb) builds the
model up in five chapters on one shared population. **Chapter 0** is the
calibration above at notebook scale; **1** runs the seven-rule quarter and
intervenes on the fitted graph; **2** adds a banking layer whose distress
propagates to a fixed point through `infer=`; **3** opens up the learned
expectations — a DuckDB history, virtualized back into SPARQL — and swaps the
estimator to show the seam is generic; **4** hands the behavioural parameters to
[black-it](https://github.com/bancaditalia/black-it) and reads the result as an
identification test on the rule set. [poledna.py](poledna.py) is the same model
as a plain script.

Elsewhere, [examples/schelling.py](examples/schelling.py) covers spatial segregation
and [notebooks/labour_automation.ipynb](notebooks/labour_automation.ipynb)
occupational mobility under an automation shock — a labour-flow network where
the model's central quantity lives on the *edge*.

## Architecture

| Module | Role |
|---|---|
| `skabm.datasets` | Eurostat loaders (IO tables, business demography) |
| `skabm.calibration.population` | Population samplers + constraint calibrators (GA, Metropolis–Hastings) with a sklearn estimator API |
| `skabm.calibration.parameters` | Model calibration: `RDFSimulator` as a black-it `model(theta, N, seed)`, plus the `noise_floor` diagnostic |
| `skabm.rules` | `map_df` (DataFrame → graph, auto-generated templates) + SPARQL `string.Template` rules + `render` (param substitution) + UDF registrars |
| `skabm.behaviour` | The rule library, grouped by economic function: `firm`, `household`, `macro`, `bank`, `labour`, `learning` |
| `skabm.history` | State history in DuckDB, virtualized back into SPARQL (chrontext) — the graph's memory |
| `skabm.simulation` | `RDFSimulator`: fit/fit_iter over SPARQL update rules |

**Two senses of "calibration".** The word means different things in the
microsimulation and ABM literatures, so `skabm.calibration` is split in two and
re-exports both. `calibration.population` fits the agent *rows* to accounting
identities and known margins — IO coefficients, census shares, Basel III ratios
— which is Poledna §4's usage and survey sampling's (Deville & Särndal 1992,
calibration estimators). `calibration.parameters` fits the *behavioural
parameters* to macro time series, the ABM literature's usual sense (Fagiolo et
al. 2019); skabm supplies the adapter and
[black-it](https://github.com/bancaditalia/black-it) supplies the search, via
`uv sync --extra model-calibration`.

The two compose in the sklearn way — population calibration is a transformer on
`X`, model calibration is a search over `get_params()` — and the ordering
follows: the population is calibrated once, *outside* the search loop, and held
frozen while `θ` moves inside it.

Design decisions, in sklearn vocabulary:

- **X is the data**: heterogeneous agent populations, passed to `fit` as
  keyword DataFrames (`Firm=`, `Household=`, ...). Predicates in the graph
  are simply column names — the DataFrame schema *is* the graph schema.
  **Relations live in the data too**: an IRI-valued column (a household's
  `employer`, `owns`) becomes a graph edge at map time, and init rules
  traverse those edges. Economic invariants that always hold — e.g. "a
  `firm_ownership_ratio` fraction of households own firms" — are *not* the
  user's job to encode per dataset: they ship as init rules
  (`FIRM_OWNERSHIP`) that create the edges in-graph, controlled by a
  hyperparameter, and only where the data left them undefined.
- **Rules are hyperparameters**: SPARQL `string.Template` objects in
  `__init__` (serializable, so `get_params`/`clone` work). `init_rules`
  set *initial conditions* (run once, after mapping); `update_rules` are
  the *dynamics* (DELETE/INSERT upserts, run every tick in the paper's
  event order). Rule *logic* lives in the templates, rule *numbers* in the
  `params` dict — merged over `rules.POLEDNA_PARAMS` and substituted into
  `$placeholders` at fit time, so overriding one value is
  `params={"total_deposits": 2.5e4}`, never a rewritten rule. The default
  rule set covers the full economy and self-scopes: at fit time, rules
  whose referenced agent classes are all absent from the populations are
  filtered out, so passing only `Firm=` and `Household=` runs exactly the
  firm and household dynamics.
- **`model_` is the fitted artifact**: the graph where data and rules
  blend into one evolving world. Cold `fit` rebuilds it; `warm_start=True`
  continues it — after an intervention (`model_.update(...)`), under
  different update rules, or on a hand-built model.
- **Structure vs state**: predicates no update rule touches (links,
  coefficients, classes) are *structure*, written at fit and edited only
  by explicit intervention; predicates the rules upsert (output, price,
  wealth, ...) are *state*, owned by the rules after t=0. The partition is
  derivable from the rule strings.
- **Randomness is a UDF**: SPARQL has no `RAND`, so skabm registers
  polars-random as SPARQL functions (`rules.register_polars_random`,
  maplib ≥ 0.20.26): `pr:uniform`/`pr:normal`, called in-rule via
  `BIND(pr:uniform(0e0, 1e0) AS ?u)`, with `RDFSimulator(random_seed=...)`
  pinning them for reproducible runs. This is what lets the Schelling
  example derive its whole population in-graph with no seed column, and
  gives Poledna genuine AR(1) innovations (`$..._sigma` params, off by
  default) — Monte-Carlo ensembles included.

## Limitations

The current simulation backend is **pure SPARQL** — deliberately, as a
stress test of how far a declarative, set-based rule engine carries an
economic ABM. The walls we hit are documented here on purpose.

**No *sequential* search-and-matching.** The paper's goods, labor, and
credit markets are *random sequential* algorithms: consumers visit firms in
random order, first-come-first-served, until stocks run out. Declarative
SPARQL can only express the simultaneous approximation (demand allocated
proportionally to supply shares). Consequently Poledna's employment links are
static — hiring and firing would require rewiring `employer` triples under
per-firm vacancy quotas, which needs an ordering no single UPDATE can
express.

What is *not* out of reach is search-and-matching as such. Where a model
specifies matching as a closed form rather than a queue, SPARQL states it
directly: `behaviour.labour` implements del Rio-Chanona et al. (2021) — an
occupational mobility network where unemployed workers apply across a
weighted job-transition graph and vacancies draw from their applicant pools —
in seven rules, urn-ball matching function included, and reproduces the
Beveridge curve. Where a model needs one-to-one assignment,
`schelling.RELOCATE`'s rank join does it in a single batch. The wall is
agent-by-agent *ordering*, not matching.

**Estimation on the model's own history is a database, not a rule.** The
paper's agents re-estimate their AR(1) expectation rules on the model's own
history every quarter (behavioral learning, eq. 6/9). A SPARQL update rule
sees one time slice — the present — so history lives in a DuckDB table that
gains one row per tracked aggregate per tick, virtualized back into SPARQL
through maplib's chrontext (`skabm.history`). Sample-Autocorrelation learning
is then exactly two things the simulator already accepts: a polars UDF passed
through `udfs=`, and a SPARQL `SELECT` passed through `history_rules=` that
runs it over the virtualized series (`behaviour.learning`). Behaviour rules
themselves stay graph-local — `firm_produce` reads one `def:forecast` triple
and falls back to its `$growth_e` parameter until enough history exists.
Nothing in `skabm.history` knows what is being learned, so a moving average or
an RL update is another query and another UDF and no engine change.

Two chrontext properties shape that design, and both fail *silently*: only
`Model.query` federates (`update` and `insert` see virtualized data as empty,
and CONSTRUCT panics), and registering a virtualization makes every
graph-local aggregate on that model return `None` — which is why the
virtualization lives on a separate learner model and results are written back
as triples. `tests/test_history.py` pins both.

SPARQL also has no `log`/`exp` of its own, so the Poledna rules turn
log-level laws of motion into linear growth factors — but that is a choice,
not a limit: `rules.register_math` supplies `math:exp` / `math:log` on the
same `add_udf` seam as the random draws, which is how `behaviour.labour`
states its urn-ball matching function and its S-curve technology shock
verbatim. Which UDFs a rule set needs is the `udfs=` hyperparameter.

**Right-associative arithmetic in maplib's SPARQL.** Operators of equal
precedence associate to the right, against the SPARQL grammar: `?a - ?b + ?c`
evaluates as `?a - (?b + ?c)` and `?a / ?b * ?c` as `?a / (?b * ?c)`, silently
and with no error. Chains led by `+` or `*` survive it algebraically, which is
why the Poledna and Schelling rules are unaffected — but bracket every mixed
chain explicitly, and give any conservation law a test that pins its invariant
(`tests/test_labour.py::test_labour_force_is_conserved` exists for exactly
this reason).

**Activation is synchronous and staged — and there is no scheduler, on
purpose.** ABM "schedulers" do two jobs: ordering behavioral *phases*
within a tick, and ordering *agents* within a phase. The ABM literature's
hard-won result concerns the second: the activation regime is a modeling
assumption, not a technicality — synchronous vs asynchronous updating can
qualitatively change outcomes (Huberman & Glance 1993). In this
architecture the engine decides it for us: a SPARQL UPDATE evaluates its
WHERE against the pre-update graph, so all agents update simultaneously
(synchronous), and rules fire in list order (staged) — exactly Mesa's
`StagedActivation`, with `RandomActivation` structurally unreachable. An
in-house scheduler could only reorder phases, which an ordered rule list
already does; the investment only becomes worthwhile with a numerical
backend, where activation regimes become a real, implementable choice.

**Reproducibility caveat.** A *simulation* is reproducible via
`RDFSimulator(random_seed=...)`, which pins the `pr:*` UDF draws. *Calibration*
is subtler: `polars_random.set_random_seed` only reaches the `size=N` Series
form, not the expression-over-frame form `make_dataset` uses for its value
pools, so a population is reproducible only when each `pr.*` sampler (and
`weighted_enum`) is given an explicit `seed=`.

## Roadmap

- **Numerical backends**: compile the graph to polars frames
  (`Model.query` → DataFrame → polars expressions, or `.to_jax()` for
  differentiable kernels), step in frame-land, re-map at observation
  points. The SPARQL path then becomes the slow, semantically transparent
  reference implementation the fast kernels are validated against — and
  the home of search-and-matching, activation regimes, and gradient-based
  calibration.
- **Rule-scoped validation**: each rule knows the predicates it traverses;
  check coverage against the graph at fit time (eventually SHACL shapes).
- **Self-describing export**: serialize `model_` together with its rule
  strings — world and behavior in a single shareable artifact.

## References

- Poledna, Miess, Hommes & Rabitsch (2023). Economic forecasting with an
  agent-based model. *European Economic Review* 151, 104306.
- del Rio-Chanona, Mealy, Beguerisse-Díaz, Lafond & Farmer (2021).
  Occupational mobility and automation: a data-driven network model.
  *J. R. Soc. Interface* 18(174), 20200898.
- Hommes & Zhu (2014). Behavioral learning equilibria. *JET* 150.
- Huberman & Glance (1993). Evolutionary games and computer simulations.
  *PNAS* 90(16).
- Deville & Särndal (1992). Calibration estimators in survey sampling.
  *JASA* 87(418).
- Fagiolo, Guerini, Lamperti, Moneta & Roventini (2019). Validation of
  agent-based models in economics and finance. In *Computer Simulation
  Validation*, Springer.
- [black-it](https://github.com/bancaditalia/black-it) — Banca d'Italia's ABM
  parameter-calibration toolbox.
- [maplib](https://github.com/DataTreehouse/maplib) — Rust knowledge-graph
  toolkit with polars-native OTTR templates and SPARQL.
- [AMBER](https://github.com/a11to1n3/AMBER) — polars-based ABM framework.
