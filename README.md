# skabm

![coverage](https://img.shields.io/badge/coverage-99%25-brightgreen)

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

```python
import polars as pl
from skabm.calibration import make_dataset
from skabm.behaviour.household import kinked_consume, household_income
from skabm.behaviour.firm import firm_produce, firm_price
from skabm.behaviour.macro import centralbank_rate
from skabm.simulation import RDFSimulator


# 1. Calibrate agent populations as DataFrames (marginals from real data)
firms = make_dataset(samplers={...}, n_agents=300)
households = make_dataset(samplers={...}, n_agents=10_000)

# 1. Choose (or build) the behavioural rules you want — each is a SPARQL
#    Template imported from skabm.behaviour; the canonical Poledna parameter
#    set lives in skabm.behaviour.params and is merged at fit time.
rules = [firm_produce, firm_price, household_income, kinked_consume, centralbank_rate]


# 2. Calibrate agent populations as DataFrames (marginals from real data)
#    and simulate: the simulator only runs rules whose referenced agent classes
#    are present, so a Firm + Household run executes exactly those dynamics.
sim = RDFSimulator(n_periods=12, update_rules=rules)
for state in sim.fit_iter(Firm=firms, Household=households):
    print(state.select((pl.col("price") * pl.col("output")).sum()))

# 3. Post-fit, sim.model_ is a regular maplib Model — SPARQL queries,
#    interventions (do-calculus style), visualization (explore()),
#    and serialization all work on it directly.
#    Here: a single do-calculus intervention — set every firm's price to a
#    fixed value (Pearl-style do-operator), then resume simulation under
#    warm_start.  The do-operator abstraction is public-domain notation;
#    production-grade identification/estimation tooling is a 3rd-party
#    (licensed) concern and out of scope for the core engine.
sim.model_.update("DELETE { ?f ex:price ?p } INSERT { ?f ex:price 42e0 } WHERE { ?f a ex:Firm }")
for state in sim.fit_iter(warm_start=True):
    print(state.select((pl.col("price") * pl.col("output")).sum()))
```

See [examples/poledna_maplib_demo.py](examples/poledna_maplib_demo.py) for
the full six-sector economy (~10,000 agents, 120 quarters in ~5 seconds),
[examples/schelling.py](examples/schelling.py) for spatial segregation, and
[examples/labour_automation.py](examples/labour_automation.py) for
occupational mobility under an automation shock — a labour-flow network where
the model's central quantity lives on the *edge*.

## Architecture

| Module | Role |
|---|---|
| `skabm.datasets` | Eurostat loaders (IO tables, business demography) |
| `skabm.calibration` | Population samplers + constraint calibrators (GA, Metropolis–Hastings) with a sklearn estimator API |
| `skabm.rules` | `map_df` (DataFrame → graph, auto-generated templates) + SPARQL `string.Template` rules + `render` (param substitution) + UDF registrars |
| `skabm.behaviour` | The rule library, grouped by economic function: `firm`, `household`, `macro`, `bank`, `labour` |
| `skabm.simulation` | `RDFSimulator`: fit/fit_iter over SPARQL update rules |

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

**No in-graph estimation.** The paper's agents re-estimate their AR(1)
expectation rules on the model's own history every quarter (behavioral
learning); SPARQL cannot run regressions, so the *expectation parameters*
are constants (the exogenous processes do carry `pr:normal` innovations —
see "Randomness is a UDF" above — they just aren't re-fit from history).
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
- [maplib](https://github.com/DataTreehouse/maplib) — Rust knowledge-graph
  toolkit with polars-native OTTR templates and SPARQL.
- [AMBER](https://github.com/a11to1n3/AMBER) — polars-based ABM framework.
