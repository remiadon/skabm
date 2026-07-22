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
from skabm.calibration import make_dataset, weighted_enum
from skabm.simulation import RDFSimulator

# 1. Calibrate agent populations as DataFrames (marginals from real data)
firms = make_dataset(samplers={...}, n_agents=300)
households = make_dataset(samplers={...}, n_agents=10_000)

# 2. Simulate: one DataFrame per agent kind; default rules are assembled
#    from the kinds you pass (only Firm + Household -> only their dynamics)
sim = RDFSimulator(n_periods=12)
for state in sim.fit_iter(Firm=firms, Household=households):
    print(state.select((pl.col("price") * pl.col("output")).sum()))

# 3. Post-fit, sim.model_ is a regular maplib model: SPARQL queries,
#    interventions, visualization (explore()), serialization
sim.model_.query("SELECT ?h ?w WHERE { ?h a ex:Household ; def:wealth ?w }")
```

See [examples/poledna_maplib_demo.py](examples/poledna_maplib_demo.py) for
the full six-sector economy (~10,000 agents, 120 quarters in ~5 seconds).

## Architecture

| Module | Role |
|---|---|
| `skabm.datasets` | Eurostat loaders (IO tables, business demography) |
| `skabm.calibration` | Population samplers + constraint calibrators (GA, Metropolis–Hastings) with a sklearn estimator API |
| `skabm.rules` | `map_df` (DataFrame → graph, auto-generated templates) + SPARQL `string.Template` rules + `render` (param substitution) |
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

## Limitations

The current simulation backend is **pure SPARQL** — deliberately, as a
stress test of how far a declarative, set-based rule engine carries an
economic ABM. The walls we hit are documented here on purpose.

**No search-and-matching.** The paper's goods, labor, and credit markets
are *random sequential* algorithms: consumers visit firms in random order,
first-come-first-served, until stocks run out. Declarative SPARQL can only
express the simultaneous approximation (demand allocated proportionally to
supply shares). Consequently employment links are static — hiring and
firing would require rewiring `employer` triples under per-firm vacancy
quotas, which needs an ordering no single UPDATE can express.

**No stochastic shocks.** maplib's SPARQL has no seedable `RAND`, so the
paper's exogenous AR(1) processes degenerate to deterministic drifts, and
Monte-Carlo ensembles (the paper runs 500 per forecast) are out of reach
in-graph. The pure-SPARQL simulator is a deterministic scenario engine,
not a forecasting engine.

**No in-graph estimation.** The paper's agents re-estimate AR(1)
expectation rules on the model's own history every quarter (behavioral
learning); SPARQL cannot run regressions, so expectations are constant
parameters. Likewise SPARQL has no `log`/`exp`, so log-level laws of
motion become linear growth factors.

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

**Reproducibility caveat.** `make_dataset` seeds its sampling draws but
polars-random value pools are not yet seedable end-to-end
([polars-random#27](https://github.com/diegoglozano/polars-random/issues/27)),
so calibrated populations differ between runs.

## Roadmap

- **Numerical backends**: compile the graph to polars frames
  (`Model.query` → DataFrame → polars expressions, or `.to_jax()` for
  differentiable kernels), step in frame-land, re-map at observation
  points. The SPARQL path then becomes the slow, semantically transparent
  reference implementation the fast kernels are validated against — and
  the home of stochastic shocks, search-and-matching, activation regimes,
  and gradient-based calibration.
- **Rule-scoped validation**: each rule knows the predicates it traverses;
  check coverage against the graph at fit time (eventually SHACL shapes).
- **Self-describing export**: serialize `model_` together with its rule
  strings — world and behavior in a single shareable artifact.

## References

- Poledna, Miess, Hommes & Rabitsch (2023). Economic forecasting with an
  agent-based model. *European Economic Review* 151, 104306.
- Hommes & Zhu (2014). Behavioral learning equilibria. *JET* 150.
- Huberman & Glance (1993). Evolutionary games and computer simulations.
  *PNAS* 90(16).
- [maplib](https://github.com/DataTreehouse/maplib) — Rust knowledge-graph
  toolkit with polars-native OTTR templates and SPARQL.
- [AMBER](https://github.com/a11to1n3/AMBER) — polars-based ABM framework.
