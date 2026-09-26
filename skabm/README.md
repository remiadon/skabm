# skabm, the engine

How a simulation works under the hood: the world, the rules, what gets measured, and
where the walls are. The rule library has its own page ([behaviour](behaviour/README.md)),
and so do populations and parameter search ([calibration](calibration/README.md)).

| Module | Role |
|---|---|
| `template` | A registry of maplib `Template`s, one per agent class (the contract `Model.map` checks a population against), plus `SCHEMA` (what each field means and where each link lands) and `DataFrame.with_iri`, which gives a frame its IRIs |
| `dsl` | Rules as SymPy dicts: `Agents`, the graph operators, and the SPARQL and JAX compilers |
| `translate` | `rules(description, templates)`: DSL rules generated from a description, under the templates' grammar |
| `sparql` | Namespaces and the UDF registrars: random draws, `exp`/`log` (a rule compiles to SPARQL through `dsl.sparql`) |
| `simulation` | `RDFSimulator`: `fit`/`fit_iter` over rule dicts compiled to SPARQL |
| `datasets` | Eurostat loaders (IO tables, business demography) |

## Design, in scikit-learn vocabulary

| | |
|---|---|
| **X is the world** | a maplib `Model`, with populations mapped into it by `skabm.template` and `Model.map`. Predicates are column names, so the DataFrame schema *is* the graph schema, and an IRI-valued column (`employer`, `owns`) is a graph edge. Structure is built in polars before mapping (`firm.ownership`, `bank.depositors`, `schelling.grid_neighbors`), never by a rule |
| **Rules are hyperparameters** | rule dicts are passed to `__init__`, so `get_params`/`clone` work. `rules` (a module's `RULES`) are the dynamics, upserted every tick in the paper's event order. There is no initialisation phase: the world a fit starts from is its opening state. The default set self-scopes, since rules whose classes are all absent match nothing |
| **`model_` is the fitted artifact** | the graph where data and rules blend into one evolving world. A cold `fit` rebuilds it; `warm_start=True` continues it |
| **Structure vs state** | predicates no rule writes (links, coefficients) are *structure*: written at fit, and edited only by intervention. Predicates the rules upsert are *state*, owned by the rules after t=0. This is derived, not asserted: a rule dict names what it writes |
| **Randomness is a UDF** | SPARQL has no `RAND`, so polars-random is registered as `pr:uniform`/`pr:normal` and pinned by `RDFSimulator(random_seed=...)`. `Normal`/`Uniform` in a DSL rule compile to it |

## The world

`fit` takes one thing: a maplib `Model`, advanced in place. It can be one you mapped, a
graph another system built, or one you deserialized and intervened on. That is also why
examples build the world in a function: each run gets a fresh one. Mapping a population is
maplib's own `Model.map`, which checks the frame before anything simulates. A missing
column, a typo or an integer column where the rules compute in doubles is refused at map
time, naming the column, rather than silently joining with nothing twelve ticks later.
Undeclared columns are refused too: drop them, or declare your own class with
`template.agent`.

Importing `skabm.template` registers `DataFrame.with_iri(*links)`, which turns `id` and the
named link columns into `ex:` IRIs.

`model_` is the world: agents, plus the signal nodes their learned expectations live on,
so `?s ?p ?o` returns what a modeller expects. `meta_` holds behaviour provenance: which
rule, from which module, citing which source.

## Rules

`skabm.dsl` writes a rule as SymPy over the template's fields:

- a link is followed by name (`Edge.src.unemployment`);
- `sum_over(link, e)` sums over the agents a link connects, and `total`/`mean` aggregate
  over a whole class;
- `lag(e)` is a value one step ago; `coalesce(a, b, …)` is the first of its arguments
  that exists;
- `Normal`/`Uniform` are fresh draws per agent;
- the relational operators `total_by`, `running_sum` and `pick` cover shared keys,
  cumulative sums by rank, and one-to-one matching.

A rule compiles to SPARQL for maplib. Through `sympy.lambdify`, it also compiles to JAX
for `jax.grad`: `dsl.jax_tick(rules)` steps `{class: {field: array}}`, and
`dsl.arrays` builds that state from the populations' frames, with links as row indices.
The tick drops into `jax.lax.scan`. The labour market runs both ways to 1e-12, and a
gradient through 25 ticks matches finite differences. The relational operators, and
scatters over many-valued links, have no JAX form yet, and `jax_tick` refuses them by
name. A simulation run on SPARQL still reaches JAX through its telemetry: a run is a
polars frame, which `polars.DataFrame.to_jax` turns into arrays.

## What gets measured

`fit_iter` yields every agent after each tick, one row each (every field it holds, with
`class` and `t`; a many-valued link, like a cell's neighbours, is a list; `state_extract=`
replaces the default frame), so a run is `pl.concat(sim.fit_iter(world))` and a macro quantity is
polars over it: `run.group_by("t").agg(...)`. There is nothing to declare to the simulator.

Mesa makes you declare `model_reporters`, Agents.jl `adata`, NetLogo a metric list. Here the
run is the data: `class` tells apart the agents that share a field, and a binding constraint
is one comparison, `(pl.col("output") >= pl.col("alpha") * pl.col("size")).mean()` for the
share of firms at Poledna's eq. 5 cap.

What to look at is the researcher's call: calling the total of firm output *GDP* is a
modelling claim, and so is choosing a Gini over a percentile ratio.

## Learned expectations

A rule naming an `ex:sig__<agg>__<Class>__<predicate>` signal reads a learned expectation,
and the simulator runs one `learner` rule per signal after every tick. SAC learning
(Hommes & Zhu 2014) is a DSL rule over seven running sums on the signal node, so it needs
no stored series. A moving average or an RL update is just another `learner` function.

## Rules from a description

`translate.rules(description, templates)` sends the description and the templates to a
local model, generating under a Lark grammar built from those templates (llguidance). The
model can only write the DSL over the fields the templates declare, so its answer runs as
the Python it is, with nothing but the DSL in scope. It returns `{name: rule}` in run
order.

The default model is Qwen2.5-Coder-7B. On Apple silicon it runs 4-bit through MLX (~4 GB);
elsewhere it runs bf16 through transformers (~15 GB). Install with `pip install "skabm[llm]"`.
Any `(prompt, lark grammar) -> text` callable can be passed as `model=` instead.

## Limitations

The backend is pure SPARQL, deliberately: it is a stress test of how far a declarative
rule engine carries an economic ABM, so the walls are documented on purpose.

| Wall | Why, and what is *not* walled |
|---|---|
| **No sequential search-and-matching** | the paper's goods, labour and credit markets are *random sequential*: consumers visit firms in random order until stocks run out. SPARQL expresses only the simultaneous approximation, so Poledna's employment links stay static. Matching as such is fine where a model states it in closed form: `behaviour.labour` does del Rio-Chanona et al. (2021), urn-ball function included, and reproduces the Beveridge curve; `schelling.RELOCATE` does one-to-one assignment with `pick` on matched ranks. The wall is agent-by-agent *ordering* |
| **The past is state, not a series** | a rule sees one time slice, so a behaviour that depends on history keeps what it needs as state: `dsl.lag` for a value one step ago, running sums for a statistic of the whole sample (SAC's mean and autocorrelation decompose exactly). A statistic over the unbounded past that does not decompose has no home |
| **Right-associative arithmetic** | in maplib's SPARQL, operators of equal precedence associate right, against the grammar: `?a - ?b + ?c` evaluates as `?a - (?b + ?c)`, silently. The `skabm.dsl` printer brackets every operation, so this only bites hand-written SPARQL |
| **Synchronous, staged activation** | and no scheduler, on purpose. A SPARQL UPDATE evaluates its WHERE against the pre-update graph, so all agents update simultaneously and rules fire in list order: exactly Mesa's `StagedActivation`, with `RandomActivation` structurally unreachable. Activation regime is a modelling assumption (Huberman & Glance 1993), and it only becomes an implementable choice with a numerical backend |
| **`log`/`exp` are UDFs** | plain SPARQL has neither. `sparql.register_math` supplies `math:exp`/`math:log` on the same `add_udf` seam as the random draws, which is how `behaviour.labour` states its urn-ball function and S-curve verbatim. Which UDFs a rule set needs is the `udfs=` hyperparameter |
| **Reproducibility** | a *simulation* is reproducible via `random_seed=`. *Calibration* is subtler: `polars_random.set_random_seed` reaches only the `size=N` Series form, not the expression form `make_dataset` uses, so a population is reproducible only when each sampler gets an explicit `seed=` |

## Development

Hooks are split by what they cost. Per commit: `ruff`, every ```python fence in the root README,
and `pytest --testmon` (~0.3 s on an untouched tree). Per push: the full suite with the
coverage gate, plus `pytest --nbmake notebooks/`, minus `notebooks/poledna`, which takes
3m24 and is the only notebook hitting the network.
