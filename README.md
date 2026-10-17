# skabm

![coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)

**Agent-based models written as equations, or in plain English, and run on a knowledge graph.**

Say what an agent does, in a paper's equations or your own, and it becomes a rule. Run
the rules over a population calibrated from public data, and get back every agent after
every tick, as polars: any macro quantity is one expression. Agents are nodes of an RDF
graph ([maplib](https://github.com/DataTreehouse/maplib)), relations are its edges,
populations are [polars](https://pola.rs) DataFrames, and the API is scikit-learn's.
Behaviour comes from published models, each cited to its tables: Poledna et al. (2023)'s
Austrian economy calibrated 1:1 from Eurostat, CANVAS (Hommes et al. 2025), del
Rio-Chanona et al. (2021)'s occupational mobility, Schelling (1971), and Pangallo et al.
(2024)'s epidemic and economy of New York in spring 2020.

**Install.**

```bash notest
pip install "skabm[voila]"
```

The coding-agent skills install as a Claude Code plugin, pinned to the release you installed
(`/plugin marketplace add remiadon/skabm#v<version>`, then `/plugin install skabm@skabm`):
`skabm-core` writes a model from a paper, `skabm-simulation` takes a request to a working
simulation with a Voilà app. They lean on Polars' own skill
(`/plugin marketplace add polars-inc/skills`, then `/plugin install polars@polars`).

**Watch one run.** Each model below is a live page on [Binder](https://mybinder.org): set it
up, press Run, and watch it unfold tick by tick (`skabm.player`). A first launch builds for a
few minutes.

| Model | Live page | Reference | Engine | Main plot |
|---|---|---|---|---|
| Schelling segregation | [launch](https://mybinder.org/v2/gh/remiadon/skabm/main?urlpath=voila%2Frender%2Fnotebooks%2Fschelling.ipynb) | Schelling (1971) | RDF, SPARQL | 2-D grid |
| Bayonne with a car-free old town | [launch](https://mybinder.org/v2/gh/remiadon/skabm/main?urlpath=voila%2Frender%2Fnotebooks%2Fbayonne.ipynb) | day-to-day route choice (Horowitz 1984; Cascetta 1989), pivot-point mode choice (Koppelman 1983), BPR delays (1964), on census 2022 flows and OpenStreetMap roads | RDF, SPARQL | geographical map |
| Occupational mobility and automation | [launch](https://mybinder.org/v2/gh/remiadon/skabm/main?urlpath=voila%2Frender%2Fnotebooks%2Flabour_automation.ipynb) | del Rio-Chanona et al. (2021), J. R. Soc. Interface 18:20200898, on the paper's US data | JAX | time series, with a per-occupation scatter or the Beveridge curve |
| Spatial evolutionary markets | [launch](https://mybinder.org/v2/gh/remiadon/skabm/main?urlpath=voila%2Frender%2Fnotebooks%2Fmarkets.ipynb) | Brock & Hommes (1998), on a lattice ([implementation](https://github.com/zakhac/Spatial-Evolutionary-Markets)), on the S&P 500 | RDF, SPARQL | 2-D grid, with price and strategy-share curves |
| COVID-19 in New York | [launch](https://mybinder.org/v2/gh/remiadon/skabm/main?urlpath=voila%2Frender%2Fnotebooks%2Fpangallo.ipynb) | Pangallo et al. (2024), Nat. Hum. Behav. 8:264-275, in a lighter form, on the paper's NY metro area and the authors' replication data | JAX | geographical map, with deaths, employment and output |

Locally, serve them all with

```bash
uv run --extra voila voila notebooks
```

## What skabm gives researchers

| | |
|---|---|
| **Behaviour as equations** | a rule is a dict of SymPy expressions, `{Firm.price: Firm.price * (1 + inflation)}`. The same rule runs on the knowledge graph and, differentiably, in JAX |
| **Rules from a paper** | the `skabm-core` skill takes a coding agent from a model section to cited, tested rules ([plugin/skills/skabm-core](plugin/skills/skabm-core/SKILL.md)); `skabm-simulation` takes it from a request to a Voilà app ([plugin/skills/skabm-simulation](plugin/skills/skabm-simulation/SKILL.md)) |
| **Observables in polars** | a run is every agent after every tick, one DataFrame, so any macro quantity, a Gini included, is a polars expression you write. Nothing is declared to the simulator |
| **Populations from public data** | calibrators that fit agent rows to known joint facts while every marginal survives exactly |
| **Citations built in** | each behaviour module carries its published parameter values next to their table and equation |
| **Interventions as graph edits** | after a fit, the world is a graph: rewire an ownership edge, delete a bank, close a street, then keep simulating |
| **Learned expectations** | agents forecast with learning rules (SAC, Hommes & Zhu 2014) that are just more rules |

### A first model

Schelling (1971) in a dozen lines: people of two groups on a grid, and whoever has too few
neighbours like them moves to an empty cell.

```python
import polars as pl
from maplib import Model

from skabm import template  # adds DataFrame.with_iri
from skabm.behaviour import person
from skabm.simulation import RDFSimulator
from skabm.space import moore_neighbors

# the world: a 20 x 20 grid of cells, four in five settled by someone of group 0 or 1
cells = pl.DataFrame({"x": range(20)}).join(pl.DataFrame({"y": range(20)}), how="cross")
cells = cells.with_columns(pl.col("x", "y").cast(pl.Float64), id=pl.format("cell_{}_{}", "x", "y"))
people = person.settle(cells)

world = Model()
world.map(template.cell, cells.with_iri())
world.map(template.person, people.with_iri("location"))
world.map(template.links("neighbor"), moore_neighbors(cells).with_iri("neighbor"))

# the model: a list of behaviours, run in order every tick
RULES = [person.occupancy, person.happiness, person.cell_rank, person.person_rank, person.relocate]
run = pl.concat(RDFSimulator(RULES, n_periods=10, random_seed=0).fit_iter(world))
```

`run` is every agent after every tick, one polars DataFrame. What you measure is a polars
expression, written after the fact: here, how alike people's neighbours are, which rises
as they sort themselves out.

```python continuation
run.filter(pl.col("class") == "Person").group_by("t", maintain_order=True).agg(
    alike=pl.col("share_similar").mean()
)  # about 0.52 after the first tick, 0.83 by the tenth
```

### Behaviours, with the values of their source

A behaviour is a rule for one type of agent (`firm`, `household`, `person`, `trader`,
`commuter`, ...), named for what it does and carrying the values of the paper that
introduced it ([skabm/behaviour](skabm/behaviour/README.md)). Override one by name:

```python
from skabm.behaviour import defaults, person

person.PARAMETERS[person.want_similar]   # 0.375: 3 of 8 neighbours, Schelling (1971)
assert defaults()["want_similar"] == 0.375
# RDFSimulator(RULES, params={"want_similar": 0.5}) asks for more like neighbours
```

Each published model is a list of behaviours, checked against its paper in
`tests/validation` and shown on its page in `notebooks/`.

### Your own rule, on the graph and in JAX

A rule is a dict of SymPy expressions, one per field it updates, and the same dict runs on
the knowledge graph and, differentiably, in JAX:

```python
import jax
import polars as pl
import sympy as sp

from skabm.dsl import Agents, arrays, jax_tick

Firm, inflation = Agents("Firm"), sp.Symbol("inflation")
reprice = {Firm.price: Firm.price * (1 + inflation)}   # every firm's price, each tick

firms = pl.DataFrame({"id": ["firm_0", "firm_1"], "price": [1.0, 1.1]})
tick = jax_tick([reprice])

def price_level(inflation):
    state = arrays({"Firm": firms}, [reprice])
    for _ in range(4):
        state = tick(state, {"inflation": inflation})
    return state["Firm"]["price"].mean()

jax.grad(price_level)(0.02)   # 4 × 1.05 × 1.02³, the exact derivative
```

`RDFSimulator([reprice], params={"inflation": 0.02})` runs it on the graph. The `skabm-core`
skill ([plugin/skills/skabm-core](plugin/skills/skabm-core/SKILL.md)) takes a coding agent
from a paper's model section to such rules, cited and tested: CANVAS's (Hommes et al. 2025)
came out of it.

| Where to look next | |
|---|---|
| [notebooks/poledna](notebooks/poledna.ipynb) | Poledna et al. (2023)'s Austrian economy, its population calibrated from Eurostat, in five chapters: **0** calibration, **1** the quarter plus an intervention, **2** a banking layer whose distress spreads a step per quarter, **3** learned expectations as running sums on the signal nodes, **4** black-it parameter calibration read as an identification test |
| [notebooks/extraction](notebooks/extraction.ipynb) | what to *do* with derived state: Gini as one polars expression, an intervention that moves it, telemetry into JAX |

## Architecture

| Package | What it is |
|---|---|
| [skabm](skabm/README.md) | the engine: templates, the rule DSL and its SPARQL and JAX compilers, the simulator, what gets measured, rules from a description |
| [skabm.behaviour](skabm/behaviour/README.md) | the agent behaviours, one module per type of agent, each rule named for its mechanism |
| `skabm.space` | neighbourhoods on a lattice or among points, and the areas a link falls in |
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
  Farmer (2021). Occupational mobility and automation. *J. R. Soc. Interface* 18(174), 20200898 ·
  Hommes, He, Poledna, Siqueira & Zhang (2025). CANVAS: a Canadian behavioral agent-based model
  for monetary policy. *J. Econ. Dyn. Control* 172, 104986.
- Hommes & Zhu (2014). Behavioral learning equilibria. *JET* 150 · Huberman & Glance (1993).
  Evolutionary games and computer simulations. *PNAS* 90(16) · Deville & Särndal (1992).
  Calibration estimators in survey sampling. *JASA* 87(418) · Fagiolo, Guerini, Lamperti,
  Moneta & Roventini (2019). Validation of ABMs. In *Computer Simulation Validation*.
- [black-it](https://github.com/bancaditalia/black-it) (ABM parameter calibration) ·
  [maplib](https://github.com/DataTreehouse/maplib) (knowledge graphs, OTTR, SPARQL) · [AMBER](https://github.com/a11to1n3/AMBER) (polars-based ABM).
