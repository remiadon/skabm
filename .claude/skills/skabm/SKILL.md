---
name: skabm
description: Writing agent-based model rules with skabm, from a paper's model section to a behaviour module — SymPy rule dicts over OTTR templates, compiled to SPARQL (maplib) and JAX, with cited parameters and tests. Use when the user asks to implement, port, extend or check an ABM (a paper, a set of equations, one agent behaviour) in skabm, to write or fix a rule in skabm.behaviour, or to add an agent class or field.
---

# skabm: from a paper to rules

A model in skabm is a behaviour module: a prose docstring, `PARAMETERS` with a citation
per value, rules as SymPy dicts listed in `RULES`, and polars functions for whatever
exists before the first step. `skabm/behaviour/canvas.py` with `tests/test_canvas.py` is
a worked example, from Hommes et al. (2025) to a module that runs on the graph and in JAX.

## Workflow

1. **Read the source itself**: the model section, the appendix, the parameter tables,
   with equation numbers. A summary, a PR description or your memory of the paper is
   not the source; check every claim you inherit against it. A public implementation of
   the model is a cross-check, and where it differs from the paper, follow the paper and
   say where they differ.
2. **Scope**: list the equations you implement and those you leave out, with the reason
   (an engine wall below, data the paper does not publish). It goes in the reply and the
   module's README row.
3. **Classes and fields**: the classes are OTTR templates in `skabm/template.py`, and
   `SCHEMA` says what each field means and where each link lands:
   `uv run python -c "from skabm.template import SCHEMA; print(SCHEMA['Firm'])"`.
   Reuse a field that means the same thing; add a missing one to the class's
   `agent(...)` call, with a `doc` (a test requires one per field); a new class is a new
   `agent(...)` added to `TEMPLATES`. A field no rule writes is `required`.
4. **Write the module** (anatomy below), one rule per step the paper takes, in its order.
5. **Verify** (checklist below), then report what runs, what is left out, and anything
   the long run shows.

## The DSL

```python
import sympy as sp
from sympy.stats import Normal, Uniform

from skabm.behaviour.learning import expect
from skabm.dsl import Agents, coalesce, lag, mean, node, pick, running_sum, sum_over, total, total_by

Firm, Sector = Agents("Firm"), Agents("Sector")  # fields from the templates; a typo raises
vat_rate = sp.Symbol("vat_rate")  # a symbol that is not a field is a parameter
reprice = {Firm.price: Firm.price * (1 + vat_rate)}  # a rule: fields of ONE class
```

- A rule updates every agent of its class at once, and **every read sees the state
  before the rule ran** (a SPARQL DELETE/INSERT). Fields decided together go in one
  rule; a value that a later rule reads is written by an earlier rule of its own.
- `Firm.sector.price_index` reads across a link. A bare field of another class in a
  rule reads through the rule class's one link to it (`Sector.demand` in a Firm rule is
  `Firm.sector.demand`); with two links to that class, name the link.
- `sum_over(Other.link, e)`: `e` summed over the `Other` agents whose link points here,
  read on them (`Input.buyer.purchases` reaches their far end).
  `sum_over(Class.link, e)`: over the agents this one's many-valued link reaches.
- `total(e)`, `mean(e)`: over every agent of `e`'s class, 0 when there is none.
- `sp.Max`, `sp.Min`, `sp.Piecewise((value, condition), ..., (value, True))` with
  conditions joined by `&`, `|`, `~` and equality as `sp.Eq` (Python's `==` is silently
  False); `sp.exp`, `sp.log` are SPARQL UDFs: `RDFSimulator(udfs=(register_math,))`.
- `coalesce(a, b, ...)`: the first that exists, the one way to handle a missing link or
  an unset field.
- `lag(e)`: `e` one step ago. `Normal("eps", 0, sigma)`, `Uniform("u", 0, 1)`: a fresh
  draw per agent (the default `register_polars_random` UDFs; seed with `random_seed=`).
- `expect("SUM" | "AVG", "Class", "field")`: the learned forecast of that aggregate's
  growth, SAC learning (Hommes & Zhu 2014) that the simulator runs after every tick.
- `node("name", "field")`: a field of one named node; a rule keyed by such fields
  updates that node alone.
- `total_by(Class.key, e)`, `running_sum(Class.order, e, Class.key)`,
  `pick(Other, where, order)`: shared keys, cumulative sums by rank, one-to-one matching.
  SPARQL only: `jax_tick` refuses them by name.

## Module anatomy

```python
"""<Model>, <Authors> (<year>): <what it covers>. N rules, run each step, in this order.

1. A firm's price becomes its price times (1 + the parameter vat_rate) ...
"""

from __future__ import annotations

import sympy as sp

from skabm.dsl import Agents

Firm = Agents("Firm")
vat_rate = sp.Symbol("vat_rate")

PARAMETERS = {
    vat_rate: 0.1529,  # τ^VAT, Poledna et al. (2023) Table 2
}

# Poledna et al. (2023) eq. 8
firm_price = {Firm.price: Firm.price * (1 + vat_rate)}

RULES = [firm_price]
```

- The docstring is the specification in prose, in the fields' own names, parameters as
  "the parameter x". Citations are comments above each rule; no other prose.
- `PARAMETERS`: every published value the rules read, with its citation (`# unsourced:`
  or `# scenario knob:` otherwise). **No published value, no entry**: the parameter stays
  required and compiling raises a `KeyError` naming it. `behaviour.defaults()` merges
  every module and refuses a name with two values, so a model re-calibrating a name
  another module cites differently needs its own name.
- Before the first step is data: a polars function (`initial`, `ownership`,
  `depositors`) filling starting values and links from the paper's initial conditions.

## Verify

- **Compile** each rule: `dsl.sparql(rule, {**defaults(), **params})` raises on a rule
  it cannot place (a field read where no link reaches it, a missing parameter).
- **Run** on a tiny world mapped the way a user does (`tests/worlds.py`'s `world()`):
  `run = pl.concat(RDFSimulator(rules=RULES, params=..., udfs=...).fit_iter(world))`.
- **Run long**, 40 steps or more, and look for NaN, negative prices, runaway ratios. A
  rule set can match the paper and still diverge; report it rather than add an
  unsourced clamp.
- **JAX**: without the SPARQL-only operators, `dsl.jax_tick(RULES)` stepped from
  `dsl.arrays(frames, RULES)` matches the graph to 1e-9
  (`test_sparql_and_jax_agree_on_canvas`).
- **Tests** in `tests/test_<module>.py`. Model tests assert only what the source says:
  a table value, an equation evaluated at the paper's values, a result the paper
  reports, with the citation next to the assertion. Engine tests pin software contracts
  and may use any numbers. A behaviour with no published calibration gets no model test;
  the test module says so.
- **Coverage** must not drop: `uv run --with pytest-cov pytest --cov=skabm`.
- Add the module to `skabm/behaviour/README.md`'s table.

## Walls

- **Synchronous, staged activation**: all agents update at once, rules fire in order.
- **No sequential search and matching**: a market is simultaneous. Write the
  expectation of the sequential process (a firm's share of demand is its chance of being
  picked) and say so; the ordering itself cannot be expressed.
- **The past is state**: `lag` for one step ago, running sums for a statistic of the
  whole sample. A coefficient the paper re-estimates each period by regression has no
  home in the graph: keep it a parameter, required when the paper reports no value.
