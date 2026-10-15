---
name: skabm-core
description: Writing agent-based models with skabm, from a paper's model section to agent behaviours in skabm.behaviour (rules named by mechanism, with the introducing source's values) with the builders of their opening state, and the paper's RULES composed in its validation test — SymPy rule dicts over the shared OTTR templates, compiled to SPARQL (maplib) and JAX, with validation tests. Use when the user asks to implement, port, extend or check an ABM (a paper, a set of equations, one agent behaviour) in skabm, to write or fix a rule in skabm.behaviour, or to add an agent class or field. A user's Voilà app around the model is the skabm-simulation skill, which loads this one.
---

# skabm-core: from a paper to behaviours and a model

skabm's library is behaviours: `skabm/behaviour/<agent>.py`, one module per type of agent
(`firm`, `household`, `person`, `labour_market`, `trader`, `commuter`, `bank`, `macro`,
`learning`). A rule is a SymPy dict named for the mechanism it implements
(`firm.adaptive_output`, `labour_market.urn_ball_flow`), never after a researcher; the
comment above it cites the source that introduced it, and the module's `PARAMETERS` hold
that source's values. The polars functions that build an agent's opening state by an
equation of a source sit with its rules (`household.opening_balance`,
`firm.steady_state`); neighbourhoods and areas are `skabm.space`. A paper is not a
module: its `RULES`, its step built from behaviours in its order, are written in
`tests/validation/test_<paper>.py` and again on its page, `notebooks/<paper>.ipynb`, which
loads the paper's data. CANVAS is a worked example: its rules and `firm.opening_quarter`
in `behaviour.firm` and `behaviour.macro`, composed and tested in
`tests/validation/test_canvas.py`, on the graph and in JAX.

## Workflow

1. **Read the source itself**: the model section, the appendix, the parameter tables,
   with equation numbers. A summary, a PR description or your memory of the paper is
   not the source; check every claim you inherit against it. A public implementation of
   the model is a cross-check, and where it differs from the paper, follow the paper and
   say where they differ.
2. **Scope**: list the equations you implement and those you leave out, with the reason
   (an engine wall below, data the paper does not publish). It goes in the reply, the test
   module's docstring and the page's opening markdown.
3. **Classes and fields**: the classes are OTTR templates in `skabm/template.py`, and
   `SCHEMA` says what each field means and where each link lands:
   `uv run python -c "from skabm.template import SCHEMA; print(SCHEMA['Firm'])"`.
   Reuse a field that means the same thing; add a missing one to the class's
   `_agent(...)` call, with a `doc` (a test requires one per field); a new class is a new
   `_agent(...)` added to `TEMPLATES`. Templates are skabm's, never a user's. A field no rule writes is `required`.
4. **Write the behaviours** (anatomy below): first look for the mechanism in
   `skabm/behaviour` and reuse it, with the paper's values passed by name. A new
   mechanism goes into the module of the agent it updates, named for what it does. Then
   **compose the paper**: `RULES`, one rule per step the paper takes, in its order, in
   `tests/validation/test_<paper>.py`.
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

## Anatomy

A behaviour, in the module of its agent:

```python
Firm = Agents("Firm")
inflation_sigma = sp.Symbol("inflation_sigma")

PARAMETERS = {
    inflation_sigma: 0.0,  # scenario knob: AR(1) innovation std, 0 = deterministic
}

# Poledna et al. (2023) eq. 8
adaptive_price = {
    Firm.price: Firm.price * (1 + expected_inflation + Normal("eps", 0, inflation_sigma))
}
```

A paper, composing behaviours, in its validation test and on its page:

```python
"""Poledna et al. (2023), Economic forecasting with an agent-based model, European
Economic Review 151, 104306. What the tests assert, with the citations."""

from skabm.behaviour import firm, household, macro

# one quarter, in the paper's order
RULES = [firm.adaptive_output, firm.adaptive_price, household.income_by_status, ...]
```

- A behaviour module's docstring is a short summary of its agents and mechanisms, with
  sources. Citations are comments above each rule; the code is
  the specification.
- `PARAMETERS`: every value the rules read, from the source that introduced the
  mechanism, with its citation (`# unsourced:` or `# scenario knob:` otherwise). **No
  published value, no entry**: the parameter stays required and compiling raises a
  `KeyError` naming it. `behaviour.defaults()` merges every module and refuses a name
  with two values, so a mechanism calibrated differently by another source needs its
  own name, or the paper passes its value by name.
- Before the first step is data, built by polars functions. One that states an equation
  of a source (an opening balance sheet, a contact network, an input-output steady
  state) goes in the module of the agent it builds, with its citation; a neighbourhood
  or an area goes in `skabm.space`; reading and reshaping the paper's files stays on the
  page.

## Verify

- **Compile** each rule: `dsl.sparql(rule, {**defaults(), **params})` raises on a rule
  it cannot place (a field read where no link reaches it, a missing parameter).
- **Run** on a tiny world mapped the way a user does (`tests/worlds.py`'s `world()`):
  `run = pl.concat(RDFSimulator(rules=rules, params=..., udfs=...).fit_iter(world))`.
- **Run long**, 40 steps or more, and look for NaN, negative prices, runaway ratios. A
  rule set can match the paper and still diverge; report it rather than add an
  unsourced clamp.
- **JAX**: without the SPARQL-only operators, `dsl.jax_tick(rules)` stepped from
  `dsl.arrays(frames, rules)` matches the graph to 1e-9
  (`tests/validation/test_canvas.py`).
- **Tests** in `tests/validation/test_<paper>.py`, run at pre-push. Model tests assert
  only what the source says: a table value, an equation evaluated at the paper's values,
  a result the paper reports, with the citation next to the assertion. Engine tests pin
  software contracts and may use any numbers. A model with no published calibration
  gets no model test; the test module says so.
- **Coverage** must not drop: `uv run --with pytest-cov pytest --cov=skabm`.
- **The page** (`notebooks/<paper>.ipynb`, the CLAUDE.md showcase rules) writes the same
  `RULES` and ends its setup with one cheap `assert` of what the paper states about its
  data, which nbmake runs at pre-push. List any new builder in
  `skabm/behaviour/README.md`.

## Walls

- **Synchronous, staged activation**: all agents update at once, rules fire in order.
- **No sequential search and matching**: a market is simultaneous. Write the
  expectation of the sequential process (a firm's share of demand is its chance of being
  picked) and say so; the ordering itself cannot be expressed.
- **The past is state**: `lag` for one step ago, running sums for a statistic of the
  whole sample. A coefficient the paper re-estimates each period by regression has no
  home in the graph: keep it a parameter, required when the paper reports no value.
