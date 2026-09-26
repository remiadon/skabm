# skabm.behaviour, the rule library

One module per model, or per part of one:

| Module | Model | Before the first step (polars) |
|---|---|---|
| `firm` | Poledna et al. (2023): production, pricing, sales, liquidity, labour, dividends | `ownership`: who owns which firm, §3.2 |
| `household` | Poledna et al. (2023): income (eq. 49) and consumption | `initial`: starting income and wealth, eq. 49 and §5.2 |
| `macro` | Poledna et al. (2023): government spending, the Taylor rule | |
| `bank` | a skabm extension: bank distress and depositor flight, spreading one step per tick. No published calibration, so no model tests | `depositors`: each depositor's bank, drawn by deposit share |
| `labour` | del Rio-Chanona et al. (2021): occupational mobility under an automation shock | `mobility_network`: edges from observed transitions |
| `learning` | SAC expectations (Hommes & Zhu 2014), as running sums on a signal node | |
| `schelling` | Schelling segregation, on a lattice or continuous geometry | `settle`, `grid_neighbors`, `geo_neighbors` |
| `traffic` | day-to-day route choice and pivot-point mode choice (the Bayonne model) | `routes`, `area_membership`; `pedestrianize` is the closure, as an intervention |

## What a module holds

- **`RULES`**: the rules, in the order each step runs them. A rule is a dict of SymPy
  expressions, `{Firm.price: ...}` (see [the engine](../README.md#rules)), with its
  citation in a comment above it.
- **`PARAMETERS = {symbol: value}`**: every published value its rules read, each with a
  citation comment (`# τ^VAT, Poledna et al. (2023) Table 2`, or `# unsourced: ...`, or
  `# scenario knob: ...`). `skabm.behaviour.defaults()` merges them and refuses a name
  with two values. No published value means no default: the parameter stays required,
  and compiling raises a `KeyError` naming it.
- **Polars functions** for everything before the first step: starting values, links,
  derived populations. There is no initialisation phase and no SPARQL text here. A knob
  of such a function (`ownership`'s `ratio`, `settle`'s `density`) is a cited default of
  its argument.

Callers override a value by name only: `RDFSimulator(params={"vat_rate": 0.2})`.

## The docstring is the specification

A module's docstring states its rules in prose: the paper's terms, parameters named "the
parameter x", no code. `translate.rules` must write the module's `RULES` back from it, in an
order that computes the same thing. `SKABM_LLM=1 pytest -k regenerates` checks this with
Qwen2.5-Coder-7B (4-bit MLX and bf16 give the same verdicts):

| Regenerates | Module |
|---|---|
| yes | `macro`, `bank`, `labour` |
| not since the rules became one `RULES` list | `household` (reads a household's dividend as the owned firm's `dividend` field), `firm` (garbles sales) |
| not yet (xfail) | `schelling`, `traffic` |
| n/a | `learning`, which has no rule of its own |

Two lessons shape how a docstring is written. The model reads a field inside a rule as
its old value, so a value one rule computes and another reuses becomes a rule of its own
that runs first (`labour_target`, `job_finding`, `firm_sales`). And numbered rules under
a "N rules, run each step, in this order" header keep the order.
