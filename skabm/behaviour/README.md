# skabm.behaviour, the rule library

One module per type of agent. A module is how that agent behaves according to each source
it cites, one list of rules per source:

| Module | Agents | Rules, by source | Before the first step (polars) |
|---|---|---|---|
| `firm` | firms, and the sectors whose goods they sell | `poledna_rules`: Poledna et al. (2023), production, pricing, sales, liquidity, labour, dividends. `canvas_rules`: CANVAS, Hommes et al. (2025), price and quantity heuristics (eqs. 39-42), cost push through a production network (eqs. 43, 48, 49, 54), demand at each buyer's first pick (A.2.3); final demand is an input, and taken literally it diverges within 15 quarters ([why](../README.md#limitations)) | `ownership`: who owns which firm, Poledna §3.2; `canvas_initial`: CANVAS's opening quarter, §3.1 |
| `household` | households | `poledna_rules`: income (eq. 49) and consumption | `initial`: starting income and wealth, eq. 49 and §5.2 |
| `macro` | government, central bank | `poledna_rules`: government spending, the Taylor rule. `canvas_rules`: the Taylor rule on forecasts (eq. 7); its coefficients are re-estimated each quarter and never reported, so they stay required | |
| `bank` | banks and their depositors | `contagion_rules`: a skabm extension, distress and depositor flight spreading one step per tick. No published calibration, so no model tests | `depositors`: each depositor's bank, drawn by deposit share |
| `labour` | occupations | `delrio_rules`: del Rio-Chanona et al. (2021), occupational mobility under an automation shock | `mobility_network`: edges from observed transitions |
| `residence` | people, and the cells they live in | `schelling_rules`: Schelling (1971) segregation, on a lattice or continuous geometry | `settle`, `grid_neighbors`, `geo_neighbors` |
| `traffic` | commuters, routes, road links | `bayonne_rules`: day-to-day route choice and pivot-point mode choice (the Bayonne model) | `routes`, `area_membership`; `pedestrianize` is the closure, as an intervention |
| `learning` | the forecasts every agent reads | SAC expectations (Hommes & Zhu 2014), as running sums on a signal node | |

## What a module holds

- **`<source>_rules`**: one list per source, in the order each step runs its rules. A rule
  is a dict of SymPy expressions, `{Firm.price: ...}` (see [the engine](../README.md#rules)),
  named after its source (`poledna_produce`), with its citation in a comment above it. A
  sub-expression the sources share is defined once (`firm.expected_growth`). A model
  takes one list, or orders rules from several: `[*firm.canvas_rules, *macro.canvas_rules]`
  is CANVAS, and `simulation.DEFAULT_RULES` interleaves Poledna's firms and households.
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

A module's docstring states its rules in prose, one section per source: the paper's
terms, the fields' names, parameters named "the parameter x", numbered in the order each
step runs them. Every read
in a rule sees the state before the rule ran, so a value one rule computes and another
reuses is a rule of its own that runs first (`delrio_target`,
`delrio_job_finding`, `poledna_sales`). Writing a module from a paper is the `skabm` skill
([.claude/skills/skabm](../../.claude/skills/skabm/SKILL.md)).
