## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## Tests

skABM is only as credible as its calibration, so tests split in two:

- **Model tests** (anything exercising `skabm/behaviour/*` or `skabm/behaviour/schelling.py` rules) assert only what a cited source says: a published parameter value (Poledna et al. 2023 Table 2 is the model), an equation of the paper evaluated at those values, or a result the paper reports (e.g. the Beveridge curve slopes down). Cite the source (table, equation, section) next to the assertion. No invented parameters, no hand-picked targets, no asserting a constant against itself, no "row exists" smoke checks.
- **Engine tests** (simulator, templates, calibration machinery, loaders) pin software contracts and may use any numbers.
- A behaviour with no published calibration (e.g. the bank extension) gets no model tests until it has one. Say so in the test module, don't fake one.
- The `poledna_params` fixture (`tests/conftest.py`) is the independent copy of Table 2. `test_default_rules_carry_the_cited_values` checks the rules' own defaults against it.
- Trimming a test must not lower line coverage: run `uv run --with pytest-cov pytest --cov=skabm` before and after.

## Parameters

A parameter's value lives in the `PARAMETERS` of the module whose rules read it, and nowhere else in `skabm/`.

- A rule in `skabm.behaviour` is a dict of SymPy expressions (`skabm.dsl`); a module lists them in `RULES`, in the order each step runs them. There is no initialisation phase: everything before the first step (starting values, links, derived populations) is a polars function the module exports (`household.initial`, `firm.ownership`, `bank.depositors`, `schelling.settle`, `schelling.grid_neighbors`, `traffic.area_membership`), and a many-valued link maps through `template.links`. No SPARQL text in `skabm/behaviour`. A module's `PARAMETERS = {symbol: value}` gives every published value its rules read, with a citation comment per value (`# τ^VAT, Poledna et al. (2023) Table 2`, or `# unsourced: ...`, or `# scenario knob: ...`). `skabm.behaviour.defaults()` merges them and refuses a name with two values; `test_every_cited_value_is_read_and_has_one_value` enforces both.
- A behaviour module's docstring is the specification of its rules, in prose (the paper's terms, parameters named "the parameter x"), numbered in the order each step runs them. Every read in a rule sees the state before the rule ran, so a value one rule computes and another field reuses is a rule of its own that runs first (`labour_target`, `job_finding`, `firm_sales`). Citations are comments above each rule. No other prose in `skabm/behaviour`. Writing a module from a paper is the `skabm` skill (`.claude/skills/skabm/SKILL.md`).
- No published value, no default. The parameter stays required and compiling raises a `KeyError` naming it. A polars function's knob (`ownership`'s `ratio`, `settle`'s `density`, `initial`'s `total_deposits`) is a cited default of its argument, since no rule reads it.
- Callers override by name only: `RDFSimulator(params={"peak_factor": 1.5})`. Read a default as `module.PARAMETERS[module.symbol]`, or `defaults()["name"]`. Never copy the full set into a notebook, app or test.
- Language namespaces: `skabm.sparql` is SPARQL plumbing (prefixes, UDFs), `skabm.template` is the OTTR templates that map DataFrames into the graph (and `SCHEMA`, what each field means), `skabm.behaviour` is the rules, `skabm.dsl` is rules as SymPy compiled to SPARQL and JAX.
- A new per-tick rule is a rule dict, so it also runs in JAX. `RDFSimulator` has no `infer`: a cascade is rules run a step per tick (`bank.RULES`).
