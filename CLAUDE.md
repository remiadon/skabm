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
- **Engine tests** (simulator, IR, history, templates, calibration machinery, loaders) pin software contracts and may use any numbers.
- A behaviour with no published calibration (e.g. the bank extension) gets no model tests until it has one. Say so in the test module, don't fake one.
- The `poledna_params` fixture (`tests/conftest.py`) is the independent copy of Table 2. `test_default_rules_carry_the_cited_values` checks the rules' own defaults against it.
- Trimming a test must not lower line coverage: run `uv run --with pytest-cov pytest --cov=skabm` before and after.

## Parameters

A parameter's value lives in the `PARAMETERS` of the module whose rules read it, and nowhere else in `skabm/`.

- A rule in `skabm.behaviour` is a dict of SymPy expressions (`skabm.dsl`), or, for a rule that builds structure or adds agents, a SPARQL `string.Template`. A module's `PARAMETERS = {symbol: value}` gives every published value its rules read, with a citation comment per value (`# τ^VAT, Poledna et al. (2023) Table 2`, or `# unsourced: ...`, or `# scenario knob: ...`). `skabm.behaviour.defaults()` merges them and refuses a name with two values; `test_every_cited_value_is_read_and_has_one_value` enforces both.
- A behaviour module's docstring is the specification of its rules, in prose (the paper's terms, parameters named "the parameter x", no code): `translate.rules` must write the module's rules back from it, in order, checked by `SKABM_LLM=1 pytest -k regenerates` (`DESCRIBED` in `tests/test_translate.py`: every module but `learning`; `schelling` and `traffic` are xfail in `BEYOND_7B` until a model can write them). A value a rule computes and another field reuses is a rule of its own that runs first (`labour_target`, `job_finding`, `firm_sales`): the model reads a field inside a rule as its old value. Citations are comments above each rule. No other prose in `skabm/behaviour`.
- No published value, no default. The parameter stays required and compiling raises a `KeyError` naming it (e.g. `firm_entry`'s `entry_barrier`).
- Callers override by name only: `RDFSimulator(params={"peak_factor": 1.5})`. Read a default as `module.PARAMETERS[module.symbol]`, or `defaults()["name"]`. Never copy the full set into a notebook, app or test.
- Language namespaces: `skabm.sparql` is SPARQL (render, prefixes, UDFs), `skabm.ottr` is the OTTR templates that map DataFrames into the graph (and `SCHEMA`, what each field means), `skabm.behaviour` is the rules, `skabm.dsl` is rules as SymPy compiled to SPARQL and JAX, `skabm.translate` generates DSL rules from a researcher's description and the OTTR templates.
- A new per-tick rule is a rule dict, so it also runs in JAX. Only a rule that adds agents stays SPARQL text.
