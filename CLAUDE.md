## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## Tests

skABM is only as credible as its calibration, so tests split in two:

- **Model tests** (a paper's `RULES` composed from behaviours, or a behaviour checked against the paper that introduced it) live in `tests/validation`, run at pre-push, and assert only what a cited source says: a published parameter value (Poledna et al. 2023 Table 2, Hommes et al. 2025 Tables 3-4), an equation of the paper evaluated at those values, or a result the paper reports (e.g. the Beveridge curve slopes down). Cite the source (table, equation, section) next to the assertion. No invented parameters, no hand-picked targets, no asserting a constant against itself, no "row exists" smoke checks.
- **Engine tests** (simulator, DSL, templates, calibration machinery, loaders) live in `tests/`, run at every commit, pin software contracts and may use any numbers.
- A model with no published calibration (e.g. `contagion`) gets no model tests until it has one. Say so in the test module, don't fake one.
- The `poledna_params` fixture (`tests/conftest.py`) is the independent copy of Table 2. `test_default_rules_carry_the_cited_values` checks the rules' own defaults against it.
- Trimming a test must not lower line coverage: run `uv run --with pytest-cov pytest --cov=skabm` before and after.

## Models and parameters

A parameter's value lives in the `PARAMETERS` of the module whose rules read it, and nowhere else in `skabm/`.

- `skabm.behaviour` is the library: one module per type of agent (`firm`, `household`, `person`, `labour_market`, `trader`, `commuter`, `bank`, `macro`, `learning`), each rule a dict of SymPy expressions (`skabm.dsl`) named for the mechanism it implements (`firm.adaptive_output`, `labour_market.urn_ball_flow`), never after a researcher; the comment above it cites the source that introduced it. A sub-expression several rules share is defined once in the module. Every read in a rule sees the state before the rule ran, so a value one rule computes and another field reuses is a rule of its own that runs first (`labour_market.s_curve_target`, `person.exposure`). A module's docstring is a short summary of its agents and mechanisms, with sources.
- A paper is evidence, not a module: its `RULES`, the paper's step composed from behaviours in the paper's order, are written in its validation test (`tests/validation/test_<paper>.py`) and again on its page (`notebooks/`), which loads and reshapes the paper's data. A polars function that builds agents by an equation of a source (`household.opening_balance`, `person.contact_population`, `firm.steady_state`, `labour_market.mobility_network`) lives in the module of the agent it builds; neighbourhoods and areas live in `skabm.space` (`moore_neighbors`, `geo_neighbors`, `area_membership`). A use case that is not a paper (Bayonne) is a page only. There is no initialisation phase, a many-valued link maps through `template.links`, and a rule is never SPARQL text. The fields any model needs live in the shared templates (`skabm.template`), optional unless no rule writes them; users never declare their own (`template._agent` is internal).
- A behaviour module's `PARAMETERS = {symbol: value}` gives every value its rules read from the introducing source, with a citation comment per value (`# τ^VAT, Table 2`, or `# unsourced: ...`, or `# scenario knob: ...`). `skabm.behaviour.defaults()` merges every module's and refuses a name with two values; `test_every_cited_value_is_read_and_has_one_value` enforces both.
- No published value, no default. The parameter stays required and compiling raises a `KeyError` naming it. A polars function's knob (`ownership`'s `ratio`, `settle`'s `density`, `initial`'s `total_deposits`) is a cited default of its argument, since no rule reads it.
- Callers override by name only: `RDFSimulator(params={"peak_factor": 1.5})`. Read a default as `module.PARAMETERS[module.symbol]`, or `defaults()["name"]`. Never copy the full set into a notebook, app or test.
- Language namespaces: `skabm.sparql` is SPARQL plumbing (prefixes, UDFs), `skabm.template` is the OTTR templates that map DataFrames into the graph (and `SCHEMA`, what each field means), `skabm.behaviour` the agent behaviours and the builders of their opening state, `skabm.space` neighbourhoods and areas, `skabm.dsl` is rules as SymPy compiled to SPARQL and JAX. Writing a model from a paper is the `skabm-core` skill (`.claude/skills/skabm-core/SKILL.md`); taking a user's request to their own Voilà app is the `skabm-simulation` skill.
- A new per-tick rule is a rule dict, so it also runs in JAX. `RDFSimulator` has no `infer`: a cascade is rules run a step per tick (`bank`'s rules).

## Showcase pages (maintainers)

The pages in `notebooks/` are skabm's showcase for stakeholders, served free on mybinder.org; a user's own app follows the `skabm-simulation` skill instead and has none of these limits.

- A page is the paper's model on the paper's own data, built on `skabm.player.Player`: nothing runs before Run, each step is drawn as it comes, a pace slider, a replay slider.
- It must fit a Binder session (1 GB guaranteed, 2 GB max). Measure with `/usr/bin/time -l uv run --extra voila jupyter execute notebooks/<name>.ipynb` plus one full run. `binder/postBuild` executes every page that imports `skabm.player`, caching its downloads (a page whose data server is down is reported, not fatal: the image serves every page); `binder/start` caps polars at two threads. `binder/check` builds what Binder builds, the committed HEAD on a cold data cache with every page strict, and runs at pre-push to main: local caches hide a failing download from nbmake.
- A page on a paper's data ends its setup with one cheap `assert` of what the paper states about that data (Pangallo's steady state, del Rio-Chanona's eq. 21 network, Poledna's opening deposits), so the page and its validation test cannot drift apart unnoticed.
- Each page has a row in the README's Binder table (Model | Live page | Reference | Engine | Main plot), and runs under `pytest --nbmake` at pre-push.
