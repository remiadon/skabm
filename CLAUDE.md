## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## Tests

skABM is only as credible as its calibration, so tests split in two:

- **Model tests** (anything exercising `skabm/behaviour/*` or `skabm/schelling.py` rules) assert only what a cited source says: a published parameter value (Poledna et al. 2023 Table 2 is the model), an equation of the paper evaluated at those values, or a result the paper reports (e.g. the Beveridge curve slopes down). Cite the source (table, equation, section) next to the assertion. No invented parameters, no hand-picked targets, no asserting a constant against itself, no "row exists" smoke checks.
- **Engine tests** (simulator, IR, history, templates, calibration machinery, loaders) pin software contracts and may use any numbers.
- A behaviour with no published calibration (e.g. the bank extension) gets no model tests until it has one. Say so in the test module, don't fake one.
- New parameters in `poledna_params` must be either in `TABLE_2` or explicitly listed in `UNSOURCED` (`tests/test_poledna_calibration.py`).
- Trimming a test must not lower line coverage: run `uv run --with pytest-cov pytest --cov=skabm` before and after.
