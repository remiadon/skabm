## Contents

- [Scan options for dirty data](#scan-options-for-dirty-data)
- [Query plan inspection with `explain()`](#query-plan-inspection-with-explain)
- [Streaming engine for larger-than-memory data](#streaming-engine-for-larger-than-memory-data)
- [Writing results out: `sink_parquet`](#writing-results-out-sink_parquet)
- [Lazy API quick reference](#lazy-api-quick-reference)

The lazy API is the default. `pl.scan_csv`/`pl.scan_parquet`/`pl.scan_ndjson` return a `LazyFrame`. Build the full chain, then call `.collect()` once. The engine optimizes the whole plan before any data is read.

---

## Scan options for dirty data

Handle dirt at the scan, not downstream. The relevant `scan_*` kwargs:

```python
lf = pl.scan_csv(
    "data.csv",
    null_values=["N/A", "", "null", "-", "NA"],   # strings to treat as null
    try_parse_dates=True,                           # attempt to parse date-like columns
    encoding="utf8",                                # default; set explicitly if unsure
    ignore_errors=True,                             # skip unparseable rows (use with care)
    truncate_ragged_lines=True,                     # skip extra fields in wide rows
    infer_schema_length=1000,                       # rows to sample for schema inference
    schema_overrides={"col_name": pl.Float64},     # force a column's dtype
)
```

`try_parse_dates=True` is the single highest-value option for real-world CSV — it handles many date formats without a manual `str.to_date` pass.

For Parquet, the scan is usually clean (schema is stored in the file). The main option:

```python
lf = pl.scan_parquet("data.parquet", n_rows=10_000_000)  # cap rows read
```

`scan_ndjson` supports similar options to `scan_csv` (`null_values`, `infer_schema_length`, `schema_overrides`).

---

## Query plan inspection with `explain()`

Before collecting, inspect the plan to catch expensive operations early:

```python
lf.explain(fmt="text")      # human-readable plan
lf.explain(fmt="json")      # machine-readable, for programmatic inspection
```

The plan shows: scan nodes, filter/join order, aggregation strategy, whether streaming is active, and projection pushdown (which columns are actually read). If a column you did not select appears in the plan's output projection, something is keeping it alive unnecessarily.

Call `explain()` before the first `.collect()` on a new chain. Re-run after major changes.

---

## Streaming engine for larger-than-memory data

When the data is too large to fit in memory comfortably, use the streaming engine. It processes data in chunks and keeps memory bounded.

```python
result = (
    pl.scan_csv("large.csv")
    .filter(pl.col("year") == 2024)
    .group_by("region")
    .agg(pl.col("revenue").sum())
    .sort("region")
    .collect(streaming=True)
)
```

Streaming is not a free substitute for a good query plan — filter early, avoid `map_elements`, collect once. Streaming makes those good habits scale to more data.

Not all operations are available in streaming mode. If a chain fails with streaming, fall back to `collect()` without streaming and check memory. Common non-streaming operations: certain `map_elements` uses, complex struct reshaping, some list operations.

---

## Writing results out: `sink_parquet`

Instead of collecting into a DataFrame and then writing, `sink_parquet` writes directly from the LazyFrame. This can skip materializing the full result in memory.

```python
(
    pl.scan_csv("large.csv")
    .filter(pl.col("year") == 2024)
    .group_by("region")
    .agg(pl.col("revenue").sum())
    .sink_parquet("output_2024.parquet")
)
```

`sink_parquet` accepts the same path and compression options as `pl.DataFrame.write_parquet`. Use it when the result is large and you do not need to inspect it as a DataFrame first.

Other sinks: `sink_csv`, `sink_ipc` (Feather/IPC format). The pattern is the same: write from the LazyFrame without an intermediate `.collect()`.

---

## Lazy API quick reference

| Operation | Lazy form | Notes |
|---|---|---|
| Read CSV | `pl.scan_csv(path, ...)` | Use scan options for dirty data |
| Read Parquet | `pl.scan_parquet(path, ...)` | Schema from file; usually clean |
| Read NDJSON | `pl.scan_ndjson(path, ...)` | Same options as scan_csv |
| Filter | `.filter(expr)` | Place early |
| Add/replace columns | `.with_columns(expr, ...)` | Batch into one call |
| Select columns | `.select(expr, ...)` | Drops unmentioned columns |
| Group and aggregate | `.group_by("col").agg(expr, ...)` | One row per group |
| Window / broadcast | `.over("group")` inside `with_columns` | Same shape as input |
| Join | `.join(other, on=..., how=...)` | `other` can be LazyFrame or DataFrame |
| Sort | `.sort("col", descending=True)` | Stable |
| Limit rows | `.head(n)` / `.tail(n)` / `.slice(offset, length)` | |
| Distinct | `.unique()` / `.unique(subset=[...])` | |
| Collect | `.collect()` | Execute once |
| Collect streaming | `.collect(streaming=True)` | For large data |
| Write parquet (lazy) | `.sink_parquet(path)` | Skip full materialization |
| Inspect plan | `.explain(fmt="text")` | Before collecting |
