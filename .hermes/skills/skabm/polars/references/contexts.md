## Contents

1. [`select` — choosing and transforming columns](#select--choosing-and-transforming-columns)
2. [`with_columns` — adding or replacing columns](#with_columns--adding-or-replacing-columns)
3. [`filter` — removing rows](#filter--removing-rows)
4. [`group_by` + `agg` — aggregating per group](#group_by--agg--aggregating-per-group)
5. [`over` — window functions, broadcast group results to every row](#over--window-functions-broadcast-group-results-to-every-row)
6. [`sort` — ordering rows](#sort--ordering-rows)
7. [`join` — combining two frames](#join--combining-two-frames)

---

## `select` — choosing and transforming columns

Returns only the columns you name. Drops everything else. Use it when the question is "give me these columns."

```python
df.select("id", "name", "score")
```

You can also transform in `select`:

```python
df.select(
    pl.col("revenue") - pl.col("cost").alias("profit"),
    pl.col("name").str.to_uppercase(),
)
```

`select` keeps the row count. Only the column set changes.

---

## `with_columns` — adding or replacing columns

Adds new columns or replaces existing ones. Returns all original columns plus the new ones (or the replacements). Use it when you're adding derived columns to the frame.

```python
df.with_columns(
    (pl.col("revenue") - pl.col("cost")).alias("profit"),
    pl.col("name").str.to_uppercase().alias("name_upper"),
)
```

**Critical distinction from `select`:** `with_columns` keeps every column you did not touch. `select` drops everything not named. Pick the one that matches intent — a stray `select` that forgets a column is a common silent data loss.

---

## `filter` — removing rows

Keeps rows where the expression is `True`. Drops rows where it is `False` or `null`. Use it to reduce the row set before expensive operations.

```python
df.filter(pl.col("year") == 2024)
df.filter((pl.col("score") > 50) & (pl.col("active")))
```

**Null handling:** `filter(pl.col("v") > 2)` silently drops nulls because `null > 2` evaluates to `null`, which is falsy. To keep nulls: `(pl.col("v") > 2) | pl.col("v").is_null()`.

---

## `group_by` + `agg` — aggregating per group

`group_by()` splits the frame into groups. `agg()` computes one value per group. The result has one row per group.

```python
df.group_by("region").agg(
    pl.col("revenue").sum().alias("total_revenue"),
    pl.col("customer_id").len().alias("customer_count"),
    pl.col("satisfaction").mean().alias("avg_satisfaction"),
)
```

Common aggregation expressions: `sum()`, `mean()`, `min()`, `max()`, `len()`, `first()`, `last()`, `count()`.

---

## `over` — window functions, broadcast group results to every row

`over("group")` computes an expression per group and broadcasts the result back to every row in that group. The output has the same shape as the input — no rows are dropped or collapsed.

```python
df.with_columns(
    pl.col("revenue").sum().over("region").alias("region_total"),
    pl.col("revenue").mean().over("region").alias("region_avg"),
)
```

Use `over` inside `with_columns` when every row needs its group's aggregate. Use `group_by().agg()` when you want one row per group.

---

## `sort` — ordering rows

Reorders rows by the given column(s). Keeps all rows. Use `descending=True` for reverse order.

```python
df.sort("total_revenue", descending=True)
df.sort(["region", "customer_count"], descending=[True, False])
```

`sort` is stable: rows with equal sort keys keep their original relative order.

---

## `join` — combining two frames

Joins two frames on matching column values. By default inner join: rows with no match in the other frame are dropped.

```python
df.join(other, on="customer_id", how="left")
```

Common `how` values: `"inner"` (default), `"left"`, `"right"`, `"full"`, `"semi"`, `"anti"`.

**Nulls in join keys:** rows with null keys silently drop out of inner joins. Pass `nulls_equal=True` to match null keys against each other.

**Duplicate column names:** if both frames have a column with the same name (and it's not the join key), the result will have suffixes or raise. Resolve with `.rename()` before joining or alias the overlapping columns.
