## Contents

- [Top-k / ranking](#top-k--ranking)
- [Period-over-period](#period-over-period)
- [Share of total](#share-of-total)
- [Distributions / bucket analysis](#distributions--bucket-analysis)
- [Time series: trend and rate of change](#time-series-trend-and-rate-of-change)
- [Cohort-style: retention / repeat rate](#cohort-style-retention--repeat-rate)
- [Conditional breakdown](#conditional-breakdown)

Each recipe is a lazy chain. Adapt column names to the frame at hand. Run `.collect()` once at the end.

---

## Top-k / ranking

Which N items have the highest value of X?

```python
top10 = (
    pl.scan_csv("orders.csv")
    .group_by("product_id")
    .agg(pl.col("amount").sum().alias("total"))
    .sort("total", descending=True)
    .head(10)
    .collect()
)
```

Top-k within each group (e.g. top 3 products per region):

```python
top_per_region = (
    df
    .with_columns(pl.col("total").rank().over("region").alias("rank"))
    .filter(pl.col("rank") <= 3)
    .collect()
)
```

---

## Period-over-period

Change vs the same period last cycle (e.g. this quarter vs last quarter, this month vs last month). Requires a time column and a way to match periods.

For a single metric with monthly data:

```python
period_over_period = (
    df.sort("month")
    .with_columns(
        pl.col("revenue").diff().alias("abs_change"),
        pl.col("revenue").pct_change().alias("pct_change"),
    )
    .collect()
)
```

For "same period last year" with monthly data, shift by 12 rows within a partition if rows are contiguous months per group:

```python
same_period_last_year = (
    df.sort(["region", "month"])
    .with_columns(
        pl.col("revenue").shift(12).over("region").alias("rev_ly"),
    )
    .with_columns(
        (pl.col("revenue") - pl.col("rev_ly")).alias("abs_change"),
        (pl.col("revenue") / pl.col("rev_ly") - 1).alias("pct_change"),
    )
    .drop("rev_ly")
    .collect()
)
```

State the definition of the period explicitly in the answer ("month-over-month", "year-over-year", "vs previous quarter").

---

## Share of total

What fraction of the total does each group represent?

```python
share_of_total = (
    df.group_by("region")
    .agg(pl.col("revenue").sum().alias("region_revenue"))
    .with_columns(
        pl.col("region_revenue")
        .truediv(pl.col("region_revenue").sum())
        .mul(100)
        .alias("pct_of_total"),
    )
    .sort("region_revenue", descending=True)
    .collect()
)
```

For share within a partition (e.g. each product's share of its category):

```python
share_within_group = (
    df
    .with_columns(
        pl.col("revenue")
        .truediv(pl.col("revenue").sum().over("category"))
        .mul(100)
        .alias("pct_of_category"),
    )
    .collect()
)
```

---

## Distributions / bucket analysis

How are values spread? Bin them.

Equal-width buckets:

```python
buckets = (
    df.with_columns(
        pl.col("revenue").bucketize(
            pl.col("revenue").min(),
            pl.col("revenue").max(),
            10,
        ).alias("bucket"),
    )
    .group_by("bucket")
    .agg(pl.len().alias("count"))
    .sort("bucket")
    .collect()
)
```

Quantile buckets (e.g. quartiles, deciles):

```python
quantiles = (
    df
    .select(pl.col("revenue").quantile([0.25, 0.5, 0.75]).alias("quartile_bounds"))
    .collect()
)
```

For a value frequency table:

```python
value_counts = (
    df.group_by("status")
    .agg(pl.len().alias("count"))
    .with_columns(
        pl.col("count").truediv(pl.col("count").sum()).alias("fraction"),
    )
    .sort("count", descending=True)
    .collect()
)
```

---

## Time series: trend and rate of change

For a metric over time, compute the trend direction and the rate.

Simple trend: value and month-over-month change.

```python
trend = (
    df.sort("date")
    .group_by("metric")
    .agg(
        pl.col("value").sum().alias("total"),
        pl.col("value").first().alias("first"),
        pl.col("value").last().alias("last"),
    )
    .with_columns(
        (pl.col("last") - pl.col("first")).alias("net_change"),
    )
    .collect()
)
```

Rolling window (e.g. 3-period moving average):

```python
rolling = (
    df.sort("date")
    .with_columns(
        pl.col("revenue")
        .rolling_mean(window_size=3, min_periods=1)
        .alias("revenue_3m_avg"),
    )
    .collect()
)
```

Other rolling ops: `.rolling_sum()`, `.rolling_min()`, `.rolling_max()`, `.rolling_std()`, `.rolling_var()`.

---

## Cohort-style: retention / repeat rate

What fraction of entities from an initial set reappear in a later period?

Assume a table of `(entity_id, period, event)` where `event` indicates activity.

```python
retention = (
    df.group_by("entity_id")
    .agg(
        pl.col("period").min().alias("first_period"),
        pl.col("period").max().alias("last_period"),
        pl.col("period").len().alias("active_periods"),
    )
    .with_columns(
        (pl.col("active_periods") > 1).alias("returned"),
    )
    .group_by("first_period")
    .agg(
        pl.len().alias("cohort_size"),
        pl.col("returned").sum().alias("returned_count"),
    )
    .with_columns(
        pl.col("returned_count")
        .truediv(pl.col("cohort_size"))
        .alias("return_rate"),
    )
    .sort("first_period")
    .collect()
)
```

State the cohort definition and the window clearly in the answer.

---

## Conditional breakdown

How does a metric differ across segments defined by a condition?

```python
breakdown = (
    df
    .with_columns(
        pl.when(pl.col("revenue") > 1000)
        .then(pl.lit("high"))
        .otherwise(pl.lit("low"))
        .alias("segment"),
    )
    .group_by("segment")
    .agg(
        pl.col("revenue").sum().alias("total_revenue"),
        pl.col("customer_id").len().alias("customer_count"),
        pl.col("revenue").mean().alias("avg_revenue"),
    )
    .collect()
)
```

For multiple segments at once, build them as separate `when`/`then` branches or compute flags first, then group.
