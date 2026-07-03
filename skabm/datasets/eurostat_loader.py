"""
Eurostat data loaders for ABM calibration.

Two public functions:

  fetch_io_components(geo, year)
      Fetches NAIO_10_CP1700 (symmetric IO table at basic prices) and returns
      one row per leaf-level CPA industry with output, wages, value added,
      depreciation, and intermediate consumption in millions of EUR.

  fetch_firm_demographics(geo, year)
      Fetches BD_9PM_R2 (business demography) and returns one row per CPA
      industry with the number of active enterprises and persons employed.
      Falls back to the nearest available year when the requested year has
      no data for a given industry.

Both return a pl.DataFrame keyed on `industry` (CPA code string, e.g.
"CPA_C10-12").  Joining them on that key gives the full input needed to
build a firm-population spec for make_heterogeneous_agents.
"""

from __future__ import annotations

import re
from functools import lru_cache

import polars as pl

try:
    import eurostat as _eurostat
except ImportError as e:
    raise ImportError("pip install eurostat") from e


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_IO_DATASET = "NAIO_10_CP1700"
_BD_DATASET = "BD_9PM_R2"

# Value-added component codes that appear as prd_ava (column) dimension.
_IO_COMPONENTS = {
    "P1": "output",
    "D1": "wages",
    "B1G": "value_added",
    "P51C": "depreciation",
    "P2_ADJ": "intermediate",
}

# Business demography indicators we need.
_BD_FIRMS = "V11960"  # number of active enterprises
_BD_EMPLOYED = "V16961"  # persons employed


def _cpa_to_nace(cpa: str) -> list[str]:
    """Strip 'CPA_' prefix; expand range codes like CPA_C10-12 → [C10, C11, C12]."""
    code = cpa.removeprefix("CPA_")
    m = re.match(r"^([A-Z])(\d+)-(\d+)$", code)
    if m:
        letter, lo, hi = m.group(1), int(m.group(2)), int(m.group(3))
        return [f"{letter}{n}" for n in range(lo, hi + 1)]
    return [code]


def _is_leaf(code: str, all_codes: set[str]) -> bool:
    """True when no other CPA code in the dataset is a strict sub-code of `code`.

    Avoids double-counting: if both CPA_C10-12 and CPA_C10 are present we
    keep CPA_C10 (and C11, C12) and drop the aggregate.
    """
    stripped = code.removeprefix("CPA_")
    for other in all_codes:
        if other == code:
            continue
        s = other.removeprefix("CPA_")
        # s is a sub-code of stripped when stripped is a range that contains s
        m = re.match(r"^([A-Z])(\d+)-(\d+)$", stripped)
        if m and s.startswith(m.group(1)):
            try:
                n = int(s[1:])
                if int(m.group(2)) <= n <= int(m.group(3)):
                    return False
            except ValueError:
                pass
    return True


@lru_cache(maxsize=32)
def _fetch_io_raw(geo: str, year: str) -> pl.DataFrame:
    """Cached raw fetch of the IO table for one country."""
    filter_pars = {
        "geo": [geo],
        "unit": ["MIO_EUR"],
        "stk_flow": ["TOTAL"],
        "prd_ava": list(_IO_COMPONENTS),
    }
    raw = _eurostat.get_data_df(_IO_DATASET, filter_pars=filter_pars)
    return (
        pl.from_pandas(raw)
        .rename({"geo\\TIME_PERIOD": "geo"})
        .filter(pl.col("prd_use").str.starts_with("CPA"))
        .select(["prd_use", "prd_ava", year])
        .rename({"prd_use": "industry", "prd_ava": "component", year: "value"})
        .with_columns(pl.col("value").cast(pl.Float64))
    )


@lru_cache(maxsize=32)
def _fetch_bd_raw(geo: str) -> pl.DataFrame:
    """Cached raw fetch of business demography for one country."""
    filter_pars = {
        "geo": [geo],
        "indic_sb": [_BD_FIRMS, _BD_EMPLOYED],
    }
    raw = _eurostat.get_data_df(_BD_DATASET, filter_pars=filter_pars)
    return (
        pl.from_pandas(raw)
        .rename({"geo\\TIME_PERIOD": "geo"})
        .select(["indic_sb", "nace_r2"] + [c for c in raw.columns if c.isdigit()])
    )


def _nearest_year(
    df: pl.DataFrame, nace: str, indicator: str, year: int
) -> float | None:
    """Return the value for `nace`/`indicator` at the nearest year with non-null data."""
    sub = df.filter((pl.col("nace_r2") == nace) & (pl.col("indic_sb") == indicator))
    if sub.is_empty():
        return None
    year_cols = [c for c in sub.columns if c.isdigit()]
    # Sort candidate years by distance to requested year; skip nulls.
    for col in sorted(year_cols, key=lambda c: abs(int(c) - year)):
        val = sub[col][0]
        if val is not None and not (isinstance(val, float) and val != val):  # not NaN
            return float(val)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_io_components(geo: str = "AT", year: int = 2010) -> pl.DataFrame:
    """Return one row per leaf-level CPA industry with IO-table flow values.

    Columns
    -------
    industry        : str   — CPA product code (e.g. "CPA_C10-12")
    output          : f64   — gross output P1 (mln EUR)
    wages           : f64   — compensation of employees D1 (mln EUR)
    value_added     : f64   — gross value added B1G (mln EUR)
    depreciation    : f64   — consumption of fixed capital P51C (mln EUR)
    intermediate    : f64   — intermediate consumption P2_ADJ (mln EUR)
    """
    raw = _fetch_io_raw(geo, str(year))
    all_industries = set(raw["industry"].unique().to_list())

    # Keep only leaf-level codes (drop aggregates that have sub-codes present).
    leaves = {c for c in all_industries if _is_leaf(c, all_industries)}

    wide = (
        raw.filter(pl.col("industry").is_in(leaves))
        .pivot(on="component", index="industry", values="value")
        .rename(_IO_COMPONENTS)
        # Drop industries with no output reported (country doesn't publish them
        # separately; their aggregate parent is kept instead).
        .filter(pl.col("output").is_not_null())
    )
    return wide.sort("industry")


def fetch_firm_demographics(geo: str = "AT", year: int = 2010) -> pl.DataFrame:
    """Return one row per leaf-level CPA industry with firm count and employment.

    For aggregate CPA codes (e.g. CPA_C10-12) the values are summed over
    component NACE codes (C10 + C11 + C12).  When the requested year has no
    data for an industry, the nearest available year is used and the actual
    source year is recorded in `year_used`.

    Columns
    -------
    industry        : str   — CPA product code
    n_firms         : i64   — number of active enterprises (V11960)
    n_employed      : i64   — persons employed (V16961)
    year_used       : i64   — actual year from which data was drawn
    """
    io_df = fetch_io_components(geo, year)
    bd_raw = _fetch_bd_raw(geo)

    rows: list[dict] = []
    for industry in io_df["industry"].to_list():
        nace_codes = _cpa_to_nace(industry)
        n_firms_total = 0
        n_employed_total = 0
        best_year = year

        for nace in nace_codes:
            v_firms = _nearest_year(bd_raw, nace, _BD_FIRMS, year)
            v_employed = _nearest_year(bd_raw, nace, _BD_EMPLOYED, year)
            if v_firms is not None:
                n_firms_total += v_firms
            if v_employed is not None:
                n_employed_total += v_employed

        rows.append(
            {
                "industry": industry,
                "n_firms": int(n_firms_total) if n_firms_total else None,
                "n_employed": int(n_employed_total) if n_employed_total else None,
                "year_used": best_year,
            }
        )

    return pl.DataFrame(rows).sort("industry")


def build_firm_io_df(geo: str = "AT", year: int = 2010) -> pl.DataFrame:
    """Join IO components with firm demographics and derive productivity coefficients.

    Derived columns
    ---------------
    alpha_s         : f64  — average labour productivity (output / persons employed)
    w_bar_s         : f64  — average annual wage (wages mln EUR * 1e6 / persons employed)
    delta_s         : f64  — depreciation rate proxy (depreciation / output)
    tech_share_s    : f64  — intermediate input share (intermediate / output)
    """
    io = fetch_io_components(geo, year)
    dem = fetch_firm_demographics(geo, year)
    return io.join(dem, on="industry", how="left").with_columns(
        # output per person (EUR, note: IO values in mln EUR, employed is persons)
        (pl.col("output") * 1e6 / pl.col("n_employed")).alias("alpha_s"),
        # annual wage per person (EUR)
        (pl.col("wages") * 1e6 / pl.col("n_employed")).alias("w_bar_s"),
        # depreciation as share of gross output
        (pl.col("depreciation") / pl.col("output")).alias("delta_s"),
        # intermediate input share of gross output
        (pl.col("intermediate") / pl.col("output")).alias("tech_share_s"),
    )
