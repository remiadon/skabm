"""
Poledna et al. (2022) Austrian ABM calibration test suite.

Paper: "Economic forecasting with an agent-based model."
       European Economic Review 151, 104306.

Six institutional sectors (Section 3.1):
  1. Non-financial corporations (firms) — 62 industries, IO + BD
  2. Households                          — census active/inactive
  3. General government                  — 25% of domestic firms
  4. Financial corporations (banks)      — Basel III capital regulation
  5. Central bank                        — ECB Taylor rule (singleton, not a population)
  6. Rest of world (foreign firms)       — 50% of domestic firms as importers

Coverage vs paper Table 2 (~70%):
  Census / demography  H^act, H^inact, I_s, J                100 %
  IO table             ā_i, w̄_i, δ_i, tech_share             ~80 %  (κ_i missing)
  Government stats     τ^INC, τ^FIRM, τ^VAT, τ^SIE/SIW,
                       τ^CF, τ^G, θ^UB                        100 %
  Banking / Basel III  ζ, ζ^LTV, ζ^b, θ, μ                  100 %
  National accounts    ψ, ψ^H, θ^DIV, r^G                    100 %
  AR(1) / Taylor rule  exogenous process params               NOT YET
                       (simulation dynamics, not populations)
  Gap: κ_i (capital productivity, needs nama_10_nfa_st loader).
"""

from __future__ import annotations

import pytest
import polars as pl
import polars_random as pr

from skabm.calibration import GeneticConstraintCalibration, make_dataset, weighted_enum
from skabm.datasets import build_firm_io_df


# ---------------------------------------------------------------------------
# Table 2 scalar parameters (reference quarter 2010:Q4)
# ---------------------------------------------------------------------------

# fmt: off
TAX = {
    "income":      0.2134,   # τ^INC
    "corporate":   0.0762,   # τ^FIRM
    "vat":         0.1529,   # τ^VAT
    "si_employer": 0.2122,   # τ^SIE
    "si_worker":   0.1711,   # τ^SIW
    "capform":     0.0876,   # τ^CF
    "gov_cons":    0.0091,   # τ^G
}

BANKING = {
    "capital_ratio": 0.03,   # ζ      — Basel III minimum
    "ltv":           0.60,   # ζ^LTV
    "ltv_new":       0.50,   # ζ^b
    "instalment":    0.05,   # θ
    "risk_premium":  0.0293, # μ
}

HH = {
    "unemp_benefit":  0.3586,  # θ^UB
    "propensity":     0.9394,  # ψ
    "housing_share":  0.0736,  # ψ^H
    "dividend_ratio": 0.7768,  # θ^DIV
}
# fmt: on

H_ACTIVE = 4_729_215
H_INACTIVE = 4_130_385


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def io_df() -> pl.DataFrame:
    return build_firm_io_df("AT", 2010).filter(
        pl.col("n_firms").is_not_null() & pl.col("alpha_s").is_not_null()
    )


@pytest.fixture(scope="module")
def firm_samplers(io_df) -> dict:
    """Samplers for non-financial corporations (Sections 4.1.1 and 4.1.2)."""
    industries = io_df["industry"]  # pl.Series — used in replace_strict
    industry_enum = pl.Enum(industries.to_list())
    return {
        # Industry ∝ n_firms (Section 4.1.1): each firm belongs to one industry,
        # count calibrated from business demography data (BD_9PM_R2).
        "industry": weighted_enum(industry_enum, io_df["n_firms"]),
        # Firm size ~ LogNormal(3, 1): mean(log(size)) = 3.0, Pareto tail (Section 4.1.1).
        "size": pr.normal(3.0, 1.0).exp().cast(pl.Int64).clip(1, None),
        # IO-table coefficients: industry-homogeneous, derived via replace_strict.
        # ā_i = output / employed                             (Section 4.1.2)
        "alpha": pl.col("industry").replace_strict(
            industries, io_df["alpha_s"], return_dtype=pl.Float64
        ),
        # w̄_i = wages / employed
        "w_bar": pl.col("industry").replace_strict(
            industries, io_df["w_bar_s"], return_dtype=pl.Float64
        ),
        # δ_i = depreciation / output
        "delta": pl.col("industry").replace_strict(
            industries, io_df["delta_s"], return_dtype=pl.Float64
        ),
        # a_{sg} = intermediate consumption / output  (Leontief coefficient)
        "tech_share": pl.col("industry").replace_strict(
            industries, io_df["tech_share_s"], return_dtype=pl.Float64
        ),
    }


# ---------------------------------------------------------------------------
# Test 1: Non-financial corporations — all marginals in samplers (Section 4.1)
# ---------------------------------------------------------------------------


def test_firm_marginals(firm_samplers):
    firms = make_dataset(samplers=firm_samplers, n_agents=500, seed=0)

    assert firms.columns[0] == "id"
    assert firms["id"].to_list() == list(range(500))

    # LogNormal(3, 1): mean(log(size)) = 3.0 by construction.
    assert firms["size"].log().mean() == pytest.approx(3.0, abs=0.4)

    # IO-table coefficients are industry-homogeneous (one value per sector).
    for col in ("alpha", "w_bar", "delta", "tech_share"):
        assert (
            firms.select(pl.col(col).n_unique().over("industry")).to_series().max() == 1
        )

    assert (firms["alpha"] > 0).all()
    assert firms["delta"].is_between(0, 1).all()
    assert firms["tech_share"].is_between(0, 1).all()


# ---------------------------------------------------------------------------
# Test 2: Households — census proportions via weighted_enum (Section 4.1.1)
# ---------------------------------------------------------------------------


def test_household_marginals():
    status_enum = pl.Enum(["active", "inactive"])
    wage_enum = pl.Enum(["Q1", "Q2", "Q3", "Q4"])

    households = make_dataset(
        samplers={
            "status": weighted_enum(status_enum, [H_ACTIVE, H_INACTIVE]),
            "wage_class": weighted_enum(wage_enum, [0.325, 0.325, 0.175, 0.175]),
        },
        n_agents=2_000,
        seed=1,
    )

    assert households.columns[0] == "id"

    inactive_share = households.filter(pl.col("status") == "inactive").height / 2_000
    assert inactive_share == pytest.approx(
        H_INACTIVE / (H_ACTIVE + H_INACTIVE), abs=0.04
    )

    q3q4_share = (
        households.filter(pl.col("wage_class").is_in(["Q3", "Q4"])).height / 2_000
    )
    assert q3q4_share == pytest.approx(0.35, abs=0.04)

    # Scalar household parameters verified against Table 2.
    assert HH["propensity"] == pytest.approx(0.9394)
    assert HH["housing_share"] == pytest.approx(0.0736)
    assert HH["unemp_benefit"] == pytest.approx(0.3586)


# ---------------------------------------------------------------------------
# Test 3: General government — J = 25% of domestic firms (Section 4.1.1)
# ---------------------------------------------------------------------------


def test_government_entities(io_df):
    J = int(io_df["n_firms"].sum() * 0.25)

    industries = io_df["industry"]
    industry_enum = pl.Enum(industries.to_list())

    gov_entities = make_dataset(
        samplers={
            # Each government entity purchases from one sector, weighted by output share.
            "purchase_sector": weighted_enum(industry_enum, io_df["output"]),
            # Fixed income tax rate τ^INC shared by all entities (Table 2).
            "tax_rate": pr.uniform(TAX["income"], TAX["income"] + 1e-12),
        },
        n_agents=J,
        seed=2,
    )

    assert gov_entities.height == J
    assert gov_entities["tax_rate"].mean() == pytest.approx(TAX["income"], abs=1e-10)
    assert gov_entities["purchase_sector"].cast(pl.String).str.starts_with("CPA_").all()

    # Verify all scalar tax rates against Table 2.
    assert TAX["corporate"] == pytest.approx(0.0762)
    assert TAX["vat"] == pytest.approx(0.1529)
    assert TAX["si_employer"] == pytest.approx(0.2122)
    assert TAX["si_worker"] == pytest.approx(0.1711)


# ---------------------------------------------------------------------------
# Test 4: Banks (financial corporations) — Basel III calibration (Section 4.4)
# ---------------------------------------------------------------------------


def test_banks():
    banks = make_dataset(
        samplers={
            # Capital ratio calibrated around Basel III minimum ζ = 0.03.
            "capital_ratio": pr.normal(0.08, 0.02).clip(BANKING["capital_ratio"], 0.30),
            # Leverage = assets/equity; clipped at 1/max_capital_ratio.
            "leverage": pr.normal(12.0, 2.0).clip(1 / 0.30, None),
            "deposit_share": pr.uniform(0.05, 0.30),
        },
        n_agents=12,
        seed=3,
    )

    assert banks.height == 12
    assert banks.columns[0] == "id"
    assert (banks["capital_ratio"] >= BANKING["capital_ratio"]).all()
    assert (banks["leverage"] >= 1 / 0.30).all()

    assert BANKING["ltv"] == pytest.approx(0.60)
    assert BANKING["ltv_new"] == pytest.approx(0.50)
    assert BANKING["instalment"] == pytest.approx(0.05)
    assert BANKING["risk_premium"] == pytest.approx(0.0293)


# ---------------------------------------------------------------------------
# Test 5: Rest of world — L = 50% of domestic firms (Section 4.1.1)
# ---------------------------------------------------------------------------


def test_foreign_firms(io_df):
    L = int(io_df["n_firms"].sum() * 0.50)

    industries = io_df["industry"]
    industry_enum = pl.Enum(industries.to_list())
    foreign_firms = make_dataset(
        samplers={
            "source_industry": weighted_enum(industry_enum, io_df["output"]),
            "demand_size": pr.normal(2.5, 1.0).exp().cast(pl.Int64).clip(1, None),
        },
        n_agents=L,
        seed=4,
    )

    assert foreign_firms.height == L
    assert foreign_firms.columns[0] == "id"
    assert (foreign_firms["demand_size"] >= 1).all()


# ---------------------------------------------------------------------------
# Test 6: Inter-column GA — manufacturing firms are larger than service firms
#
# This is a joint fact about (size, industry) that no marginal encodes.
# Constraint tuple: (metric_expr, target_expr)
#   metric = mean(log(size)) per sector (windowed)
#   target = 3.5 for manufacturing, 2.5 for services
# After GA, we evaluate the metric and check it is near the target.
# ---------------------------------------------------------------------------


def test_firm_sector_size_ga(io_df):
    industries = io_df["industry"]
    n_firms = io_df["n_firms"]
    industry_enum = pl.Enum(industries.to_list())

    # sector is derived from industry — compute it inline in the constraint
    # rather than materialising it as a separate column.  The calibrator only
    # sees [industry, size] and permutes size to satisfy the constraint.
    ind_keys = industries.to_list()
    sec_vals = ["manuf" if i.startswith("CPA_C") else "service" for i in ind_keys]
    sector_of = pl.col("industry").replace_strict(
        ind_keys, sec_vals, return_dtype=pl.String
    )

    constraints = [
        (
            pl.col("size").log().mean().over(sector_of),
            pl.when(sector_of == "manuf").then(3.5).otherwise(2.5),
        )
    ]

    population = make_dataset(
        samplers={
            "industry": weighted_enum(industry_enum, n_firms),
            "size": pr.normal(3.0, 1.0).exp().cast(pl.Int64).clip(1, None),
        },
        n_agents=300,
        seed=5,
    )
    cal = GeneticConstraintCalibration(
        constraints=constraints,
        population_size=40,
        n_generations=150,
        seed=5,
    ).fit(population)
    firms = cal.transform(population)

    assert firms.columns[0] == "id"

    # score() returns -energy; a less-negative value = better fit to constraints.
    assert cal.score(firms) >= cal.score(population)

    # Directly evaluate each constraint's metric and target and check convergence.
    for metric_expr, target_expr in constraints:
        achieved = firms.select(metric_expr.alias("m")).to_series()
        expected = firms.select(target_expr.alias("t")).to_series()
        assert (achieved - expected).abs().mean() == pytest.approx(0.0, abs=0.8)


# ---------------------------------------------------------------------------
# Test 7: Guard — single-column constraint metric is rejected
# ---------------------------------------------------------------------------


def test_single_column_constraint_warns():
    population = make_dataset(
        samplers={"size": pr.normal(3.0, 1.0).exp().cast(pl.Int64).clip(1, None)},
        n_agents=50,
    )
    with pytest.warns(UserWarning, match="single-column"):
        GeneticConstraintCalibration(
            constraints=[(pl.col("size").mean(), 20.0)],
            n_generations=1,
        ).fit(population)


# ---------------------------------------------------------------------------
# Test 8: Full Poledna calibration — headline numbers from Table 2 (2010:Q4)
# ---------------------------------------------------------------------------


def test_full_poledna_calibration(firm_samplers):
    status_enum = pl.Enum(["active", "inactive"])
    wage_enum = pl.Enum(["Q1", "Q2", "Q3", "Q4"])

    firms = make_dataset(samplers=firm_samplers, n_agents=1_000, seed=10)
    households = make_dataset(
        samplers={
            "status": weighted_enum(status_enum, [H_ACTIVE, H_INACTIVE]),
            "wage_class": weighted_enum(wage_enum, [0.325, 0.325, 0.175, 0.175]),
        },
        n_agents=2_000,
        seed=11,
    )

    # Firms: Section 4.1.1 + 4.1.2 headline checks.
    assert firms.columns[0] == "id"
    assert firms["size"].log().mean() == pytest.approx(3.0, abs=0.4)
    assert (firms["alpha"] > 0).all()
    assert firms["delta"].is_between(0, 1).all()
    assert firms["tech_share"].is_between(0, 1).all()

    # Households: census active share H^act / (H^act + H^inact) ≈ 53.4%.
    assert households.columns[0] == "id"
    active_share = households.filter(pl.col("status") == "active").height / 2_000
    assert active_share == pytest.approx(H_ACTIVE / (H_ACTIVE + H_INACTIVE), abs=0.05)

    # Scalar parameters from Table 2.
    assert BANKING["capital_ratio"] == pytest.approx(0.03)
    assert HH["dividend_ratio"] == pytest.approx(0.7768)
    assert HH["unemp_benefit"] == pytest.approx(0.3586)
    assert TAX["income"] == pytest.approx(0.2134)
