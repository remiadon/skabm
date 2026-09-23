"""Poledna et al. (2023) — every number here is the paper's, nothing is invented.

Paper: "Economic forecasting with an agent-based model."
       European Economic Review 151, 104306.  Austria, reference quarter 2010:Q4.

Covered: the shipped parameter set against Table 2 (and a tripwire so no
unsourced parameter slips in), census activity shares (Section 4.1.1), and the
IO-table firm coefficients built from the same Eurostat tables the paper uses
(Section 4.1.2).
"""

from __future__ import annotations

import polars as pl
import pytest

from skabm.behaviour.params import poledna_params
from skabm.calibration import make_dataset, weighted_enum
from skabm.datasets import build_firm_io_df

# fmt: off
TABLE_2 = {
    "dividend_ratio":      0.7768,     # θ^DIV
    "benefit_replacement": 0.3586,     # θ^UB
    "vat_rate":            0.1529,     # τ^VAT
    "total_deposits":      222_933.2e6,  # D^H
    "rho":                 0.9263,     # Taylor-rule smoothing
    "r_star":             -0.0034,     # r*
    "pi_star":             0.005,      # π*
    "xi_pi":               0.3214,     # ξ^π
    "xi_gamma":            1.2994,     # ξ^γ
}
# Parameters with no Table 2 entry.  Adding one here is a decision, not a default.
UNSOURCED = {
    "firm_ownership_ratio", "gov_growth",
    "growth_sigma", "inflation_sigma", "gov_growth_sigma",  # 0 = deterministic
    "distress_threshold", "bank_asset_scale", "flee_amount_threshold",  # bank extension
}
# fmt: on

H_ACTIVE = 4_729_215  # H^act, census
H_INACTIVE = 4_130_385  # H^inact, census


def test_params_are_table_2():
    assert {k: poledna_params[k] for k in TABLE_2} == TABLE_2
    assert set(poledna_params) == set(TABLE_2) | UNSOURCED


def test_household_census_shares():
    status = pl.Enum(["active", "inactive"])
    households = make_dataset(
        samplers={"status": weighted_enum(status, [H_ACTIVE, H_INACTIVE], seed=110)},
        n_agents=2_000,
        seed=1,
    )
    inactive = households.filter(pl.col("status") == "inactive").height / 2_000
    assert inactive == pytest.approx(H_INACTIVE / (H_ACTIVE + H_INACTIVE), abs=0.02)


def test_firm_io_coefficients():
    """ā_i, w̄_i, δ_i, a_i: one value per industry, off the Eurostat IO table."""
    io = build_firm_io_df("AT", 2010).filter(
        pl.col("n_firms").is_not_null() & pl.col("alpha_s").is_not_null()
    )
    industries = io["industry"]
    coef = {
        col: pl.col("industry").replace_strict(
            industries, io[f"{col}_s"], return_dtype=pl.Float64
        )
        for col in ("alpha", "w_bar", "delta", "tech_share")
    }
    firms = make_dataset(
        samplers={
            # industry ∝ n_firms, from business demography (Section 4.1.1)
            "industry": weighted_enum(
                pl.Enum(industries.to_list()), io["n_firms"], seed=100
            ),
            **coef,
        },
        n_agents=500,
        seed=0,
    )

    for col in coef:
        assert firms.select(pl.col(col).n_unique().over("industry")).max().item() == 1
    assert (firms["alpha"] > 0).all()
    assert firms["delta"].is_between(0, 1).all()
    assert firms["tech_share"].is_between(0, 1).all()
