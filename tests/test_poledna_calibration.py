"""Poledna et al. (2023) — every number here is the paper's, nothing is invented.

Paper: "Economic forecasting with an agent-based model."
       European Economic Review 151, 104306.  Austria, reference quarter 2010:Q4.

Covered: the default rule set's own defaults are the values the ``poledna_params``
fixture cites, one for every placeholder; census activity shares (Section 4.1.1), and the IO-table firm
coefficients built from the same Eurostat tables the paper uses (Section 4.1.2).
"""

from __future__ import annotations

import polars as pl
import pytest

from skabm.behaviour import defaults
from skabm.calibration import make_dataset, weighted_enum
from skabm.datasets import build_firm_io_df
from skabm.simulation import DEFAULT_INIT_RULES, DEFAULT_UPDATE_RULES
from skabm.sparql import parameters

H_ACTIVE = 4_729_215  # H^act, census
H_INACTIVE = 4_130_385  # H^inact, census


def test_default_rules_carry_the_cited_values(poledna_params):
    """Every parameter the default rules read defaults to the paper's number, and no
    cited value goes unread."""
    rules = (*DEFAULT_INIT_RULES, *DEFAULT_UPDATE_RULES)
    read = set().union(*map(parameters, rules))
    assert read == set(poledna_params)
    assert {k: defaults()[k] for k in read} == poledna_params


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
