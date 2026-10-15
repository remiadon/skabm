"""Poledna et al. (2023) — every number here is the paper's, nothing is invented.

Paper: "Economic forecasting with an agent-based model."
       European Economic Review 151, 104306.  Austria, reference quarter 2010:Q4.

Covered: the quarter's (``worlds.POLEDNA``) own defaults are the values the ``poledna_params``
fixture cites, one for every placeholder; census activity shares (Section 4.1.1).  The
IO-table coefficients (Section 4.1.2) are built from Eurostat on notebooks/poledna.ipynb.
"""

from __future__ import annotations

import polars as pl
import pytest
from worlds import POLEDNA

from skabm.behaviour import defaults
from skabm.behaviour.household import TOTAL_DEPOSITS
from skabm.calibration import make_dataset, weighted_enum
from skabm.dsl import _Rule

H_ACTIVE = 4_729_215  # H^act, census
H_INACTIVE = 4_130_385  # H^inact, census


def test_default_rules_carry_the_cited_values(poledna_params):
    """Every parameter the default rules read defaults to the paper's number, and no
    cited value goes unread: D^H seeds the starting wealth (``household.opening_balance``)."""
    read = set().union(*(_Rule(r).parameters() for r in POLEDNA))
    assert read | {"total_deposits"} == set(poledna_params)
    assert {k: defaults()[k] for k in read} == {k: poledna_params[k] for k in read}
    assert poledna_params["total_deposits"] == TOTAL_DEPOSITS


def test_household_census_shares():
    status = pl.Enum(["active", "inactive"])
    households = make_dataset(
        samplers={"status": weighted_enum(status, [H_ACTIVE, H_INACTIVE], seed=110)},
        n_agents=2_000,
        seed=1,
    )
    inactive = households.filter(pl.col("status") == "inactive").height / 2_000
    assert inactive == pytest.approx(H_INACTIVE / (H_ACTIVE + H_INACTIVE), abs=0.02)
