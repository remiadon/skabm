"""Poledna et al. (2022) Austrian ABM calibration test suite.

Paper: "Economic forecasting with an agent-based model."
       European Economic Review 151, 104306.

Six institutional sectors (Section 3.1):
  1. Non-financial corporations (firms) - 62 industries, IO + BD
  2. Households                          - census active/inactive
  3. General government                  - 25% of domestic firms
  4. Financial corporations (banks)     - Basel III capital regulation
  5. Central bank                        - ECB Taylor rule (singleton)
  6. Rest of world (foreign firms)      - 50% of domestic firms as importers

Coverage vs paper Table 2 (~70%):
  Census / demography  H^act, H^inact, I_s, J                100 %
  IO table             a_i, w_i, delta_i, tech_share         ~80 %  (kappa_i missing)
  Government stats     tau^INC, tau^FIRM, tau^VAT, tau^SIE/SIW,
                       tau^CF, tau^G, theta^UB                 100 %
  Banking / Basel III  zeta, zeta^LTV, zeta^b, theta, mu     100 %
  National accounts    psi, psi^H, theta^DIV, r^G             100 %
  AR(1) / Taylor rule  exogenous process params               NOT YET
                       (simulation dynamics, not populations)
  Gap: kappa_i (capital productivity, needs nama_10_nfa_st loader).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import polars_random as pr
import pytest
from maplib import Model

from skabm.calibration import (
    GeneticConstraintCalibration,
    make_dataset,
    noise_floor,
    simulator_model,
    weighted_enum,
)
from skabm.datasets import build_firm_io_df
from skabm.templates import firm_template, household_template

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
        # count calibrated from business demography data (BD_9BD_SZ_CL_R2).
        "industry": weighted_enum(industry_enum, io_df["n_firms"], seed=100),
        # Firm size ~ LogNormal(3, 1): mean(log(size)) = 3.0, Pareto tail (Section 4.1.1).
        "size": pr.normal(3.0, 1.0, seed=101).exp().cast(pl.Int64).clip(1, None),
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

    # LogNormal(3, 1): mean(log(size)) = 3.0 by construction.  The sampler now
    # carries an explicit seed, so the draw is reproducible and this bound can
    # be tight (realized 2.933) instead of absorbing run-to-run RNG noise.
    assert firms["size"].log().mean() == pytest.approx(3.0, abs=0.15)

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
            "status": weighted_enum(status_enum, [H_ACTIVE, H_INACTIVE], seed=110),
            "wage_class": weighted_enum(
                wage_enum, [0.325, 0.325, 0.175, 0.175], seed=111
            ),
        },
        n_agents=2_000,
        seed=1,
    )

    assert households.columns[0] == "id"

    # Seeded weighted_enum draws → deterministic shares, so these bounds are
    # tight (realized: inactive 0.4565, Q3+Q4 0.3515) rather than ±0.04.
    inactive_share = households.filter(pl.col("status") == "inactive").height / 2_000
    assert inactive_share == pytest.approx(
        H_INACTIVE / (H_ACTIVE + H_INACTIVE), abs=0.02
    )

    q3q4_share = (
        households.filter(pl.col("wage_class").is_in(["Q3", "Q4"])).height / 2_000
    )
    assert q3q4_share == pytest.approx(0.35, abs=0.015)

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
            "purchase_sector": weighted_enum(industry_enum, io_df["output"], seed=120),
            # Fixed income tax rate τ^INC shared by all entities (Table 2).
            "tax_rate": pr.uniform(TAX["income"], TAX["income"] + 1e-12, seed=121),
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
            "capital_ratio": pr.normal(0.08, 0.02, seed=130).clip(
                BANKING["capital_ratio"], 0.30
            ),
            # Leverage = assets/equity; clipped at 1/max_capital_ratio.
            "leverage": pr.normal(12.0, 2.0, seed=131).clip(1 / 0.30, None),
            "deposit_share": pr.uniform(0.05, 0.30, seed=132),
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
            "source_industry": weighted_enum(industry_enum, io_df["output"], seed=140),
            "demand_size": pr.normal(2.5, 1.0, seed=141)
            .exp()
            .cast(pl.Int64)
            .clip(1, None),
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
#   target = a large value for manufacturing, the residual for services
# After GA, we evaluate the metric and check it is near the target.
#
# The service target is *derived*, not chosen, and that is load-bearing.  The GA
# permutes rows, so the per-column multiset — and hence the global mean of
# log(size) — is invariant.  A pair of sector targets is therefore reachable
# only when it respects
#
#     n_manuf * t_manuf + n_service * t_service == n * mean(log(size))
#
# Austria's real business demography is 293:7 services:manufacturing by firm
# count (retail, construction and professional services dominate), so a
# hand-picked pair like (3.5, 2.5) is arithmetically unreachable and the GA
# plateaus at a positive energy no budget clears.  Deriving one target from the
# other keeps the test measuring the optimiser rather than the infeasibility.
# ---------------------------------------------------------------------------

T_MANUF = 4.5  # manufacturing firms are the large ones


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

    population = make_dataset(
        samplers={
            "industry": weighted_enum(industry_enum, n_firms, seed=150),
            "size": pr.normal(3.0, 1.0, seed=151).exp().cast(pl.Int64).clip(1, None),
        },
        n_agents=300,
        seed=5,
    )

    # Derive the service target from the invariant the permutation preserves.
    n = population.height
    n_manuf = (
        population.select(sector_of.alias("s")).filter(pl.col("s") == "manuf").height
    )
    mean_log_size = float(population["size"].log().mean())
    t_service = (n * mean_log_size - n_manuf * T_MANUF) / (n - n_manuf)
    assert t_service < T_MANUF  # manufacturing really is the larger sector

    constraints = [
        (
            pl.col("size").log().mean().over(sector_of),
            pl.when(sector_of == "manuf").then(T_MANUF).otherwise(t_service),
        )
    ]

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

    # Directly evaluate each constraint's metric and target and check
    # convergence.  Seeded population + seeded GA (seed=5) make the whole
    # optimisation deterministic, so this bound is tight (realized 0.072)
    # rather than the ±0.8 that non-reproducible runs needed.
    for metric_expr, target_expr in constraints:
        achieved = firms.select(metric_expr.alias("m")).to_series()
        expected = firms.select(target_expr.alias("t")).to_series()
        assert (achieved - expected).abs().mean() == pytest.approx(0.0, abs=0.15)


# ---------------------------------------------------------------------------
# Test 7: Guard — single-column constraint metric is rejected
# ---------------------------------------------------------------------------


def test_single_column_constraint_warns():
    population = make_dataset(
        samplers={
            "size": pr.normal(3.0, 1.0, seed=160).exp().cast(pl.Int64).clip(1, None)
        },
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
            "status": weighted_enum(status_enum, [H_ACTIVE, H_INACTIVE], seed=170),
            "wage_class": weighted_enum(
                wage_enum, [0.325, 0.325, 0.175, 0.175], seed=171
            ),
        },
        n_agents=2_000,
        seed=11,
    )

    # Firms: Section 4.1.1 + 4.1.2 headline checks.  Seeded samplers → tight,
    # deterministic bound (realized 2.916).
    assert firms.columns[0] == "id"
    assert firms["size"].log().mean() == pytest.approx(3.0, abs=0.15)
    assert (firms["alpha"] > 0).all()
    assert firms["delta"].is_between(0, 1).all()
    assert firms["tech_share"].is_between(0, 1).all()

    # Households: census active share H^act / (H^act + H^inact) ≈ 53.4%.
    assert households.columns[0] == "id"
    active_share = households.filter(pl.col("status") == "active").height / 2_000
    assert active_share == pytest.approx(H_ACTIVE / (H_ACTIVE + H_INACTIVE), abs=0.02)

    # Scalar parameters from Table 2.
    assert BANKING["capital_ratio"] == pytest.approx(0.03)
    assert HH["dividend_ratio"] == pytest.approx(0.7768)
    assert HH["unemp_benefit"] == pytest.approx(0.3586)
    assert TAX["income"] == pytest.approx(0.2134)


# ---------------------------------------------------------------------------
# calibration.parameters — the (N, D) adapter a parameter search consumes
# ---------------------------------------------------------------------------

CAL_FIRMS = pl.DataFrame({"id": [f"firm_{i}" for i in range(6)]}).with_columns(
    output=pl.lit(100.0),
    price=pl.lit(1.0),
    alpha=pl.lit(12.0),
    size=pl.lit(10.0),
    margin=pl.lit(0.15),
    liquidity=pl.lit(50.0),
    profit=pl.lit(5.0),
    w_bar=pl.lit(30.0),
)
CAL_HH = pl.DataFrame(
    {
        "id": [f"hh_{i}" for i in range(12)],
        "psi": [0.9] * 12,
        "employer": [f"firm_{i % 6}" for i in range(12)],
    }
)


def cal_world() -> Model:
    world = Model()
    world.map(firm_template, CAL_FIRMS.with_iri())
    world.map(household_template, CAL_HH.with_iri("employer"))
    return world


CAL_PARAMS = {"total_deposits": 4.0e5}


def _summarise(state: pl.DataFrame) -> list[float]:
    active = state.drop_nulls("output")
    return [active["output"].sum(), active["price"].mean()]


def _model(**kwargs):
    return simulator_model(
        cal_world,
        free=["growth_sigma"],
        summarise=_summarise,
        params=CAL_PARAMS,
        **kwargs,
    )


def test_summarise_defaults_to_the_derived_observables():
    """The yielded observables *are* the per-tick summary statistics.

    A method-of-moments loss wants a fixed vector per tick, which is what the
    rule set already implies — so the common case passes no ``summarise`` and
    the columns come out named.
    """
    model = simulator_model(cal_world, free=["growth_sigma"], params=CAL_PARAMS)
    out = model([0.0], 6, 0)

    assert out.shape == (6, len(model.observables_))
    assert "sig__SUM__Firm__output" in model.observables_
    assert model.observables_ == tuple(sorted(model.observables_))  # stable columns
    assert np.isfinite(out).all()

    # track=False narrows the vector to the signals the rules read back
    narrow = simulator_model(
        cal_world, free=["growth_sigma"], params=CAL_PARAMS, track=False
    )
    assert narrow([0.0], 6, 0).shape == (6, 3)
    assert set(narrow.observables_) < set(model.observables_)


def test_simulator_model_shape_and_guards():
    """``model(theta, N, seed) -> (N, D)``, and the two ways to misuse it."""
    model = _model()
    out = model([0.0], 5, 0)
    assert out.shape == (5, 2) and model.ticks_ == 5 and model.free == ("growth_sigma",)
    # deterministic without shocks: same theta and seed, same array
    assert (model([0.0], 5, 0) == out).all()

    with pytest.raises(ValueError, match="set per call"):
        _model(n_periods=5)
    with pytest.raises(ValueError, match="not in the parameter set"):
        simulator_model(
            cal_world, free=["no_such_knob"], summarise=_summarise, params=CAL_PARAMS
        )


def test_early_stopping_preserves_the_shape_contract():
    """A stopped run is padded, so a calibrator still gets its ``(N, D)``.

    The padding carries the last row forward rather than writing a sentinel, so
    the diverged values stay in the array and the loss stays finite — the
    candidate is penalised on its own numbers.
    """
    diverged = simulator_model(
        cal_world,
        free=["growth_sigma"],
        summarise=_summarise,
        params=CAL_PARAMS,
        stop=lambda rows: rows[-1][0] <= 0,
    )
    out = diverged([0.9], 40, 0)

    assert out.shape == (40, 2)  # contract held
    assert diverged.ticks_ < 40  # ... but the run was cut short
    assert np.isfinite(out).all()  # no sentinel leaked in
    ran = diverged.ticks_
    assert (out[ran - 1] == out[-1]).all()  # the tail is the carried last row

    # a criterion that never fires costs nothing and changes nothing
    never = simulator_model(
        cal_world,
        free=["growth_sigma"],
        summarise=_summarise,
        params=CAL_PARAMS,
        stop=lambda rows: False,
    )
    assert (never([0.9], 40, 0) == _model()([0.9], 40, 0)).all()
    assert never.ticks_ == 40


def test_noise_floor_measures_seed_spread():
    """The precision floor: how much of a loss difference is just the seed.

    Uses a stub loss so the diagnostic is tested without pulling in a calibration
    toolbox — it only ever calls ``loss_function.compute_loss(simulated, real)``.
    """

    class MeanGap:
        def compute_loss(self, simulated, real_data):
            return float(np.abs(np.mean(simulated, axis=0) - real_data).mean())

    model = simulator_model(cal_world, free=["growth_sigma"], params=CAL_PARAMS)
    real = np.zeros((4, len(model([0.0], 4, 0)[0])))

    losses = noise_floor(
        model,
        theta=[0.02],
        real_data=real,
        loss_function=MeanGap(),
        ensemble_size=2,
        n_repeats=3,
    )
    assert len(losses) == 3 and all(np.isfinite(losses))
    # disjoint seed sets, so the repeats are not identical by construction
    assert len(set(losses)) > 1


def test_derived_summarise_needs_something_to_derive():
    """A rule set that implies no observable says so instead of returning (N, 0)."""
    model = simulator_model(
        cal_world,
        free=["growth_sigma"],
        params=CAL_PARAMS,
        init_rules=(),
        update_rules=(),
    )
    # no rules at all, so Firm is inert and warns before the failure we are after
    with (
        pytest.warns(UserWarning, match="not referenced"),
        pytest.raises(RuntimeError, match="implies no observable"),
    ):
        model([0.0], 3, 0)
