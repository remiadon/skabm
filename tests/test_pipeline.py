"""
Integration tests for the make_dataset → calibrator pipeline.

Key properties verified:
  1. GeneticConstraintCalibration preserves per-column multisets exactly.
  2. MetropolisHastingsConstraintCalibration approximately preserves per-column
     distributions (mean / std within tolerance).
  3. Both calibrators satisfy inter-column constraints after fit+transform.
  4. score() detects distribution drift: calibrated data scores better than
     random data, which scores better than deliberately broken data.
  5. transform() generalises: fitted calibrator handles a population of a
     different size from the training data.
  6. make_dataset(calibrator.samplers_) generates fresh populations that
     inherit the learned inter-column structure.
  7. Single-column constraints trigger a UserWarning, not an error.
"""

from __future__ import annotations

import warnings

import polars as pl
import polars_random as pr
import pytest

from skabm.calibration import (
    GeneticConstraintCalibration,
    MetropolisHastingsConstraintCalibration,
    energy,
    make_dataset,
    weighted_enum,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SECTOR_ENUM = pl.Enum(["manuf", "service"])
AGE_CLASS_ENUM = pl.Enum(["young", "mature"])

SAMPLERS = {
    "sector": weighted_enum(SECTOR_ENUM, [1, 1]),  # 50 / 50
    "age_class": weighted_enum(AGE_CLASS_ENUM, [2, 1]),  # 67 % young
    "size": pr.normal(mean=4.0, std=1.0).exp().cast(pl.Int64).clip(1, None),
    "wage": pr.normal(mean=2.5, std=0.5).exp(),
}

# Two inter-column constraints, each pointing to a different enum anchor:
#   1. manufacturing firms are larger than service firms  (size ~ sector)
#   2. mature firms pay higher wages than young firms     (wage ~ age_class)
CONSTRAINTS = [
    (
        pl.col("size").log().mean().over("sector"),
        pl.when(pl.col("sector") == "manuf").then(5.0).otherwise(3.0),
    ),
    (
        pl.col("wage").log().mean().over("age_class"),
        pl.when(pl.col("age_class") == "mature").then(3.0).otherwise(2.0),
    ),
]


@pytest.fixture(scope="module")
def population():
    return make_dataset(SAMPLERS, n_agents=400, seed=0)


@pytest.fixture(scope="module")
def fitted_ga(population):
    return GeneticConstraintCalibration(
        constraints=CONSTRAINTS,
        population_size=30,
        n_generations=120,
        seed=0,
    ).fit(population)


@pytest.fixture(scope="module")
def fitted_mh(population):
    return MetropolisHastingsConstraintCalibration(
        constraints=CONSTRAINTS,
        n_steps=8_000,
        seed=1,
    ).fit(population)


# ---------------------------------------------------------------------------
# 1. make_dataset sanity
# ---------------------------------------------------------------------------


def test_make_dataset_shape_and_id(population):
    assert population.height == 400
    assert population.columns[0] == "id"
    assert population["id"].to_list() == list(range(400))


def test_make_dataset_column_types(population):
    assert population["sector"].dtype == SECTOR_ENUM
    assert population["age_class"].dtype == AGE_CLASS_ENUM
    assert population["size"].dtype == pl.Int64
    assert population["wage"].dtype == pl.Float64


def test_make_dataset_sector_balance(population):
    counts = population["sector"].value_counts()
    manuf = counts.filter(pl.col("sector") == "manuf")["count"][0]
    assert manuf == pytest.approx(200, abs=40)  # 50 % ± 10 %


def test_make_dataset_age_class_balance(population):
    counts = population["age_class"].value_counts()
    young = counts.filter(pl.col("age_class") == "young")["count"][0]
    assert young == pytest.approx(267, abs=40)  # 67 % ± 10 %


# ---------------------------------------------------------------------------
# 2. GA — fitted attributes and samplers_
# ---------------------------------------------------------------------------


def test_ga_has_samplers_after_fit(fitted_ga):
    assert hasattr(fitted_ga, "samplers_")
    assert set(fitted_ga.samplers_.keys()) == {"sector", "age_class", "size", "wage"}
    for pool in fitted_ga.samplers_.values():
        assert isinstance(pool, pl.Series)


def test_ga_anchor_cols(fitted_ga):
    assert set(fitted_ga.anchor_cols_) == {"sector", "age_class"}


def test_ga_best_energy_recorded(fitted_ga):
    assert hasattr(fitted_ga, "best_energy_")
    assert fitted_ga.best_energy_ >= 0.0


# ---------------------------------------------------------------------------
# 3. GA — transform satisfies constraints
# ---------------------------------------------------------------------------


def test_ga_transform_satisfies_constraints(population, fitted_ga):
    calibrated = fitted_ga.transform(population)
    for metric_expr, target_expr in CONSTRAINTS:
        achieved = calibrated.select(metric_expr.alias("m")).to_series()
        expected = calibrated.select(target_expr.alias("t")).to_series()
        assert (achieved - expected).abs().mean() == pytest.approx(0.0, abs=0.8)


def test_ga_transform_output_shape(population, fitted_ga):
    calibrated = fitted_ga.transform(population)
    assert calibrated.height == population.height
    assert calibrated.columns[0] == "id"
    assert set(calibrated.columns) == set(population.columns)


# ---------------------------------------------------------------------------
# 4. GA — per-column distributions preserved after conditional transform
#
# transform() does conditional sampling: for each anchor value, free-column
# values are drawn from the fitted population rows that share that anchor value.
# The anchor itself is kept exactly from X.
# The multiset of free columns may differ (sampling with replacement), but
# the per-column marginal distributions are preserved in expectation.
# ---------------------------------------------------------------------------


def test_ga_preserves_anchor_distributions(population, fitted_ga):
    # All anchor columns kept from X unchanged — exact distribution match.
    calibrated = fitted_ga.transform(population)
    for col in fitted_ga.anchor_cols_:
        orig = population[col].value_counts().sort(col)["count"]
        cal = calibrated[col].value_counts().sort(col)["count"]
        assert (orig == cal).all(), f"distribution mismatch for anchor column {col!r}"


def test_ga_preserves_size_distribution_approximately(population, fitted_ga):
    # Conditional sampling preserves per-column marginals in expectation.
    calibrated = fitted_ga.transform(population)
    orig_log_mean = population["size"].log().mean()
    cal_log_mean = calibrated["size"].log().mean()
    assert cal_log_mean == pytest.approx(orig_log_mean, abs=0.5)


# ---------------------------------------------------------------------------
# 5. GA — transform generalises to a different population size
# ---------------------------------------------------------------------------


def test_ga_transform_different_size(fitted_ga):
    bigger = make_dataset(SAMPLERS, n_agents=800, seed=99)
    calibrated = fitted_ga.transform(bigger)
    assert calibrated.height == 800
    # The calibrated version should score at least as well as uncalibrated.
    assert fitted_ga.score(calibrated) >= fitted_ga.score(bigger)


# ---------------------------------------------------------------------------
# 6. score() for drift detection
# ---------------------------------------------------------------------------


def test_score_calibrated_beats_random(population, fitted_ga):
    calibrated = fitted_ga.transform(population)
    assert fitted_ga.score(calibrated) >= fitted_ga.score(population)


def test_score_detects_broken_data(population, fitted_ga):
    calibrated = fitted_ga.transform(population)
    # Deliberately break the size-sector relationship: set all sizes to 1.
    broken = calibrated.with_columns(pl.lit(1).cast(pl.Int64).alias("size"))
    assert fitted_ga.score(calibrated) > fitted_ga.score(broken)


def test_score_on_drift(fitted_ga):
    # Two populations: one satisfying both constraints (good), one violating size~sector.
    good = pl.DataFrame(
        {
            "sector": pl.Series(["manuf"] * 200 + ["service"] * 200, dtype=SECTOR_ENUM),
            "age_class": pl.Series(
                ["young"] * 267 + ["mature"] * 133, dtype=AGE_CLASS_ENUM
            ),
            "size": pl.Series([500] * 200 + [10] * 200, dtype=pl.Int64),
            "wage": pl.Series([3.0] * 400),
        }
    )
    drifted = pl.DataFrame(
        {
            "sector": pl.Series(["manuf"] * 200 + ["service"] * 200, dtype=SECTOR_ENUM),
            "age_class": pl.Series(
                ["young"] * 267 + ["mature"] * 133, dtype=AGE_CLASS_ENUM
            ),
            "size": pl.Series([10] * 200 + [500] * 200, dtype=pl.Int64),  # reversed
            "wage": pl.Series([3.0] * 400),
        }
    )
    assert fitted_ga.score(good) > fitted_ga.score(drifted)


# ---------------------------------------------------------------------------
# 7. make_dataset(calibrator.samplers_) — generate new populations
#
# samplers_ holds per-column value pools sampled independently.  This
# preserves marginal distributions but NOT inter-column joint structure:
# make_dataset samples each column independently, so conditional constraints
# are only approximately satisfied.  For exact constraint satisfaction, use
# cal.transform() instead.
# ---------------------------------------------------------------------------


def test_make_dataset_from_fitted_samplers_preserves_marginals(fitted_ga):
    """make_dataset with fitted samplers_ preserves per-column marginals."""
    new_pop = make_dataset(fitted_ga.samplers_, n_agents=400, seed=7)
    fitted_size_mean = fitted_ga.samplers_["size"].log().mean()
    assert new_pop["size"].log().mean() == pytest.approx(fitted_size_mean, abs=0.5)


def test_transform_gives_better_score_than_make_dataset_from_samplers(
    fitted_ga, population
):
    """transform() (conditional) scores better than make_dataset() (marginal)."""
    new_pop = make_dataset(fitted_ga.samplers_, n_agents=400, seed=7)
    calibrated = fitted_ga.transform(population)
    assert fitted_ga.score(calibrated) >= fitted_ga.score(new_pop)


# ---------------------------------------------------------------------------
# 8. MH — approximately preserves per-column distributions
# ---------------------------------------------------------------------------


def test_mh_transform_satisfies_constraints(population, fitted_mh):
    calibrated = fitted_mh.transform(population)
    for metric_expr, target_expr in CONSTRAINTS:
        achieved = calibrated.select(metric_expr.alias("m")).to_series()
        expected = calibrated.select(target_expr.alias("t")).to_series()
        assert (achieved - expected).abs().mean() == pytest.approx(0.0, abs=1.0)


def test_mh_approximately_preserves_size_mean(population, fitted_mh):
    calibrated = fitted_mh.transform(population)
    orig_log_mean = population["size"].log().mean()
    cal_log_mean = calibrated["size"].log().mean()
    assert cal_log_mean == pytest.approx(orig_log_mean, abs=0.5)


def test_mh_approximately_preserves_size_std(population, fitted_mh):
    calibrated = fitted_mh.transform(population)
    orig_std = population["size"].log().std()
    cal_std = calibrated["size"].log().std()
    assert cal_std == pytest.approx(orig_std, abs=0.5)


def test_mh_score_improves_after_fit(population, fitted_mh):
    calibrated = fitted_mh.transform(population)
    assert fitted_mh.score(calibrated) >= fitted_mh.score(population)


# ---------------------------------------------------------------------------
# 9. Single-column constraints warn, not raise
# ---------------------------------------------------------------------------


def test_single_column_constraint_warns_at_fit(population):
    cal = GeneticConstraintCalibration(
        constraints=[(pl.col("size").mean(), 20.0)],
        n_generations=1,
    )
    with pytest.warns(UserWarning, match="single-column"):
        cal.fit(population)


def test_single_column_constraint_still_runs(population):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cal = GeneticConstraintCalibration(
            constraints=[(pl.col("size").mean(), 20.0)],
            population_size=5,
            n_generations=5,
            seed=0,
        ).fit(population)
    assert hasattr(cal, "samplers_")


# ---------------------------------------------------------------------------
# 10. energy() utility
# ---------------------------------------------------------------------------


def test_energy_zero_with_no_constraints(population):
    assert energy(population, []) == 0.0


def test_energy_positive_when_constraint_violated(population):
    bad_constraint = [
        (
            pl.col("size").log().mean().over("sector"),
            pl.when(pl.col("sector") == "manuf").then(100.0).otherwise(0.0),
        )
    ]
    assert energy(population, bad_constraint) > 0.0
