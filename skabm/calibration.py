"""
Agent-population sampler and constraint calibrators for agent-based models.

Calibration in ABM vs calibration in machine learning
------------------------------------------------------
In traditional ML, "calibration" refers to aligning predicted probabilities
with empirical frequencies (e.g. Platt scaling, isotonic regression).

Here the term carries its economic / ABM meaning: finding a synthetic
micro-level population whose aggregate statistics reproduce observed macro-level
stylized facts (IO-table coefficients, census shares, Basel III ratios, etc.).
There is no training set, no label, and no generalisation error — the goal is
internal consistency of a simulated economy, not predictive accuracy on held-out
data.

The two concepts share the sklearn estimator interface purely for
interoperability with pipelines, parameter search, and cross-validation
scaffolding.  score() returns -energy (higher = better fit to constraints),
which gives a meaningful optimisation signal even though it is not a
classification or regression metric.

Public API
----------
    make_dataset(samplers, n_agents, seed)
        Draw a synthetic population from marginal distributions.
        Returns a polars DataFrame with a leading `id` column.
        ABM equivalent of sklearn.datasets.make_*; no inter-column constraints.

    GeneticConstraintCalibration(constraints, ...)
        Fits by permuting rows of the population (OX1 crossover).
        Learns samplers_ — the optimal per-column value pools (including anchor).

    MetropolisHastingsConstraintCalibration(constraints, ...)
        Fits by mutating individual cell values (MH / simulated annealing).
        Proposals are drawn from the empirical distribution of X seen at fit.

    weighted_enum(enum, weights) -> pl.Expr
        CDF-inversion sampler for a pl.Enum with given weights.

    energy(df, constraints) -> float
        Total MSE energy of df against all (metric_expr, target) constraints.

Fitted attributes (both calibrators)
-------------------------------------
    samplers_ : dict[str, pl.Series]
        Optimal per-column value pools, including the anchor column.
        Passing samplers_ to make_dataset() generates new populations that
        reflect the learned marginal distributions (joint structure is
        approximate; use transform() for exact conditional structure).

    anchor_col_ : str
        The first column of X passed to fit() — kept from X in transform().

    best_energy_ : float
        Energy achieved at the end of optimisation.

transform() semantics
---------------------
    transform(X_new) keeps the anchor column from X_new unchanged and
    conditionally resamples each free column from samplers_: for each unique
    anchor value v, free-column values are drawn (with replacement) from the
    subset of the learned pool where anchor == v.  Works for any X size.

Constraints
-----------
    Sequence[tuple[pl.Expr, float | pl.Expr]]

    Single-column constraints are accepted with a UserWarning (may erode the
    marginal distribution set by make_dataset).  Multi-column constraints are
    the primary use case: GA preserves per-column multisets exactly; MH
    preserves per-column distributions approximately.

sklearn compatibility note
--------------------------
    pl.Expr objects are serialised internally via pl.Expr.meta.serialize /
    pl.Expr.deserialize so that get_params() / clone() work correctly.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import polars as pl
import polars_random as pr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

_POOL_SIZE = 10_000

# TODO : pl.random_seed and pr.random_seed : https://github.com/diegoglozano/polars-random/issues/27


# ---------------------------------------------------------------------------
# Expr serialisation — sklearn clone / pickle safety
# ---------------------------------------------------------------------------


def _ser(v) -> bytes | float:
    return v.meta.serialize(format="binary") if isinstance(v, pl.Expr) else v


def _de(v) -> pl.Expr | float:
    return pl.Expr.deserialize(v, format="binary") if isinstance(v, bytes) else v


def _ser_constraints(constraints: Sequence[tuple]) -> list[tuple]:
    return [(_ser(m), _ser(t)) for m, t in constraints]


def _de_constraints(constraints: Sequence[tuple]) -> list[tuple]:
    return [(_de(m), _de(t)) for m, t in constraints]


# ---------------------------------------------------------------------------
# Sampling primitives
# ---------------------------------------------------------------------------


def weighted_enum(enum: pl.Enum, weights: Sequence[float] | pl.Series) -> pl.Expr:
    """CDF-inversion sampler for a pl.Enum with given weights.

    Returns a row-context pl.Expr (polars-random).  Pass directly as a sampler
    value; _pool() materialises it via a dummy frame.
    """
    if isinstance(weights, pl.Series):
        weights = weights.to_list()
    cats = list(enum.categories)
    total = sum(weights)
    breaks = [sum(weights[: i + 1]) / total for i in range(len(cats) - 1)]
    return pr.uniform(0, 1).cut(breaks=breaks, labels=cats)


def _pool(entry: pl.Series | pl.Expr) -> pl.Series:
    if isinstance(entry, pl.Series):
        return entry
    return (  # TODO : express this in pure pl.Expression form via pl.select and inline it. There should be no need for an intermediate DataFrame here -> reduce memory usage and speed up execution
        pl.int_range(_POOL_SIZE, eager=True)
        .to_frame("_idx")
        .select(entry.alias("_v"))
        .to_series()
    )


def _is_derived(col: str, entry, sampler_keys: set[str]) -> bool:
    if not isinstance(entry, pl.Expr):
        return False
    return bool(set(entry.meta.root_names()) & (sampler_keys - {col}))


def _split(samplers: dict) -> tuple[dict, list]:
    keys = set(samplers.keys())
    mutable = {k: v for k, v in samplers.items() if not _is_derived(k, v, keys)}
    derived = [(k, v) for k, v in samplers.items() if _is_derived(k, v, keys)]
    return mutable, derived


def energy(df: pl.DataFrame, constraints: Sequence[tuple]) -> float:
    """Total MSE energy of df against all (metric_expr, target) constraints.

    Accepts raw pl.Expr or serialised bytes.  Returns 0.0 when empty.
    """
    if not constraints:
        return 0.0
    total = 0.0
    for metric_expr, target in _de_constraints(constraints):
        metric = df.select(metric_expr.alias("_m")).to_series()
        if isinstance(target, pl.Expr):
            t = df.select(target.alias("_t")).to_series()
            t = float(t[0]) if t.len() == 1 else t
        else:
            t = float(target)
        diff = metric - t
        total += float(diff.pow(2).mean() if diff.len() > 1 else diff.pow(2)[0])
    return total


def _warn_single_col_constraints(constraints: Sequence[tuple]) -> None:
    for i, (metric_expr, _) in enumerate(_de_constraints(constraints)):
        roots = metric_expr.meta.root_names()
        if len(roots) < 2:
            warnings.warn(
                f"Constraint {i} references only {roots}.  Optimising a "
                "single-column constraint may shift that column's global "
                "distribution away from what make_dataset established.  "
                "Consider encoding single-column facts in samplers instead.",
                UserWarning,
                stacklevel=3,
            )


# ---------------------------------------------------------------------------
# Conditional transform helper — shared by both calibrators
# ---------------------------------------------------------------------------


def _anchor_cols(df: pl.DataFrame) -> list[str]:
    """Return columns whose dtype is Enum, Categorical, or Boolean."""
    return [
        c
        for c in df.columns
        if isinstance(df[c].dtype, (pl.Enum, pl.Categorical))
        or df[c].dtype == pl.Boolean
    ]


def _conditional_transform(
    X: pl.DataFrame,
    samplers_: dict[str, pl.Series],
    anchor_cols: list[str],
    seed: int,
) -> pl.DataFrame:
    """Conditionally resample free columns of X from learned pools.

    Groups X by every anchor column (enum/categorical/boolean).  For each
    unique combination of anchor values, draws free-column values (with
    replacement) from the rows of samplers_ that share the same combination.

    Uses polars concat_str composite keys + arg_true() + pl.Series.sample() —
    no numpy random generators.
    """
    working = X.drop("id") if "id" in X.columns else X
    free_cols = [c for c in working.columns if c not in set(anchor_cols)]

    if not anchor_cols:
        pool_size = next(iter(samplers_.values())).len()
        indices = pl.int_range(pool_size, eager=True).sample(
            working.height, with_replacement=True, seed=seed
        )
        return pl.DataFrame(
            {col: samplers_[col].gather(indices) for col in working.columns}
        ).with_row_index("id")

    _SEP = "\x00"
    pool_keys = pl.DataFrame({col: samplers_[col] for col in anchor_cols}).select(
        pl.concat_str(anchor_cols, separator=_SEP).alias("_k")
    )["_k"]

    chunks = []
    for group_key, group in working.with_row_index("_orig").group_by(anchor_cols):
        key_str = _SEP.join("" if v is None else str(v) for v in group_key)
        fitted_indices = (pool_keys == key_str).arg_true()
        sampled = fitted_indices.sample(group.height, with_replacement=True, seed=seed)
        chunks.append(
            group.select(["_orig"] + anchor_cols).with_columns(
                pl.Series(col, samplers_[col].gather(sampled)) for col in free_cols
            )
        )
    return (
        pl.concat(chunks)
        .sort("_orig")
        .drop("_orig")
        .select(working.columns)
        .with_row_index("id")
    )


# ---------------------------------------------------------------------------
# make_dataset — pure marginal sampler, no inter-column constraints
# ---------------------------------------------------------------------------


def make_dataset(
    samplers: dict[str, "pl.Series | pl.Expr"],
    n_agents: int = 500,
    seed: int = 0,
) -> pl.DataFrame:
    """Draw a synthetic agent population from marginal samplers.

    Samples each column independently from its pool and resolves derived
    columns (those referencing other sampler columns) in insertion order.
    No inter-column constraints — pass the result to a calibrator.

    Can be called with a fitted calibrator's samplers_ to generate new
    populations reflecting the learned distributions:

        new_pop = make_dataset(calibrator.samplers_, n_agents=1_000)

    Parameters
    ----------
    samplers : dict[str, pl.Series | pl.Expr]
        Column name → pool.  pl.Series and context-free pl.Expr are sampled
        with replacement.  Row-context pl.Expr (polars-random, weighted_enum)
        are materialised via a dummy frame.  Column-context Expr are resolved
        last, in insertion order.
    n_agents : int
    seed : int

    Returns
    -------
    pl.DataFrame with a leading ``id`` column (UInt32, 0-based).
    """
    mutable, derived = _split(samplers)
    col_series = {
        col: _pool(entry).sample(n_agents, with_replacement=True, seed=seed + i)
        for i, (col, entry) in enumerate(mutable.items())
    }
    schema = pl.Schema({col: s.dtype for col, s in col_series.items()})
    df = pl.DataFrame(
        {col: s.to_list() for col, s in col_series.items()}, schema=schema
    )
    for (
        col,
        expr,
    ) in (
        derived
    ):  # FIXME : remove me, this should be done via a plain .with_columns() call
        df = df.with_columns(expr.alias(col))
    return df.with_row_index(
        "id"
    )  # TODO `id` should be a string, that would make it eaaier for downstream code.


# ---------------------------------------------------------------------------
# GeneticConstraintCalibration
# ---------------------------------------------------------------------------


class GeneticConstraintCalibration(BaseEstimator, TransformerMixin):
    """Constraint calibrator using a genetic algorithm over row permutations.

    Learns samplers_ by permuting rows of the input population to minimise
    constraint energy.  Permutation leaves per-column value multisets exactly
    intact, so global marginal distributions are perfectly preserved.

    Parameters
    ----------
    constraints : Sequence[tuple[pl.Expr, float | pl.Expr]]
        Stored in serialised form for sklearn clone() / pickle safety.
    population_size, n_generations, mutation_rate, elite_frac,
    tournament_size, seed, verbose : optimisation hyper-parameters.
    """

    def __init__(
        self,
        constraints: Sequence[tuple] = (),
        population_size: int = 80,
        n_generations: int = 300,
        mutation_rate: float = 0.02,
        elite_frac: float = 0.1,
        tournament_size: int = 5,
        seed: int = 0,
        verbose: bool = False,
    ):
        self.constraints = _ser_constraints(constraints)
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.elite_frac = elite_frac
        self.tournament_size = tournament_size
        self.seed = seed
        self.verbose = verbose

    def get_params(self, deep: bool = True) -> dict:
        return {  # pragma: no cover
            "constraints": self.constraints,
            "population_size": self.population_size,
            "n_generations": self.n_generations,
            "mutation_rate": self.mutation_rate,
            "elite_frac": self.elite_frac,
            "tournament_size": self.tournament_size,
            "seed": self.seed,
            "verbose": self.verbose,
        }

    def fit(self, X: pl.DataFrame, y=None) -> "GeneticConstraintCalibration":
        """Permute rows of X to minimise constraint energy.  Populates samplers_."""
        constraints = _de_constraints(self.constraints)
        _warn_single_col_constraints(constraints)

        base_df = X.drop("id") if "id" in X.columns else X
        anchors = _anchor_cols(base_df)
        free_cols = [c for c in base_df.columns if c not in set(anchors)]
        n_agents = base_df.height
        n_elite = max(1, int(np.ceil(self.elite_frac * self.population_size)))
        n_swaps = max(1, int(np.ceil(self.mutation_rate * n_agents)))

        def apply_genome(genome: dict) -> pl.DataFrame:
            if not free_cols:
                return base_df
            return base_df.with_columns(
                pl.col(c).gather(pl.Series(genome[c])) for c in free_cols
            )

        n_free = max(1, len(free_cols))

        # Pre-generate all random indices via polars (no numpy RNG).
        # OX1 is called len(free_cols) times per genome; tournament twice per genome.
        budget = self.n_generations * self.population_size
        _rints = lambda n, hi, s: (  # noqa: E731
            pl.int_range(hi, eager=True)
            .sample(n, with_replacement=True, seed=s)
            .to_numpy()
        )
        ab_pool = _rints(budget * 2 * n_free, n_agents, self.seed)
        # 2 tournament calls per (genome, free_col): each parent per col is a separate call
        tour_pool = _rints(
            budget * self.tournament_size * 2 * n_free,
            self.population_size,
            self.seed + 1,
        )
        swap_pool = _rints(budget * n_swaps * 2 * n_free, n_agents, self.seed + 2)
        ab_ptr = tour_ptr = swap_ptr = 0

        def random_genome(idx: int) -> dict:
            return {
                c: pl.int_range(n_agents, eager=True)
                .sample(
                    n_agents,
                    with_replacement=False,
                    seed=self.seed + idx * len(free_cols) + j,
                )
                .to_numpy()
                for j, c in enumerate(free_cols)
            }

        def ox1(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
            nonlocal ab_ptr
            a, b = sorted(ab_pool[ab_ptr : ab_ptr + 2])
            ab_ptr += 2
            child = np.full(n_agents, -1, dtype=np.intp)
            child[a:b] = p1[a:b]
            in_slice = set(child[a:b].tolist())
            fill = [v for v in p2 if v not in in_slice]
            child[:a] = fill[:a]
            child[b:] = fill[a:]
            return child

        def mutate(g: dict) -> dict:
            nonlocal swap_ptr
            out = {}
            for c in free_cols:
                perm = g[c].copy()
                for _ in range(n_swaps):
                    i, j = swap_pool[swap_ptr], swap_pool[swap_ptr + 1]
                    swap_ptr += 2
                    perm[i], perm[j] = perm[j], perm[i]
                out[c] = perm
            return out

        def tournament(scored: list) -> dict:
            nonlocal tour_ptr
            indices = tour_pool[tour_ptr : tour_ptr + self.tournament_size]
            tour_ptr += self.tournament_size
            return min([scored[k % len(scored)] for k in indices], key=lambda x: x[0])[
                1
            ]

        initial = [random_genome(i) for i in range(self.population_size)]
        scored = [(energy(apply_genome(g), constraints), g) for g in initial]
        scored.sort(key=lambda x: x[0])
        best_energy_val, best_genome = scored[0]

        EPS = 1e-9
        for gen in range(self.n_generations):
            if best_energy_val <= EPS:
                break  # pragma: no cover
            next_pop = [g for _, g in scored[:n_elite]]
            while len(next_pop) < self.population_size:
                next_pop.append(
                    mutate(
                        {
                            c: ox1(tournament(scored)[c], tournament(scored)[c])
                            for c in free_cols
                        }
                    )
                )
            scored = [(energy(apply_genome(g), constraints), g) for g in next_pop]
            scored.sort(key=lambda x: x[0])
            if scored[0][0] < best_energy_val:
                best_energy_val, best_genome = scored[0]

        if best_energy_val > EPS:
            print(
                f"[GA] warning: energy {best_energy_val:.4f} after {self.n_generations} generations."
            )

        best = apply_genome(best_genome)
        self.samplers_ = {col: best[col] for col in base_df.columns}
        self.anchor_cols_ = anchors
        self.best_energy_ = best_energy_val
        return self

    def transform(self, X: pl.DataFrame, y=None) -> pl.DataFrame:
        check_is_fitted(self, "samplers_")
        return _conditional_transform(X, self.samplers_, self.anchor_cols_, self.seed)

    def score(self, X: pl.DataFrame, y=None) -> float:
        """-energy(X, constraints).  Higher is better.  Use for drift detection."""
        return -energy(X, self.constraints)


# ---------------------------------------------------------------------------
# MetropolisHastingsConstraintCalibration
# ---------------------------------------------------------------------------


class MetropolisHastingsConstraintCalibration(BaseEstimator, TransformerMixin):
    """Constraint calibrator using Metropolis-Hastings / simulated annealing.

    Mutates individual cell values by drawing proposals from the empirical
    distribution of X seen at fit (X[col] is both prior and proposal pool).
    Per-column distributions are approximately preserved.

    Parameters
    ----------
    constraints : Sequence[tuple[pl.Expr, float | pl.Expr]]
        Stored in serialised form for sklearn clone() / pickle safety.
    n_steps, t0, cooling, seed, verbose : optimisation hyper-parameters.
    """

    def __init__(
        self,
        constraints: Sequence[tuple] = (),
        n_steps: int = 20_000,
        t0: float = 2.0,
        cooling: float = 0.9995,
        seed: int = 0,
        verbose: bool = False,
    ):
        self.constraints = _ser_constraints(constraints)
        self.n_steps = n_steps
        self.t0 = t0
        self.cooling = cooling
        self.seed = seed
        self.verbose = verbose

    def get_params(self, deep: bool = True) -> dict:
        return {  # pragma: no cover
            "constraints": self.constraints,
            "n_steps": self.n_steps,
            "t0": self.t0,
            "cooling": self.cooling,
            "seed": self.seed,
            "verbose": self.verbose,
        }

    def fit(self, X: pl.DataFrame, y=None) -> "MetropolisHastingsConstraintCalibration":
        """Run MH on X using X[col] as the proposal pool.  Populates samplers_."""
        constraints = _de_constraints(self.constraints)
        _warn_single_col_constraints(constraints)

        working = X.drop("id") if "id" in X.columns else X
        anchors = _anchor_cols(working)
        free_cols = [c for c in working.columns if c not in set(anchors)]
        n_agents = working.height

        schema = pl.Schema({col: working[col].dtype for col in free_cols})
        data = {col: working[col].to_list() for col in free_cols}

        def to_df(d: dict) -> pl.DataFrame:
            df = pl.DataFrame(d, schema=schema)
            return pl.concat([working.select(anchors), df], how="horizontal_extend")

        # Pre-generate all random indices via polars (no numpy RNG).
        _rints = lambda n, hi, s: (  # noqa: E731
            pl.int_range(hi, eager=True)
            .sample(n, with_replacement=True, seed=s)
            .to_list()
        )
        row_idx = _rints(self.n_steps, n_agents, self.seed)
        col_idx = _rints(self.n_steps, len(free_cols), self.seed + 1)
        log_u = (
            pl.int_range(self.n_steps, eager=True)
            .to_frame("_i")
            .select(pr.uniform(0, 1).alias("u"))
            .to_series()
            .log()
            .to_list()
        )

        # Proposal queues — drawn from X[col] (empirical prior).
        from collections import Counter

        col_hit_counts = Counter(col_idx)
        proposal_queues = {
            free_cols[j]: iter(
                working[free_cols[j]]
                .sample(count, with_replacement=True, seed=self.seed + j + 10)
                .to_list()
            )
            for j, count in col_hit_counts.items()
        }

        current_energy = energy(to_df(data), constraints)
        best_data = {k: list(v) for k, v in data.items()}
        best_energy_val = current_energy
        T = self.t0
        EPS = 1e-9

        for step in range(self.n_steps):
            if best_energy_val <= EPS:
                break
            i = row_idx[step]
            col = free_cols[col_idx[step]]
            new_val = next(proposal_queues[col])
            old_val = data[col][i]
            if new_val == old_val:
                continue
            data[col][i] = new_val
            trial = energy(to_df(data), constraints)
            delta = trial - current_energy
            if delta <= 0 or log_u[step] < -delta / max(T, 1e-9):
                current_energy = trial
                if trial < best_energy_val:
                    best_energy_val = trial
                    best_data = {k: list(v) for k, v in data.items()}
            else:
                data[col][i] = old_val
            T *= self.cooling

        if best_energy_val > EPS:
            print(
                f"[MH] warning: energy {best_energy_val:.4f} after {self.n_steps} steps."
            )

        best = to_df(best_data)
        self.samplers_ = {col: best[col] for col in working.columns}
        self.anchor_cols_ = anchors
        self.best_energy_ = best_energy_val
        return self

    def transform(self, X: pl.DataFrame, y=None) -> pl.DataFrame:
        check_is_fitted(self, "samplers_")
        return _conditional_transform(X, self.samplers_, self.anchor_cols_, self.seed)

    def score(self, X: pl.DataFrame, y=None) -> float:
        """-energy(X, constraints).  Higher is better.  Use for drift detection."""
        return -energy(X, self.constraints)
