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

The two concepts share the sklearn estimator interface (fit / transform / score)
purely for interoperability with pipelines, cross-validation scaffolding, and
parameter search.  score() returns -energy (higher = better fit to constraints),
which gives a meaningful optimisation signal even though it is not a
classification or regression metric.

Public API
----------
    make_dataset(samplers, n_agents, seed)
        Draw a synthetic population from marginal distributions.
        Returns a polars DataFrame with a leading `id` column.
        This is the ABM equivalent of sklearn.datasets.make_*; no constraints
        are applied here.

    MetropolisHastingsConstraintCalibration(samplers, constraints, ...)
        Fits by mutating individual cell values (Metropolis-Hastings /
        simulated annealing).  Needs `samplers` for the proposal distribution.

    GeneticConstraintCalibration(constraints, ...)
        Fits by permuting rows of the population (OX1 crossover).
        Does not need `samplers`; the gene pool is the population itself.

    weighted_enum(enum, weights) -> pl.Expr
        Convenience sampler: CDF-inversion of a pl.Enum with given weights.

    energy(df, constraints) -> float
        Total MSE energy of df against all (metric_expr, target) constraints.

Samplers
--------
    dict[str, pl.Series | pl.Expr]

    Every entry is a pool from which n_agents rows are drawn with replacement:
    - pl.Series            → sampled directly.
    - context-free pl.Expr → pl.select() gives a non-empty Series.
    - row-context pl.Expr  → pl.select() returns 0 rows (polars-random,
                             weighted_enum); materialised via a _POOL_SIZE-row
                             dummy frame.
    Column-context Expr (those referencing other sampler columns via
    replace_strict etc.) are detected via .meta.root_names() and resolved in
    insertion order after all other columns.

Constraints
-----------
    Sequence[tuple[pl.Expr, float | pl.Expr]]

    Each entry is (metric_expr, target).  metric_expr must reference ≥ 2
    sampler columns (single-column marginal facts belong in `samplers`).
    Energy contribution = MSE(metric - target).

sklearn compatibility note
--------------------------
    polars Expr objects are not guaranteed to be hashable or deepcopy-able by
    Python's standard mechanisms, which can trip up sklearn's clone() utility.
    Both calibrators therefore store constraints in serialised form internally
    (pl.Expr.meta.serialize / pl.Expr.deserialize) and expose them transparently
    via get_params / set_params.  From the caller's perspective the API is
    unchanged: pass plain pl.Expr, get pl.Expr back.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
import polars as pl
import polars_random as pr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

_POOL_SIZE = 10_000


# ---------------------------------------------------------------------------
# Serialisation helpers — make pl.Expr safe for sklearn clone / get_params
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
    if len(cats) != len(weights):
        raise ValueError(f"len(weights)={len(weights)} must equal len(categories)={len(cats)}")
    total = sum(weights)
    breaks = [sum(weights[: i + 1]) / total for i in range(len(cats) - 1)]
    return pr.uniform(0, 1).cut(breaks=breaks, labels=cats)


def _pool(entry: pl.Series | pl.Expr) -> pl.Series:
    if isinstance(entry, pl.Series):
        return entry
    s = pl.select(entry).to_series()
    if s.len() > 0:
        return s
    return (
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

    Constraints may contain raw pl.Expr or serialised bytes (auto-detected).
    Returns 0.0 when constraints is empty.  Higher energy = worse fit.
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


def _validate_constraints(constraints: Sequence[tuple]) -> None:
    for i, (metric_expr, _) in enumerate(_de_constraints(constraints)):
        roots = metric_expr.meta.root_names()
        if len(roots) < 2:
            raise ValueError(
                f"Constraint {i} metric references only {roots}. "
                "Single-column marginal facts belong in `samplers`, not `constraints`."
            )


# ---------------------------------------------------------------------------
# make_dataset — pure ABM population sampler, no constraints
# ---------------------------------------------------------------------------

def make_dataset(
    samplers: dict[str, "pl.Series | pl.Expr"],
    n_agents: int = 500,
    seed: int = 0,
) -> pl.DataFrame:
    """Draw a synthetic agent population from marginal samplers.

    This is the ABM equivalent of sklearn.datasets.make_*: it samples each
    column independently from its pool and resolves derived columns (those
    referencing other sampler columns) in insertion order.  No inter-column
    constraints are applied; pass the result to a calibrator to enforce joint
    facts.

    Parameters
    ----------
    samplers : dict[str, pl.Series | pl.Expr]
        Column name → pool.  See module docstring for pool types.
    n_agents : int
        Population size.
    seed : int
        Seed for reproducible pool sampling.

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
    df = pl.DataFrame({col: s.to_list() for col, s in col_series.items()}, schema=schema)
    for col, expr in derived:
        df = df.with_columns(expr.alias(col))
    return df.with_row_index("id")


# ---------------------------------------------------------------------------
# MetropolisHastingsConstraintCalibration
# ---------------------------------------------------------------------------

class MetropolisHastingsConstraintCalibration(BaseEstimator, TransformerMixin):
    """Constraint calibrator using Metropolis-Hastings / simulated annealing.

    Mutates individual cell values in the population by drawing proposals from
    the sampler pools, accepting or rejecting each proposal according to the
    Boltzmann criterion.  Requires `samplers` to define the proposal distribution.

    ABM context: used to enforce aggregate stylized facts (e.g. mean firm size
    per industry matches IO-table coefficients) on a population already drawn by
    make_dataset().

    Parameters
    ----------
    samplers : dict[str, pl.Series | pl.Expr]
        Proposal pools — same dict passed to make_dataset().
    constraints : Sequence[tuple[pl.Expr, float | pl.Expr]]
        (metric_expr, target) pairs; metric_expr must reference ≥ 2 columns.
        Stored in serialised form internally for sklearn compatibility.
    n_steps, t0, cooling, seed, verbose : see module docstring.
    """

    def __init__(
        self,
        samplers: dict,
        constraints: Sequence[tuple] = (),
        n_steps: int = 20_000,
        t0: float = 2.0,
        cooling: float = 0.9995,
        seed: int = 0,
        verbose: bool = False,
    ):
        self.samplers = samplers
        self.constraints = _ser_constraints(constraints)
        self.n_steps = n_steps
        self.t0 = t0
        self.cooling = cooling
        self.seed = seed
        self.verbose = verbose

    def get_params(self, deep: bool = True) -> dict:
        return {
            "samplers":    self.samplers,
            "constraints": self.constraints,   # bytes — safe for clone / pickle
            "n_steps":     self.n_steps,
            "t0":          self.t0,
            "cooling":     self.cooling,
            "seed":        self.seed,
            "verbose":     self.verbose,
        }

    def fit(self, X: pl.DataFrame, y=None) -> "MetropolisHastingsConstraintCalibration":
        """Run MH optimisation on population X."""
        constraints = _de_constraints(self.constraints)
        _validate_constraints(constraints)
        mutable, derived = _split(self.samplers)
        mutable_cols = list(mutable.keys())
        n_agents = X.height

        working = X.drop("id") if "id" in X.columns else X
        schema = pl.Schema({col: working[col].dtype for col in mutable_cols})

        def to_df(d: dict) -> pl.DataFrame:
            df = pl.DataFrame(d, schema=schema)
            for col, expr in derived:
                df = df.with_columns(expr.alias(col))
            for col in working.columns:
                if col not in df.columns:
                    df = df.with_columns(working[col].alias(col))
            return df.select(working.columns)

        data = {col: working[col].to_list() for col in mutable_cols}

        if not constraints:
            self.best_df_ = X
            self.best_energy_ = 0.0
            return self

        rng = np.random.default_rng(self.seed)
        row_idx = rng.integers(0, n_agents, size=self.n_steps)
        col_idx = rng.integers(0, len(mutable_cols), size=self.n_steps)
        log_u = np.log(rng.random(self.n_steps))

        col_hit_counts = Counter(col_idx.tolist())
        proposal_queues = {
            mutable_cols[j]: iter(
                _pool(mutable[mutable_cols[j]]).sample(count, with_replacement=True).to_list()
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
            i = int(row_idx[step])
            col = mutable_cols[int(col_idx[step])]
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
            if self.verbose and step % 1_000 == 0:
                print(f"step {step:6d}  T={T:.4f}  energy={current_energy:.4f}  best={best_energy_val:.4f}")

        if best_energy_val > EPS:
            print(f"[MH] warning: energy {best_energy_val:.4f} after {self.n_steps} steps.")

        best = to_df(best_data)
        self.best_df_ = best.with_row_index("id") if "id" not in best.columns else best
        self.best_energy_ = best_energy_val
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:  # noqa: ARG002
        check_is_fitted(self, "best_df_")
        return self.best_df_

    def score(self, X: pl.DataFrame, y=None) -> float:
        """-energy(X, constraints); higher is better (sklearn sign convention)."""
        return -energy(X, self.constraints)


# ---------------------------------------------------------------------------
# GeneticConstraintCalibration
# ---------------------------------------------------------------------------

class GeneticConstraintCalibration(BaseEstimator, TransformerMixin):
    """Constraint calibrator using a genetic algorithm over row permutations.

    Optimises by rearranging existing rows of the population via OX1 crossover
    and swap mutation.  Does not need `samplers`; the gene pool is the
    population X passed to fit().

    ABM context: enforces inter-column joint facts (e.g. larger firms
    concentrated in capital-intensive industries) on a population whose
    marginals were already set by make_dataset().

    Parameters
    ----------
    constraints : Sequence[tuple[pl.Expr, float | pl.Expr]]
        Stored in serialised form internally for sklearn compatibility.
    population_size, n_generations, mutation_rate, elite_frac,
    tournament_size, seed, verbose : see module docstring.
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
        return {
            "constraints":    self.constraints,
            "population_size": self.population_size,
            "n_generations":  self.n_generations,
            "mutation_rate":  self.mutation_rate,
            "elite_frac":     self.elite_frac,
            "tournament_size": self.tournament_size,
            "seed":           self.seed,
            "verbose":        self.verbose,
        }

    def fit(self, X: pl.DataFrame, y=None) -> "GeneticConstraintCalibration":
        """Run GA optimisation, permuting rows of X to minimise constraint energy."""
        constraints = _de_constraints(self.constraints)
        _validate_constraints(constraints)
        base_df = X.drop("id") if "id" in X.columns else X
        free_cols = base_df.columns[1:]
        n_agents = base_df.height

        def apply_genome(genome: dict) -> pl.DataFrame:
            if not free_cols:
                return base_df
            return base_df.with_columns(
                pl.col(c).gather(pl.Series(genome[c])) for c in free_cols
            )

        rng = np.random.default_rng(self.seed)
        n_elite = max(1, int(np.ceil(self.elite_frac * self.population_size)))
        n_swaps = max(1, int(np.ceil(self.mutation_rate * n_agents)))

        def random_genome() -> dict:
            return {c: rng.permutation(n_agents) for c in free_cols}

        def ox1(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
            a, b = sorted(rng.integers(0, n_agents, size=2))
            child = np.full(n_agents, -1, dtype=np.intp)
            child[a:b] = p1[a:b]
            in_slice = set(child[a:b].tolist())
            fill = [v for v in p2 if v not in in_slice]
            child[:a] = fill[:a]
            child[b:] = fill[a:]
            return child

        def mutate(g: dict) -> dict:
            out = {}
            for c in free_cols:
                perm = g[c].copy()
                for _ in range(n_swaps):
                    i, j = rng.integers(0, n_agents, size=2)
                    perm[i], perm[j] = perm[j], perm[i]
                out[c] = perm
            return out

        def tournament(scored: list) -> dict:
            return min(
                [scored[k] for k in rng.integers(0, len(scored), size=self.tournament_size)],
                key=lambda x: x[0],
            )[1]

        scored = [(energy(apply_genome(g), constraints), g)
                  for g in [random_genome() for _ in range(self.population_size)]]
        scored.sort(key=lambda x: x[0])
        best_energy_val, best_genome = scored[0]

        EPS = 1e-9
        for gen in range(self.n_generations):
            if best_energy_val <= EPS:
                break
            next_pop = [g for _, g in scored[:n_elite]]
            while len(next_pop) < self.population_size:
                next_pop.append(mutate({c: ox1(tournament(scored)[c], tournament(scored)[c])
                                        for c in free_cols}))
            scored = [(energy(apply_genome(g), constraints), g) for g in next_pop]
            scored.sort(key=lambda x: x[0])
            if scored[0][0] < best_energy_val:
                best_energy_val, best_genome = scored[0]
            if self.verbose and gen % 50 == 0:
                print(f"gen {gen:4d}  energy={best_energy_val:.4f}")

        if best_energy_val > EPS:
            print(f"[GA] warning: energy {best_energy_val:.4f} after {self.n_generations} generations.")

        self.best_df_ = apply_genome(best_genome).with_row_index("id")
        self.best_energy_ = best_energy_val
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:  # noqa: ARG002
        check_is_fitted(self, "best_df_")
        return self.best_df_

    def score(self, X: pl.DataFrame, y=None) -> float:
        """-energy(X, constraints); higher is better (sklearn sign convention)."""
        return -energy(X, self.constraints)
