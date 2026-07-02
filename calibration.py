"""
Synthetic agent-population sampler for ABM calibration.

Architecture
------------
  Marginal (single-column) facts  →  encode in `samplers`.
  Joint (inter-column) facts      →  encode in `constraints` as (metric, target) tuples.

samplers: dict[str, pl.Series | pl.Expr]
    Every entry is treated as a pool from which n_agents rows are drawn with
    replacement.  pl.Series pools are used directly.  Context-free pl.Expr (e.g.
    pl.int_range(1, 100)) are materialised first via pl.select().  Column-context
    Expr that reference other sampler columns (e.g. replace_strict) are detected
    via .meta.root_names() and resolved last, in insertion order.

constraints: Sequence[tuple[pl.Expr, float | pl.Expr]]
    Each entry is (metric_expr, target).  metric_expr must reference ≥ 2 columns.
    Energy contribution = MSE(metric - target).

Output: DataFrame with a leading `id` column (UInt32, 0-based).
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
import polars as pl
import polars_random as pr


# ---------------------------------------------------------------------------
# weighted_enum: CDF-inversion via polars-random
# ---------------------------------------------------------------------------

def weighted_enum(enum: pl.Enum, weights: Sequence[float] | pl.Series) -> pl.Expr:
    """Sample a pl.Enum column with given weights using CDF inversion.

    Returns a row-context pl.Expr (polars-random); pass it directly as a
    sampler value and _pool() will materialise it via a dummy frame.

    Example
    -------
    >>> status_enum = pl.Enum(["active", "inactive"])
    >>> weighted_enum(status_enum, [4_729_215, 4_130_385])  # census proportions
    """
    if isinstance(weights, pl.Series):
        weights = weights.to_list()
    cats = list(enum.categories)
    if len(cats) != len(weights):
        raise ValueError(f"len(weights)={len(weights)} must equal len(categories)={len(cats)}")
    total = sum(weights)
    breaks = [sum(weights[: i + 1]) / total for i in range(len(cats) - 1)]
    return pr.uniform(0, 1).cut(breaks=breaks, labels=cats)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_derived(col: str, entry, sampler_keys: set[str]) -> bool:
    """True when entry is an Expr that references another sampler column."""
    if not isinstance(entry, pl.Expr):
        return False
    return bool(set(entry.meta.root_names()) & (sampler_keys - {col}))


_POOL_SIZE = 10_000


def _pool(entry: pl.Series | pl.Expr) -> pl.Series:
    """Materialise a sampler entry into a pool Series.

    pl.Series            → returned as-is.
    context-free pl.Expr → pl.select() gives a non-empty Series directly.
    row-context pl.Expr  → pl.select() returns 0 rows (polars-random, weighted_enum);
                           evaluated in a _POOL_SIZE-row dummy frame instead.
    """
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


def _energy(df: pl.DataFrame, constraints: Sequence[tuple]) -> float:
    """Evaluate all (metric, target) constraints and return total MSE energy."""
    if not constraints:
        return 0.0
    total = 0.0
    for metric_expr, target in constraints:
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
    for i, (metric_expr, _) in enumerate(constraints):
        roots = metric_expr.meta.root_names()
        if len(roots) < 2:
            raise ValueError(
                f"Constraint {i} metric references only {roots}. "
                f"Single-column facts belong in `samplers`, not `constraints`."
            )


def _split(samplers: dict) -> tuple[dict, list]:
    keys = set(samplers.keys())
    mutable = {k: v for k, v in samplers.items() if not _is_derived(k, v, keys)}
    derived = [(k, v) for k, v in samplers.items() if _is_derived(k, v, keys)]
    return mutable, derived


# ---------------------------------------------------------------------------
# make_agents  (Metropolis-Hastings / simulated annealing)
# ---------------------------------------------------------------------------

def make_agents(
    samplers: dict[str, "pl.Series | pl.Expr"],
    constraints: Sequence[tuple] = (),
    n_agents: int = 500,
    n_steps: int = 20_000,
    t0: float = 2.0,
    cooling: float = 0.9995,
    seed: int = 0,
    verbose: bool = False,
) -> pl.DataFrame:
    """Sample a synthetic agent population matching aggregate stylized facts.

    Parameters
    ----------
    samplers : dict[str, pl.Series | pl.Expr]
        Column name → pool.  pl.Series and context-free pl.Expr are sampled
        with replacement.  Column-context Expr (e.g. replace_strict) are
        resolved in insertion order after all other columns are drawn.
    constraints : Sequence[tuple[pl.Expr, float | pl.Expr]]
        (metric_expr, target) pairs; metric_expr must reference ≥ 2 columns.
    """
    _validate_constraints(constraints)
    mutable, derived = _split(samplers)
    mutable_cols = list(mutable.keys())

    col_series = {
        col: _pool(entry).sample(n_agents, with_replacement=True, seed=seed + i)
        for i, (col, entry) in enumerate(mutable.items())
    }
    schema = pl.Schema({col: s.dtype for col, s in col_series.items()})
    data = {col: s.to_list() for col, s in col_series.items()}

    def to_df(d: dict) -> pl.DataFrame:
        df = pl.DataFrame(d, schema=schema)
        for col, expr in derived:
            df = df.with_columns(expr.alias(col))
        return df

    if not constraints:
        return to_df(data).with_row_index("id")

    rng = np.random.default_rng(seed)
    row_idx = rng.integers(0, n_agents, size=n_steps)
    col_idx = rng.integers(0, len(mutable_cols), size=n_steps)
    log_u = np.log(rng.random(n_steps))

    col_hit_counts = Counter(col_idx.tolist())
    proposal_queues = {
        mutable_cols[j]: iter(
            _pool(mutable[mutable_cols[j]]).sample(count, with_replacement=True).to_list()
        )
        for j, count in col_hit_counts.items()
    }

    current_energy = _energy(to_df(data), constraints)
    best_data = {k: list(v) for k, v in data.items()}
    best_energy = current_energy
    T = t0
    EPS = 1e-9

    for step in range(n_steps):
        if best_energy <= EPS:
            break
        i = int(row_idx[step])
        col = mutable_cols[int(col_idx[step])]
        new_value = next(proposal_queues[col])
        old_value = data[col][i]
        if new_value == old_value:
            continue
        data[col][i] = new_value
        trial_energy = _energy(to_df(data), constraints)
        delta = trial_energy - current_energy
        if delta <= 0 or log_u[step] < -delta / max(T, 1e-9):
            current_energy = trial_energy
            if trial_energy < best_energy:
                best_energy = trial_energy
                best_data = {k: list(v) for k, v in data.items()}
        else:
            data[col][i] = old_value
        T *= cooling
        if verbose and step % 1000 == 0:
            print(f"step {step:6d}  T={T:.4f}  energy={current_energy:.4f}  best={best_energy:.4f}")

    if best_energy > EPS:
        print(f"[make_agents] warning: energy {best_energy:.4f} after {n_steps} steps.")

    return to_df(best_data).with_row_index("id")


# ---------------------------------------------------------------------------
# make_agents_ga  (genetic algorithm over row permutations)
# ---------------------------------------------------------------------------

def make_agents_ga(
    samplers: dict[str, "pl.Series | pl.Expr"],
    constraints: Sequence[tuple] = (),
    n_agents: int = 500,
    population_size: int = 80,
    n_generations: int = 300,
    mutation_rate: float = 0.02,
    elite_frac: float = 0.1,
    tournament_size: int = 5,
    seed: int = 0,
    verbose: bool = False,
) -> pl.DataFrame:
    """Genetic-algorithm variant: searches row permutations to satisfy inter-column constraints."""
    _validate_constraints(constraints)
    mutable, derived = _split(samplers)
    cols = list(mutable.keys())
    if not cols:
        raise ValueError("samplers must have at least one non-derived column")

    col_series = {
        col: _pool(entry).sample(n_agents, with_replacement=True, seed=seed + i)
        for i, (col, entry) in enumerate(mutable.items())
    }
    schema = pl.Schema({col: s.dtype for col, s in col_series.items()})
    base_df = pl.DataFrame({col: s.to_list() for col, s in col_series.items()}, schema=schema)
    free_cols = cols[1:]

    def apply_genome(genome: dict[str, np.ndarray]) -> pl.DataFrame:
        df = base_df if not free_cols else base_df.with_columns(
            pl.col(c).gather(pl.Series(genome[c])) for c in free_cols
        )
        for col, expr in derived:
            df = df.with_columns(expr.alias(col))
        return df

    rng = np.random.default_rng(seed)
    n_elite = max(1, int(np.ceil(elite_frac * population_size)))
    n_swaps = max(1, int(np.ceil(mutation_rate * n_agents)))

    def random_genome() -> dict[str, np.ndarray]:
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
            [scored[k] for k in rng.integers(0, len(scored), size=tournament_size)],
            key=lambda x: x[0],
        )[1]

    scored = [(_energy(apply_genome(g), constraints), g)
              for g in [random_genome() for _ in range(population_size)]]
    scored.sort(key=lambda x: x[0])
    best_energy, best_genome = scored[0]

    EPS = 1e-9
    for gen in range(n_generations):
        if best_energy <= EPS:
            break
        next_pop = [g for _, g in scored[:n_elite]]
        while len(next_pop) < population_size:
            next_pop.append(mutate({c: ox1(tournament(scored)[c], tournament(scored)[c])
                                    for c in free_cols}))
        scored = [(_energy(apply_genome(g), constraints), g) for g in next_pop]
        scored.sort(key=lambda x: x[0])
        if scored[0][0] < best_energy:
            best_energy, best_genome = scored[0]
        if verbose and gen % 50 == 0:
            print(f"gen {gen:4d}  energy={best_energy:.4f}")

    if best_energy > EPS:
        print(f"[make_agents_ga] warning: energy {best_energy:.4f} after {n_generations} generations.")

    return apply_genome(best_genome).with_row_index("id")
