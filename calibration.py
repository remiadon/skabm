"""
Synthetic agent-population sampler for ABM calibration.

Given a `spec` dict (column name -> dtype-or-pool-or-sampler) and a list of
constraints, `make_agents` samples a population matching them via single-cell
Metropolis-Hastings / simulated annealing.  `make_agents_ga` is an alternative
engine that keeps marginal distributions exact and searches over row permutations
instead (better for inter-column structural constraints).  `make_heterogeneous_agents`
orchestrates multiple populations in one call.

`spec` entries, per column, are one of:
  - a pl.Enum(...) dtype: categories come from the dtype itself.
  - a pl.Series: sampled (with replacement) as an empirical bootstrap.
  - a context-free pl.Expr, e.g. `pl.int_range(0, 11)`: evaluated once via
    `pl.select(expr)` into a pool Series, then bootstrap-sampled.
  - a context-dependent pl.Expr, e.g. `pl.col("industry").replace_strict(map)`:
    detected automatically (pl.select raises) and resolved after sampling as a
    derived column -- no stochastic search needed for these.
  - any object exposing `.sample(n, seed) -> Sequence` for custom distributions.

`constraints` entries are pl.Expr values that evaluate to either:
  - Boolean (scalar or row-wise/windowed): energy = number of violated rows/cells.
  - Scalar Float64 from `.calibration.target(value, tol)`: energy = ((v-target)/tol)^2,
    which gives the search a smooth gradient toward a calibration moment.  Use this
    instead of `.eq()` on wide-range aggregates -- a flat boolean plateau gives no
    gradient to climb.

Polars namespace extension:
    pl.col("a").sum().calibration.target(15.0, tol=2.0)
    # equivalent to: ((pl.col("a").sum() - 15.0) / 2.0).pow(2)
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Polars namespace: .calibration.target()
# ---------------------------------------------------------------------------

@pl.api.register_expr_namespace("calibration")
class CalibrationNamespace:
    """Extends pl.Expr with calibration-specific scoring helpers.

    Accessible on any expression via the `.calibration` attribute after this
    module is imported.
    """

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def target(self, value: float, tol: float = 1.0) -> pl.Expr:
        """Soft calibration target: returns ((metric - value) / tol)^2.

        The expression must reduce to a single scalar when evaluated on a
        DataFrame (i.e. it must be an aggregate like `.sum()`, `.mean()`,
        `.std()`).  The result is added directly to the energy as a smooth
        squared penalty, giving the search a gradient toward `value`.

        Parameters
        ----------
        value : float
            Calibration target (the stylized fact to match).
        tol : float
            Tolerance: energy = 1 when the metric is exactly `tol` away from
            `value`.  Smaller tol = steeper penalty = more pressure to hit the
            target precisely.

        Example
        -------
        >>> pl.col("size").log().mean().calibration.target(3.5, tol=0.5)
        # energy contribution: ((mean(log(size)) - 3.5) / 0.5)^2
        """
        return ((self._expr - pl.lit(value)) / pl.lit(tol)).pow(2)


# ---------------------------------------------------------------------------
# Sampler protocol and helpers
# ---------------------------------------------------------------------------

@runtime_checkable
class Sampler(Protocol):
    def sample(self, n: int, seed: int) -> Sequence[Any]: ...


SpecEntry = "pl.DataType | pl.Series | pl.Expr | Sampler"


def _is_context_free(expr: pl.Expr) -> bool:
    """True when the expression can be evaluated without a DataFrame context."""
    try:
        pl.select(expr)
        return True
    except Exception:
        return False


def _pool(entry) -> pl.Series:
    if isinstance(entry, pl.Enum):
        return pl.Series(entry.categories.to_list())
    if isinstance(entry, pl.Series):
        return entry
    if isinstance(entry, pl.Expr):
        return pl.select(entry).to_series()
    raise TypeError(f"{entry!r} is not a pool-like spec entry (Enum / Series / Expr).")


def _draw(name: str, entry, n: int, seed: int) -> list:
    if isinstance(entry, pl.Categorical):
        raise TypeError(
            f"Column {name!r} is pl.Categorical, which doesn't carry a fixed "
            f"category list. Use pl.Enum([...]) instead."
        )
    if isinstance(entry, pl.DataType) and not isinstance(entry, pl.Enum):
        raise TypeError(
            f"Column {name!r} was given the bare dtype {entry}, which carries "
            f"no value domain. Pass a pool expression or a Sampler instead."
        )
    if isinstance(entry, (pl.Enum, pl.Series, pl.Expr)):
        return _pool(entry).sample(n=n, with_replacement=True, seed=seed).to_list()
    if isinstance(entry, Sampler):
        return list(entry.sample(n=n, seed=seed))
    raise TypeError(
        f"Column {name!r}: don't know how to sample from {entry!r}. Expected "
        f"a pl.Enum dtype, a pl.Series, a pl.Expr, or an object with `.sample(n, seed)`."
    )


def _dtype_of(name: str, entry, probe_seed: int) -> pl.DataType:
    if isinstance(entry, pl.DataType):
        return entry
    if isinstance(entry, (pl.Series, pl.Expr)):
        return _pool(entry).dtype
    dtype = getattr(entry, "dtype", None)
    if dtype is not None:
        return dtype
    return pl.Series(_draw(name, entry, 1, probe_seed)).dtype


def _energy(df: pl.DataFrame, constraints: Sequence[pl.Expr]) -> float:
    """Evaluate all constraints on `df` and return total energy (0 = perfect).

    Boolean constraint  → energy += number of False/null cells (hard violation count).
    Scalar Float64      → energy += the value directly (expected to be a
                          `.calibration.target()` penalty, already normalised).
    Anything else       → TypeError with a diagnostic message.
    """
    total = 0.0
    for i, c in enumerate(constraints):
        s = df.select(c.alias(f"_c{i}")).to_series()
        if s.dtype == pl.Boolean:
            total += float(s.fill_null(False).not_().cast(pl.UInt32).sum())
        elif s.len() == 1:
            v = s[0]
            total += 1.0 if v is None else float(v)
        else:
            raise TypeError(
                f"Constraint {i} returned a non-boolean, non-scalar Series "
                f"(dtype={s.dtype}, len={s.len()}). "
                f"Use `.calibration.target(value, tol)` for numeric moments, "
                f"or a boolean expression for row-wise/windowed constraints."
            )
    return total


# ---------------------------------------------------------------------------
# make_agents  (Metropolis-Hastings / simulated annealing)
# ---------------------------------------------------------------------------

def make_agents(
    spec: dict[str, "SpecEntry"],
    constraints: Sequence[pl.Expr] = (),
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
    spec : dict[str, SpecEntry]
        Column name -> how to sample it (Enum, Series, Expr, or Sampler).
        Context-dependent Expr (e.g. `pl.col("x").replace_strict(map)`) are
        detected automatically and resolved after sampling as derived columns.
    constraints : Sequence[pl.Expr]
        Boolean or `.calibration.target()` expressions.  See module docstring.
    n_agents, n_steps, t0, cooling, seed, verbose : see module docstring.
    """
    samplable = {k: v for k, v in spec.items()
                 if not (isinstance(v, pl.Expr) and not _is_context_free(v))}
    derived = [(k, v) for k, v in spec.items()
               if isinstance(v, pl.Expr) and not _is_context_free(v)]

    cols = list(samplable.keys())
    schema = pl.Schema({name: _dtype_of(name, samplable[name], seed - 1) for name in cols})

    data = {name: _draw(name, samplable[name], n_agents, seed + i)
            for i, name in enumerate(cols)}

    def to_df(d: dict) -> pl.DataFrame:
        return pl.DataFrame(d, schema=schema)

    row_idx = (
        pl.int_range(0, n_agents, eager=True)
        .sample(n=n_steps, with_replacement=True, seed=seed + 100)
        .to_list()
    )
    col_idx = (
        pl.int_range(0, len(cols), eager=True)
        .sample(n=n_steps, with_replacement=True, seed=seed + 101)
        .to_list()
    )
    col_hit_counts = Counter(col_idx)
    proposal_queues = {
        cols[j]: iter(_draw(cols[j], samplable[cols[j]], n_hits, seed + 200 + j))
        for j, n_hits in col_hit_counts.items()
    }
    U = 1_000_000
    log_u = (
        (pl.int_range(1, U, eager=True)
         .sample(n=n_steps, with_replacement=True, seed=seed + 300) / U)
        .log()
        .to_list()
    )

    current_energy = _energy(to_df(data), constraints)
    best_data = {k: list(v) for k, v in data.items()}
    best_energy = current_energy
    T = t0
    EPS = 1e-9

    for step in range(n_steps):
        if best_energy <= EPS:
            break
        i = row_idx[step]
        col = cols[col_idx[step]]
        new_value = next(proposal_queues[col])
        old_value = data[col][i]
        if new_value == old_value:
            continue
        data[col][i] = new_value
        trial_energy = _energy(to_df(data), constraints)
        delta = trial_energy - current_energy
        accept = delta <= 0 or log_u[step] < -delta / max(T, 1e-9)
        if accept:
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
        print(
            f"[make_agents] warning: energy {best_energy:.4f} after {n_steps} steps. "
            f"Try raising n_steps/t0, loosening tol, or using is_between() for ranges."
        )

    df = to_df(best_data)
    for col, expr in derived:
        df = df.with_columns(expr.alias(col))
    return df


# ---------------------------------------------------------------------------
# make_agents_ga  (genetic algorithm over row permutations)
# ---------------------------------------------------------------------------

def make_agents_ga(
    spec: dict[str, "SpecEntry"],
    constraints: Sequence[pl.Expr] = (),
    n_agents: int = 500,
    population_size: int = 80,
    n_generations: int = 300,
    mutation_rate: float = 0.02,
    elite_frac: float = 0.1,
    tournament_size: int = 5,
    seed: int = 0,
    verbose: bool = False,
) -> pl.DataFrame:
    """Genetic-algorithm variant of make_agents.

    Keeps every column's marginal distribution exactly as drawn and searches
    over row permutations (via `gather()`).  Permutation-invariant single-column
    statistics are satisfied by construction; the GA focuses on inter-column
    structural constraints (e.g. `n_unique().over('b').eq(k)`).

    Genome
    ------
    One permutation of [0, n_agents) per non-anchor column (anchor = first
    column in spec).  `pl.col(c).gather(genome[c])` reorders c without
    changing its values.

    Operators: OX1 crossover per column, swap mutation, tournament selection,
    elite carry-over.
    """
    samplable = {k: v for k, v in spec.items()
                 if not (isinstance(v, pl.Expr) and not _is_context_free(v))}
    derived = [(k, v) for k, v in spec.items()
               if isinstance(v, pl.Expr) and not _is_context_free(v)]

    cols = list(samplable.keys())
    if not cols:
        raise ValueError("spec must have at least one samplable column")
    schema = pl.Schema({name: _dtype_of(name, samplable[name], seed - 1) for name in cols})

    base_df = pl.DataFrame(
        {name: _draw(name, samplable[name], n_agents, seed + i)
         for i, name in enumerate(cols)},
        schema=schema,
    )
    free_cols = cols[1:]

    def apply_genome(genome: dict[str, np.ndarray]) -> pl.DataFrame:
        if not free_cols:
            return base_df
        return base_df.with_columns(
            pl.col(c).gather(pl.Series(genome[c])) for c in free_cols
        )

    def energy(genome: dict[str, np.ndarray]) -> float:
        return _energy(apply_genome(genome), constraints)

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

    def crossover(g1: dict, g2: dict) -> dict:
        return {c: ox1(g1[c], g2[c]) for c in free_cols}

    def mutate(g: dict) -> dict:
        out = {}
        for c in free_cols:
            perm = g[c].copy()
            for _ in range(n_swaps):
                i, j = rng.integers(0, n_agents, size=2)
                perm[i], perm[j] = perm[j], perm[i]
            out[c] = perm
        return out

    def tournament(scored: list[tuple[float, dict]]) -> dict:
        contestants = [scored[k] for k in rng.integers(0, len(scored), size=tournament_size)]
        return min(contestants, key=lambda x: x[0])[1]

    scored: list[tuple[float, dict]] = [(energy(g), g) for g in
                                         [random_genome() for _ in range(population_size)]]
    scored.sort(key=lambda x: x[0])
    best_energy, best_genome = scored[0]

    EPS = 1e-9
    for gen in range(n_generations):
        if best_energy <= EPS:
            break
        next_pop = [g for _, g in scored[:n_elite]]
        while len(next_pop) < population_size:
            next_pop.append(mutate(crossover(tournament(scored), tournament(scored))))
        scored = [(energy(g), g) for g in next_pop]
        scored.sort(key=lambda x: x[0])
        if scored[0][0] < best_energy:
            best_energy, best_genome = scored[0]
        if verbose and gen % 50 == 0:
            print(f"gen {gen:4d}  energy={best_energy:.4f}")

    if best_energy > EPS:
        print(
            f"[make_agents_ga] warning: energy {best_energy:.4f} after {n_generations} "
            f"generations. Try raising n_generations/population_size or loosening tol."
        )

    df = apply_genome(best_genome)
    for col, expr in derived:
        df = df.with_columns(expr.alias(col))
    return df


# ---------------------------------------------------------------------------
# make_heterogeneous_agents
# ---------------------------------------------------------------------------

def make_heterogeneous_agents(
    populations: dict[str, dict[str, Any]],
    seed: int = 0,
    verbose: bool = False,
) -> dict[str, pl.DataFrame]:
    """Sample one DataFrame per agent type for a multi-population ABM.

    Parameters
    ----------
    populations : dict[str, dict]
        Keys are population names (e.g. "firms", "households").  Each value
        is a dict with:
          - spec        : dict[str, SpecEntry]   (required)
          - n_agents    : int                    (required)
          - constraints : Sequence[pl.Expr]      (default: [])
          - method      : "mh" | "ga"            (default: "mh")
          - **kwargs    : forwarded to make_agents / make_agents_ga

    seed : int
        Base RNG seed; population i gets seed + i.
    verbose : bool
        Passed through to the underlying sampler.

    Returns
    -------
    dict[str, pl.DataFrame]
        One DataFrame per population, keyed by name.

    Example
    -------
    >>> result = make_heterogeneous_agents({
    ...     "firms": dict(
    ...         spec={"industry": industry_pool, "size": pl.int_range(1, 500)},
    ...         n_agents=1_000,
    ...         constraints=[pl.col("size").log().mean().calibration.target(3.5, tol=0.5)],
    ...         method="mh",
    ...         n_steps=30_000,
    ...     ),
    ...     "households": dict(
    ...         spec={"active": pl.Enum(["yes", "no"])},
    ...         n_agents=4_700_000,
    ...     ),
    ... })
    """
    results: dict[str, pl.DataFrame] = {}
    for i, (name, cfg) in enumerate(populations.items()):
        spec        = cfg["spec"]
        n_agents    = cfg["n_agents"]
        constraints = cfg.get("constraints", [])
        method      = cfg.get("method", "mh")
        kwargs      = {k: v for k, v in cfg.items()
                       if k not in ("spec", "n_agents", "constraints", "method")}
        sampler = make_agents if method == "mh" else make_agents_ga
        results[name] = sampler(
            spec=spec,
            constraints=constraints,
            n_agents=n_agents,
            seed=seed + i,
            verbose=verbose,
            **kwargs,
        )
    return results


# ---------------------------------------------------------------------------
# __main__: test suite against Austrian Eurostat data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from eurostat_loader import build_firm_io_df

    print("=" * 60)
    print("Loading Austrian IO + business demography (2010)...")
    io_df = build_firm_io_df("AT", 2010).filter(
        pl.col("n_firms").is_not_null() & pl.col("alpha_s").is_not_null()
    )
    print(f"  {io_df.height} industries with full data")

    alpha_map  = dict(zip(io_df["industry"].to_list(), io_df["alpha_s"].to_list()))
    w_bar_map  = dict(zip(io_df["industry"].to_list(), io_df["w_bar_s"].to_list()))
    delta_map  = dict(zip(io_df["industry"].to_list(), io_df["delta_s"].to_list()))
    # Proportional pool: repeat each industry label n_firms times
    industry_pool = pl.concat([
        pl.Series([ind] * max(1, n))
        for ind, n in zip(io_df["industry"].to_list(), io_df["n_firms"].to_list())
    ])

    # ------------------------------------------------------------------
    # Test 1: MH -- soft target on mean log-size (power-law shape) +
    #              derived productivity columns from IO maps
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 1: MH, 400 firms, size power-law + derived alpha/w_bar/delta")
    # Pool int_range(1, 100): initial mean(log) ≈ 3.65; target 3.0 is reachable
    # quickly (exp(3) ≈ 20 employees, reasonable SME population).
    firms_mh = make_agents(
        spec={
            "industry": industry_pool,
            "size":     pl.int_range(1, 100),
            # context-dependent exprs: auto-detected as derived, resolved after sampling
            "alpha":    pl.col("industry").replace_strict(alpha_map, return_dtype=pl.Float64),
            "w_bar":    pl.col("industry").replace_strict(w_bar_map, return_dtype=pl.Float64),
            "delta":    pl.col("industry").replace_strict(delta_map, return_dtype=pl.Float64),
        },
        constraints=[
            # Pareto-like firm size: mean(log(size)) ≈ 3.0 (exp(3) ≈ 20 employees)
            pl.col("size").log().mean().calibration.target(3.0, tol=0.3),
        ],
        n_agents=400,
        n_steps=20_000,
        seed=0,
        verbose=True,
    )
    achieved = firms_mh["size"].log().mean()
    print(f"  mean(log(size)) achieved: {achieved:.3f}  target: 3.0")
    assert abs(achieved - 3.0) < 0.9, f"MH failed: mean(log(size))={achieved:.3f}"
    # Derived columns: exactly one unique alpha per industry (IO calibration fact)
    n_unique_per_industry = firms_mh.select(
        pl.col("alpha").n_unique().over("industry")
    ).to_series().unique()
    assert n_unique_per_industry.to_list() == [1], \
        "alpha not homogeneous within industry"
    print(f"  alpha homogeneous per industry: OK (n_unique={n_unique_per_industry.to_list()})")
    print(firms_mh.head(5))

    # ------------------------------------------------------------------
    # Test 2: GA -- structural inter-column constraint
    #   Each industry should have at most 3 unique income classes.
    #   GA finds the row pairing via gather(); MH would fight itself here.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 2: GA, 200 firms, n_unique(income_class).over(industry) <= 3")
    # Income classes: 4 levels; we want at most 3 unique per industry group
    firms_ga = make_agents_ga(
        spec={
            "industry":     industry_pool,
            "income_class": pl.Enum(["low", "mid", "high", "top"]),
        },
        constraints=[
            pl.col("income_class").n_unique().over("industry").le(3),
        ],
        n_agents=200,
        population_size=60,
        n_generations=200,
        seed=1,
        verbose=True,
    )
    max_nunique = firms_ga.select(
        pl.col("income_class").n_unique().over("industry")
    ).to_series().max()
    print(f"  max n_unique(income_class) per industry: {max_nunique}")
    print(firms_ga.head(5))

    # ------------------------------------------------------------------
    # Test 3: make_heterogeneous_agents -- firms + active households
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 3: make_heterogeneous_agents -- firms + households")
    result = make_heterogeneous_agents(
        {
            "firms": dict(
                spec={
                    "industry": industry_pool,
                    "size":     pl.int_range(1, 500),
                    "alpha":    pl.col("industry").replace_strict(
                        alpha_map, return_dtype=pl.Float64
                    ),
                },
                n_agents=300,
                constraints=[
                    pl.col("size").log().mean().calibration.target(2.5, tol=0.3),
                ],
                method="mh",
                n_steps=10_000,
            ),
            "households": dict(
                spec={
                    # active vs inactive proportional to Austrian census (4.7M / 4.1M)
                    "status": pl.Series(["active"] * 47 + ["inactive"] * 41),
                    "wage_class": pl.Enum(["Q1", "Q2", "Q3", "Q4"]),
                },
                n_agents=500,
                # active households are more likely in upper wage quartiles
                constraints=[
                    pl.col("wage_class").is_in(["Q3", "Q4"])
                    .mean()
                    .calibration.target(0.35, tol=0.05),
                ],
                method="mh",
                n_steps=8_000,
            ),
        },
        seed=42,
        verbose=False,
    )
    for pop_name, df in result.items():
        print(f"\n  {pop_name}: {df.height} agents, columns={df.columns}")
        print(df.head(4))

    print("\n" + "=" * 60)
    print("All tests passed.")
    sys.exit(0)
