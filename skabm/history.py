"""
State history: what the rules imply is worth measuring, measuring it, and the DuckDB
table that can keep it.

``measure`` is the primary path and needs no database: one row of aggregates read out
of the model, which ``RDFSimulator.fit_iter`` yields.  A database opens for one reason,
a caller who wants the series to outlive the process (``duckdb_connection=``).

Nothing reads the table back.  A behaviour that depends on the model's own past keeps
that past as state on the graph: ``dsl.lag`` for a value one run ago, running sums for
a statistic of the whole sample (``behaviour.learning.sac``).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import NamedTuple

import polars as pl
import sympy as sp
from sympy.stats.rv import RandomSymbol

from skabm.dsl import (
    OPERATORS,
    _arithmetic,
    _Rule,
    coalesce,
    is_rule,
    mean,
    owner,
    total,
)
from skabm.sparql import _PREFIXES
from skabm.template import SCHEMA, Link

TABLE = "skabm_state"

# A signal's key is (aggregate, agent class, field): several classes share a field
# (firms, households and government all carry a price).
signal_name = "sig__{}__{}__{}".format


class Observable(NamedTuple):
    """``AGG(expr)`` over every agent of ``klass``, recorded as ``signal``."""

    agg: str
    klass: str
    name: str
    expr: str

    @property
    def signal(self) -> str:
        return signal_name(self.agg, self.klass, self.name)


def consumed(rules) -> set:
    """``(agg, class, field)`` of every signal the rules read back (``learning.expect``)."""
    return {
        tuple(s.name[1:].split(".")[0].split("__", 3)[1:])
        for rule in rules
        if is_rule(rule)
        for e in rule.values()
        for s in sp.sympify(e).atoms(sp.Symbol)
        if s.name.startswith("@sig__")
    }


def observables(rules, params: dict) -> tuple:
    """What the rules imply is worth measuring, name-sorted.

    Every field a rule writes, under the aggregate the rules apply to it (SUM for
    ``total``, AVG for ``mean``) or both; every signal they read back; and for a write
    capped by ``Max``/``Min``, ``binds__<field>``, the share of agents at the cap.
    """
    found = {signal: f"?{signal[2]}" for signal in consumed(rules)}
    rules = [r for r in map(_Rule, filter(is_rule, rules)) if not r.subject]
    applied: dict = {}
    for rule in rules:
        for e in rule.writes.values():
            for op in e.atoms(total, mean):
                if op.args[0].is_Symbol and owner(op.args[0]):
                    agg = "SUM" if isinstance(op, total) else "AVG"
                    applied.setdefault(owner(op.args[0]), set()).add(agg)
    for rule in rules:
        for name, e in rule.writes.items():
            if name.startswith("lag_") or isinstance(
                SCHEMA.get(rule.klass, {}).get(name, (None,))[0], Link
            ):
                continue
            for agg in applied.get((rule.klass, name), ("SUM", "AVG")):
                found[agg, rule.klass, name] = f"?{name}"
            caps = [
                cap
                for cap in (e.args if isinstance(e, (sp.Max, sp.Min)) else ())
                if not cap.atoms(RandomSymbol, coalesce, *OPERATORS)
                and all(
                    owner(s) is None or owner(s)[0] == rule.klass
                    for s in cap.free_symbols
                )
            ]
            if caps:
                cap = _flat(caps[0].subs({sp.Symbol(k): v for k, v in params.items()}))
                found["AVG", rule.klass, f"binds__{name}"] = (
                    f"IF(ABS(?{name} - {cap}) <= 1e-9 + 1e-9 * ABS({cap}), 1e0, 0e0)"
                )
    return tuple(
        sorted((Observable(*k, v) for k, v in found.items()), key=lambda o: o.signal)
    )


def _flat(e) -> str:
    """*e*, its parameters substituted, over one agent's fields, ``?field``."""
    if e.is_Symbol:
        return f"?{owner(e)[1]}"
    return _arithmetic(e, _flat, lambda text: text)


def measure(model, observables: Sequence) -> dict:
    """Every observable, measured out of *model*: one query per agent class.

    A field no agent carries comes back ``None`` rather than missing, so the row is
    the same width every tick and a run stacks rectangular.
    """
    measured: dict = {}
    for klass in sorted({o.klass for o in observables}):
        batch = [o for o in observables if o.klass == klass]
        needs = sorted({v for o in batch for v in re.findall(r"\?(\w+)", o.expr)})
        frame = model.query(
            f"{_PREFIXES}SELECT "
            + " ".join(f"({o.agg}({o.expr}) AS ?{o.signal})" for o in batch)
            + f"\nWHERE {{ ?a a ex:{klass} . "
            + " ".join(f"OPTIONAL {{ ?a def:{p} ?{p} }}" for p in needs)
            + " }"
        )
        if frame.height:
            measured.update(zip(frame.columns, frame.row(0)))
    return {o.signal: measured.get(o.signal) for o in observables}


def connect(connection: str | object | None = None):
    """Open (or adopt) the DuckDB connection holding the state history.

    ``None`` opens a fresh ``:memory:`` database — history lives as long as the
    simulator does.  A string is a file path, so a long run persists tick by
    tick and outlives the process.  An already-open connection is used as given
    and never closed here, which is how two simulators share one store.  The
    table is created if absent, so pointing at an existing file resumes it.
    """
    try:
        import duckdb
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on install
        raise ModuleNotFoundError(
            "persisting the state history needs the optional 'history' extra "
            "(pip install 'skabm[history]'), and is only reached by passing "
            "duckdb_connection="
        ) from exc

    con = (
        duckdb.connect(connection or ":memory:")
        if connection is None or isinstance(connection, str)
        else connection
    )
    con.execute(
        f"CREATE TABLE IF NOT EXISTS {TABLE} (signal VARCHAR, t INTEGER, level DOUBLE)"
    )
    return con


def record(con, row: dict, t: int) -> None:
    """Append the measured half of *row* to the table at *t*, long."""
    rows = [
        (signal, int(t), float(level))
        for signal, level in row.items()
        if level is not None
    ]
    if rows:
        con.executemany(f"INSERT INTO {TABLE} VALUES (?, ?, ?)", rows)


def state_frame(con) -> pl.DataFrame:
    """The whole history as polars (``signal``, ``t``, ``level``) — realized series."""
    return con.execute(f"SELECT signal, t, level FROM {TABLE} ORDER BY signal, t").pl()
