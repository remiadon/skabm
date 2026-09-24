"""Sample-autocorrelation learning of the model's own aggregates, Hommes & Zhu (2014)."""

from __future__ import annotations

import sympy as sp

from skabm.dsl import DSLError, coalesce, lag, mean, node, total
from skabm.history import signal_name


def sac(agg: str, klass: str, predicate: str) -> dict:
    """SAC learning of ``agg`` of ``predicate`` over ``klass``, on the signal's node:
    a + b (g_last - a), a the mean of the growth rates observed so far and b their
    first-order autocorrelation, clipped to (-0.999, 0.999).  Hommes & Zhu (2014)
    J. Econ. Theory 149; Poledna et al. (2023) eqs. 6, 9.

    Seven running sums replace the sample, each 0 until the node carries it:
    ``tests/test_history.py`` checks them against the two-pass estimate.
    """
    levels = {"SUM": total, "AVG": mean}
    if agg not in levels:
        raise DSLError(f"SAC learns SUM and AVG signals, not {agg}")
    name = signal_name(agg, klass, predicate)
    field = {f: node(name, f) for f in ("k", "s1", "s2", "p", "first", "last")}
    k, s1_, s2_, p, first_, last_ = (coalesce(v, 0) for v in field.values())
    level = levels[agg](sp.Symbol(f"{klass}.{predicate}"))
    g = level / lag(level) - 1
    s1, s2, last = s1_ + g, s2_ + g**2, g
    first = sp.Piecewise((g, sp.Eq(k, 1)), (first_, True))
    a = s1 / sp.Max(k, 1)
    num = (p + g * last_) - a * ((s1 - first) + (s1 - last)) + (k - 1) * a**2
    den = s2 - k * a**2
    ratio = num / sp.Piecewise((den, den > 0), (1, True))  # no 0/0, even untaken
    b = sp.Piecewise((sp.Max(-0.999, sp.Min(0.999, ratio)), den > 0), (0, True))
    return {
        field["k"]: k + 1,
        field["s1"]: s1,
        field["s2"]: s2,
        field["p"]: p + g * last_,
        field["first"]: first,
        field["last"]: last,
        node(name, "forecast"): sp.Piecewise((a + b * (last - a), k > 0), (0, True)),
    }


def expect(agg: str, klass: str, predicate: str):
    """The learned forecast of the growth of ``agg`` of ``predicate`` over ``klass``,
    0 until a learner has run: ``expect("SUM", "Firm", "output")``."""
    return coalesce(node(signal_name(agg, klass, predicate), "forecast"), 0)
