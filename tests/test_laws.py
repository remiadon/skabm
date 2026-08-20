"""Equivalence tests: the plain-polars laws of motion (``skabm.laws``) must
reproduce, to floating point, the non-relational SPARQL update rules they
replace (``skabm.rules``).

The SPARQL path is treated as the semantically transparent reference (as
``simulation.py`` frames it): each test maps a tiny population into a maplib
model, runs the reference rule once via ``Model.update``, and checks the
polars expression applied to the same starting frame with ``.with_columns``
lands on the same numbers.  Stochastic laws are checked with ``*_sigma = 0``
(deterministic) for exact equality, plus a separate reproducibility check.
"""

from __future__ import annotations

import polars as pl
import polars_random as pr
import pytest

from skabm import laws, rules
from skabm.rules import DEF_NS, EX_NS, POLEDNA_PARAMS, map_df, render
from maplib import Model

P0 = {
    **POLEDNA_PARAMS,
    "growth_sigma": 0.0,
    "inflation_sigma": 0.0,
    "gov_growth_sigma": 0.0,
}


def _sparql_attr(
    kind: str, df: pl.DataFrame, rule, attr: str, extra_kinds=None
) -> dict:
    """Run one reference SPARQL update and read back ``def:<attr>`` per agent."""
    m = Model()
    rules.register_polars_random(m)
    map_df(m, df, kind)
    for k, extra in (extra_kinds or {}).items():
        map_df(m, extra, k)
    m.update(render(rule, P0))
    out = m.query(f"PREFIX def:<{DEF_NS}> SELECT ?id ?v WHERE {{ ?id def:{attr} ?v }}")
    return {
        row["id"].strip("<>").removeprefix(EX_NS): row["v"]
        for row in out.iter_rows(named=True)
    }


def _polars_attr(df: pl.DataFrame, expr) -> dict:
    got = df.with_columns(expr if isinstance(expr, list) else [expr])
    exprs = expr if isinstance(expr, list) else [expr]
    attr = exprs[0].meta.output_name()
    return dict(zip(df["id"], got[attr]))


@pytest.mark.parametrize(
    "kind, df, rule, law, attr",
    [
        (
            "Firm",
            pl.DataFrame(
                {
                    "id": ["firm_0", "firm_1"],
                    "output": [10.0, 5.0],
                    "alpha": [3.0, 100.0],
                    "size": [4.0, 2.0],
                }
            ),
            rules.FIRM_PRODUCTION,
            laws.firm_production,
            "output",
        ),
        (
            "Firm",
            pl.DataFrame({"id": ["firm_0", "firm_1"], "price": [1.0, 2.5]}),
            rules.FIRM_PRICING,
            laws.firm_pricing,
            "price",
        ),
        (
            "Household",
            pl.DataFrame(
                {
                    "id": ["hh_0", "hh_1"],
                    "wealth": [100.0, 50.0],
                    "psi": [0.9, 0.5],
                    "income": [20.0, 8.0],
                }
            ),
            rules.HOUSEHOLD_UPDATE,
            laws.household_update,
            "wealth",
        ),
        (
            "Government",
            pl.DataFrame({"id": ["gov_0"], "budget": [1000.0]}),
            rules.GOVERNMENT_CONSUMPTION,
            laws.government_consumption,
            "budget",
        ),
    ],
)
def test_law_matches_sparql(kind, df, rule, law, attr):
    pr.set_random_seed(0)
    ref = _sparql_attr(kind, df, rule, attr)
    got = _polars_attr(df, law(P0))
    for agent, value in got.items():
        assert value == pytest.approx(ref[agent]), agent


def test_taylor_rule_matches_sparql():
    firms = pl.DataFrame(
        {"id": ["firm_0", "firm_1"], "output": [10.0, 6.0], "price": [1.2, 0.8]}
    )
    cb = pl.DataFrame(
        {
            "id": ["cb_0"],
            "policy_rate": [0.01],
            "prev_output": [15.0],
            "prev_price": [0.9],
        }
    )

    m = Model()
    rules.register_polars_random(m)
    map_df(m, firms, "Firm")
    map_df(m, cb, "CentralBank")
    m.update(render(rules.TAYLOR_RULE, P0))
    ref = m.query(
        f"PREFIX def:<{DEF_NS}> SELECT ?r WHERE {{ ?cb def:policy_rate ?r }}"
    )["r"][0]

    got = cb.with_columns(
        *laws.taylor_rule(P0, firms["output"].sum(), firms["price"].mean())
    )
    assert got["policy_rate"][0] == pytest.approx(ref)


def test_stochastic_law_is_seed_reproducible():
    df = pl.DataFrame({"id": ["firm_0", "firm_1", "firm_2"], "price": [1.0, 2.0, 3.0]})
    p = {**POLEDNA_PARAMS, "inflation_sigma": 0.02}

    pr.set_random_seed(123)
    a = df.with_columns(laws.firm_pricing(p))["price"].to_list()
    pr.set_random_seed(123)
    b = df.with_columns(laws.firm_pricing(p))["price"].to_list()
    det = df.with_columns(laws.firm_pricing(P0))["price"].to_list()

    assert a == b  # same seed -> same trajectory
    assert a != det  # sigma>0 actually perturbs
