"""
Housing market: the composition claims of ``skabm.housing``, pinned.

Each test names the property the decomposition is supposed to buy — the rules compose
by tick order, each one owns exactly one predicate, either can be ablated on its own,
and the mixing weight is an ordinary parameter.
"""

import polars as pl
import pytest
from sklearn.base import clone

from skabm.housing import (
    ASK_PRICE,
    COMPARABLE_PRICE,
    HOUSING_INIT_RULES,
    HOUSING_PARAMS,
    HOUSING_UPDATE_RULES,
    MARKUP_HEURISTIC,
    NO_COMPARABLES_RULES,
    NO_HEURISTIC_RULES,
    housing_extract,
    make_listings,
    neighbourhood_mean,
)
from skabm.ir import analyse, rule_name
from skabm.rules import register_polars_random, render
from skabm.simulation import RDFSimulator


def simulator(update_rules=HOUSING_UPDATE_RULES, n_periods=12, **params):
    return RDFSimulator(
        init_rules=HOUSING_INIT_RULES,
        update_rules=update_rules,
        params={**HOUSING_PARAMS, **params},
        n_periods=n_periods,
        state_extract=housing_extract,
        udfs=(register_polars_random,),
        history_rules=(),
        random_seed=0,
    )


@pytest.fixture(scope="module")
def listings():
    return make_listings(side=6)


def test_one_rule_owns_each_predicate(listings):
    """The composition rests on it: two rules writing def:ask would clobber, not compose."""
    ir = analyse(
        (rule_name(rule, i), render(rule, HOUSING_PARAMS))
        for i, rule in enumerate(HOUSING_UPDATE_RULES)
    )
    written = [p for rule in ir.rules for _, p in rule.writes]
    assert sorted(written) == sorted(set(written))
    assert set(written) == {"reference", "weeks_on_market", "markup", "ask", "sold"}


def test_combinator_is_the_only_writer_of_the_price():
    """Both motives determine the price; only ASK_PRICE states it."""
    for rule in (COMPARABLE_PRICE, MARKUP_HEURISTIC):
        text = render(rule, HOUSING_PARAMS)
        assert "def:ask ?a1" not in text
    assert "INSERT { ?l def:ask ?a1 }" in render(ASK_PRICE, HOUSING_PARAMS)


def test_heuristic_cuts_the_markup_only_after_patience(listings):
    """Rule 3 reads the clock rule 2 advanced in the same tick — staged activation."""
    sim = clone(simulator()).set_params(n_periods=8)
    panel = pl.concat(
        sim.extract().with_columns(t=pl.lit(row["t"]))
        for row in sim.fit_iter({"Listing": listings})
    )
    unsold = panel.filter(pl.col("sold") < 1.0)
    early = unsold.filter(pl.col("weeks_on_market") <= HOUSING_PARAMS["patience"])
    late = unsold.filter(pl.col("weeks_on_market") > HOUSING_PARAMS["patience"] + 1)
    assert (early["markup"] == 0.10).all()
    assert (late["markup"] < 0.10).all()


def test_ablating_the_heuristic_freezes_the_markup(listings):
    """Drop one rule and its predicate stops moving; nothing else needs editing."""
    sim = simulator(update_rules=NO_HEURISTIC_RULES).fit({"Listing": listings})
    assert (sim.extract()["markup"] == 0.10).all()


def test_ablating_the_comparables_leaves_the_reference_at_the_opening_ask(listings):
    """Without rule 1 no listing ever sees its neighbourhood: def:reference is never written."""
    sim = simulator(update_rules=NO_COMPARABLES_RULES).fit({"Listing": listings})
    assert sim.extract()["reference"].null_count() == sim.extract().height


def test_anchor_weight_moves_prices_toward_the_neighbourhood(listings):
    """The mixing weight is a parameter, so 'how much each rule explains' is calibratable."""
    spread = {}
    for anchor in (0.0, 1.0):
        sim = simulator(anchor=anchor, n_periods=6).fit({"Listing": listings})
        spread[anchor] = sim.extract()["ask"].std()
    assert spread[1.0] < spread[0.0]


def test_both_channels_are_live(listings):
    """The full model does what neither ablation does: prices converge *and* fall."""
    sim = simulator(n_periods=10).fit({"Listing": listings})
    state = sim.extract()
    assert state["sold"].sum() > 0
    assert state.filter(pl.col("sold") < 1.0)["markup"].min() < 0.10


def test_neighbourhood_mean_fragment_is_reusable():
    """The within-rule seam: same fragment, different predicate and different link."""
    fragment = neighbourhood_mean("weeks_on_market", out="stale", link="comparable")
    assert "?stale" in fragment and "def:weeks_on_market ?__v" in fragment
    assert neighbourhood_mean("ask", out="r", link="street").count("def:street") == 1
