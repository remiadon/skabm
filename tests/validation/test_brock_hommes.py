"""Brock & Hommes (1998)'s adaptive belief system on a lattice.

Engine tests only: its parameter values come from an open-source implementation, not
from the paper or any published calibration, so there is no model test until one exists.  They pin the lattice, the days of data, and
that the price follows the closes when the traders have no impact.
"""

import numpy as np
import polars as pl
from maplib import Model

from skabm import template
from skabm.behaviour import trader
from skabm.simulation import RDFSimulator
from skabm.space import moore_neighbors
from skabm.sparql import register_math, register_polars_random

# one trading day, as the implementation's simulation.py runs it
RULES = [
    trader.market_day,
    trader.strategy_demand,
    trader.impact_price,
    trader.logit_imitation,
]


def test_the_lattice_is_a_torus_of_eight_neighbours():
    traders = trader.lattice(4, seed=0)
    around = moore_neighbors(traders, torus=True)
    assert traders.height == 16 and set(traders["strategy"]) <= {0.0, 1.0, 2.0}
    counts = around.group_by("id").agg(n=pl.col("neighbor").n_unique())
    assert counts["n"].to_list() == [8] * 16
    assert around.filter(pl.col("id") == pl.col("neighbor")).height == 0
    corner = around.filter(pl.col("id") == "trader_0_0")["neighbor"]
    assert "trader_3_3" in corner.to_list()  # wrapped round both edges


def test_days_carry_the_trend_and_the_log_returns():
    close = pl.Series(100 * np.exp(0.01 * np.arange(6)))  # exactly log-linear
    days = trader.trading_days(close)
    assert np.allclose(days["fundamental"], close)
    assert np.allclose(days["ret"], [0.0] + [0.01] * 5)
    assert trader.opening_market(close)["price"][0] == close[0]


def test_without_price_impact_the_price_is_the_closes():
    close = pl.Series(
        100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 12)))
    )
    traders = trader.lattice(4, seed=0)
    world = Model()
    world.map(template.market, trader.opening_market(close).with_iri())
    world.map(template.day, trader.trading_days(close).with_iri("market"))
    world.map(template.trader, traders.with_iri("market"))
    world.map(
        template.links("neighbor"),
        moore_neighbors(traders, torus=True).with_iri("neighbor"),
    )
    sim = RDFSimulator(
        rules=RULES,
        params={"price_impact": 0.0},
        udfs=(register_polars_random, register_math),
        n_periods=10,
        random_seed=0,
    )
    run = pl.concat(sim.fit_iter(world), how="diagonal")
    market = run.filter(pl.col("class") == "Market").sort("t")
    assert np.allclose(market["price"], close[1:11], rtol=1e-12)
    assert market["momentum"][0] == 0.0  # chartists wait for a second day
    strategies = run.filter(pl.col("class") == "Trader")["strategy"]
    assert set(strategies) <= {0.0, 1.0, 2.0}
