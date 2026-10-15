"""Traders on a lattice, and the market they trade in: each holds one of three
strategies (fundamentalist, chartist, noise), demands accordingly, and switches by a
logit in the strategies' past profits, pulled towards what its neighbours hold.

Logit switching on fitness with memory: Brock & Hommes (1998), J. Econ. Dyn. Control 22.
The demands, the price on a real asset's returns plus the traders' impact, and the
neighbour pull: the open-source implementation github.com/zakhac/Spatial-Evolutionary-Markets.
"""

import numpy as np
import polars as pl
import sympy as sp
from sympy.stats import Normal, Uniform

from skabm.dsl import Agents, sum_over

Trader, Market, Day = Agents("Trader"), Agents("Market"), Agents("Day")
FUND, CHART, NOISE = 0, 1, 2

choice_intensity, social_pressure = sp.symbols("choice_intensity social_pressure")
noise_sd, price_impact, memory = sp.symbols("noise_sd price_impact memory")
chart_gain, max_return = sp.symbols("chart_gain max_return")
PARAMETERS = {
    choice_intensity: 0.3,  # β, intensity of choice; the implementation's main.py, BETA
    social_pressure: 3.0,  # γ, neighbour pull; the implementation's main.py, GAMMA
    noise_sd: 0.05,  # σ; the implementation's main.py, SIGMA
    price_impact: 0.0005,  # λ; the implementation's main.py, LAMBDA
    memory: 0.9,  # memory in past profits; the implementation's simulation.py, step
    chart_gain: 10.0,  # chartist demand per unit of return; the implementation's simulation.py
    max_return: 0.2,  # bound on a day's return; the implementation's market.py, execute_orders
}

# What each strategy asks for, as a share of the price
fundamentalist = (Market.fundamental - Market.price) / Market.price
chartist = chart_gain * Market.momentum


def holding(strategy) -> sp.Piecewise:
    return sp.Piecewise((1, sp.Eq(Trader.strategy, strategy)), (0, True))


# The implementation's market.py, Market.update_fundamental
today = sp.Eq(Day.t, Day.market.t + 1)
market_day = {
    Market.t: Market.t + 1,
    Market.fundamental: sum_over(
        Day.market, sp.Piecewise((Day.fundamental, today), (0, True))
    ),
    Market.ret: sum_over(Day.market, sp.Piecewise((Day.ret, today), (0, True))),
}

# The implementation's simulation.py, SpatialSimulation.get_agent_demands
strategy_demand = {
    Trader.demand: sp.Piecewise(
        (fundamentalist, sp.Eq(Trader.strategy, FUND)),
        (chartist, sp.Eq(Trader.strategy, CHART)),
        (Normal("noise", 0, noise_sd), True),
    )
}

# The implementation's market.py, Market.execute_orders; Brock & Hommes (1998), fitness with memory
excess = sum_over(Trader.market, Trader.demand) / sum_over(Trader.market, 1)
day_return = sp.Max(-max_return, sp.Min(max_return, price_impact * excess + Market.ret))
change = Market.price * (sp.exp(day_return) - 1)
noise_demand = sum_over(Trader.market, Trader.demand * holding(NOISE)) / sp.Max(
    sum_over(Trader.market, holding(NOISE)), 1
)
impact_price = {
    Market.price: Market.price * sp.exp(day_return),
    Market.momentum: sp.Piecewise((change / Market.price, Market.t >= 2), (0, True)),
    Market.fitness_fund: memory * Market.fitness_fund + fundamentalist * change,
    Market.fitness_chart: memory * Market.fitness_chart + chartist * change,
    Market.fitness_noise: memory * Market.fitness_noise + noise_demand * change,
}


# Brock & Hommes (1998), logit switching; the neighbour pull, the implementation's
# simulation.py, SpatialSimulation._evolve_grid_spatial
best = sp.Max(Market.fitness_fund, Market.fitness_chart, Market.fitness_noise)
pull = {
    s: social_pressure * sum_over(Trader.neighbor, holding(s)) / 8
    for s in (FUND, CHART)
}
weight = {
    FUND: sp.exp(choice_intensity * (Market.fitness_fund - best) + pull[FUND]),
    CHART: sp.exp(choice_intensity * (Market.fitness_chart - best) + pull[CHART]),
    NOISE: sp.exp(choice_intensity * (Market.fitness_noise - best)),
}
u = Uniform("u", 0, 1)
logit_imitation = {
    Trader.strategy: sp.Piecewise(
        (FUND, u < weight[FUND] / sum(weight.values())),
        (CHART, u < (weight[FUND] + weight[CHART]) / sum(weight.values())),
        (NOISE, True),
    )
}


def lattice(
    size: int = 200,  # GRID_SIZE, the implementation's main.py
    shares: tuple = (
        0.33,
        0.33,
        0.34,
    ),  # opening strategy shares, the implementation's simulation.py
    seed: int = 0,
) -> pl.DataFrame:
    """The Trader population: one per cell of a *size* x *size* lattice, its strategy drawn
    with *shares*, every one trading in the market ``market``."""
    rng = np.random.default_rng(seed)
    x, y = np.divmod(np.arange(size * size), size)
    return pl.DataFrame(
        {
            "id": [f"trader_{i}_{j}" for i, j in zip(x, y)],
            "x": x.astype(float),
            "y": y.astype(float),
            "strategy": rng.choice(3, size=size * size, p=shares).astype(float),
            "market": "market",
        }
    )


def trading_days(close: pl.Series) -> pl.DataFrame:
    """The Day population from a series of closing prices: each day's fundamental is the
    log-linear trend fitted to the whole series, and its ret the close's log return (0 on
    the first day), as the implementation's market.py has them."""
    log = np.log(close.to_numpy())
    t = np.arange(len(log))
    slope, intercept = np.polyfit(t, log, 1)
    return pl.DataFrame(
        {
            "id": [f"day_{i}" for i in t],
            "t": t.astype(float),
            "fundamental": np.exp(intercept + slope * t),
            "ret": np.diff(log, prepend=log[0]),
            "market": "market",
        }
    )


def opening_market(close: pl.Series) -> pl.DataFrame:
    """The Market on its first day: price and fundamental at the first close, as
    the implementation's market.py opens it, no momentum and no fitness yet."""
    return pl.DataFrame({"id": ["market"], "price": [float(close[0])]}).with_columns(
        fundamental="price",
        t=pl.lit(0.0),
        ret=pl.lit(0.0),
        momentum=pl.lit(0.0),
        fitness_fund=pl.lit(0.0),
        fitness_chart=pl.lit(0.0),
        fitness_noise=pl.lit(0.0),
    )
