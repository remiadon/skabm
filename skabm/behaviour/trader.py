"""Traders on a lattice, and the market they trade in. Brock & Hommes (1998),
``brock_hommes_rules``: their adaptive belief system on a lattice, as the open-source
implementation by Z. Hachemi writes it (github.com/zakhac/Spatial-Evolutionary-Markets,
``simulation.py`` and ``market.py`` in its S&P 500 mode), five rules, in this order. The
paper's: each trader picks a strategy by a logit, of intensity the parameter
choice_intensity, in the strategies' past profits with the parameter memory. The
implementation's: the three strategies' demands, a price moved by the real asset's return
plus the traders' impact, the noise traders, and the pull of the eight neighbours.

1. A market's day advances: its t becomes t plus 1, and its fundamental and ret become
the fundamental and ret of the day of this market whose t is the market's t plus 1: the
sum, over the market's days, of the day's value for that day and 0 for any other.

2. A trader's demand becomes, for a fundamentalist (strategy 0), its market's fundamental
minus its price, over its price; for a chartist (strategy 1), the parameter chart_gain
times its market's momentum; for a noise trader (strategy 2), a fresh normal draw of mean
0 and standard deviation the parameter noise_sd.

3. A market's price, momentum and three fitnesses update together, in one rule. Its
return is the parameter price_impact times the mean demand of its traders, plus its ret,
bounded by the parameter max_return either way; its price becomes the price times the
exponential of that return, and the change is the new price minus the price. Its
momentum becomes the change over the price from its second day on, and 0 before. Each
fitness becomes the parameter memory times itself, plus a demand times the change: the
fundamentalists' demand, the chartists' demand, and the mean demand of the market's noise
traders (0 when it has none).

4. A trader's fund_around and chart_around become the shares of its eight neighbours who
are fundamentalists and chartists.

5. A trader's strategy becomes a fresh draw among the three, each with a weight: the
exponential of the parameter choice_intensity times the strategy's fitness less the
largest of the three, plus, for the fundamentalist and chartist strategies, the parameter
social_pressure times its fund_around or chart_around. A uniform
draw between 0 and 1 below the fundamentalist's share of the weights picks it, below the
fundamentalist's and the chartist's shares together the chartist, else the noise trader.
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
brock_hommes_day = {
    Market.t: Market.t + 1,
    Market.fundamental: sum_over(
        Day.market, sp.Piecewise((Day.fundamental, today), (0, True))
    ),
    Market.ret: sum_over(Day.market, sp.Piecewise((Day.ret, today), (0, True))),
}

# The implementation's simulation.py, SpatialSimulation.get_agent_demands
brock_hommes_demand = {
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
brock_hommes_price = {
    Market.price: Market.price * sp.exp(day_return),
    Market.momentum: sp.Piecewise((change / Market.price, Market.t >= 2), (0, True)),
    Market.fitness_fund: memory * Market.fitness_fund + fundamentalist * change,
    Market.fitness_chart: memory * Market.fitness_chart + chartist * change,
    Market.fitness_noise: memory * Market.fitness_noise + noise_demand * change,
}


# The implementation's simulation.py, SpatialSimulation._evolve_grid_spatial: the neighbour
# counts, a rule of their own because the draw that reads them is a condition
around = Trader.neighbor
brock_hommes_around = {
    Trader.fund_around: sum_over(around, sp.Piecewise((1, sp.Eq(around.strategy, FUND)), (0, True))) / 8,
    Trader.chart_around: sum_over(around, sp.Piecewise((1, sp.Eq(around.strategy, CHART)), (0, True))) / 8,
}  # fmt: skip

# Brock & Hommes (1998), logit switching; the neighbour pull, the implementation's
# simulation.py, SpatialSimulation._evolve_grid_spatial
best = sp.Max(Market.fitness_fund, Market.fitness_chart, Market.fitness_noise)
weight = {
    FUND: sp.exp(choice_intensity * (Market.fitness_fund - best) + social_pressure * Trader.fund_around),
    CHART: sp.exp(choice_intensity * (Market.fitness_chart - best) + social_pressure * Trader.chart_around),
    NOISE: sp.exp(choice_intensity * (Market.fitness_noise - best)),
}  # fmt: skip
u = Uniform("u", 0, 1)
brock_hommes_imitate = {
    Trader.strategy: sp.Piecewise(
        (FUND, u < weight[FUND] / sum(weight.values())),
        (CHART, u < (weight[FUND] + weight[CHART]) / sum(weight.values())),
        (NOISE, True),
    )
}

brock_hommes_rules = [
    brock_hommes_day,
    brock_hommes_demand,
    brock_hommes_price,
    brock_hommes_around,
    brock_hommes_imitate,
]


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


def torus_neighbors(traders: pl.DataFrame) -> pl.DataFrame:
    """``(id, neighbor)``: the eight traders around each, the lattice wrapped into a torus,
    for ``template.links("neighbor")``."""
    size = int(traders["x"].max()) + 1
    steps = (
        pl.DataFrame({"dx": [-1, 0, 1]})
        .join(pl.DataFrame({"dy": [-1, 0, 1]}), how="cross")
        .filter((pl.col("dx") != 0) | (pl.col("dy") != 0))
    )
    return (
        traders.select("id", "x", "y")
        .join(steps, how="cross")
        .select("id", x=(pl.col("x") + pl.col("dx")) % size, y=(pl.col("y") + pl.col("dy")) % size)
        .join(traders.select(neighbor="id", x="x", y="y"), on=["x", "y"])
        .select("id", "neighbor")
    )  # fmt: skip


def days(close: pl.Series) -> pl.DataFrame:
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


def opening(close: pl.Series) -> pl.DataFrame:
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
