"""Firms, by source: Poledna et al. (2023) and CANVAS, Hommes et al. (2025).

Poledna et al. (2023), ``poledna_rules``: six rules, run each step, in this order.

1. A firm's output becomes the smaller of alpha times its size and its output multiplied
by (1 + the expected growth of the total over firms of their output + a normal shock
with mean 0 and standard deviation the parameter growth_sigma).

2. A firm's price is multiplied by (1 + the expected growth of the mean over firms of
their price + a normal shock with mean 0 and standard deviation the parameter
inflation_sigma).

3. A firm's profit becomes its margin times the smaller of two amounts: its price times
its output; and total demand times its price times its output divided by the total over
firms of their price times their output. Total demand is the sum of three totals: the
total over households of their psi times their income, divided by (1 + the parameter
vat_rate); the total over governments of their budget; and the total over foreign firms
of their demand_size.

4. Then a firm's liquidity becomes its liquidity plus its profit.

5. A firm's size becomes its size plus the larger of 0 and 0.1 times (its output divided
by alpha, minus its size): it hires towards the workforce its output needs and never
fires.

6. A firm's dividend and profit update together, in one rule. Its dividend becomes the
parameter dividend_ratio times the larger of 0 and its profit. Its profit becomes its
profit minus dividend_ratio times the larger of 0 and its profit.

CANVAS, Hommes, He, Poledna, Siqueira & Zhang (2025), ``canvas_rules``: firms setting
prices and quantities in a production network. Nine rules, run each step, in this order.

1. A firm's gamma_d and pi_d update together, in one rule, from last step's market. Its
gamma_d becomes its demand divided by its supply, minus 1, in two cases: when its supply
is at most its demand and its price is at least its sector's price_index; and when its
supply is above its demand and its price is below its sector's price_index. Otherwise its
gamma_d becomes 0. Its pi_d becomes 0 in those two cases, and otherwise its demand
divided by its supply, minus 1.

2. A firm's pi_c becomes the sum of three unit costs: (1 + the parameter tau_sif) times
its sector's labour_cost times its w_bar, divided by its alpha; its tech_share times its
sector's material_cost; and its delta times its sector's capital_cost.

3. A firm's output, supply and price update together, in one rule. Its output becomes its
supply times (1 + the expected growth of the total over firms of their output) times (1 +
its gamma_d), and its supply that output plus its inventory. Its price becomes its price
times (1 + its pi_d) times (1 + its pi_c) times (1 + the expected growth of the mean over
firms of their price).

4. A sector's price_weight, output and purchases update together, in one rule. Its
price_weight becomes the sum, over the firms whose sector is this one, of the exponential
of minus 2 times the firm's price. Its output becomes the sum of output over those firms,
and its purchases the sum over those firms of their tech_share times their output.

5. A sector's demand becomes its final_demand plus its price_index times the sum, over the
inputs whose supplier is this sector, of the input's share times the purchases of its
buyer.

6. A firm's demand becomes its sector's demand divided by its price, times half the sum
of two chances: the exponential of minus 2 times its price, divided by its sector's
price_weight; and its output divided by its sector's output.

7. A firm's sales and inventory update together, in one rule. Its sales become the
smaller of its supply and its demand, and its inventory becomes its inventory plus (1 -
the parameter inventory_depreciation) times (its output minus those sales).

8. A sector's price_index becomes the sum, over the firms whose sector is this one, of
their price times their sales, divided by the sum over those firms of their sales.

9. A sector's labour_cost, material_cost and capital_cost update together, in one rule.
Its labour_cost becomes (the total over sectors of their b_hh times their price_index)
divided by its price_index, minus 1. Its material_cost becomes (the sum, over the inputs
whose buyer is this sector, of the input's share times the price_index of the input's
supplier) divided by its price_index, minus 1. Its capital_cost becomes (the total over
sectors of their b_cf times their price_index) divided by its price_index, minus 1.
"""

from __future__ import annotations

import polars as pl
import sympy as sp
from sympy.stats import Normal

from skabm.behaviour.learning import expect
from skabm.dsl import Agents, sum_over, total
from skabm.sparql import register_math

# CANVAS's rules call math:exp: RDFSimulator(udfs=CANVAS_UDFS)
CANVAS_UDFS = (register_math,)

Firm, Household = Agents("Firm"), Agents("Household")
Government, ForeignFirm = Agents("Government"), Agents("ForeignFirm")
Sector, Input = Agents("Sector"), Agents("Input")
growth_sigma, inflation_sigma = sp.symbols("growth_sigma inflation_sigma")
vat_rate, dividend_ratio = sp.symbols("vat_rate dividend_ratio")
tau_sif, inventory_depreciation = sp.symbols("tau_sif inventory_depreciation")

PARAMETERS = {
    growth_sigma: 0.0,  # scenario knob: AR(1) innovation std, 0 = deterministic
    inflation_sigma: 0.0,  # scenario knob: AR(1) innovation std, 0 = deterministic
    vat_rate: 0.1529,  # τ^VAT, Poledna et al. (2023) Table 2
    dividend_ratio: 0.7768,  # θ^DIV, Poledna et al. (2023) Table 2
    tau_sif: 0.0,  # τ^SIF, Hommes et al. (2025) Table 4
    inventory_depreciation: 1.0,  # δ^S_s, Hommes et al. (2025) Table 3, every sector
}

# Both sources forecast with AR(1) rules fitted to the sample mean and first-order
# autocorrelation (Hommes & Zhu 2014): SAC learning, learning.sac.  Poledna et al. (2023)
# eqs. 6, 9; Hommes et al. (2025) eq. 20.
expected_growth = expect("SUM", "Firm", "output")
expected_inflation = expect("AVG", "Firm", "price")


# Poledna et al. (2023) eqs. 5, 12
poledna_produce = {
    Firm.output: sp.Min(
        Firm.output * (1 + expected_growth + Normal("eps", 0, growth_sigma)),
        Firm.alpha * Firm.size,
    )
}
# Poledna et al. (2023) eq. 8
poledna_price = {
    Firm.price: Firm.price
    * (1 + expected_inflation + Normal("eps", 0, inflation_sigma))
}
# Poledna et al. (2023) eqs. 1, 2, 27
demand = (
    total(Household.psi * Household.income) / (1 + vat_rate)
    + total(Government.budget)
    + total(ForeignFirm.demand_size)
)
supplied = Firm.price * Firm.output
poledna_sales = {
    Firm.profit: Firm.margin * sp.Min(demand * supplied / total(supplied), supplied)
}
# Poledna et al. (2023) eq. 31
poledna_liquidity = {Firm.liquidity: Firm.liquidity + Firm.profit}
# Poledna et al. (2023) eqs. 9, 11
poledna_labor = {
    Firm.size: Firm.size + sp.Max(0, (Firm.output / Firm.alpha - Firm.size) * 0.1)
}
# Poledna et al. (2023) eq. 14
dividend = dividend_ratio * sp.Max(Firm.profit, 0)
poledna_dividends = {Firm.dividend: dividend, Firm.profit: Firm.profit - dividend}

poledna_rules = [
    poledna_produce,
    poledna_price,
    poledna_sales,
    poledna_liquidity,
    poledna_labor,
    poledna_dividends,
]

# Hommes et al. (2025), J. Econ. Dyn. Control 172, 104986, Appendix A
# eqs. 39, 41: the quantity moves on excess demand at a price at or above the sector's,
# or on excess supply at a price below it (Fig. 10's (c) and (b)); otherwise the price
ratio = Firm.demand / Firm.supply - 1
short, dear = Firm.supply <= Firm.demand, Firm.price >= Firm.sector.price_index
quantity_moves = (short & dear) | (~short & ~dear)
canvas_heuristic = {
    Firm.gamma_d: sp.Piecewise((ratio, quantity_moves), (0, True)),
    Firm.pi_d: sp.Piecewise((0, quantity_moves), (ratio, True)),
}
# eq. 43, on last step's prices
canvas_cost_push = {
    Firm.pi_c: (1 + tau_sif) * Firm.sector.labour_cost * Firm.w_bar / Firm.alpha
    + Firm.tech_share * Firm.sector.material_cost
    + Firm.delta * Firm.sector.capital_cost
}
# eqs. 40, 42, and eq. 21 with every input on hand (labour hired as planned, capital
# and materials never short), so output is the planned supply; supply is Q^o, eq. 36
planned = Firm.supply * (1 + expected_growth) * (1 + Firm.gamma_d)
canvas_price_quantity = {
    Firm.output: planned,
    Firm.supply: planned + Firm.inventory,
    Firm.price: Firm.price
    * (1 + Firm.pi_d)
    * (1 + Firm.pi_c)
    * (1 + expected_inflation),
}
# Appendix A.2.3's selection probabilities, and eq. 31's intermediate inputs
canvas_sector_totals = {
    Sector.price_weight: sum_over(Firm.sector, sp.exp(-2 * Firm.price)),
    Sector.output: sum_over(Firm.sector, Firm.output),
    Sector.purchases: sum_over(Firm.sector, Firm.tech_share * Firm.output),
}
# eq. 32 at last step's price: final demand, plus the buyers' intermediate demand
used = sum_over(Input.supplier, Input.share * Input.buyer.purchases)
canvas_sector_demand = {Sector.demand: Sector.final_demand + Sector.price_index * used}
# Appendix A.2.3's search and matching, in expectation: a buyer spends its budget at the
# firm it picks first, with probability pr^cum, the mean of pr^price and pr^size
pr_price = sp.exp(-2 * Firm.price) / Firm.sector.price_weight
pr_size = Firm.output / Firm.sector.output
canvas_demand = {
    Firm.demand: Firm.sector.demand / Firm.price * (pr_price + pr_size) / 2
}
# eqs. 36-38
sold = sp.Min(Firm.supply, Firm.demand)
canvas_inventory = {
    Firm.sales: sold,
    Firm.inventory: Firm.inventory
    + (1 - inventory_depreciation) * (Firm.output - sold),
}
# eq. 49, without imports: no foreign firm sells here
canvas_price_index = {
    Sector.price_index: sum_over(Firm.sector, Firm.price * Firm.sales)
    / sum_over(Firm.sector, Firm.sales)
}
# eq. 43's price ratios, from the CPI (eq. 48) and the capital price index (eq. 54)
own = Sector.price_index
bought = sum_over(Input.buyer, Input.share * Input.supplier.price_index)
canvas_costs = {
    Sector.labour_cost: total(Sector.b_hh * Sector.price_index) / own - 1,
    Sector.material_cost: bought / own - 1,
    Sector.capital_cost: total(Sector.b_cf * Sector.price_index) / own - 1,
}

canvas_rules = [
    canvas_heuristic,
    canvas_cost_push,
    canvas_price_quantity,
    canvas_sector_totals,
    canvas_sector_demand,
    canvas_demand,
    canvas_inventory,
    canvas_price_index,
    canvas_costs,
]


def ownership(
    households: pl.DataFrame,
    firms: pl.DataFrame,
    ratio: float = 0.03,  # unsourced: investor share of households, Poledna et al. (2023) §3.2
) -> pl.DataFrame:
    """*households* with ``owns`` filled in: the firm in row j owned by the household in
    row floor(j / ratio), unless the data already names the firm's owner or gives that
    household a firm.

    ponytail: one firm per household (``owns`` is a single link); with ratio > 1 the
    later firms of a household stay unowned.
    """
    if "owns" not in households.columns:
        households = households.with_columns(owns=pl.lit(None, pl.String))
    taken = set(households["owns"].drop_nulls())
    rows = households.height
    pairs = {}
    for j, firm in enumerate(firms["id"]):
        i = int(j // ratio)
        if firm not in taken and i < rows:
            pairs.setdefault(i, firm)
    assigned = pl.Series([pairs.get(i) for i in range(rows)], dtype=pl.String)
    return households.with_columns(owns=pl.coalesce("owns", assigned))


def canvas_initial(
    firms: pl.DataFrame, sectors: pl.DataFrame, inputs: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """*firms* and *sectors* in the opening quarter, wherever the data has none.

    A firm's price is one, and it makes and sells alpha times its size, holding no
    inventory (Hommes et al. (2025) §3.1).  A sector's price_index is eq. 49 over its
    firms, and its cost ratios are eq. 43's at those prices.  Its final_demand is what
    its firms make less what the sectors buy from them, the input-output identity, so
    each sector's market clears in the opening quarter.  *inputs* are the ``Input``
    edges, ``buyer``, ``supplier`` and ``share``, by bare sector id.
    """

    def fill(frame: pl.DataFrame, **columns) -> pl.DataFrame:
        return frame.with_columns(
            (pl.coalesce(n, v) if n in frame.columns else v).alias(n)
            for n, v in columns.items()
        )

    made = pl.col("alpha") * pl.col("size")
    firms = fill(firms, price=pl.lit(1.0), output=made, inventory=pl.lit(0.0))
    firms = fill(firms, supply=pl.col("output"), demand=pl.col("output"))
    firms = fill(firms, sales=pl.col("output"))
    value = (pl.col("price") * pl.col("sales")).sum()
    books = firms.group_by("sector").agg(
        index=value / pl.col("sales").sum(),
        value=value,
        purchases=(pl.col("tech_share") * pl.col("output")).sum(),
    )
    sectors = sectors.join(books.rename({"sector": "id"}), on="id", how="left")
    sectors = fill(sectors, price_index=pl.col("index"))
    price = sectors.select("id", "price_index")
    cost = (
        inputs.join(price.rename({"id": "supplier"}), on="supplier")
        .group_by("buyer")
        .agg(bought=(pl.col("share") * pl.col("price_index")).sum())
    )
    use = (
        inputs.join(books.rename({"sector": "buyer"}), on="buyer")
        .group_by("supplier")
        .agg(used=(pl.col("share") * pl.col("purchases")).sum())
    )
    cpi = sectors.select((pl.col("b_hh") * pl.col("price_index")).sum()).item()
    capital = sectors.select((pl.col("b_cf") * pl.col("price_index")).sum()).item()
    sectors = (
        sectors.join(cost.rename({"buyer": "id"}), on="id", how="left")
        .join(use.rename({"supplier": "id"}), on="id", how="left")
        .with_columns(pl.col("bought", "used").fill_null(0.0))
    )
    index = pl.col("price_index")
    sectors = fill(
        sectors,
        labour_cost=cpi / index - 1,
        material_cost=pl.col("bought") / index - 1,
        capital_cost=capital / index - 1,
        final_demand=pl.col("value") - index * pl.col("used"),
    )
    return firms, sectors.drop("index", "value", "purchases", "bought", "used")
