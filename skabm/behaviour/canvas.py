"""CANVAS, Hommes, He, Poledna, Siqueira & Zhang (2025): firms setting prices and quantities
in a production network, and the central bank's policy rate. Ten rules, run each step,
in this order.

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

10. The central bank's policy_rate becomes the parameter boc_rho times its policy_rate
plus (1 - boc_rho) times the sum of four terms: the parameter boc_r_star; the parameter
pi_star; the parameter boc_xi_pi times the gap between the expected growth of the mean
over firms of their price and pi_star; and the parameter boc_xi_gamma times the
expected growth of the total over firms of their output.
"""

from __future__ import annotations

import polars as pl
import sympy as sp

from skabm.behaviour.learning import expect
from skabm.dsl import Agents, sum_over, total
from skabm.sparql import register_math

# the rules call math:exp: RDFSimulator(udfs=CANVAS_UDFS)
CANVAS_UDFS = (register_math,)

Firm, Sector, Input = Agents("Firm"), Agents("Sector"), Agents("Input")
CentralBank = Agents("CentralBank")
tau_sif, inventory_depreciation, pi_star = sp.symbols(
    "tau_sif inventory_depreciation pi_star"
)
boc_rho, boc_r_star, boc_xi_pi, boc_xi_gamma = sp.symbols(
    "boc_rho boc_r_star boc_xi_pi boc_xi_gamma"
)

# The central bank re-estimates boc_rho, boc_r_star, boc_xi_pi and boc_xi_gamma every
# quarter on the model's own history (eq. 7), and the paper reports none: no default.
PARAMETERS = {
    tau_sif: 0.0,  # τ^SIF, Hommes et al. (2025) Table 4
    inventory_depreciation: 1.0,  # δ^S_s, Hommes et al. (2025) Table 3, every sector
    pi_star: 0.005,  # π*, Hommes et al. (2025) Table 4
}

# Hommes et al. (2025), J. Econ. Dyn. Control 172, 104986, Appendix A.  Eq. 20's AR(1)
# forecasts are fitted to the sample mean and first-order autocorrelation (Hommes & Zhu
# 2014), which is SAC learning (learning.sac): the growth of total output and of the
# mean price.
growth, inflation = expect("SUM", "Firm", "output"), expect("AVG", "Firm", "price")

# eqs. 39, 41: the quantity moves on excess demand at a price at or above the sector's,
# or on excess supply at a price below it (Fig. 10's (c) and (b)); otherwise the price
ratio = Firm.demand / Firm.supply - 1
short, dear = Firm.supply <= Firm.demand, Firm.price >= Firm.sector.price_index
quantity_moves = (short & dear) | (~short & ~dear)
firm_heuristic = {
    Firm.gamma_d: sp.Piecewise((ratio, quantity_moves), (0, True)),
    Firm.pi_d: sp.Piecewise((0, quantity_moves), (ratio, True)),
}
# eq. 43, on last step's prices
firm_cost_push = {
    Firm.pi_c: (1 + tau_sif) * Firm.sector.labour_cost * Firm.w_bar / Firm.alpha
    + Firm.tech_share * Firm.sector.material_cost
    + Firm.delta * Firm.sector.capital_cost
}
# eqs. 40, 42, and eq. 21 with every input on hand (labour hired as planned, capital
# and materials never short), so output is the planned supply; supply is Q^o, eq. 36
planned = Firm.supply * (1 + growth) * (1 + Firm.gamma_d)
firm_price_quantity = {
    Firm.output: planned,
    Firm.supply: planned + Firm.inventory,
    Firm.price: Firm.price * (1 + Firm.pi_d) * (1 + Firm.pi_c) * (1 + inflation),
}
# Appendix A.2.3's selection probabilities, and eq. 31's intermediate inputs
sector_totals = {
    Sector.price_weight: sum_over(Firm.sector, sp.exp(-2 * Firm.price)),
    Sector.output: sum_over(Firm.sector, Firm.output),
    Sector.purchases: sum_over(Firm.sector, Firm.tech_share * Firm.output),
}
# eq. 32 at last step's price: final demand, plus the buyers' intermediate demand
used = sum_over(Input.supplier, Input.share * Input.buyer.purchases)
sector_demand = {Sector.demand: Sector.final_demand + Sector.price_index * used}
# Appendix A.2.3's search and matching, in expectation: a buyer spends its budget at the
# firm it picks first, with probability pr^cum, the mean of pr^price and pr^size
pr_price = sp.exp(-2 * Firm.price) / Firm.sector.price_weight
pr_size = Firm.output / Firm.sector.output
firm_demand = {Firm.demand: Firm.sector.demand / Firm.price * (pr_price + pr_size) / 2}
# eqs. 36-38
sold = sp.Min(Firm.supply, Firm.demand)
firm_inventory = {
    Firm.sales: sold,
    Firm.inventory: Firm.inventory
    + (1 - inventory_depreciation) * (Firm.output - sold),
}
# eq. 49, without imports: no foreign firm sells here
sector_price_index = {
    Sector.price_index: sum_over(Firm.sector, Firm.price * Firm.sales)
    / sum_over(Firm.sector, Firm.sales)
}
# eq. 43's price ratios, from the CPI (eq. 48) and the capital price index (eq. 54)
own = Sector.price_index
bought = sum_over(Input.buyer, Input.share * Input.supplier.price_index)
sector_costs = {
    Sector.labour_cost: total(Sector.b_hh * Sector.price_index) / own - 1,
    Sector.material_cost: bought / own - 1,
    Sector.capital_cost: total(Sector.b_cf * Sector.price_index) / own - 1,
}
# eq. 7, on the forecasts, with no lower bound (footnote 30)
gap = boc_xi_pi * (inflation - pi_star) + boc_xi_gamma * growth
augmented_taylor = {
    CentralBank.policy_rate: boc_rho * CentralBank.policy_rate
    + (1 - boc_rho) * (boc_r_star + pi_star + gap)
}

RULES = [
    firm_heuristic,
    firm_cost_push,
    firm_price_quantity,
    sector_totals,
    sector_demand,
    firm_demand,
    firm_inventory,
    sector_price_index,
    sector_costs,
    augmented_taylor,
]


def initial(
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
