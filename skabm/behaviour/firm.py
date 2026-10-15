"""Firms, the sectors whose goods they sell, and industries run as one representative firm.

- Output and prices that follow learned growth and inflation, capped by capacity; profit
  from a share of demand; retained profit, hiring towards the workforce output needs,
  dividends: Poledna et al. (2023), European Economic Review 151, 104306.
- Price and quantity heuristics on last step's market, cost push through a production
  network, demand at each buyer's first pick, inventories, sector price indices: CANVAS,
  Hommes et al. (2025), J. Econ. Dyn. Control 172, 104986, Appendix A.
- Industries that adjust their in-person and from-home workforce towards demand and a
  lockdown's limit, order inputs from each other and ration output pro rata, with
  consumers' fear of infection and of unemployment: Pangallo et al. (2024), Nat. Hum.
  Behav. 8, 264-275, SI §2.

Pangallo et al. (2024)'s hire_speed, fire_speed, fear_infection, fear_ratio,
fear_unemployment and reallocation_share are published only as posteriors (SI Fig. 13),
so they stay required.
"""

from __future__ import annotations

import polars as pl
import sympy as sp
from sympy.stats import Normal

from skabm.behaviour.learning import expect
from skabm.dsl import Agents, sum_over, total

Firm, Household = Agents("Firm"), Agents("Household")
Government, ForeignFirm = Agents("Government"), Agents("ForeignFirm")
Sector, Input = Agents("Sector"), Agents("Input")
Industry, Order, Region = Agents("Industry"), Agents("Order"), Agents("Region")
Person = Agents("Person")

growth_sigma, inflation_sigma = sp.symbols("growth_sigma inflation_sigma")
vat_rate, dividend_ratio = sp.symbols("vat_rate dividend_ratio")
tau_sif, inventory_depreciation = sp.symbols("tau_sif inventory_depreciation")
hire_speed, fire_speed = sp.symbols("hire_speed fire_speed")
fear_infection, fear_ratio = sp.symbols("fear_infection fear_ratio")
fear_unemployment, reallocation_share = sp.symbols(
    "fear_unemployment reallocation_share"
)
government_shock, other_shock = sp.symbols("government_shock other_shock")

PARAMETERS = {
    growth_sigma: 0.0,  # scenario knob: AR(1) innovation std, 0 = deterministic
    inflation_sigma: 0.0,  # scenario knob: AR(1) innovation std, 0 = deterministic
    vat_rate: 0.1529,  # τ^VAT, Poledna et al. (2023) Table 2
    dividend_ratio: 0.7768,  # θ^DIV, Poledna et al. (2023) Table 2
    tau_sif: 0.0,  # τ^SIF, Hommes et al. (2025) Table 4
    inventory_depreciation: 1.0,  # δ^S_s, Hommes et al. (2025) Table 3, every sector
    government_shock: -0.05,  # G_t, Pangallo et al. (2024) SI §3.2.4
    other_shock: 0.30,  # S_t, Pangallo et al. (2024) SI §3.2.4
}

# SAC learning (Hommes & Zhu 2014): Poledna et al. (2023) eqs. 6, 9; Hommes et al.
# (2025) eq. 20
expected_growth = expect("SUM", "Firm", "output")
expected_inflation = expect("AVG", "Firm", "price")

# Poledna et al. (2023) eqs. 5, 12
adaptive_output = {
    Firm.output: sp.Min(
        Firm.output * (1 + expected_growth + Normal("eps", 0, growth_sigma)),
        Firm.alpha * Firm.size,
    )
}
# Poledna et al. (2023) eq. 8
adaptive_price = {
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
market_share_profit = {
    Firm.profit: Firm.margin * sp.Min(demand * supplied / total(supplied), supplied)
}
# Poledna et al. (2023) eq. 31
retained_profit = {Firm.liquidity: Firm.liquidity + Firm.profit}
# Poledna et al. (2023) eqs. 9, 11
hire_to_capacity = {
    Firm.size: Firm.size + sp.Max(0, (Firm.output / Firm.alpha - Firm.size) * 0.1)
}
# Poledna et al. (2023) eq. 14
dividend = dividend_ratio * sp.Max(Firm.profit, 0)
dividend_payout = {Firm.dividend: dividend, Firm.profit: Firm.profit - dividend}

# Hommes et al. (2025) eqs. 39, 41: the quantity moves on excess demand at a price at
# or above the sector's, or on excess supply at a price below it (Fig. 10's (c) and
# (b)); otherwise the price
ratio = Firm.demand / Firm.supply - 1
short, dear = Firm.supply <= Firm.demand, Firm.price >= Firm.sector.price_index
quantity_moves = (short & dear) | (~short & ~dear)
price_quantity_signal = {
    Firm.gamma_d: sp.Piecewise((ratio, quantity_moves), (0, True)),
    Firm.pi_d: sp.Piecewise((0, quantity_moves), (ratio, True)),
}
# Hommes et al. (2025) eq. 43, on last step's prices
cost_push = {
    Firm.pi_c: (1 + tau_sif) * Firm.sector.labour_cost * Firm.w_bar / Firm.alpha
    + Firm.tech_share * Firm.sector.material_cost
    + Firm.delta * Firm.sector.capital_cost
}
# Hommes et al. (2025) eqs. 40, 42, and eq. 21 with every input on hand (labour hired
# as planned, capital and materials never short), so output is the planned supply;
# supply is Q^o, eq. 36
planned = Firm.supply * (1 + expected_growth) * (1 + Firm.gamma_d)
heuristic_supply = {
    Firm.output: planned,
    Firm.supply: planned + Firm.inventory,
    Firm.price: Firm.price
    * (1 + Firm.pi_d)
    * (1 + Firm.pi_c)
    * (1 + expected_inflation),
}
# Hommes et al. (2025) Appendix A.2.3's selection probabilities, and eq. 31's
# intermediate inputs
sector_totals = {
    Sector.price_weight: sum_over(Firm.sector, sp.exp(-2 * Firm.price)),
    Sector.output: sum_over(Firm.sector, Firm.output),
    Sector.purchases: sum_over(Firm.sector, Firm.tech_share * Firm.output),
}
# Hommes et al. (2025) eq. 32 at last step's price: final demand, plus the buyers'
# intermediate demand
used = sum_over(Input.supplier, Input.share * Input.buyer.purchases)
sector_demand = {Sector.demand: Sector.final_demand + Sector.price_index * used}
# Hommes et al. (2025) Appendix A.2.3's search and matching, in expectation: a buyer
# spends its budget at the firm it picks first, with probability pr^cum, the mean of
# pr^price and pr^size
pr_price = sp.exp(-2 * Firm.price) / Firm.sector.price_weight
pr_size = Firm.output / Firm.sector.output
first_pick_demand = {
    Firm.demand: Firm.sector.demand / Firm.price * (pr_price + pr_size) / 2
}
# Hommes et al. (2025) eqs. 36-38
sold = sp.Min(Firm.supply, Firm.demand)
inventory_sales = {
    Firm.sales: sold,
    Firm.inventory: Firm.inventory
    + (1 - inventory_depreciation) * (Firm.output - sold),
}
# Hommes et al. (2025) eq. 49, without imports: no foreign firm sells here
sector_price_index = {
    Sector.price_index: sum_over(Firm.sector, Firm.price * Firm.sales)
    / sum_over(Firm.sector, Firm.sales)
}
# Hommes et al. (2025) eq. 43's price ratios, from the CPI (eq. 48) and the capital
# price index (eq. 54)
own = Sector.price_index
bought = sum_over(Input.buyer, Input.share * Input.supplier.price_index)
sector_costs = {
    Sector.labour_cost: total(Sector.b_hh * Sector.price_index) / own - 1,
    Sector.material_cost: bought / own - 1,
    Sector.capital_cost: total(Sector.b_cf * Sector.price_index) / own - 1,
}


def adjusted(employed, target):
    """Pangallo et al. (2024) SI eqs. 23-24: a share hire_speed of the gap to the target
    when hiring, else fire_speed."""
    speed = sp.Piecewise((hire_speed, target >= employed), (fire_speed, True))
    return employed + speed * (target - employed)


# Pangallo et al. (2024) SI eqs. 19-22, with SI §3.2.1's supply shock s = 1 - ESS while
# the lockdown holds (macro.region_day); the people's own count, for job draws after
lockdown, mandates = total(Region.lockdown), total(Region.mandates)
excess = (Industry.demand - Industry.capacity) / Industry.output0
inperson, fromhome = Industry.inperson, Industry.fromhome
industry_labour = {
    Industry.inperson: adjusted(
        inperson,
        sp.Min(
            (1 - lockdown * (1 - Industry.essential)) * Industry.inperson0,
            inperson + Industry.inperson0 * excess,
        ),
    ),
    Industry.fromhome: adjusted(
        fromhome, sp.Min(Industry.fromhome0, fromhome + Industry.fromhome0 * excess)
    ),
    Industry.staff_inperson: sum_over(
        Person.industry, (1 - Person.wfh) * Person.employed
    ),
    Industry.staff_fromhome: sum_over(Person.industry, Person.wfh * Person.employed),
}
# Pangallo et al. (2024) Methods eqs. 2-4 (phi_ECO = phi_EPI x phi-tilde) and SI eqs.
# 26-28 for one cohort: customer-facing demand falls with yesterday's deaths, and a
# share reallocation_share of what fear saves goes to the other goods; SI eq. 30's
# income effect, on the region's own employment
calm = sp.exp(-fear_infection * fear_ratio * total(Region.deaths))
facing = Industry.customer_facing
reallocated = (
    reallocation_share
    * (1 - calm)
    * total(facing * Industry.consumption0)
    / total((1 - facing) * Industry.consumption0)
)
working = total(Industry.inperson + Industry.fromhome) / total(
    Industry.inperson0 + Industry.fromhome0
)
households = (
    Industry.consumption0
    * (facing * calm + (1 - facing) * (1 + reallocated))
    * (1 - fear_unemployment * (1 - working))
)
# SI eqs. 32-35: orders O = A x_{t-1}, total demand, capacity from the workforce, output
# the smaller, with SI §3.2.4's shocks to government and other final demand
ordered = (
    sum_over(Order.supplier, Order.coefficient * Order.buyer.output)
    + households
    + Industry.government0 * (1 - mandates * government_shock)
    + Industry.other0 * (1 - lockdown * other_shock)
)
capacity = (
    Industry.output0 * (inperson + fromhome) / (Industry.inperson0 + Industry.fromhome0)
)
industry_production = {
    Industry.consumption: households,
    Industry.demand: ordered,
    Industry.capacity: capacity,
    Industry.output: sp.Min(ordered, capacity),
}


def opening_quarter(
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


def steady_state(
    flows: pl.DataFrame, industries: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The ``Industry`` and ``Order`` frames in the steady state the model starts from
    (Pangallo et al. (2024) SI §3.4.3).

    *flows* are the intermediate flows between the region's industries before the
    pandemic, ``supplier``, ``buyer`` and ``amount`` (Z); *industries* hold the
    template's required columns but output0.  Output is all that is demanded, x_0 =
    sum_l Z_k,l + c + G + f, so demand, capacity and output start equal, and an order's
    coefficient is A_k,l = Z_k,l / x_l (SI Table 1).
    """
    sold = (
        flows.group_by("supplier")
        .agg(sold=pl.col("amount").sum())
        .rename({"supplier": "id"})
    )
    x0 = (
        pl.col("sold")
        + pl.col("consumption0")
        + pl.col("government0")
        + pl.col("other0")
    )
    industries = (
        industries.join(sold, on="id", how="left", maintain_order="left")
        .with_columns(pl.col("sold").fill_null(0.0))
        .with_columns(output0=x0)
        .with_columns(
            output="output0",
            demand="output0",
            capacity="output0",
            inperson="inperson0",
            fromhome="fromhome0",
            consumption="consumption0",
        )
        .drop("sold")
    )
    orders = flows.join(
        industries.select(buyer="id", output0="output0"), on="buyer"
    ).select(
        id=pl.format("{}-{}", "supplier", "buyer"),
        supplier="supplier",
        buyer="buyer",
        coefficient=pl.col("amount") / pl.col("output0"),
    )
    return industries, orders
