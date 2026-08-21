"""
Poledna-style macro ABM with interbank contagion — a ``skabm`` extension.

Poledna et al. (2023, European Economic Review 151, 104306) calibrates six
sectors (firms, households, government, banks, central bank, rest of world)
but leaves banks passive: no interbank market, no contagion, no deposit flight.
This demo adds that layer and exercises the new ``RDFSimulator(infer=...)`` hook
so the reader can see where ``Model.infer`` (Datalog / recursive CONSTRUCT,
licensed) simplifies contagion modelling.

What changes versus the base Poledna demo (examples/poledna.py):

1. ``bank_depositors`` (init CONSTRUCT) — assigns each household/firm to a
   bank weighted by ``def:deposit_share``.  This is new structure, not in the
   paper.
2. ``bank_capital`` (update rule) — capital ratio responds to deposit outflows
   each tick.  The paper has static Basel III calibration only.
3. ``interbank_contagion`` (Datalog rule, passed to ``infer``) — distress
   propagates through the depositor network to a fixed point, one
   ``infer()`` call per tick.  This replaces what would otherwise be a
   hand-tuned sub-tick propagation loop with guessed depth.
4. The tick loop is unchanged from ``RDFSimulator.fit_iter`` — the simulator
   runs update rules first, then ``infer`` once per tick, then yields state.

Architecture note (see ``skabm.simulation`` for the full contract):

    per tick:
      1. SPARQL update rules for per-agent state replacement
         (production, pricing, sales, income, consumption, bank capital,
          Taylor rule)
      2. Model.infer() for relational propagation — optional, only when
         ``infer=`` is passed to RDFSimulator.  The inference runs to a
         fixed point so a single call propagates contagion across the whole
         graph without hand-tuned sub-tick passes.
      3. yield state for extraction / telemetry

Non-contagion parts (firm_produce, firm_sales, centralbank_rate, etc.) stay
SPARQL DELETE/INSERT upserts — they are replacement dynamics, not accumulation,
and do not benefit from ``infer``.  Only the relational propagation layer uses
Datalog / recursive CONSTRUCT.
"""

from time import time

import polars as pl
import polars_random as pr

from skabm.behaviour.bank import bank_depositors, bank_capital, interbank_contagion
from skabm.behaviour.firm import firm_ownership, firm_produce, firm_price, firm_sales
from skabm.behaviour.household import (
    household_income,
    household_income_init,
    household_wealth_init,
    satisificing_consume,
)
from skabm.behaviour.macro import government_spend, centralbank_rate
from skabm.behaviour.params import poledna_params
from skabm.simulation import RDFSimulator

# Keep populations small so the demo runs in seconds and the contagion
# propagation is visible in the output.
N_FIRMS = 50
N_HOUSEHOLDS = 500
N_BANKS = 5
N_GOVERNMENTS = 3
N_FOREIGN = 10

# ---------------------------------------------------------------------------
# Banks: Basel III calibration with a deliberately weak bank to seed contagion
# ---------------------------------------------------------------------------
banks = pl.DataFrame(
    {
        "id": [f"bank_{i}" for i in range(N_BANKS)],
        "capital_ratio": [
            0.02,  # bank_0 starts distressed — seeds the contagion
            0.10,
            0.09,
            0.08,
            0.07,
        ],
        "leverage": [12.0, 12.0, 12.0, 12.0, 12.0],
        "deposit_share": [0.20, 0.25, 0.20, 0.20, 0.15],
    }
).cast(
    {
        "capital_ratio": pl.Float64,
        "leverage": pl.Float64,
        "deposit_share": pl.Float64,
    }
)

# ---------------------------------------------------------------------------
# Firms: simplified — IO table omitted, homogeneous parameters
# ---------------------------------------------------------------------------
pr.set_random_seed(0)
firms = pl.DataFrame(
    {
        "id": [f"firm_{i}" for i in range(N_FIRMS)],
        "size": pr.normal(100.0, 30.0, size=N_FIRMS).clip(10, None).round(0),
        "alpha": 1.0,
        "w_bar": 1.0,
        "delta": 0.05,
        "tech_share": 0.1,
        "output": 0.0,
        "price": 1.0,
        "margin": 0.2,
        "profit": 0.0,
        "liquidity": 0.0,
    }
).with_columns(
    pl.col("size").cast(pl.Float64),
    (pl.col("size") * pl.col("alpha") * 0.9).alias("output"),
    (0.2 * pl.col("output")).alias("profit"),
).with_columns(
    pl.when(pl.col("profit") > 0)
    .then(pl.col("profit"))
    .otherwise(0.0)
    .alias("pos_profit"),
).with_columns(
    pl.when(pl.col("pos_profit") > 0)
    .then(pl.col("pos_profit").cast(pl.Float64) / pl.col("pos_profit").cast(pl.Float64).sum() * 1000.0)
    .otherwise(0.0)
    .alias("liquidity"),
).drop("pos_profit")

# ---------------------------------------------------------------------------
# Households: simplified — enough to show wealth + deposits
# ---------------------------------------------------------------------------
pr.set_random_seed(1)
households = pl.DataFrame(
    {
        "id": [f"hh_{i}" for i in range(N_HOUSEHOLDS)],
        "psi": 0.9,
        "income": 0.0,
        "wealth": 0.0,
    }
).with_columns(
    pl.col("psi").cast(pl.Float64),
    (pr.uniform(0.5, 2.0, size=N_HOUSEHOLDS) * 10.0).round(2).alias("income"),
)

# ---------------------------------------------------------------------------
# Government and foreign — minimal stand-ins for the goods market
# ---------------------------------------------------------------------------
governments = pl.DataFrame(
    {
        "id": [f"gov_{i}" for i in range(N_GOVERNMENTS)],
        "budget": [50.0, 50.0, 50.0],
        "tax_rate": 0.2134,
    }
).cast({"budget": pl.Float64, "tax_rate": pl.Float64})

foreign_firms = pl.DataFrame(
    {
        "id": [f"foreign_{i}" for i in range(N_FOREIGN)],
        "demand_size": [20.0] * N_FOREIGN,
    }
).cast({"demand_size": pl.Float64})

# ---------------------------------------------------------------------------
# Central bank: singleton with Taylor-rule state
# ---------------------------------------------------------------------------
central_bank = pl.DataFrame(
    {
        "id": ["central_bank"],
        "policy_rate": [0.009],
        "inflation_target": [0.005],
        "prev_output": [float(firms["output"].sum())],
        "prev_price": [1.0],
    }
).cast(
    {
        "policy_rate": pl.Float64,
        "inflation_target": pl.Float64,
        "prev_output": pl.Float64,
        "prev_price": pl.Float64,
    }
)

# ---------------------------------------------------------------------------
# Simulation — the key addition is ``infer=[interbank_contagion]``
# ---------------------------------------------------------------------------
init_rules = (
    firm_ownership,
    household_income_init,
    household_wealth_init,
    bank_depositors,
)

update_rules = (
    firm_produce,
    firm_price,
    household_income,
    satisificing_consume,
    firm_sales,
    government_spend,
    centralbank_rate,
    bank_capital,
)

params = {
    **poledna_params,
    "total_deposits": 2000.0,
    "firm_ownership_ratio": N_FIRMS / N_HOUSEHOLDS,
    "distress_threshold": 0.03,
    "flee_amount_threshold": 5.0,
}

sim = RDFSimulator(
    init_rules=init_rules,
    update_rules=update_rules,
    infer=interbank_contagion,
    params=params,
    n_periods=8,
    random_seed=0,
)

# ---------------------------------------------------------------------------
# Run and report
# ---------------------------------------------------------------------------
print("=" * 78)
print("Poledna + interbank contagion demo")
print(f"  firms={N_FIRMS}  households={N_HOUSEHOLDS}  banks={N_BANKS}")
print(f"  contagion via Model.infer (licensed): {bool(sim.infer)}")
print("=" * 78)

t0 = time()
for t, state in enumerate(sim.fit_iter(
    Firm=firms,
    Household=households,
    Government=governments,
    Bank=banks,
    ForeignFirm=foreign_firms,
    CentralBank=central_bank,
)):
    # Banks: how many are distressed this tick?
    distressed_df = sim.model_.query(
        """
        PREFIX ex: <http://example.net/skabm#>
        SELECT (COUNT(?b) AS ?n) WHERE {
            ?b a ex:Bank ; ex:is_distressed true .
        }
        """
    )
    distressed = int(distressed_df["n"][0])

    # Banks: capital ratios this tick
    bank_cr = sim.model_.query(
        """
        PREFIX ex: <http://example.net/skabm#>
        SELECT ?b ?cr WHERE {
            ?b a ex:Bank ; def:capital_ratio ?cr .
        }
        """
    )

    # Firm output and price level
    gdp = state.select(
        (pl.col("price") * pl.col("output") * (1 - pl.col("tech_share")))
        .sum()
        .alias("gdp")
    )["gdp"][0]
    price_level = state["price"].mean()

    cr_str = "  ".join(
        f"{r['b'].rsplit('_', 1)[1].rstrip('>')}:{r['cr']:.3f}"
        for r in bank_cr.sort("b").iter_rows(named=True)
    )

    print(
        f"t={t:2d}  gdp={gdp:8.1f}  price_level={price_level:.3f}  "
        f"distressed_banks={distressed}  "
        f"{cr_str}"
    )

    if distressed > 1:
        print(f"    *** contagion spread to {distressed} banks at t={t} ***")

t1 = time()
print("=" * 78)
print(f"{sim.n_periods} ticks in {t1 - t0:.2f}s ({(t1 - t0) / sim.n_periods:.3f} s/tick)")
print("=" * 78)

# Final observation
print("\nFinal state:")
distressed_final = sim.model_.query(
    """
    PREFIX ex: <http://example.net/skabm#>
    SELECT ?b WHERE { ?b ex:is_distressed true }
    """
)
print("  Distressed banks:", distressed_final["b"].to_list())

flight = sim.model_.query(
    """
    PREFIX ex: <http://example.net/skabm#>
    SELECT ?agent ?from ?to ?amount WHERE {
        ?agent def:flees_to ?to ;
               def:flees_amount ?amount .
        ?from ex:is_distressed true .
        FILTER EXISTS { ?agent def:holds_at ?from }
    }
    """
)
print("  Flight events:")
for row in flight.iter_rows(named=True):
    print(f"    {row['agent']} -> {row['to'].split('_')[1]} : {row['amount']:.2f}")
