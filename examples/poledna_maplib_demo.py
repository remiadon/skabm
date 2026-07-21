"""Poledna-style macro ABM as a maplib knowledge graph, simulated in SPARQL.

All six institutional sectors of Poledna et al. (2023) are instantiated with
`skabm.calibration.make_dataset` from the same Eurostat-calibrated samplers
as tests/test_poledna_calibration.py (IO table NAIO_10_CP1700 + business
demography BD_9PM_R2, Austria 2010):

  1. non-financial corporations (firms)   IO table + business demography
  2. households                           census active/inactive shares
  3. general government                   J = 25% of domestic firms
  4. financial corporations (banks)       Basel III calibration
  5. central bank                         singleton, Taylor-rule state
  6. rest of world (foreign firms)        L = 50% of domestic firms

Construction: one `map_df` call per population (auto-generated template +
class tag; predicates are column names), then the initialization rules
(`household_income`, `household_wealth`) are inserted in-graph.
Simulation: `RDFSimulator` is iterated; each tick applies the SPARQL
DELETE/INSERT upserts from `skabm.rules` in the paper's event order and
yields raw per-agent state, summarized here with plain polars expressions.
See skabm/rules.py for the deliberate simplifications this pure-SPARQL
architecture imposes (no search-and-matching, no stochastic shocks, no
AR(1) re-estimation).
"""

import polars as pl
import polars_random as pr
from maplib import Model
from time import time

from skabm.calibration import make_dataset, weighted_enum
from skabm.datasets import build_firm_io_df
from skabm.rules import (
    EX_NS,
    map_df,
    firm_pricing,
    firm_production,
    firm_sales,
    government_consumption,
    household_income,
    household_income_update,
    household_update,
    household_wealth,
    taylor_rule,
)
from skabm.simulation import RDFSimulator

# --- Table 2 / Table 7 scalars (reference quarter 2010:Q4) -------------------
TAU_SIF = 0.2122  # employers' social insurance contributions
TAU_VAT = 0.1529  # value-added tax rate
TAU_INC = 0.2134  # income tax rate
PSI = 0.9394  # propensity to consume out of expected income
THETA_DIV = 0.7768  # dividend payout ratio
THETA_UB = 0.3586  # unemployment benefit replacement rate
D_I = 52_141.2e6  # initial firm-sector deposits (EUR)
D_H = 222_933.2e6  # initial household-sector deposits (EUR)
H_ACTIVE = 4_729_215  # economically active persons (census)
H_INACTIVE = 4_130_385  # economically inactive persons (census)

N_FIRMS = 300
N_HOUSEHOLDS = 10_000
N_BANKS = 12
J = N_FIRMS // 4  # government entities: 25% of domestic firms (Section 4.1.1)
L = N_FIRMS // 2  # foreign importers: 50% of domestic firms

# TODO set pl.random_seed
# TODO set pr.random_seed , see https://github.com/diegoglozano/polars-random/issues/27


# --- Firms: calibrated from IO table + business demography (Section 4.1) -----
io_df = build_firm_io_df("AT", 2010).filter(
    pl.col("n_firms").is_not_null() & pl.col("alpha_s").is_not_null()
)
industries = io_df["industry"]
industry_enum = pl.Enum(industries.to_list())

firms = (
    make_dataset(
        samplers={
            # industry proportional to n_firms; size lognormal (Section 4.1.1)
            "industry": weighted_enum(industry_enum, io_df["n_firms"]),
            "size": pr.normal(3.0, 1.0).exp().cast(pl.Int64).clip(1, None),
            # IO-table coefficients, industry-homogeneous (Section 4.1.2)
            "alpha": pl.col("industry").replace_strict(
                industries, io_df["alpha_s"], return_dtype=pl.Float64
            ),
            "w_bar": pl.col("industry").replace_strict(
                industries, io_df["w_bar_s"], return_dtype=pl.Float64
            ),
            "delta": pl.col("industry").replace_strict(
                industries, io_df["delta_s"], return_dtype=pl.Float64
            ),
            "tech_share": pl.col("industry").replace_strict(
                industries, io_df["tech_share_s"], return_dtype=pl.Float64
            ),
        },
        n_agents=N_FIRMS,
        seed=0,
    )
    .with_columns(
        pl.format(EX_NS + "firm_{}", pl.col("id")).alias("id"),
        pl.format(EX_NS + "{}", pl.col("industry").cast(pl.String)).alias("industry"),
        # start below labor capacity alpha*size so firm_production has headroom
        (0.9 * pl.col("alpha") * pl.col("size")).alias("output"),  # ~Y_i(0)
        pl.lit(1.0).alias("price"),  # P_i(0) = 1
        # operating margin: 1 - (1+tau^SIF) w_bar/alpha - delta/kappa - 1/beta
        # (delta and tech_share are already per unit of output)
        (
            1.0
            - (1.0 + TAU_SIF) * pl.col("w_bar") / pl.col("alpha")
            - pl.col("delta")
            - pl.col("tech_share")
        ).alias("margin"),
    )
    .with_columns((pl.col("margin") * pl.col("output")).alias("profit"))
    .with_columns(
        # D_i(0) = D^I max(Pi_i, 0) / sum max(Pi_i, 0), with D^I scaled down by
        # the demo's sampling fraction (the paper is 1:1, N_FIRMS is not)
        (
            (D_I * N_FIRMS / io_df["n_firms"].sum())
            * pl.col("profit").clip(0, None)
            / pl.col("profit").clip(0, None).sum()
        ).alias("liquidity")
    )
)

gva0 = float((firms["output"] * (1 - firms["tech_share"])).sum())

# --- Households: census status shares + links to firms (Section 3.2) ---------
households = (
    make_dataset(
        samplers={
            "status": weighted_enum(
                pl.Enum(["active", "inactive"]), [H_ACTIVE, H_INACTIVE]
            ),
            # search-and-matching init: employer probability grows with firm size
            "employer": weighted_enum(pl.Enum(firms["id"].to_list()), firms["size"]),
        },
        n_agents=N_HOUSEHOLDS,
        seed=1,
    )
    .with_columns(
        pl.format(EX_NS + "hh_{}", pl.col("id")).alias("id"),
        # one investor per firm (Section 3.2): household k owns firm k
        firms["id"].rename("owns").extend_constant(None, N_HOUSEHOLDS - N_FIRMS),
        pl.lit(PSI).alias("psi"),
    )
    .with_columns(
        # investors do not supply labor; inactive households neither
        pl.when(pl.col("owns").is_null() & (pl.col("status") == "active"))
        .then(pl.col("employer").cast(pl.String))
        .otherwise(None)
        .alias("employer"),
    )
    .drop("status")
)

# --- Government entities: J consumers with sector-weighted purchases ---------
# Total government consumption ~20% of initial GVA, split evenly (eq. 52).
governments = make_dataset(
    samplers={
        "purchase_sector": weighted_enum(industry_enum, io_df["output"]),
    },
    n_agents=J,
    seed=2,
).with_columns(
    pl.format(EX_NS + "gov_{}", pl.col("id")).alias("id"),
    pl.format(EX_NS + "{}", pl.col("purchase_sector").cast(pl.String)).alias(
        "purchase_sector"
    ),
    pl.lit(TAU_INC).alias("tax_rate"),
    pl.lit(0.20 * gva0 / J).alias("budget"),
)

# --- Banks: Basel III calibration (Section 4.4) ------------------------------
banks = make_dataset(
    samplers={
        "capital_ratio": pr.normal(0.08, 0.02).clip(0.03, 0.30),
        "leverage": pr.normal(12.0, 2.0).clip(1 / 0.30, None),
        "deposit_share": pr.uniform(0.05, 0.30),
    },
    n_agents=N_BANKS,
    seed=3,
).with_columns(pl.format(EX_NS + "bank_{}", pl.col("id")).alias("id"))

# --- Rest of world: L foreign consumers, exports ~52% of GVA -----------------
foreign_firms = (
    make_dataset(
        samplers={
            "source_industry": weighted_enum(industry_enum, io_df["output"]),
            "demand_weight": pr.normal(2.5, 1.0).exp(),
        },
        n_agents=L,
        seed=4,
    )
    .with_columns(
        pl.format(EX_NS + "row_{}", pl.col("id")).alias("id"),
        pl.format(EX_NS + "{}", pl.col("source_industry").cast(pl.String)).alias(
            "source_industry"
        ),
        (0.52 * gva0 * pl.col("demand_weight") / pl.col("demand_weight").sum()).alias(
            "demand_size"
        ),
    )
    .drop("demand_weight")
)

# --- Central bank: singleton with Taylor-rule state (Section 3.5) ------------
central_bank = pl.DataFrame(
    {
        "id": [EX_NS + "central_bank"],
        "policy_rate": [0.009],
        "inflation_target": [0.005],
        "prev_output": [float(firms["output"].sum())],
        "prev_price": [1.0],
    }
)

# --- Model construction: one map_df per population + init rules --------------
m = Model()
map_df(m, firms, "Firm")
map_df(m, households, "Household")
map_df(m, governments, "Government")
map_df(m, banks, "Bank")
map_df(m, foreign_firms, "ForeignFirm")
map_df(m, central_bank, "CentralBank")

m.insert(household_income(THETA_DIV, THETA_UB), transient=True)
m.insert(household_wealth(D_H * N_HOUSEHOLDS / (H_ACTIVE + H_INACTIVE)))

# --- Simulation: 12 quarters of SPARQL upserts, paper event ordering ---------
# Taylor-rule coefficients from the published Table 2 (2010:Q4, quarterly).
sim = RDFSimulator(
    update_rules=[
        firm_production(growth_e=0.005),  # (i) supply choice, eq. 5 + 12
        firm_pricing(inflation_e=0.005),  # (i) price setting, eq. 8
        household_income_update(THETA_DIV, THETA_UB),  # eq. 49
        household_update(TAU_VAT),  # (v) consumption + savings, eqs. 40 + 50
        firm_sales(TAU_VAT),  # (iv) goods market, eqs. 1-2 + 27 + 31
        government_consumption(0.005),  # eq. 51
        taylor_rule(
            rho=0.9263, r_star=-0.0034, pi_star=0.005, xi_pi=0.3214, xi_gamma=1.2994
        ),  # eq. 69
    ],
    n_periods=120,
).fit(m)

# Each tick yields raw per-agent state; the macro summary is plain polars:
# GDP by the production approach (Appendix B.1), CPI as mean price.
t0 = time()
for t, state in enumerate(sim):
    print(
        state.select(
            pl.lit(t).alias("t"),
            (pl.col("price") * pl.col("output") * (1 - pl.col("tech_share")))
            .sum()
            .alias("gdp"),
            pl.col("price").mean().alias("price_level"),
            pl.col("wealth").sum().alias("hh_wealth"),
            pl.col("policy_rate").max().alias("policy_rate"),
        )
    )
t1 = time()
print(f"Simulated {sim.n_periods} ticks in {t1 - t0:.2f} seconds ({(t1 - t0) / sim.n_periods:.3f} s/tick)")