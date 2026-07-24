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

Construction and simulation are one call: the populations are passed to
`RDFSimulator.fit_iter` as keyword arguments (one DataFrame per agent
class; templates are auto-generated, predicates are column names), the
init rules (`HOUSEHOLD_INCOME`, `HOUSEHOLD_WEALTH`) run in-graph right
after mapping, and each tick applies the SPARQL DELETE/INSERT upserts from
`skabm.rules` in the paper's event order and yields raw per-agent state,
summarized here with plain polars expressions.
See skabm/rules.py for the deliberate simplifications this pure-SPARQL
architecture imposes (no search-and-matching, no stochastic shocks, no
AR(1) re-estimation).
"""

import polars as pl
import polars_random as pr
from time import time

from skabm.calibration import make_dataset, weighted_enum
from skabm.datasets import build_firm_io_df
from skabm.simulation import RDFSimulator

# Everything is bare local names ("firm_0"): map_df prefixes ids, and a link
# column (whose values name already-mapped agents) resolves automatically.

# --- Table 2 / Table 7 scalars (reference quarter 2010:Q4) -------------------
# (behavioral parameters live in rules.POLEDNA_PARAMS; these are only the
# ones the calibration itself needs)
TAU_SIF = 0.2122  # employers' social insurance contributions
TAU_INC = 0.2134  # income tax rate
PSI = 0.9394  # propensity to consume out of expected income
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
        pl.col("id").cast(pl.String).alias("id"),
        pl.col("industry").cast(pl.String).alias("industry"),
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
        pl.lit(PSI).alias("psi"),
        # inactive households do not supply labor.  Firm ownership is NOT a
        # column: the FIRM_OWNERSHIP init rule assigns it in-graph (Section
        # 3.2), controlled by the firm_ownership_ratio hyperparameter.
        pl.when(pl.col("status") == "active")
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
    pl.col("purchase_sector").cast(pl.String).alias("purchase_sector"),
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
).with_columns(pl.format("bank_{}", pl.col("id")).alias("id"))

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
        pl.col("source_industry").cast(pl.String).alias("source_industry"),
        (0.52 * gva0 * pl.col("demand_weight") / pl.col("demand_weight").sum()).alias(
            "demand_size"
        ),
    )
    .drop("demand_weight")
)

# --- Central bank: singleton with Taylor-rule state (Section 3.5) ------------
central_bank = pl.DataFrame(
    {
        "id": ["central_bank"],
        "policy_rate": [0.009],
        "inflation_target": [0.005],
        "prev_output": [float(firms["output"].sum())],
        "prev_price": [1.0],
    }
)

# --- Simulation: default Poledna rules, one parameter override ---------------
# Rule logic lives in skabm.rules Templates; every behavioral parameter
# takes its published Table 2 value from rules.POLEDNA_PARAMS.  The single
# substitute this demo needs is total_deposits (D^H in HOUSEHOLD_WEALTH),
# rescaled to the demo's sampling fraction because the paper is 1:1 with
# Austria and this demo is not.
sim = RDFSimulator(
    params={
        "total_deposits": D_H * N_HOUSEHOLDS / (H_ACTIVE + H_INACTIVE),
        # one investor per firm (Section 3.2)
        "firm_ownership_ratio": N_FIRMS / N_HOUSEHOLDS,
    },
    n_periods=120,
)

# Each tick yields raw per-agent state; the macro summary is plain polars:
# GDP by the production approach (Appendix B.1), CPI as mean price.
# Bank is mapped but inert (no rule references ex:Bank yet) — fit warns.
t0 = time()
for t, state in enumerate(
    sim.fit_iter(
        Firm=firms,
        Household=households,
        Government=governments,
        Bank=banks,
        ForeignFirm=foreign_firms,
        CentralBank=central_bank,
    )
):
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
print(
    f"Simulated {sim.n_periods} ticks in {t1 - t0:.2f} seconds ({(t1 - t0) / sim.n_periods:.3f} s/tick)"
)
