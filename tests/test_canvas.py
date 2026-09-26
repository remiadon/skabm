"""CANVAS, Hommes, He, Poledna, Siqueira & Zhang (2025), J. Econ. Dyn. Control 172, 104986.

Model tests assert what the paper says: its Table 3 and 4 values, the price-quantity
scenarios of Appendix A.2.4 (Fig. 10) through eqs. 39-42, that a firm never moves price
and quantity together, and that a cheaper or larger firm is picked more (A.2.3).  The
Taylor rule's coefficients are re-estimated every quarter and never reported (eq. 7), so
it gets no model test: the rest runs without it.  Engine tests pin the SPARQL and JAX
runs to each other and ``initial`` to a clearing opening quarter; their numbers are any.
"""

import jax
import numpy as np
import polars as pl
import pytest
from worlds import world

from skabm.behaviour import canvas, defaults
from skabm.behaviour.learning import sac
from skabm.dsl import arrays, jax_tick, sparql
from skabm.simulation import RDFSimulator

jax.config.update("jax_enable_x64", True)

WITHOUT_TAYLOR = [rule for rule in canvas.RULES if rule is not canvas.augmented_taylor]
TAYLOR = {"boc_rho": 0.9, "boc_r_star": 0.005, "boc_xi_pi": 1.5, "boc_xi_gamma": 0.5}

# One good is the whole basket, capital and input: the weights of Table 3's columns
# sum to one, so a one-sector economy's are one.
ONE_SECTOR = pl.DataFrame({"id": ["all"], "b_hh": [1.0], "b_cf": [1.0]})
SELF = pl.DataFrame(
    {"id": ["in"], "buyer": ["all"], "supplier": ["all"], "share": [1.0]}
)


def firms(**columns) -> pl.DataFrame:
    """Firms, of the one sector unless *columns* say otherwise.  Their coefficients are
    any: every cost ratio is zero while the sectors' prices are equal."""
    n = len(next(iter(columns.values())))
    coefficients = {
        "alpha": 10.0,
        "size": 10.0,
        "margin": 0.1,
        "w_bar": 5.0,
        "tech_share": 0.5,
    }
    return pl.DataFrame(
        {"id": [f"f{i}" for i in range(n)], "sector": ["all"] * n, "delta": [0.1] * n}
        | {name: [value] * n for name, value in coefficients.items()}
        | columns
    )


# Two sectors buying from each other; the numbers are any.
NETWORK = pl.DataFrame({"id": ["a", "b"], "b_hh": [0.7, 0.3], "b_cf": [0.2, 0.8]})
EDGES = pl.DataFrame(
    {
        "id": ["aa", "ab", "ba", "bb"],
        "buyer": ["a", "a", "b", "b"],
        "supplier": ["a", "b", "a", "b"],
        "share": [0.6, 0.4, 0.3, 0.7],
    }
)
MIXED = firms(
    size=[5.0, 20.0, 8.0, 12.0],
    price=[1.0, 1.1, 0.9, 1.05],
    sector=["a", "a", "b", "b"],
    w_bar=[3.0, 6.0, 4.0, 5.0],
)


def run(firm_frame, sectors, inputs, rules=WITHOUT_TAYLOR, ticks=1, **params):
    """Every agent after each tick, from *firm_frame* in its opening quarter."""
    firm_frame, sectors = canvas.initial(firm_frame, sectors, inputs)
    sim = RDFSimulator(
        rules=rules, params=params, n_periods=ticks, udfs=canvas.CANVAS_UDFS
    )
    graph = world(Firm=firm_frame, Sector=sectors, Input=inputs)
    return pl.concat(sim.fit_iter(graph), how="diagonal").with_columns(
        id=pl.col("agent").str.extract(r"#(.*)>$")
    )


def test_the_rules_default_to_the_papers_values():
    """Table 3's inventory depreciation (1 in every sector) and Table 4's τ^SIF and π*;
    eq. 7's coefficients have no published value, so compiling without them raises."""
    assert {
        k: defaults()[k] for k in ("tau_sif", "inventory_depreciation", "pi_star")
    } == {
        "tau_sif": 0.0,  # τ^SIF, Table 4
        "inventory_depreciation": 1.0,  # δ^S_s, Table 3
        "pi_star": 0.005,  # π*, Table 4
    }
    with pytest.raises(KeyError, match="boc_r_star.*boc_rho.*boc_xi_gamma.*boc_xi_pi"):
        sparql(canvas.augmented_taylor, defaults())


def test_the_four_scenarios_of_fig_10():
    """A.2.4 (a)-(d), by eqs. 39-42: excess supply at a price above the sector's cuts the
    price, below it cuts production; excess demand above it raises production, below it
    raises the price; the one that moves, moves by demand over supply."""
    supply, demand = [100.0, 100.0, 80.0, 80.0], [80.0, 80.0, 100.0, 100.0]
    price = [1.2, 0.8, 1.2, 0.8]  # the sector's index is 1: every firm sold 80
    opening = firms(
        price=price, supply=supply, demand=demand, output=supply, sales=[80.0] * 4
    )
    after = run(opening, ONE_SECTOR, SELF).filter(pl.col("class") == "Firm").sort("id")
    assert after["price"].to_list() == pytest.approx([1.2 * 0.8, 0.8, 1.2, 0.8 * 1.25])
    assert after["output"].to_list() == pytest.approx(
        [100.0, 100 * 0.8, 80 * 1.25, 80.0]
    )


def test_a_firm_never_moves_price_and_quantity_together():
    """ "We also assume that firms cannot change their quantity and price at the same
    time" (A.2.4), in a production network run for twelve quarters."""
    ticks = run(MIXED, NETWORK, EDGES, ticks=12).filter(pl.col("class") == "Firm")
    assert ((ticks["gamma_d"] != 0) | (ticks["pi_d"] != 0)).any()  # something moved
    assert (ticks["gamma_d"] * ticks["pi_d"] == 0).all()


def test_a_cheaper_or_larger_firm_is_picked_more():
    """A.2.3: "a firm charging a relatively lower price than its competitors is more
    likely to be picked by consumers", and "a larger firm tends to have a higher
    probability".  Every market clears in the opening quarter, so nothing moves before
    buyers choose."""
    output, price = [100.0, 100.0, 200.0], [1.0, 0.9, 1.0]  # a base, cheaper, larger
    opening = firms(price=price, output=output)
    after = run(opening, ONE_SECTOR, SELF).filter(pl.col("class") == "Firm").sort("id")
    base, cheaper, larger = after["demand"].to_list()
    assert cheaper > base and larger > base


def test_table_3s_weights_price_the_opening_basket_at_one():
    """Eqs. 48 and 54 at §3.1's opening prices, all one, and Table 3's b^HH and b^CF
    columns: the CPI and the capital price index are one, to the table's rounding, so
    neither wages nor capital push costs in the opening quarter (eq. 43)."""
    # fmt: off
    b_hh = [0.0143, 0.0023, 0.0172, 0.0004, 0.1967, 0.0393, 0.1194, 0.0359, 0.0361,
            0.3117, 0.0068, 0.0055, 0.0035, 0.0282, 0.0171, 0.0678, 0.0256, 0.0397, 0.0326]
    b_cf = [0.0003, 0.0148, 0.0024, 0.5142, 0.2612, 0.0425, 0.0, 0.0086, 0.0172,
            0.0147, 0.0705, 0.0012, 0.0001, 0.0004, 0.0003, 0.0001, 0.0007, 0.0006, 0.0502]
    # fmt: on
    ids = [f"s{i}" for i in range(19)]
    sectors = pl.DataFrame({"id": ids, "b_hh": b_hh, "b_cf": b_cf})
    opening = firms(output=[1.0] * 19, sector=ids)
    none = pl.DataFrame(
        schema={"buyer": pl.String, "supplier": pl.String, "share": float}
    )
    _, sectors = canvas.initial(opening, sectors, none)
    assert sectors["labour_cost"].to_list() == pytest.approx([0.0] * 19, abs=1e-4)
    assert sectors["capital_cost"].to_list() == pytest.approx([0.0] * 19, abs=1e-12)


# --- engine -------------------------------------------------------------------------

BANK = pl.DataFrame({"id": ["cb"], "policy_rate": [0.01]})


def test_the_opening_quarter_clears_every_sector():
    """``initial``: each sector's final demand plus what the others buy from it is what
    its firms make, so its first quarter's demand is its opening sales' value, and data
    it already has is kept."""
    kept = NETWORK.with_columns(final_demand=pl.Series([None, 7.0]))
    opening, sectors = canvas.initial(MIXED, kept, EDGES)
    assert sectors.filter(pl.col("id") == "b")["final_demand"].item() == 7.0
    first = run(MIXED, NETWORK, EDGES).filter(pl.col("class") == "Sector").sort("id")
    made = opening.group_by("sector").agg((pl.col("price") * pl.col("output")).sum())
    assert first["demand"].to_list() == pytest.approx(
        made.sort("sector")["price"].to_list()
    )


def test_sparql_and_jax_agree_on_canvas():
    """Every rule, the Taylor rule included, and the SAC learners, six quarters on the
    graph and in JAX from the same opening frames."""
    ticks = 6
    opening, sectors = canvas.initial(MIXED, NETWORK, EDGES)
    sim = RDFSimulator(
        rules=canvas.RULES, params=TAYLOR, n_periods=ticks, udfs=canvas.CANVAS_UDFS
    )
    graph = world(Firm=opening, Sector=sectors, Input=EDGES, CentralBank=BANK)
    *_, last = sim.fit_iter(graph)
    last = last.with_columns(id=pl.col("agent").str.extract(r"#(.*)>$"))

    frames = {"Firm": opening, "Sector": sectors, "Input": EDGES, "CentralBank": BANK}
    learners = [sac(*signal) for signal in sorted(sim._consumed)]
    state = arrays(frames, [*canvas.RULES, *learners])
    params = {**defaults(), **TAYLOR}
    learn, quarter = jax_tick(learners), jax_tick(canvas.RULES)
    state = learn(state, params)
    for _ in range(ticks):
        state = learn(quarter(state, params), params)

    for klass, fields in {
        "Firm": ("price", "output", "demand", "sales", "pi_c"),
        "Sector": ("price_index", "demand", "material_cost"),
        "CentralBank": ("policy_rate",),
    }.items():
        agents = last.filter(pl.col("id").is_in(frames[klass]["id"].to_list())).sort(
            "id"
        )
        order = np.argsort(frames[klass]["id"].to_numpy())
        for field in fields:
            got = np.asarray(state[klass][field])[order]
            assert got == pytest.approx(agents[field].to_numpy(), rel=1e-9), field
