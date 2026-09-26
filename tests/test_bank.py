"""Banks (``skabm.behaviour.bank``): a skabm extension with no published calibration,
so no model tests, only engine ones.  They pin the depositor draw and the timing the
per-tick contagion rules give: one step of the cascade per tick.
"""

import polars as pl
from worlds import world

from skabm.behaviour.bank import RULES, depositors
from skabm.simulation import RDFSimulator
from skabm.sparql import _PREFIXES

BANKS = pl.DataFrame(
    {
        "id": ["weak", "sound"],
        "capital_ratio": [0.01, 0.10],  # weak starts below the 0.03 threshold
        "leverage": [10.0, 10.0],
        "deposit_share": [0.5, 0.5],
    }
)


def test_each_depositor_banks_at_one_bank_drawn_by_share():
    agents = pl.DataFrame({"id": [f"hh_{i}" for i in range(200)]})
    drawn = depositors(agents, BANKS, seed=1)
    assert set(drawn["holds_at"]) == {"weak", "sound"}
    assert drawn["holds_at"].equals(depositors(agents, BANKS, seed=1)["holds_at"])
    only_sound = BANKS.with_columns(deposit_share=pl.Series([0.0, 1.0]))
    assert set(depositors(agents, only_sound)["holds_at"]) == {"sound"}


def test_contagion_spreads_one_step_per_tick():
    households = pl.DataFrame(
        {"id": ["a", "b"], "wealth": [50.0, 1.0], "psi": [0.9, 0.9]}
    ).with_columns(holds_at=pl.Series(["weak", "sound"]))

    def distressed(n_periods: int) -> dict:
        sim = RDFSimulator(rules=RULES, n_periods=n_periods, random_seed=0).fit(
            world(links=("holds_at",), Bank=BANKS, Household=households)
        )
        rows = sim.model_.query(
            _PREFIXES + "SELECT ?b ?d WHERE { ?b a ex:Bank ; def:distressed ?d }"
        )
        return {b.split("#")[1].rstrip(">"): d for b, d in rows.iter_rows()}

    # tick 1: the weak bank is distressed and its depositor flees 50, above the
    # 10.0 flight threshold; tick 2: that flight distresses the sound bank too
    assert distressed(1) == {"weak": 1.0, "sound": 0.0}
    assert distressed(2) == {"weak": 1.0, "sound": 1.0}
