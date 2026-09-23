"""The simulator's ``infer=`` hook, driven by the interbank contagion rules.

The bank rules are a skabm extension with no published calibration, so there is
no literature value to pin them to: this is an engine test, not a model test.
It needs the licensed (academic) maplib build and is skipped otherwise (see
``_infer_licensed``).
"""

import polars as pl
import pytest
from worlds import world

from skabm.behaviour.bank import bank_capital, bank_depositors, interbank_contagion
from skabm.behaviour.params import poledna_params
from skabm.simulation import RDFSimulator


def _infer_licensed() -> bool:
    """Probe whether this maplib build ships the ``infer`` reasoning add-on.

    ``infer`` (Datalog / recursive CONSTRUCT) is a licensed maplib feature -
    free for academic use, but absent from the stock PyPI wheels, where the
    first call panics ``not implemented: Contact Data Treehouse``.  The panic
    is a pyo3 ``PanicException`` (subclasses ``BaseException``, so a plain
    ``except Exception`` would miss it).  Everything else in ``skabm`` - the
    ``bank_depositors`` / ``bank_capital`` rules included - runs on the free
    core; only ``interbank_contagion`` via ``infer=`` needs the add-on.
    """
    from maplib import Model

    try:
        Model().infer(
            "PREFIX ex: <http://x#> CONSTRUCT { ?s ex:p ?o } WHERE { ?s ex:p ?o }"
        )
        return True
    except BaseException:  # noqa: BLE001 — pyo3's PanicException is not an Exception
        return False


LICENSED = _infer_licensed()


@pytest.mark.skipif(
    not LICENSED, reason="Model.infer needs the licensed (academic) maplib build"
)
def test_infer_called():
    """The simulator should call infer when infer= is passed."""
    banks = pl.DataFrame(
        {
            "id": ["bank_0"],
            "capital_ratio": [0.02],  # below distress threshold
            "leverage": [12.0],
            "deposit_share": [1.0],
        }
    ).cast(
        {
            "capital_ratio": pl.Float64,
            "leverage": pl.Float64,
            "deposit_share": pl.Float64,
        }
    )

    households = pl.DataFrame(
        {
            "id": ["hh_0"],
            "wealth": [100.0],
            "income": [10.0],
            "psi": 0.9,
        }
    ).cast(
        {
            "wealth": pl.Float64,
            "income": pl.Float64,
            "psi": pl.Float64,
        }
    )

    sim = RDFSimulator(
        init_rules=(bank_depositors,),
        update_rules=(bank_capital,),
        infer=interbank_contagion,
        params={
            **poledna_params,
            "distress_threshold": 0.03,
            "flee_amount_threshold": 5.0,
        },
        n_periods=1,
        random_seed=0,
    )
    sim.fit(world(Household=households, Bank=banks))

    distressed = sim.model_.query(
        """
        PREFIX ex: <http://example.net/skabm#>
        SELECT (COUNT(?b) AS ?n) WHERE {
            ?b ex:is_distressed true .
        }
        """
    )
    n = int(distressed["n"][0])
    assert n >= 1, f"Expected at least 1 distressed bank, got {n}"
    print(f"  infer produced {n} distressed bank(s)")
