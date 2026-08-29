"""
Minimal ingest test for skabm interbank contagion demo.

Tests that the bank module templates map correctly and that the simulator
can run with the new ``infer`` hook, before running the full demo.

Catches:

- Template metadata injection
- ``bank_depositors`` CONSTRUCT (init) - free maplib core
- ``bank_capital`` UPDATE (per-tick) - free maplib core
- ``interbank_contagion`` CONSTRUCT via ``infer`` (per-tick) - licensed add-on

The first three run on the stock PyPI maplib; only ``test_infer_called``
needs the licensed (academic) build and is skipped otherwise (see
``_infer_licensed``).  So the suite is green without a license and complete
with one.
"""

import polars as pl
import pytest

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
    except BaseException:  # noqa: BLE001 - a pyo3 panic is not an Exception
        return False


LICENSED = _infer_licensed()


def test_bank_depositors_maps():
    """bank_depositors CONSTRUCT should map agents to banks."""
    banks = pl.DataFrame(
        {
            "id": ["bank_0", "bank_1"],
            "capital_ratio": [0.10, 0.09],
            "leverage": [12.0, 12.0],
            "deposit_share": [0.30, 0.70],
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
            "id": ["hh_0", "hh_1", "hh_2"],
            "wealth": [100.0, 200.0, 150.0],
            "income": [10.0, 20.0, 15.0],
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
        params={"total_deposits": 1000.0},
        n_periods=0,
        random_seed=0,
    )
    sim.fit({"Household": households, "Bank": banks})

    holds = sim.model_.query(
        """
        PREFIX ex: <http://example.net/skabm#>
        PREFIX def: <urn:maplib_default:>
        SELECT ?agent ?bank WHERE {
            ?agent def:holds_at ?bank .
        }
        """
    )
    assert holds.height > 0
    # maplib returns full IRIs ('<http://example.net/skabm#bank_0>'); compare
    # on local names so the test never spells out the namespace.
    local = {iri.rsplit("#", 1)[-1].rstrip(">") for iri in holds["bank"].to_list()}
    assert local <= {"bank_0", "bank_1"}
    print(f"  bank_depositors mapped {holds.height} agent->bank links")


def test_bank_capital_updates():
    """bank_capital should update capital ratios when there are flight amounts."""
    banks = pl.DataFrame(
        {
            "id": ["bank_0"],
            "capital_ratio": [0.10],
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
        params={
            **poledna_params,
            "distress_threshold": 0.03,
            "flee_amount_threshold": 5.0,
        },
        n_periods=1,
        random_seed=0,
    )
    sim.fit({"Household": households, "Bank": banks})

    cr = sim.model_.query(
        """
        PREFIX ex: <http://example.net/skabm#>
        PREFIX def: <urn:maplib_default:>
        SELECT ?b ?cr WHERE {
            ?b a ex:Bank ; def:capital_ratio ?cr .
        }
        """
    )
    assert cr.height == 1
    print(f"  bank_capital: bank_0 capital_ratio = {cr['cr'][0]:.4f}")


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
    sim.fit({"Household": households, "Bank": banks})

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
