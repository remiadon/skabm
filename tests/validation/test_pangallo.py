"""Pangallo, Aleta, del Rio-Chanona et al. (2024), Nat. Hum. Behav. 8, 264-275, in the
lighter form of ``notebooks/pangallo.ipynb`` (one region), on a tiny one, in JAX.

Model tests assert what the paper states: the steady state its economy starts in (SI
footnote 2), SI footnote 5's reallocation identity, the fear scenarios of the main text,
and the pre-symptomatic share k of SI Table 5.  The seven parameters the paper calibrates
by ABC (SI §4.3) have no published value, so the tests pass arbitrary ones where a value
is needed and assert nothing that depends on it.  An engine test pins the place-based
contact sums.
"""

import jax
import numpy as np
import polars as pl
import pytest

from skabm.behaviour import defaults, firm, macro, person
from skabm.dsl import arrays, jax_tick

jax.config.update("jax_enable_x64", True)

# Methods ("Economic module", "Coupling"): the day and yesterday's deaths, labour
# demand, contacts and infections on yesterday's employment, hiring and firing,
# consumption, orders and production
RULES = [
    macro.region_day,
    firm.industry_labour,
    person.place_contacts,
    person.exposure,
    person.compartment_progression,
    person.job_draws,
    firm.industry_production,
]

CALIBRATED = {  # SI §4.3 calibrates these; any value does for an engine test
    "transmissibility": 0.2,
    "fear_infection": 0.1,
    "fear_ratio": 0.07,
    "fear_unemployment": 0.3,
    "reallocation_share": 0.5,
    "hire_speed": 0.05,
    "fire_speed": 0.3,
}
NEVER = 1e6  # a lockdown_start no run reaches

REGION = pl.DataFrame({"id": ["ny"]}).with_columns(
    t=pl.lit(0.0),
    deaths=pl.lit(0.0),
    reported=pl.lit(0.0),
    lockdown=pl.lit(0.0),
    mandates=pl.lit(0.0),
)
# two people per household, a worker heading it; a is customer-facing
PEOPLE = pl.DataFrame(
    {
        "id": [f"p{i}" for i in range(12)],
        "household": [f"h{i // 2}" for i in range(12)],
        "tract": ["t1"] * 6 + ["t2"] * 6,
        "county": ["c"] * 12,
        "age": [40.0, 8.0, 35.0, 70.0, 50.0, 12.0, 28.0, 30.0, 45.0, 16.0, 60.0, 85.0],
        "worker": [1, 0] * 6,
        "wfh": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        "industry": ["a", None, "b", None, "a", None] * 2,
    }
)
FLOWS = pl.DataFrame(
    {"supplier": ["a", "b", "b"], "buyer": ["b", "a", "b"], "amount": [3.0, 2.0, 1.0]}
)


def frames(**people) -> dict:
    persons, places = person.contact_population(PEOPLE, **people)
    staff = (
        PEOPLE.drop_nulls("industry")
        .group_by(id="industry")
        .agg(
            inperson0=(1 - pl.col("wfh")).sum().cast(pl.Float64),
            fromhome0=pl.col("wfh").sum().cast(pl.Float64),
        )
    )
    industries, orders = firm.steady_state(
        FLOWS,
        pl.DataFrame(
            {
                "id": ["a", "b"],
                "essential": [0.4, 1.0],
                "customer_facing": [1.0, 0.0],
                "community_weight": [30.0, 5.0],
                "consumption0": [6.0, 4.0],
                "government0": [1.0, 2.0],
                "other0": [2.0, 1.0],
            }
        ).join(staff, on="id", maintain_order="left"),
    )
    return {
        "Region": REGION,
        "Industry": industries,
        "Order": orders,
        "Person": persons,
        "Place": places,
    }


def params(**overrides) -> dict:
    return {**defaults(), **CALIBRATED, **overrides}


def jax_run(state: dict, ticks: int, **overrides) -> list[dict]:
    tick, key, states = jax_tick(RULES), jax.random.key(0), []
    for t in range(ticks):
        state = tick(state, params(**overrides), jax.random.fold_in(key, t))
        states.append(state)
    return states


def consumption(deaths: float, **overrides) -> np.ndarray:
    """Industry consumption after one production rule, at *deaths* yesterday."""
    state = arrays(frames(latent=0), RULES)
    state["Region"]["deaths"] = state["Region"]["deaths"].at[0].set(deaths)
    out = jax_tick([firm.industry_production])(state, params(**overrides))
    return np.asarray(out["Industry"]["consumption"])


def test_the_economy_starts_in_its_steady_state():
    """SI footnote 2: with demand at initial capacity, labour demand (eqs. 19-20) is
    initial employment, so without shocks nothing moves."""
    start = frames(latent=0)
    *_, last = jax_run(arrays(start, RULES), 3, lockdown_start=NEVER)
    industries = start["Industry"]
    for field, initial in (
        ("output", "output0"),
        ("demand", "output0"),
        ("inperson", "inperson0"),
        ("fromhome", "fromhome0"),
    ):
        assert np.asarray(last["Industry"][field]) == pytest.approx(
            industries[initial].to_numpy()
        ), field
    assert np.asarray(last["Person"]["employed"]).tolist() == [1.0, 0.0] * 6


def test_reallocation_moves_a_share_of_what_fear_saves():
    """SI footnote 5: without income effects, demand for goods bought without contact is
    its usual demand plus Delta s times what fear saves on customer-facing ones."""
    before = frames(latent=0)["Industry"]["consumption0"].to_numpy()
    now = consumption(5.0, fear_unemployment=0.0)
    facing = frames(latent=0)["Industry"]["customer_facing"].to_numpy() == 1
    saved = (before - now)[facing].sum()
    assert saved > 0
    assert now[~facing].sum() == pytest.approx(
        before[~facing].sum() + CALIBRATED["reallocation_share"] * saved
    )


def test_tenfold_fear_cuts_customer_facing_demand_by_three_quarters():
    """Main text, counterfactuals: the calibrated fear cuts consumption from
    customer-facing industries by 14% at the epidemic's peak, and ten times that fear by
    77% (eq. 2: 1 - 0.86^10 = 0.779)."""
    industries = frames(latent=0)["Industry"]
    facing = industries["customer_facing"].to_numpy() == 1
    before = industries["consumption0"].to_numpy()[facing]
    phi_eco = -np.log(1 - 0.14)
    for scale, cut in ((1, 0.14), (10, 0.77)):
        now = consumption(1.0, fear_infection=scale * phi_eco, fear_ratio=1.0)
        assert 1 - now[facing] / before == pytest.approx(cut, abs=0.01)


def test_presymptomatic_cases_carry_a_share_k_of_transmission():
    """SI Table 5: a share k of a symptomatic case's transmission happens before
    symptoms; beta_S over beta times the pre-symptomatic period gamma, against beta
    times the infectious period mu, gives that share."""
    p = defaults()
    before = before_symptoms(p)
    share = (
        before
        * p["presymptomatic_days"]
        / (before * p["presymptomatic_days"] + p["infectious_days"])
    )
    assert share == pytest.approx(p["presymptomatic_share"])
    value = person.infectiousness.subs({**p, "Person.state": 2})
    assert float(value) == pytest.approx(before)


def before_symptoms(p: dict) -> float:
    return (
        p["presymptomatic_share"]
        * p["infectious_days"]
        / (p["presymptomatic_days"] * (1 - p["presymptomatic_share"]))
    )


def test_exposure_is_the_pairwise_contact_sum():
    """Engine: summing infectiousness at places, then over a person's places, is SI
    eq. 1's sum over contacts j of omega_ij I_j, with omega_ij the weights of the places
    i and j share."""
    start = frames(latent=0)
    start["Person"] = start["Person"].with_columns(
        state=pl.Series([3.0, 0, 0, 2, 0, 4, 0, 0, 3, 0, 0, 0])
    )
    p = params(fear_infection=0.0)
    out = jax_tick([person.place_contacts, person.exposure])(arrays(start, RULES), p)[
        "Person"
    ]
    persons = start["Person"]
    weight = dict(zip(start["Place"]["id"], start["Place"]["weight"]))
    pulls = {2.0: before_symptoms(p), 3.0: 1.0, 4.0: p["asymptomatic_infectiousness"]}
    pull = persons["state"].replace_strict(pulls, default=0.0).to_numpy()
    rows = list(persons.iter_rows(named=True))
    expected = [
        sum(
            weight[me[layer]] * pull[j]
            for j, other in enumerate(rows)
            if other["id"] != me["id"]
            for layer in ("home", "school", "community", "work")
            if me[layer] is not None and me[layer] == other[layer]
        )
        for me in rows
    ]
    susceptible = persons["state"].to_numpy() == 0
    assert np.asarray(out["exposure"])[susceptible] == pytest.approx(
        np.asarray(expected)[susceptible]
    )


def test_an_epidemic_runs_its_course_and_reports_its_deaths():
    """Engine: seeded, the epidemic moves people only forward through its states, and
    the region counts the deaths reported."""
    start = frames(latent=6)
    start["Person"] = start["Person"].with_columns(fatality=pl.lit(1.0))
    states = jax_run(arrays(start, RULES), 60, transmissibility=2.0)
    path = np.stack([np.asarray(s["Person"]["state"]) for s in states])
    assert (np.diff(path, axis=0) >= 0).all()
    reported = (path == person.REPORTED).sum(axis=1)
    assert reported[-1] > 0
    regions = [float(s["Region"]["reported"][0]) for s in states]
    assert regions == np.r_[0, reported[:-1]].tolist()


def test_the_lockdown_closes_and_reopens_on_the_paper_dates():
    """Methods, Timeline: closures from 16 March to 15 May 2020, schools and work from
    home from 16 March on, counted from 12 February."""
    states = jax_run(arrays(frames(latent=0), RULES), 100)
    day = {int(s["Region"]["t"][0]): s["Region"] for s in states}
    assert float(day[32]["lockdown"][0]) == 0 and float(day[33]["lockdown"][0]) == 1
    assert float(day[92]["lockdown"][0]) == 1 and float(day[93]["lockdown"][0]) == 0
    assert float(day[93]["mandates"][0]) == 1
