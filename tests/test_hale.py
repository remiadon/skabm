"""HALE (Moon et al. 2026) on the RDFSimulator machinery — the two claims that matter.

The disease side is standard SIR and is checked by making it deterministic
(`beta=1`, `gamma=0`, unit weights), so a contact either transmits or it does
not and there is nothing statistical to tolerate.

The claim under test is the coupling: an LLM answer written on a *group* node
reaches every agent through `def:group`, and switching it off silences exactly
the deliberate contacts.  Household transmission surviving a lockdown is the
paper's deliberate/non-deliberate split, and it is the one thing a wrong join
would quietly break — a `?gs def:goes_out` that failed to match would look like
a working simulation with a slower epidemic.
"""

import polars as pl
import pytest
from maplib import Model, xsd

from skabm.hale import HALE_PARAMS, HALE_UPDATE_RULES
from skabm.ottr import map_populations
from skabm.rules import _PREFIXES, LLM_NS, register_llm, register_polars_random
from skabm.simulation import RDFSimulator


def decider(answer: float):
    """A stub `llm:choose` that always answers the same way — the offline stand-in
    for a model, registered on the same `Model.add_udf` seam."""

    def register(model):
        model.add_udf(
            LLM_NS + "choose",
            lambda df: pl.select(out=pl.repeat(answer, len(df), dtype=pl.Float64))[
                "out"
            ],
            xsd.double,
            [xsd.string],
        )

    return register


@pytest.fixture
def world():
    """p0 is infected; p1 meets it only outdoors, p2 only at home."""
    persons = pl.DataFrame(
        {
            "id": ["p0", "p1", "p2"],
            "group": ["g0", "g0", "g0"],
            "infected": [1.0, 0.0, 0.0],
            "recovered": [0.0, 0.0, 0.0],
        }
    )
    groups = pl.DataFrame(
        {
            "id": ["g0"],
            "municipality": ["Bluffdale"],
            "age_band": ["25-34"],
            "sex": ["female"],
            "race": ["Asian"],
        }
    )
    edges = pl.DataFrame(
        {
            "id": ["e_out", "e_home"],
            "src": ["p1", "p2"],  # susceptible -> infected, as INFECT reads them
            "dst": ["p0", "p0"],
            "weight": [1.0, 1.0],
            "kind": ["out", "home"],
        }
    )
    return {"Person": persons, "Group": groups, "Edge": edges}


def infected(sim) -> set:
    frame = sim.model_.query(
        _PREFIXES + "SELECT ?p WHERE { ?p a ex:Person ; def:infected 1e0 }"
    )
    return {s.strip("<>").rsplit("#", 1)[-1] for s in frame["p"]}


def run(world, answer: float):
    sim = RDFSimulator(
        init_rules=(),
        update_rules=HALE_UPDATE_RULES,
        # deterministic disease: every contact transmits, nobody recovers, so
        # whatever fails to spread was silenced by behaviour and not by luck
        params=dict(HALE_PARAMS, beta=1.0, gamma=0.0),
        n_periods=1,
        udfs=(register_polars_random, decider(answer)),
        random_seed=0,
    )
    sim.fit(world)
    return sim


def test_going_out_spreads_on_both_kinds_of_contact(world):
    assert infected(run(world, 1.0)) == {"p0", "p1", "p2"}


def test_staying_home_silences_only_the_deliberate_contact(world):
    # the LLM said no: the outdoor contact is gone, the household one is not.
    assert infected(run(world, 0.0)) == {"p0", "p2"}


def test_prevalence_reaches_the_prompt(world):
    """The group carries the county figure the prompt is built from.

    PREVALENCE runs first in the tick, so the number the prompt sees is the one
    the decision is actually taken on — the state *before* this tick's
    transmission, not after it.  One of three infected at t=0."""
    sim = run(world, 0.0)
    prevalence = sim.model_.query(
        _PREFIXES + "SELECT ?v WHERE { ?g a ex:Group ; def:prevalence ?v }"
    )
    assert prevalence["v"][0] == pytest.approx(1 / 3)  # p0 alone, pre-transmission


# ---------------------------------------------------------------------------
# The decider seam itself — rules.register_llm, with the provider patched out
# ---------------------------------------------------------------------------

ASK = (
    _PREFIXES
    + """
DELETE { ?g def:goes_out ?o0 }
INSERT { ?g def:goes_out ?o1 }
WHERE {
    ?g a ex:Group ; def:municipality ?city .
    OPTIONAL { ?g def:goes_out ?o0 }
    BIND(CONCAT("Does someone in ", ?city, " go out?") AS ?q)
    BIND(llm:choose(?q) AS ?o1)
}
"""
)


@pytest.fixture
def two_groups(monkeypatch):
    """Two groups in one city and one in another — three subjects, two prompts.

    The provider is replaced by an offline decider recording what reaches it,
    which is the list the memo is measured on.
    """
    import polars_llm  # noqa: F401  — registers the `.llm` namespace being patched

    sent: list[str] = []

    def _fake(self, **kwargs):
        def _answer(s: pl.Series) -> pl.Series:
            sent.extend(s.to_list())
            return pl.Series(["no" if "Alta" in p else "yes" for p in s])

        return self._prompt.map_batches(_answer, return_dtype=pl.String)

    monkeypatch.setattr(pl.Expr.llm, "aanthropic", _fake)

    model = Model()
    map_populations(
        model,
        {
            "Group": pl.DataFrame(
                {
                    "id": ["g0", "g1", "g2"],
                    "municipality": ["Bluffdale", "Bluffdale", "Alta"],
                }
            )
        },
    )
    return model, sent


def test_the_word_never_reaches_the_graph(two_groups):
    """`choices` maps yes/no to a double inside the UDF, so `def:goes_out` is a
    number the transmission rule can filter on — and an aggregate the IR can
    measure, which a string predicate would not be."""
    model, _ = two_groups
    register_llm(model)
    model.update(ASK)

    out = model.query(
        _PREFIXES + "SELECT ?g ?o WHERE { ?g a ex:Group ; def:goes_out ?o }"
    ).sort("g")
    assert out["o"].to_list() == [1.0, 1.0, 0.0]


def test_each_distinct_prompt_is_asked_once(two_groups):
    """Three groups, two situations, two ticks — two calls.  This is HALE's
    group reduction falling out of the memo rather than being built."""
    model, sent = two_groups
    register_llm(model)
    model.update(ASK)
    model.update(ASK)

    assert sorted(sent) == [
        "Does someone in Alta go out?",
        "Does someone in Bluffdale go out?",
    ]
