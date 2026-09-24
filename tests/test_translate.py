"""``skabm.translate`` — rules from a description and templates.

Engine tests: the grammar built from templates is the DSL over their fields and
nothing else, and what a model writes under it becomes the same rule dicts
``behaviour/labour.py`` builds by hand.  No model is called, except by
``test_a_description_regenerates_its_rule``: ``SKABM_LLM=1`` runs it on the tiny local
model, ``SKABM_LLM=<hugging face name>`` on another; either is downloaded.
"""

import importlib
import os
import re

import pytest
import sympy as sp
from sympy.stats.rv import RandomSymbol

from skabm.behaviour.labour import applications, clock_tick, labour_flow
from skabm.dsl import OPERATORS, DSLError, _Rule, coalesce, is_rule, lag, sparql
from skabm.ottr import TEMPLATES, clock_template, edge_template, occupation_template
from skabm.translate import _accepts, grammar, local, rules

LABOUR = [occupation_template, edge_template, clock_template]

# Three rules, one per class in the templates' order, as a model writing under the grammar would.
WRITTEN = """\
applications = Rule({Occupation.applications: Occupation.vacancies * sum_over(Edge.dst, Edge.weight * sp.Piecewise((Edge.src.unemployment / Edge.src.app_norm, Edge.src.app_norm > 0), (0, True)))})
labour_flow = Rule({Edge.flow: sp.Piecewise((Edge.src.unemployment * Edge.dst.vacancies ** 2 * Edge.weight * (1 - sp.exp(-Edge.dst.applications / Edge.dst.vacancies)) / (Edge.dst.applications * Edge.src.app_norm), (Edge.dst.vacancies > 0) & (Edge.dst.applications > 0) & (Edge.src.app_norm > 0)), (0, True))})
clock_tick = Rule({Clock.t: Clock.t + 1})
"""


def test_the_grammar_is_the_dsl_over_the_templates():
    lark = grammar(LABOUR)
    assert _accepts(WRITTEN, lark)
    for outside in (
        "x = __import__('os')\n",
        "x = Occupation.__class__\n",
        "x = Firm.output\n",  # not among the templates
        "x = Occupation.employmnet\n",
        "x = Edge.src\n",  # a link is not a number
        "x = sp.Piecewise((1, Clock.t == 0), (0, True))\n",  # == is Python's, not SymPy's
        "import os\n",
        "x = Occupation.employment\n",  # a statement is a rule
    ):
        assert not _accepts(outside, lark), outside
    # a key, a class and a many-valued link, where the templates have them
    town = grammar([TEMPLATES[k] for k in ("Commuter", "Route", "Link", "Option")])
    chooses = (
        "x = Rule({Commuter.route: coalesce(pick(Route, sp.Eq(Route.od, Commuter.od)"
        " & (Route.prob > 0), Route.cum), Commuter.route)})\n"
        "y = Rule({Route.time: sum_over(Route.via, Route.via.t0) + total_by(Route.od, 1)})\n"
    )
    assert _accepts(chooses, town)


def test_rules_come_from_the_description_and_the_templates():
    seen = {}

    def model(prompt, lark):
        seen.update(prompt=prompt, grammar=lark)
        return answer

    answer = WRITTEN
    written = rules("how occupations hire", LABOUR, model=model)
    assert list(written) == ["applications", "labour_flow", "clock_tick"]
    assert [sparql(r) for r in written.values()] == [
        sparql(applications),
        sparql(labour_flow),
        sparql(clock_tick),
    ]
    assert seen["grammar"] == grammar(LABOUR, "how occupations hire")
    assert "how occupations hire" in seen["prompt"]
    assert "src (link to Occupation: occupation the job seeker last" in seen["prompt"]
    # a parameter is a name the description introduces, even across a line break
    answer = "churn = Rule({Occupation.separations: delta_u * Occupation.employment})"
    (churn,) = rules("separate at the parameter\ndelta_u", LABOUR, model=model).values()
    assert _Rule(churn).parameters() == {"delta_u"}
    with pytest.raises(DSLError, match="not in the rule language"):
        rules("separate at some rate", LABOUR, model=model)


def test_an_answer_outside_the_grammar_never_runs():
    def model(prompt, lark):
        return answer

    answer = "x = __import__('os').system('true')\n"
    with pytest.raises(DSLError, match="not in the rule language"):
        rules("anything", LABOUR, model=model)
    answer = (
        "x = Rule({Edge.flow: Occupation.vacancies})\n"  # in the grammar, still wrong
    )
    with pytest.raises(DSLError, match="reach it by a link"):
        rules("anything", LABOUR, model=model)
    answer = "x = Rule({Edge.flow: 1, Edge.flow: 2})\n"  # a dict keeps the last
    with pytest.raises(DSLError, match="same field twice"):
        rules("anything", LABOUR, model=model)


# modules whose docstring is the specification of their rules
DESCRIBED = ("macro", "household", "firm", "bank", "labour", "schelling", "traffic")
BEYOND_7B = {  # the docstring is right; the 7B cannot yet write these rules from it
    "schelling": "reads 'the people whose location is this cell' as comparing coordinates",
    "traffic": "merges the two Commuter rules and runs away on the leg times",
}


def module_rules(name: str) -> list:
    """A behaviour module's rules, in the order it defines them."""
    module = importlib.import_module(f"skabm.behaviour.{name}")
    return [v for v in vars(module).values() if is_rule(v)]


def same(ours: dict, theirs: dict) -> bool:
    """Same class, same fields, each written to an equal expression: "a tenth" may be
    0.1 or 1/10, and a draw may have any name."""

    def plain(e):
        draws = sorted(e.atoms(RandomSymbol), key=str)
        e = e.xreplace({d: sp.Symbol(f"draw{i}") for i, d in enumerate(draws)})
        return sp.nsimplify(e, rational=True)

    ours, theirs = _Rule(ours), _Rule(theirs)
    if ours.klass != theirs.klass or set(ours.writes) != set(theirs.writes):
        return False
    return all(
        sp.simplify(plain(ours.writes[f]) - plain(theirs.writes[f])) == 0
        for f in ours.writes
    )


@pytest.mark.parametrize("name", DESCRIBED)
def test_a_module_docstring_is_prose(name):
    """A description the model could copy would prove nothing: no code in it."""
    names = "|".join(op.__name__ for op in (*OPERATORS, coalesce, lag))
    code = re.compile(rf"\b[A-Z]\w*\.\w+|\b(?:{names})\(")
    doc = importlib.import_module(f"skabm.behaviour.{name}").__doc__
    assert not code.search(doc), code.search(doc)


@pytest.mark.skipif(
    not os.environ.get("SKABM_LLM"), reason="SKABM_LLM=1 runs the model"
)
@pytest.mark.parametrize(
    "name",
    [
        pytest.param(n, marks=pytest.mark.xfail(reason=BEYOND_7B[n]))
        if n in BEYOND_7B
        else n
        for n in DESCRIBED
    ],
)
def test_a_module_docstring_regenerates_its_rules(name):
    """The docstring is the specification: the module's rules, in order, come back
    from it and the templates of the classes they touch."""
    shipped = module_rules(name)
    classes = {
        s.name.split(".")[0]
        for rule in map(_Rule, shipped)
        for e in rule.writes.values()
        for s in {*e.atoms(sp.Symbol), sp.Symbol(f"{rule.klass}.x")}
        if "." in s.name and s.name[0] != "@"
    }
    templates = [TEMPLATES[k] for k in sorted(classes)]
    model = os.environ["SKABM_LLM"]
    doc = importlib.import_module(f"skabm.behaviour.{name}").__doc__
    written = list(
        rules(doc, templates, local(*([model] if model != "1" else []))).values()
    )
    assert len(written) == len(shipped), written
    assert all(map(same, shipped, written)), written
