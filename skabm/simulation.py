"""
RDFSimulator: advance a maplib knowledge-graph ABM, one step of its rules per tick.

``fit`` takes one argument, the world: a maplib ``Model``, simulated as given and
advanced in place.  How the agents got into it — ``skabm.template`` and
``Model.map``, a deserialized file, another system — is not the simulator's business.

    world = Model()
    world.map(template.firm, firms.with_iri())
    RDFSimulator(params=params, n_periods=12).fit(world)   # Poledna rules by default

The world is the opening state, starting values included: what happens before the first
step is data (``household.initial``, ``firm.ownership``), not a rule.

``fit_iter`` yields every agent's state after each tick, one frame with ``t`` and
``class`` columns: a run is ``pl.concat(sim.fit_iter(world))``, and a macro quantity is
polars over it, written by whoever knows what to look for.  ``model_`` is current
at each yield.

A rule naming an ``ex:sig__<agg>__<Class>__<predicate>`` signal reads a learned
expectation of that aggregate.  The simulator builds one ``learner`` rule per signal
(``behaviour.learning.sac`` by default) and runs it after every tick, so the forecast is
state on the signal's node, re-estimated from running sums rather than from a stored
series.

**Two models.**  ``model_`` is the world, agents plus the signal nodes their expectations
live on; ``meta_`` is behaviour provenance, so a ``?s ?p ?o`` over the world is not
cluttered with it.

Rules written as ``skabm.dsl`` dicts also compile to JAX (``dsl.jax_tick``), where the
SPARQL run is the reference the differentiable one is checked against.
"""

from __future__ import annotations

import importlib
import pkgutil
import warnings
from collections.abc import Callable, Iterator, Sequence
from functools import cache

import polars as pl
import polars_random as pr
import sympy as sp
from maplib import Model
from sklearn.base import BaseEstimator

from skabm.behaviour import defaults
from skabm.behaviour.firm import (
    firm_liquidity,
    firm_price,
    firm_produce,
    firm_sales,
)
from skabm.behaviour.household import (
    household_income,
    satisificing_consume,
)
from skabm.behaviour.learning import consumed, sac
from skabm.behaviour.macro import (
    centralbank_rate,
    government_spend,
)
from skabm.dsl import is_rule, sparql
from skabm.sparql import _PREFIXES, DEF_NS, EX_NS, register_polars_random

# Canonical Poledna (2023) rule composition, sourced from behaviour/.
# Users override via __init__(rules=..., params=...).
DEFAULT_RULES = (
    firm_produce,  # supply choice, eq. 5 + 12
    firm_price,  # price setting, eq. 8
    household_income,  # income refresh, eq. 49
    satisificing_consume,  # consumption + savings, eqs. 40 + 50
    firm_sales,  # goods market, eqs. 1-2 + 27
    firm_liquidity,  # eq. 31
    government_spend,  # AR(1), eq. 51
    centralbank_rate,  # Taylor rule, eq. 69
)


def rule_name(rule, index: int) -> str:
    """A rule's module-level name in ``skabm.behaviour``, or its slot."""
    return _shipped().get(id(rule), (f"rule_{index}", ""))[0]


@cache
def _shipped() -> dict:
    """``id(rule) -> (module-level name, the module's first docstring line)`` for every
    rule in skabm.behaviour."""
    from skabm import behaviour

    shipped: dict = {}
    for module in pkgutil.iter_modules(behaviour.__path__):
        found = importlib.import_module(f"skabm.behaviour.{module.name}")
        source = (found.__doc__ or "").strip().split("\n")[0]
        shipped.update(
            {id(v): (k, source) for k, v in vars(found).items() if is_rule(v)}
        )
    return shipped


def _inject_metadata(model: Model, rules: Sequence) -> None:
    """Each rule as an ``ex:Behaviour`` of the class it updates, with its module's
    source."""
    links, sources = [], []
    for i, rule in enumerate(rules):
        klass = str(sp.sympify(next(iter(rule)))).split(".")[0]
        iri = EX_NS + rule_name(rule, i)
        links += [
            (EX_NS + klass, EX_NS + "behaviour", iri),
            (iri, _RDF_TYPE, _BEHAVIOUR),
        ]
        source = _shipped().get(id(rule), ("", ""))[1]
        if source:
            sources.append((iri, "http://purl.org/dc/terms/source", source))
    columns = ["subject", "predicate", "object"]
    if links:
        model.map_triples(pl.DataFrame(links, schema=columns, orient="row"))
    if sources:
        from rdflib import Literal

        frame = pl.DataFrame(sources, schema=columns, orient="row")
        model.map_triples(frame.with_columns(pl.col("object").map_elements(Literal)))


_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
_BEHAVIOUR = EX_NS + "Behaviour"


def _extract(model: Model) -> pl.DataFrame:
    """Every agent's class and every field, one row per agent, null where it has none:
    one UNION branch per class, then grouped by agent, since a many-valued link (a
    cell's neighbours) repeats the agent once per value.  That link is the list of
    them; every other field is its one value."""
    fields: dict = {}
    pairs = model.query("SELECT DISTINCT ?c ?p WHERE { ?a a ?c ; ?p ?o }")
    for c, p in pairs.iter_rows():
        c, p = c.strip("<>"), p.strip("<>")
        if c.startswith(EX_NS) and p.startswith(DEF_NS):
            fields.setdefault(c[len(EX_NS) :], set()).add(p[len(DEF_NS) :])
    columns = sorted(set().union(*fields.values()))
    branches = "\n  UNION ".join(
        f'{{ ?agent a ex:{klass} BIND("{klass}" AS ?class) '
        + " ".join(f"OPTIONAL {{ ?agent def:{p} ?{p} }}" for p in sorted(ps))
        + " }"
        for klass, ps in sorted(fields.items())
    )
    rows = model.query(
        f"{_PREFIXES}SELECT ?agent ?class {' '.join('?' + c for c in columns)}\n"
        f"WHERE {{\n  {branches}\n}}"
    )
    grouped = rows.group_by("agent", maintain_order=True).agg(
        pl.exclude("agent").drop_nulls().unique(maintain_order=True)
    )
    many = [c for c in grouped.columns[1:] if (grouped[c].list.len() > 1).any()]
    return grouped.with_columns(pl.exclude("agent", *many).list.first())


def _rules(learned) -> tuple:
    """A learner may return one rule or several."""
    return (learned,) if isinstance(learned, dict) else tuple(learned)


class RDFSimulator(BaseEstimator):
    """Advance a maplib knowledge-graph ABM with SPARQL update rules.

    Parameters
    ----------
    rules : Sequence[dict]
        Rule dicts (a module's ``RULES``), applied in order at every tick: the model's event sequence (Poledna Section 3.5).  Rules over
        unmapped agent classes no-op harmlessly.
    params : dict | None
        Parameter values by name, laid over ``skabm.behaviour.defaults()``, every
        module's cited ``PARAMETERS``.  A parameter with no published value has no
        default and must be passed; a missing one raises at fit time, naming it.
    n_periods : int
        Number of ticks ``fit`` runs (and ``fit_iter`` yields).
    warm_start : bool
        When True, ``fit``/``fit_iter`` continue ticking a model that already
        exists — ``model_``, or the one passed as ``X`` — instead of rebuilding
        and re-initialising it.
    state_extract : callable | None
        ``Model -> pl.DataFrame``, called by ``fit_iter`` after each tick to
        yield raw per-agent state.  ``None`` (the default) projects every field
        the graph holds, one UNION branch per agent class.  Pass
        one only when that is not enough — a model whose state hangs off untyped
        link targets, like ``schelling.state_extract`` reaching ``def:x``
        through ``def:location``.
    udfs : Sequence[Callable[[Model], None]]
        Registrars called on ``model_`` before mapping, each installing SPARQL
        UDFs via ``Model.add_udf``.  SPARQL's built-in function set is fixed and
        small, so a rule needing anything beyond it (a random draw, ``exp``)
        depends on a registrar having run first — which makes *which UDFs exist*
        a property of the rule set, exactly like the rules themselves, and
        therefore a hyperparameter rather than a hard-coded call.  Defaults to
        ``pr:uniform`` / ``pr:normal`` for the Poledna rules.  Schelling needs only
        ``rules.register_polars_random``; the labour-market rules additionally
        need ``rules.register_math`` (``behaviour.labour.LABOUR_UDFS``).
    random_seed : int | None
        When set, ``pr.set_random_seed`` is called at the start of each
        ``fit``/``fit_iter`` so the ``pr:uniform`` / ``pr:normal`` SPARQL UDFs
        (registered on ``model_`` via ``rules.register_polars_random``) draw a
        reproducible sequence.  Leave None for entropy-seeded stochastic runs.
    learner : Callable[[str, str, str], dict] | None
        ``(agg, class, predicate) -> rule``, called once per signal the rules read
        (``ex:sig__<agg>__<Class>__<pred>``, ``learning.consumed``).
        The rules it returns run after every tick, the opening state included,
        and write ``def:forecast`` on the signal's node, which is what
        ``behaviour.learning.expect`` reads.  Defaults to
        ``behaviour.learning.sac``, Poledna's Sample-Autocorrelation-learned
        expectations (eq. 6/9); another estimator is another function.  ``None``
        learns nothing, and every expectation stays at its structural zero.

    Attributes
    ----------
    model_ : maplib.Model
        The world state: empty after ``__init__``, populated and evolved by
        ``fit`` / ``fit_iter``.
    meta_ : maplib.Model
        Behaviour provenance.  Never holds agents — see "Two models" above.
    """

    def __init__(
        self,
        rules: Sequence[dict] = DEFAULT_RULES,
        params: dict | None = None,
        n_periods: int = 12,
        warm_start: bool = False,
        state_extract: Callable[[Model], pl.DataFrame] | None = None,
        udfs: Sequence[Callable[[Model], None]] = (register_polars_random,),
        random_seed: int | None = None,
        learner: Callable[[str, str, str], dict] | None = sac,
    ):
        self.rules = rules
        self.params = params
        self.n_periods = n_periods
        self.warm_start = warm_start
        self.state_extract = state_extract
        self.udfs = udfs
        self.random_seed = random_seed
        self.learner = learner
        self.model_ = Model()
        self.meta_ = Model()
        self._learners, self._consumed, self._tick = [], set(), 0

    def _cold_start(
        self,
        world: Model | None,
        rules: list[str],
    ) -> None:
        """Adopt the world and inject provenance.

        Runs only on a cold ``fit``/``fit_iter`` (warm_start=False); the tick
        loop in ``fit_iter`` is shared by both paths.  *world* is a maplib
        ``Model``, used as given and advanced in place; ``None`` is an empty one.

        The learners then run on the opening state, so the first tick already has a
        base period to measure growth against.
        """
        self.model_ = Model() if world is None else world
        self.meta_ = Model()
        self._register_udfs()
        rules_text = "\n".join(rules)
        for klass in sorted(self._classes()):
            if f"ex:{klass}" not in rules_text:
                warnings.warn(
                    f"population {klass!r} is not referenced by any rule "
                    "(no 'ex:' + kind pattern): it is in the model but "
                    "stays inert during simulation.",
                    UserWarning,
                    stacklevel=4,
                )
        _inject_metadata(self.meta_, self.rules)
        self._bind_graph()
        self._tick = 0
        self._learn()

    def fit_iter(self, X=None) -> Iterator[pl.DataFrame]:
        """Take the world, then yield every agent's state after each of
        the ``n_periods`` ticks, with a ``t`` column: a run is ``pl.concat(...)``.

        *X* is a maplib ``Model`` — advanced in place, so whatever it already
        holds is the opening state.  ``None`` continues ``model_`` under
        ``warm_start``, and is an empty world otherwise.
        """
        if X is not None and not isinstance(X, Model):
            raise TypeError(
                f"X is a maplib Model, not {type(X).__name__}: map populations into "
                "one with skabm.template and Model.map, then pass the model."
            )
        if self.random_seed is not None:
            pr.set_random_seed(self.random_seed)
        merged = {**defaults(), **(self.params or {})}
        rules = [sparql(rule, merged) for rule in self.rules]
        self._consumed = consumed(self.rules)
        if self.warm_start:
            if X is not None:
                self.model_ = X
            self._register_udfs()
            self._bind_graph()
        else:
            self._cold_start(X, rules)
        for _ in range(self.n_periods):
            for rule in rules:
                self.model_.update(rule)
            self._tick += 1
            self._learn()
            state = (self.state_extract or _extract)(self.model_)
            yield state.with_columns(t=pl.lit(self._tick))

    def _register_udfs(self) -> None:
        """Install every UDF registrar in ``self.udfs`` on ``model_``.

        That is the seam a researcher's own routine arrives through: an ordinary
        ``Model.add_udf`` callable, named in their SPARQL like any other function.
        """
        for register in self.udfs:
            register(self.model_)

    def _learn(self) -> None:
        """Run the learners on this period, so the next tick's rules read forecasts
        that include it."""
        for rule in self._learners:
            self.model_.update(rule)

    def _bind_graph(self) -> None:
        """Settle what gets learned, from the classes the graph holds.

        A signal over a class the graph lacks gets no learner: there is no level to
        learn from, and its expectation stays at the structural zero ``expect``
        defaults to.
        """
        present = self._classes()
        params = {**defaults(), **(self.params or {})}
        self._learners = [
            sparql(rule, params)
            for signal in sorted(self._consumed if self.learner else ())
            if signal[1] in present
            for rule in _rules(self.learner(*signal))
        ]

    def _classes(self) -> set:
        """The ``ex:`` agent classes the graph holds, whoever put them there."""
        return {
            iri.strip("<>")[len(EX_NS) :]
            for iri in self.model_.query("SELECT DISTINCT ?c WHERE { ?s a ?c }")["c"]
            if iri.strip("<>").startswith(EX_NS)
        }

    def fit(self, X=None) -> RDFSimulator:
        """Take the world, run all ticks; return self."""
        for _ in self.fit_iter(X):
            pass
        return self
