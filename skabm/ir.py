"""
Rule IR: what the SPARQL says about itself.

A behaviour rule is a string until something reads it.  This module reads it through a
real SPARQL algebra (``rdflib``), not a regex, and answers three questions no ABM
framework can normally answer without being told: which predicates each rule reads and
writes (``RuleIR.reads`` / ``writes``, keyed by ``(class, predicate)``); what follows at
model level (``writes`` is what the rules own after t=0, ``structure`` everything they
read but never write — links, coefficients, the knobs a parameter search moves); and
what should therefore be measured (``observables()``, using the aggregate the rules
themselves apply to a predicate, falling back to SUM+AVG where none does).

**Regimes.**  A rule of the form ``BIND(IF(a < b, a, b) AS ?v)`` writing ``?v`` to
predicate ``P`` has a binding constraint: post-tick, the agents where ``P = b`` are the
constrained ones.  When ``b`` reconstructs into an expression over predicates alone
(``?alpha * ?size`` does, ``pr:normal(...)`` does not) that share becomes an observable
nobody wrote down.  Deliberately conservative: a branch touching a random draw, an
aggregate or another class's predicate is dropped rather than guessed at.

**Scope, honestly.**  Class membership is resolved from ``?v a ex:Class`` within each
basic graph pattern, falling back to the rule-wide map.  A predicate hung on a variable
never typed — a link target reached only through another agent, as in the Schelling
lattice — has no class and is skipped, which is why a genuinely relational model still
wants ``RDFSimulator(state_extract=...)``.

**And what it does not do.**  It proposes *scalar* aggregates per agent class; a
distributional statistic is not derivable from a rule and cannot be composed from
class-level scalars, so it belongs to the caller, in polars over ``extract()``.
``signature()`` gives the dtypes for ``polars.DataFrame.to_jax`` — the whole JAX story.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import cache
from string import Template

import polars as pl
from rdflib.plugins.sparql.algebra import translateQuery, translateUpdate
from rdflib.plugins.sparql.parser import parseQuery, parseUpdate
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.term import Literal, URIRef, Variable

from skabm.rules import _PREFIXES, DEF_NS, EX_NS

RDF_TYPE = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")

# The aggregate functions the IR recognises, in both directions: read off a
# rule that already aggregates a predicate, and emitted when proposing one.
_AGGREGATES = {
    "Aggregate_Sum": "SUM",
    "Aggregate_Avg": "AVG",
    "Aggregate_Min": "MIN",
    "Aggregate_Max": "MAX",
}
# Applied to a predicate no rule ever aggregates: a level and a mean, because
# nothing in the rule text says whether the quantity is extensive or intensive.
_DEFAULT_AGGREGATES = ("SUM", "AVG")

# Infix algebra nodes, spelled back out with their operator list interleaved.
_INFIX = ("MultiplicativeExpression", "AdditiveExpression", "RelationalExpression")

# A constrained agent's predicate *equals* its bound, because the rule assigned
# the bound to it — but the bound is recomputed in SPARQL from the same floats,
# so the test is equality within a relative tolerance rather than on the nose.
_AT_BOUND = "IF(ABS(?{p} - {e}) <= 1e-9 + 1e-9 * ABS({e}), 1e0, 0e0)"

_VARIABLE_RE = re.compile(r"\?([A-Za-z_]\w*)")

# ``ex:sig__<agg>__<Class>__<predicate>`` — how a rule declares that it reads an
# aggregate of the model's own past (``behaviour.learning.expect``).  Matched on
# the IRI as it appears in the parsed algebra, so a signal named in a comment or
# a string does not count as a dependency.
_SIGNAL_RE = re.compile(r"^sig__([A-Za-z]+)__([A-Za-z][A-Za-z0-9]*)__([A-Za-z_]\w*)$")


# ---------------------------------------------------------------------------
# SPARQL algebra front end
# ---------------------------------------------------------------------------


def _nodes(node) -> Iterator[CompValue]:
    """Every algebra node under *node*, depth first."""
    if isinstance(node, CompValue):
        yield node
        for value in node.values():
            yield from _nodes(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _nodes(value)


def _local(iri) -> str | None:
    """Local name of a ``def:``/``ex:`` IRI; None for anything else.

    Predicates outside the two model namespaces (``rdf:type``, ``dcterms:``,
    chrontext's) are not agent state and are not part of the IR.
    """
    text = str(iri)
    for namespace in (DEF_NS, EX_NS):
        if text.startswith(namespace):
            return text[len(namespace) :]
    return None


def _triple_blocks(algebra) -> Iterator[tuple[list, bool]]:
    """Every triple block in the rule, flagged True when it is a write.

    Reads come from basic graph patterns, writes from the DELETE/INSERT clauses
    of an update and from the template of a CONSTRUCT.  A DELETE and an INSERT
    on the same predicate is the upsert idiom, so both count as writes and the
    distinction is not worth keeping.
    """
    for node in _nodes(algebra):
        if node.name in ("BGP", "TriplesBlock"):
            yield list(node.get("triples") or []), False
        elif node.name in ("DeleteClause", "InsertClause"):
            yield list(node.get("triples") or []), True
        elif node.name == "ConstructQuery":
            yield list(node.get("template") or []), True


# ---------------------------------------------------------------------------
# Per-rule analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuleIR:
    """One rule's data flow, as read off its SPARQL algebra."""

    name: str
    reads: frozenset  # {(class, predicate)}
    writes: frozenset  # {(class, predicate)}
    aggregates: frozenset  # {(agg, class, predicate)} the rule itself applies
    regimes: frozenset  # {(class, predicate, expression)} binding constraints
    consumes: frozenset = frozenset()  # {(agg, class, predicate)} read from history
    links: frozenset = frozenset()  # {(class, predicate)} pointing at another agent
    orphans: frozenset = frozenset()  # {(predicate, is_write)} with no class in scope

    def adopt(self, owners: dict) -> RuleIR:
        """Re-attach orphan predicates to the classes other rules give them.

        A rule can touch a predicate on a subject it never types — an owner
        minted with ``BIND(IRI(...))``, a lattice cell reached only through
        ``def:location``.  Locally that predicate has no class; model-wide it
        usually does, because some other rule types the same subject.  Adopting
        them is what stops ``firm_ownership`` looking like it writes nothing,
        and therefore what stops ``schedule()`` claiming it commutes with the
        rule that reads what it wrote.
        """
        if not (self.orphans or any(k is None for k, _ in self.links)):
            return self
        reads = set(self.reads)
        writes = set(self.writes)
        links = {pair for pair in self.links if pair[0] is not None}
        for predicate, is_write in self.orphans:
            target = writes if is_write else reads
            target.update((klass, predicate) for klass in owners.get(predicate, ()))
        for klass, predicate in ((k, p) for k, p in self.links if k is None):
            links.update((owner, predicate) for owner in owners.get(predicate, ()))
        return RuleIR(
            self.name,
            frozenset(reads),
            frozenset(writes),
            self.aggregates,
            self.regimes,
            self.consumes,
            frozenset(links),
            self.orphans,
        )


def _class_map(blocks: Sequence[tuple[list, bool]]) -> dict:
    """Variable to agent classes, over the whole rule.

    A type outside the ``ex:``/``def:`` namespaces is not an agent class and
    leaves its subject unclassed, to be picked up as an orphan.
    """
    types: dict = {}
    for triples, _ in blocks:
        for subject, predicate, obj in triples:
            if predicate == RDF_TYPE and isinstance(subject, Variable) and _local(obj):
                types.setdefault(subject, set()).add(_local(obj))
    return types


def _read_write_sets(blocks, types) -> tuple[set, set, dict]:
    """Read set, write set, and the variable-to-``(class, predicate)`` binding.

    Class membership is resolved inside each block first — a UNION branch is
    its own block, so ``?agent a ex:Firm`` in one branch never leaks onto
    ``?agent a ex:Household`` in the next — and falls back to the rule-wide map
    for a variable typed in a block it is joined to rather than declared in.
    A predicate whose subject is typed nowhere in the rule is set aside as an
    orphan for ``RuleIR.adopt`` to place once the whole model is known.
    """
    reads, writes, bound, orphans, links = set(), set(), {}, set(), set()
    for triples, is_write in blocks:
        local_types = {
            s: {_local(o)}
            for s, p, o in triples
            if p == RDF_TYPE and isinstance(s, Variable) and _local(o)
        }
        for subject, predicate, obj in triples:
            name = _local(predicate)
            if predicate == RDF_TYPE or name is None:
                continue
            if not isinstance(subject, Variable):
                continue  # a triple on a named node (a signal) is not agent state
            classes = local_types.get(subject) or types.get(subject) or ()
            if not classes:
                orphans.add((name, is_write))
                if types.get(obj):
                    links.add((None, name))
            for klass in classes:
                (writes if is_write else reads).add((klass, name))
                if types.get(obj):
                    links.add((klass, name))  # object is itself an agent
                elif isinstance(obj, Variable) and not is_write:
                    bound.setdefault(obj, (klass, name))
    return reads, writes, bound, orphans, links


def _reconstruct(expr, bound: dict, binds: dict, klass: str) -> str | None:
    """Spell an algebra expression back out over predicate-named variables.

    Returns None the moment the expression leaves what can be re-evaluated
    against a post-tick graph: a UDF call (the random draw is gone), an
    aggregate, or a predicate belonging to another class.  That conservatism is
    the point — a regime share is only worth recording when it is exact.
    """
    if isinstance(expr, Literal):
        return f"{float(expr):.6e}"
    if isinstance(expr, Variable):
        if expr in bound:
            owner, predicate = bound[expr]
            return f"?{predicate}" if owner == klass else None
        return _reconstruct(binds[expr], bound, binds, klass) if expr in binds else None
    if isinstance(expr, CompValue) and expr.name in _INFIX:
        parts = [_reconstruct(expr["expr"], bound, binds, klass)]
        for op, other in zip(expr["op"], expr["other"]):
            parts += [str(op), _reconstruct(other, bound, binds, klass)]
        return None if None in parts else "(" + " ".join(parts) + ")"
    return None


def _regimes(algebra, bound: dict, written: dict) -> set:
    """Binding constraints: ``(class, predicate, expression)`` triples.

    A ``BIND(IF(cond, a, b) AS ?v)`` whose ``?v`` is written to predicate ``P``
    puts every agent on one of two branches.  Whichever branch reconstructs
    becomes a post-tick test — ``P = <branch>`` selects the agents it bound —
    and the share satisfying it is the regime indicator.
    """
    binds = {n["var"]: n["expr"] for n in _nodes(algebra) if n.name == "Extend"}
    found = set()
    for var, expr in binds.items():
        if (
            var not in written
            or not isinstance(expr, CompValue)
            or expr.name != "Builtin_IF"
        ):
            continue
        klass, predicate = written[var]
        for branch in ("arg2", "arg3"):
            rebuilt = _reconstruct(expr[branch], bound, binds, klass)
            if rebuilt is not None and rebuilt != f"?{predicate}":
                found.add((klass, predicate, rebuilt))
    return found


def analyse_rule(text: str, name: str) -> RuleIR:
    """Read one rendered rule's data flow off its SPARQL algebra."""
    # update rules (DELETE/INSERT) parse as an update; CONSTRUCT and SELECT do
    # not and fall through to the query grammar.  Trying both is cheaper than
    # sniffing, and a rule that parses as neither raises here — at fit time
    try:
        algebra = translateUpdate(parseUpdate(text)).algebra
    except Exception:  # noqa: BLE001 - not an update; try the query grammar
        algebra = [translateQuery(parseQuery(text)).algebra]
    blocks = list(_triple_blocks(algebra))
    types = _class_map(blocks)
    reads, writes, bound, orphans, links = _read_write_sets(blocks, types)

    aggregates = set()
    for node in _nodes(algebra):
        var = node.get("vars") if node.name in _AGGREGATES else None
        if isinstance(var, Variable) and var in bound:
            aggregates.add((_AGGREGATES[node.name], *bound[var]))

    written = {
        o: (klass, predicate)
        for triples, is_write in blocks
        if is_write
        for s, p, o in triples
        if isinstance(o, Variable)
        for klass, predicate in [(next(iter(types.get(s, {None}))), _local(p))]
        if klass is not None and predicate is not None
    }
    consumes = {
        match.groups()
        for triples, _ in blocks
        for term in (t for triple in triples for t in triple)
        if isinstance(term, URIRef)
        for match in [_SIGNAL_RE.match(_local(term) or "")]
        if match
    }
    return RuleIR(
        name=name,
        reads=frozenset(reads),
        writes=frozenset(writes),
        aggregates=frozenset(aggregates),
        regimes=frozenset(_regimes(algebra, bound, written)),
        consumes=frozenset(consumes),
        links=frozenset(links),
        orphans=frozenset(orphans),
    )


# ---------------------------------------------------------------------------
# Model level
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Observable:
    """One measurement the rule set implies: an aggregate over a class.

    ``expr`` is the SPARQL expression to aggregate and ``needs`` the predicates
    it must have bound, so a batch of observables over one class compiles into
    a single query.  ``kind`` is why it was proposed — which is what lets a
    caller plot the sinks, track the cycles and ignore the rest.
    """

    agg: str
    klass: str
    name: str
    expr: str
    needs: frozenset
    kind: str  # "state" | "structure" | "regime"

    @property
    def signal(self) -> str:
        """The history table's key for this observable."""
        return f"sig__{self.agg}__{self.klass}__{self.name}"


@dataclass(frozen=True)
class ModelIR:
    """A rule set's data flow, and everything that follows from it.

    ``carried`` holds ``(class, predicate)`` pairs the *graph* has but no rule
    mentions — a calibrated column like ``def:tech_share`` that the dynamics
    read through no rule of their own.  Nothing writes them, so they are
    structure by definition: they widen the extract and the schema, and they
    are never recorded, because a constant measured every tick measures
    nothing.  ``RDFSimulator`` fills this in from the mapped populations.
    """

    rules: tuple
    carried: frozenset = frozenset()
    graph_links: frozenset = frozenset()

    def with_graph(self, model) -> ModelIR:
        """This IR, plus what only a mapped graph can say.

        Two things: predicates that arrived with a population and are named by
        no rule (``carried``), and predicates whose values are IRIs rather than
        literals (``graph_links``).  Both are cheap — one query — and both are
        facts about the data, so they are kept apart from what the rules say.
        """
        carried, links = graph_predicates(model)
        return ModelIR(self.rules, frozenset(carried) - self.writes, frozenset(links))

    # -- the partitions ----------------------------------------------------

    @property
    def writes(self) -> frozenset:
        return frozenset().union(*(r.writes for r in self.rules))

    @property
    def reads(self) -> frozenset:
        return frozenset().union(*(r.reads for r in self.rules))

    @property
    def structure(self) -> frozenset:
        """Read but never written: links, coefficients — closed-over constants."""
        return (self.reads - self.writes) | self.carried

    @property
    def links(self) -> frozenset:
        """Predicates whose object is another agent, not a quantity.

        Excluded from ``observables()``: the mean of a set of IRIs is not a
        number, and asking maplib for one panics rather than erring.  Kept in
        the extract, where a link column is exactly what a relational plot
        needs.
        """
        return frozenset().union(*(r.links for r in self.rules)) | self.graph_links

    @property
    def consumed(self) -> frozenset:
        """Signals the rules read back out of history — ``(agg, class, predicate)``.

        These are the aggregates a behaviour *depends* on, so they must be
        recorded for the model to run at all; the rest of ``observables()`` is
        recorded because it is worth looking at.  A rule set consuming nothing
        needs no database.
        """
        return frozenset().union(*(r.consumes for r in self.rules))

    @property
    def known(self) -> frozenset:
        """Every ``(class, predicate)`` pair the model touches at all."""
        return self.reads | self.writes | self.carried

    @property
    def classes(self) -> tuple:
        return tuple(sorted({k for k, _ in self.known}))

    def signature(self) -> dict:
        """Per class, the ordered predicate schema — one array per column."""
        return {
            klass: tuple(sorted(p for k, p in self.known if k == klass))
            for klass in self.classes
        }

    # -- what to measure ---------------------------------------------------

    def observables(self) -> tuple:
        """The measurements the rule set implies, deduplicated and sorted.

        State predicates get the aggregate their own rules apply, or SUM and
        AVG when none does; regimes get the share of agents at the bound.
        Structure is deliberately excluded — it does not move during a run, so
        recording it every tick measures nothing.
        """
        chosen: dict = {}
        for rule in self.rules:
            for agg, klass, predicate in rule.aggregates:
                chosen.setdefault((klass, predicate), set()).add(agg)

        found = []
        for klass, predicate in sorted(self.writes - self.links):
            for agg in sorted(chosen.get((klass, predicate)) or _DEFAULT_AGGREGATES):
                found.append(
                    Observable(
                        agg,
                        klass,
                        predicate,
                        f"?{predicate}",
                        frozenset({predicate}),
                        "state",
                    )
                )
        for agg, klass, predicate in sorted(
            self.consumed
            - {(a, k, p) for a in _DEFAULT_AGGREGATES for k, p in self.links}
        ):
            if (agg, klass, predicate) not in {(o.agg, o.klass, o.name) for o in found}:
                found.append(
                    Observable(
                        agg,
                        klass,
                        predicate,
                        f"?{predicate}",
                        frozenset({predicate}),
                        "state",
                    )
                )
        named = set()
        for klass, predicate, expression in sorted(
            {r for rule in self.rules for r in rule.regimes}
        ):
            # IF(c, 1e0, 0e0) has two bounds and one partition: the second share
            # is the first's complement, and a repeated name is a query that fails
            if (klass, predicate) in named:
                continue
            named.add((klass, predicate))
            found.append(
                Observable(
                    "AVG",
                    klass,
                    f"binds__{predicate}",
                    _AT_BOUND.format(p=predicate, e=expression),
                    frozenset(_needs(expression) | {predicate}),
                    "regime",
                )
            )
        return tuple(found)

    # -- queries -----------------------------------------------------------

    def extract_query(self) -> str:
        """Per-agent state, one UNION branch per class — the generic extract.

        Replaces the hand-written ``state_extract`` a model family would
        otherwise carry: the columns are exactly the predicates its own rules
        touch, plus whatever the populations brought.  Every predicate binds
        through OPTIONAL, so an agent missing one keeps its row and gets a null
        — the sparse wide frame the hand-written extracts produced, without the
        writing.  A model whose state is genuinely relational (a predicate on an
        untyped link target) still wants its own — see the module docstring.
        """
        signature = self.signature()
        branches = [
            f"{{ ?agent a ex:{klass} "
            + " ".join(f"OPTIONAL {{ ?agent def:{p} ?{p} }}" for p in predicates)
            + " }"
            for klass, predicates in signature.items()
            if predicates
        ]
        columns = sorted({p for ps in signature.values() for p in ps})
        return (
            f"{_PREFIXES}SELECT ?agent "
            + " ".join(f"?{c}" for c in columns)
            + "\nWHERE {\n  "
            + "\n  UNION ".join(branches)
            + "\n}"
        )

    def extract(self, model) -> pl.DataFrame:
        """Run ``extract_query`` — the IR-derived ``state_extract``."""
        return model.query(self.extract_query())


def measure_queries(observables: Sequence[Observable]) -> dict:
    """One aggregate query per agent class, covering every observable.

    Batching matters: the same measurement one signal at a time is one SPARQL
    round trip per signal per tick, which is what made the narrow ``ex:sig__``
    set the only affordable one to record.  Predicates bind through OPTIONAL so
    an agent missing one still counts toward the others, and each aggregate is
    projected under its own signal name — which makes the result frame's columns
    the history table's keys, and recording an unpivot.
    """
    queries = {}
    for klass in sorted({o.klass for o in observables}):
        batch = [o for o in observables if o.klass == klass]
        needs = sorted(frozenset().union(*(o.needs for o in batch)))
        projections = " ".join(f"({o.agg}({o.expr}) AS ?{o.signal})" for o in batch)
        optionals = "\n  ".join(f"OPTIONAL {{ ?a def:{p} ?{p} }}" for p in needs)
        queries[klass] = (
            f"{_PREFIXES}SELECT {projections}\nWHERE {{\n"
            f"  ?a a ex:{klass} .\n  {optionals}\n}}"
        )
    return queries


def _needs(expression: str) -> set:
    """Predicate names a reconstructed expression binds."""
    return set(_VARIABLE_RE.findall(expression))


def analyse(rules: Iterable[tuple[str, str]]) -> ModelIR:
    """Build the model IR from ``(name, rendered SPARQL)`` pairs.

    Two passes: each rule is read on its own, then the model-wide map from
    predicate to owning classes is fed back so unclassed predicates find their
    subject (``RuleIR.adopt``).
    """
    analysed = [analyse_rule(text, name) for name, text in rules]
    owners: dict = {}
    for rule in analysed:
        for klass, predicate in rule.reads | rule.writes:
            owners.setdefault(predicate, set()).add(klass)
    return ModelIR(tuple(rule.adopt(owners) for rule in analysed))


def graph_predicates(model) -> tuple:
    """``(all pairs, IRI-valued pairs)`` actually present in a mapped graph.

    Feeds ``ModelIR.with_graph``: the IR reads rules, so a column that arrives
    with the population and is never named by one is invisible to it until the
    graph is asked — and so is the difference between a quantity and a link,
    which the rule text only shows when it types the object.
    """
    found = model.query(
        _PREFIXES
        + "SELECT DISTINCT ?c ?p (isIRI(?o) AS ?link) WHERE { ?a a ?c ; ?p ?o }"
    )
    pairs, links = set(), set()
    for row in found.iter_rows(named=True):
        pair = (_local(row["c"].strip("<>")), _local(row["p"].strip("<>")))
        if not all(pair):
            continue
        pairs.add(pair)
        if row["link"]:
            links.add(pair)
    return frozenset(pairs), frozenset(links)


def rule_name(rule, index: int) -> str:
    """A stable name for a rule: its ``@id``, its module-level name, or its slot.

    Names are what the IR triples hang off, so they should survive a rename of
    the variable holding the rule — which ``metadata["@id"]`` does and the
    registry lookup does not.  The registry is the courtesy path for the rules
    ``skabm.behaviour`` ships without metadata.
    """
    declared = getattr(rule, "metadata", {}).get("@id")
    return declared or _shipped().get(id(rule)) or f"rule_{index}"


@cache
def _shipped() -> dict:
    """``id(template) -> module-level name`` for every rule in skabm.behaviour."""
    from skabm import behaviour

    shipped: dict = {}
    for module in pkgutil.iter_modules(behaviour.__path__):
        found = importlib.import_module(f"skabm.behaviour.{module.name}")
        shipped.update(
            {id(v): k for k, v in vars(found).items() if isinstance(v, (Template, str))}
        )
    return shipped
