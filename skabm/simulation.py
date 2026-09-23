"""
RDFSimulator: advance a maplib knowledge-graph ABM with SPARQL update rules.

``fit`` takes one argument, the world: a maplib ``Model``, simulated as given and
advanced in place.  How the agents got into it — ``skabm.ottr`` and
``Model.map``, a deserialized file, another system — is not the simulator's business.

    world = Model()
    world.map(firm_template, firms.with_iri())
    RDFSimulator(params=params, n_periods=12).fit(world)   # Poledna rules by default

``fit_iter`` yields one ``{signal: level}`` row per tick — the observables
``skabm.ir`` derives from the rules, so nothing declares a reporter.  ``model_`` is
current at each yield, so ``sim.extract()`` in the loop body gives the per-agent
frame a distributional statistic needs.

A rule naming an ``ex:sig__<agg>__<Class>__<predicate>`` signal reads the model's own
past, and that is what opens a DuckDB history: ``history_rules`` re-run over the
accumulated series each tick and their result columns are upserted back as ``def:``
predicates (``skabm.history``).  Measuring itself never needed a database.

**Two models, on purpose.**  ``model_`` is the world, agents and nothing else; ``meta_`` is
the sidecar — rule IR, behaviour metadata, signal nodes, the virtualized history, any UDF a
history rule calls.  Forced rather than stylistic: registering a virtualization silently
nulls every graph-local aggregate on the model carrying it, and the behaviour rules are
built out of those.  maplib has no cross-graph join, so spanning both costs a polars join.

TODO: numerical backends — compile the graph to polars frames (or ``.to_jax()`` for
differentiable kernels), step in frame-land, re-map at observation points.  The SPARQL
path then becomes the reference the fast kernels are validated against.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator, Sequence
from string import Template

import polars as pl
import polars_random as pr
from maplib import Model
from sklearn.base import BaseEstimator

from skabm.behaviour.firm import (
    firm_ownership,
    firm_price,
    firm_produce,
    firm_sales,
)
from skabm.behaviour.household import (
    household_income,
    household_income_init,
    household_wealth_init,
    satisificing_consume,
)
from skabm.behaviour.learning import register_sac, sac_learning
from skabm.behaviour.macro import (
    centralbank_rate,
    government_spend,
)
from skabm.history import TABLE as HISTORY_TABLE
from skabm.history import (
    apply_history_rules,
    attach_history,
    connect,
    measure,
    record,
    signal_name,
    state_frame,
)
from skabm.ir import analyse, rule_name
from skabm.sparql import EX_NS, register_polars_random, render

# Canonical Poledna (2023) rule composition, sourced from behaviour/.
# Users override via __init__(init_rules=..., update_rules=..., params=...).
DEFAULT_INIT_RULES = (firm_ownership, household_income_init, household_wealth_init)
DEFAULT_UPDATE_RULES = (
    firm_produce,  # supply choice, eq. 5 + 12
    firm_price,  # price setting, eq. 8
    household_income,  # income refresh, eq. 49
    satisificing_consume,  # consumption + savings, eqs. 40 + 50
    firm_sales,  # goods market, eqs. 1-2 + 27 + 31
    government_spend,  # AR(1), eq. 51
    centralbank_rate,  # Taylor rule, eq. 69
)
# Expectations are learned rather than assumed: firm_produce, firm_price and
# satisificing_consume read ex:sig__* forecasts, and this rule re-estimates them
# from the recorded history each tick (behaviour.learning).
DEFAULT_HISTORY_RULES = (sac_learning,)


def _inject_metadata(model: Model, rules: Sequence) -> None:
    """Insert behaviour provenance triples into *model* for templates that carry metadata.

    Covers the ``template.metadata`` injection path used by
    ``RDFSimulator.fit_iter``.  Used directly by tests so the block is
    exercised without a full cold-fit.
    """
    for rule in rules:
        if hasattr(rule, "metadata"):
            m = rule.metadata
            kind = m.get("agentClass", "")
            # Accept both ex:Firm and http://example.net/skabm#Firm forms
            if kind.startswith("ex:"):
                kind = EX_NS + kind[3:]
            if kind.startswith(EX_NS):
                kind_name = kind[len(EX_NS) :]
                class_iri = EX_NS + kind_name
                beh_iri = EX_NS + m["@id"]
                model.map_triples(
                    pl.DataFrame(
                        {
                            "subject": [class_iri],
                            "predicate": [EX_NS + "behaviour"],
                            "object": [beh_iri],
                        }
                    )
                )
                model.map_triples(
                    pl.DataFrame(
                        {
                            "subject": [beh_iri],
                            "predicate": [
                                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
                            ],
                            "object": [EX_NS + "Behaviour"],
                        }
                    )
                )
                if "source" in m:
                    from rdflib import Literal

                    model.map_triples(
                        pl.DataFrame(
                            {
                                "subject": [beh_iri],
                                "predicate": ["http://purl.org/dc/terms/source"],
                                "object": [Literal(m["source"])],
                            }
                        )
                    )


class RDFSimulator(BaseEstimator):
    """Advance a maplib knowledge-graph ABM with SPARQL update rules.

    Parameters
    ----------
    init_rules : Sequence[Template | str]
        SPARQL CONSTRUCT rules applied once through ``Model.insert`` right
        after the populations are mapped (e.g. ``behaviour.household.household_income_init``).
        ``string.Template`` rules get their ``$placeholders`` substituted
        from ``params``; plain strings pass through.  Rules anchored on
        unmapped agent classes no-op harmlessly.
    update_rules : Sequence[Template | str]
        SPARQL UPDATE rules (DELETE/INSERT upserts) applied in order within
        each tick — the model's event sequence (Poledna Section 3.5).
    infer : str | Sequence[Template | str] | None
        Optional Datalog or recursive SPARQL CONSTRUCT rules evaluated by
        ``Model.infer`` (licensed feature: maplib commercial add-on, free for
        academic / personal use) once per tick, **after** the update rules have
        advanced state and **before** the next tick begins.  The inference runs
        to a fixed point — derived triples feed back into the rule bodies until
        nothing new is produced — so a single ``infer`` call propagates relational
        effects (contagion, transitive closure, reachability) across the whole
        graph without hand-tuned sub-tick passes.  ``infer=None`` (default) keeps
        the simulator in its original Poledna (2023) configuration where banks
        are passive and no multi-hop propagation exists.  Because the feature is
        licensed, users must ensure their maplib installation carries the add-on;
        calling ``infer`` on a build without it raises at the ``Model.infer`` call.
        Pass a single rule string or a sequence of ``Template``/string rules;
        templates get their ``$placeholders`` substituted from ``params`` exactly
        like ``update_rules``, and plain strings pass through unrendered.
    params : dict | None
        Overrides for the rules' ``$placeholders``, by name, laid over each rule's
        own cited defaults (``behaviour.DefaultTemplate``: ``rule.default``).  A
        placeholder with no published value has no default and must be passed;
        a missing one raises at fit time, naming it.
    n_periods : int
        Number of ticks ``fit`` runs (and ``fit_iter`` yields).
    warm_start : bool
        When True, ``fit``/``fit_iter`` continue ticking a model that already
        exists — ``model_``, or the one passed as ``X`` — instead of rebuilding
        and re-initialising it.
    state_extract : callable | None
        ``Model -> pl.DataFrame``, called by ``fit_iter`` after each tick to
        yield raw per-agent state.  ``None`` (the default) uses the IR-derived
        extract: one UNION branch per agent class, projecting exactly the
        predicates the rules touch plus whatever the populations carried.  Pass
        one only when that is not enough — a model whose state hangs off untyped
        link targets, like ``schelling.state_extract`` reaching ``def:x``
        through ``def:location``.
    track : bool
        Whether to measure every observable the IR derives, or only the signals
        the rules read back.  The default measures everything: it is one
        aggregate query per agent class per tick and it needs no database, so
        the telemetry is there whether or not anything consumes it.  ``False``
        is for a rule set whose aggregates are expensive and whose series
        nobody wants.
    udfs : Sequence[Callable[[Model], None]]
        Registrars called on ``model_`` before mapping, each installing SPARQL
        UDFs via ``Model.add_udf``.  SPARQL's built-in function set is fixed and
        small, so a rule needing anything beyond it (a random draw, ``exp``)
        depends on a registrar having run first — which makes *which UDFs exist*
        a property of the rule set, exactly like the rules themselves, and
        therefore a hyperparameter rather than a hard-coded call.  Registered on ``model_``
        and, when there is one, on the history learner.  Defaults to
        ``pr:uniform`` / ``pr:normal`` for the Poledna rules plus ``sac:forecast``
        for the default history rule.  Schelling needs only
        ``rules.register_polars_random``; the labour-market rules additionally
        need ``rules.register_math`` (``behaviour.labour.LABOUR_UDFS``).
    random_seed : int | None
        When set, ``pr.set_random_seed`` is called at the start of each
        ``fit``/``fit_iter`` so the ``pr:uniform`` / ``pr:normal`` SPARQL UDFs
        (registered on ``model_`` via ``rules.register_polars_random``) draw a
        reproducible sequence.  Leave None for entropy-seeded stochastic runs.
    history_rules : Sequence[Template | str]
        SPARQL SELECT queries evaluated once per tick against the *history* —
        the DuckDB table of per-tick aggregates, virtualized into a dedicated
        model (``skabm.history``).  Each projects a subject IRI first and any
        number of value columns after it, which are upserted onto that subject
        as ``def:`` predicates, so a quantity derived from the model's own past
        becomes an ordinary triple the update rules read.  Which aggregates get
        recorded is detected from the rule text (``ex:sig__<agg>__<Class>__<pred>``),
        so a rule set that never mentions one opens no database at all.

        This is the generic seam for any history-dependent behaviour: the
        simulator knows only "run these queries, write back their columns".
        Defaults to ``(behaviour.learning.sac_learning,)`` — Poledna's
        Sample-Autocorrelation-learned expectations (eq. 6/9), whose entire
        implementation is that query plus the ``sac:forecast`` UDF in ``udfs``.
    duckdb_connection : str | duckdb.DuckDBPyConnection | None
        Where the recorded state history is kept.  ``None`` (default) opens a
        fresh ``:memory:`` database per fit; a string is a file path, so a long
        run persists tick by tick and can be inspected after the process exits;
        an already-open connection is used as given and never closed here.

    Attributes
    ----------
    model_ : maplib.Model
        The world state: empty after ``__init__``, populated and evolved by
        ``fit`` / ``fit_iter``.
    observables_ : tuple[skabm.ir.Observable, ...]
        The measurements this run takes, derived from the rules and name-sorted.
        Their names are the keys of every row ``fit_iter`` yields.
    meta_ : maplib.Model
        The sidecar: rule IR triples, behaviour metadata, signal nodes, the
        virtualized history, and the UDFs a history rule calls.  Never holds
        agents — see "Two models, on purpose" above.
    connection_ : duckdb.DuckDBPyConnection | None
        The DuckDB connection holding the realized series behind the learned
        expectations — one row per measured signal per tick, virtualized into
        ``meta_`` so a history rule can select over graph and history at once.
        ``None`` unless a rule reads an expectation or a ``duckdb_connection``
        was named.  ``history()`` returns the whole table as polars.
    """

    def __init__(
        self,
        init_rules: Sequence[Template | str] = DEFAULT_INIT_RULES,
        update_rules: Sequence[Template | str] = DEFAULT_UPDATE_RULES,
        infer: Sequence[Template | str] | None = None,
        params: dict | None = None,
        n_periods: int = 12,
        warm_start: bool = False,
        state_extract: Callable[[Model], pl.DataFrame] | None = None,
        track: bool = True,
        udfs: Sequence[Callable[[Model], None]] = (
            register_polars_random,
            register_sac,
        ),
        random_seed: int | None = None,
        history_rules: Sequence[Template | str] = DEFAULT_HISTORY_RULES,
        duckdb_connection: str | object | None = None,
    ):
        self.init_rules = init_rules
        self.update_rules = update_rules
        self.infer = infer
        self.params = params
        self.n_periods = n_periods
        self.warm_start = warm_start
        self.state_extract = state_extract
        self.track = track
        self.udfs = udfs
        self.random_seed = random_seed
        self.history_rules = history_rules
        self.duckdb_connection = duckdb_connection
        self.model_ = Model()
        self.meta_ = Model()
        self.virtualized_ = False
        self.connection_ = None
        self._ir = None
        self._tick = 0
        self.observables_ = ()

    def _cold_start(
        self,
        world: Model | None,
        init_rules: list[str],
        update_rules: list[str],
        infer_rules: list[str] | None,
    ) -> None:
        """Adopt the world, inject provenance, apply init rules.

        Runs only on a cold ``fit``/``fit_iter`` (warm_start=False); the tick
        loop in ``fit_iter`` is shared by both paths.  *world* is a maplib
        ``Model``, used as given and advanced in place; ``None`` is an empty one.

        When the rules read expectations, this also opens the DuckDB history,
        virtualizes it into the model, seeds each signal's prior, and records
        the t=0 state — so the first tick already has a base period to measure
        growth against and a forecast to read.
        """
        self.model_ = Model() if world is None else world
        self.meta_ = Model()
        self.virtualized_ = False
        self._register_udfs()
        self._register_udfs(self.meta_)
        rules_text = "\n".join((*init_rules, *update_rules, *(infer_rules or ())))
        for klass in sorted(self._classes()):
            if f"ex:{klass}" not in rules_text:
                warnings.warn(
                    f"population {klass!r} is not referenced by any init/update/"
                    "infer rule (no 'ex:' + kind pattern): it is in the model but "
                    "stays inert during simulation.",
                    UserWarning,
                    stacklevel=4,
                )
        all_rules = (*self.init_rules, *self.update_rules, *(self.infer or ()))
        _inject_metadata(self.meta_, all_rules)
        for rule in init_rules:
            self.model_.insert(rule)
        # only now is the graph complete enough to finish the IR: the init rules
        # write links (firm ownership) that no population carried
        self._bind_graph()
        self._tick = 0
        if self._ir.consumed or self.duckdb_connection is not None:
            # a cold fit restarts the world, memory included: a fresh in-memory
            # database starts empty, and a file the user pointed us at is
            # cleared of this model's signals rather than silently continued.
            self.connection_ = connect(self.duckdb_connection)
            self.connection_.execute(f"DELETE FROM {HISTORY_TABLE}")
            self._attach_history()
            self._observe(t=0)

    def fit_iter(self, X=None) -> Iterator[dict]:
        """Take the world, apply init rules, then yield one ``{signal: level}``
        row after each of the ``n_periods`` ticks, ``t`` included.

        *X* is a maplib ``Model`` — advanced in place, so whatever it already
        holds is the opening state.  ``None`` continues ``model_`` under
        ``warm_start``, and is an empty world otherwise.

        The row is ``observables_``, settled at fit time and so the same width
        every tick: a whole run is ``pl.DataFrame(sim.fit_iter(...))``.  For
        per-agent state call ``sim.extract()`` in the loop body — ``model_`` is
        current at each yield, and a loop that does not need the frame pays
        nothing.
        """
        if X is not None and not isinstance(X, Model):
            raise TypeError(
                f"X is a maplib Model, not {type(X).__name__}: map populations into "
                "one with skabm.ottr and Model.map, then pass the model."
            )
        if self.random_seed is not None:
            pr.set_random_seed(self.random_seed)
        merged = self.params or {}
        init_rules = [render(rule, merged) for rule in self.init_rules]
        update_rules = [render(rule, merged) for rule in self.update_rules]
        infer_rules: list[str] | None = None
        if self.infer is not None:
            infer_rules = [render(rule, merged) for rule in self.infer]
        self._ir = analyse(
            (rule_name(rule, i), text)
            for i, (rule, text) in enumerate(
                zip(
                    (*self.init_rules, *self.update_rules, *(self.infer or ())),
                    (*init_rules, *update_rules, *(infer_rules or ())),
                )
            )
        )
        if self.warm_start:
            if X is not None:
                self.model_ = X
            self._register_udfs()
            self._register_udfs(self.meta_)
            self._bind_graph()
            if (self._ir.consumed or self.duckdb_connection is not None) and (
                self.connection_ is None
            ):
                # a hand-built or externally-assigned model_ gains the history
                # machinery here, continuing from whatever the database holds.
                self.connection_ = connect(self.duckdb_connection)
                # continue the tick count from a persisted run; empty means 0
                last = self.connection_.execute(
                    f"SELECT MAX(t) FROM {HISTORY_TABLE}"
                ).fetchone()[0]
                self._tick = -1 if last is None else int(last)
                self._attach_history()
                self._tick += 1
                self._observe(t=self._tick)
        else:
            self._cold_start(X, init_rules, update_rules, infer_rules)
        for _ in range(self.n_periods):
            for rule in update_rules:
                self.model_.update(rule)
            if self.infer is not None:
                self.model_.infer(infer_rules)  # type: ignore[arg-type]
            self._tick += 1
            yield self._observe(t=self._tick)

    def _attach_history(self) -> None:
        """Virtualize the recorded history onto ``meta_``, over consumed signals.

        Only the signals the rules actually read back are exposed — everything
        else recorded is telemetry, and virtualizing it would make each history
        rule (SAC re-estimates every signal it can see) do work no behaviour
        reads.  A rule set that consumes nothing gets no virtualization, and
        ``_observe`` then records without deriving.
        """
        consumed = sorted(signal_name(*s) for s in self._ir.consumed)
        if not consumed:
            return
        attach_history(self.meta_, self.connection_, consumed)
        self.virtualized_ = True

    def _register_udfs(self, model: Model | None = None) -> None:
        """Install every UDF registrar in ``self.udfs`` on the current model_.

        Applied to ``model_`` by default, and to ``meta_`` once it exists — a
        history rule is evaluated against ``meta_``, so any UDF it calls has to
        be registered there too.  That is the seam a researcher's own
        forecasting or reinforcement-learning routine arrives through: an
        ordinary ``Model.add_udf`` callable, named in their SPARQL like any
        other function.
        """
        for register in self.udfs:
            register(model if model is not None else self.model_)

    def _observe(self, t: int) -> dict:
        """Measure this period, persist it if there is a database, learn from it.

        The measurement is the return value — the row ``fit_iter`` yields.  When
        a history is open the row is appended to it, and each ``history_rules``
        query reads the accumulated series back through the virtualization, its
        result columns upserted into ``model_`` for the next tick's update rules.
        That write-back is skipped while the table is empty: maplib panics
        resolving a UDF projection over a zero-row frame.
        """
        row = measure(self.model_, self.observables_)
        if self.connection_ is not None:
            record(self.connection_, row, t)
            if (
                self.virtualized_
                and self.connection_.execute(
                    f"SELECT COUNT(*) FROM {HISTORY_TABLE}"
                ).fetchone()[0]
            ):
                apply_history_rules(
                    self.model_,
                    self.meta_,
                    [render(rule, self.params or {}) for rule in self.history_rules],
                )
        return {"t": t, **row}

    def _bind_graph(self) -> None:
        """Finish the IR against the mapped graph and settle what gets recorded.

        Split out because it has to happen after the populations are in — the
        graph is what distinguishes a quantity from a link, and what reveals a
        predicate no rule mentions — but before the history is opened, which
        needs the final observable list.
        """
        self._ir = self._ir.with_graph(self.model_)
        self.observables_ = self._recorded()

    def _classes(self) -> set:
        """The ``ex:`` agent classes the graph holds, whoever put them there."""
        return {
            iri.strip("<>")[len(EX_NS) :]
            for iri in self.model_.query("SELECT DISTINCT ?c WHERE { ?s a ?c }")["c"]
            if iri.strip("<>").startswith(EX_NS)
        }

    def _recorded(self) -> tuple:
        """Which observables this run measures, name-sorted.

        Sorting is the ``(N, D)`` contract a calibrator reads off the yielded
        rows: column *j* means the same thing on every tick and across every
        candidate.  Classes the graph does not have are dropped — the default
        rule set implies a central bank whether or not one was passed, and a
        column of nulls is not a measurement.
        """
        present = self._classes()
        observables = sorted(
            (o for o in self._ir.observables() if o.klass in present),
            key=lambda o: o.signal,
        )
        if self.track:
            return tuple(observables)
        required = self._ir.consumed
        return tuple(o for o in observables if (o.agg, o.klass, o.name) in required)

    def extract(self) -> pl.DataFrame:
        """Per-agent state: ``state_extract`` if one was given, else the IR's.

        The IR-derived extract projects exactly the predicates the rules touch
        plus whatever the populations carried, one UNION branch per class — the
        same frame the hand-written extracts used to build, without the writing.
        A model whose state hangs off untyped link targets (the Schelling
        lattice reaches ``def:x`` only through ``def:location``) still wants its
        own, which is what the parameter is for.
        """
        if self.state_extract is not None:
            return self.state_extract(self.model_)
        return self._ir.extract(self.model_)

    def history(self) -> pl.DataFrame:
        """Every persisted observable, long — ``signal``, ``t``, ``level``.

        The sidecar: the same series ``fit_iter`` yielded, after a database also
        kept it.
        """
        if self.connection_ is None:
            raise RuntimeError(
                "no history was opened: this rule set reads no ex:sig__ signal "
                "and no duckdb_connection was given, so nothing was persisted. "
                "The observables are yielded by fit_iter; pass duckdb_connection "
                "(needs the 'history' extra) to keep them in a database too."
            )
        return state_frame(self.connection_)

    def fit(self, X=None) -> RDFSimulator:
        """Take the world, apply init rules, run all ticks; return self."""
        for _ in self.fit_iter(X):
            pass
        return self
