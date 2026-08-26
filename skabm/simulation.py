"""
RDFSimulator: advance a maplib knowledge-graph ABM with SPARQL update rules.

The sklearn contract, adapted to ABM: X is a set of keyword DataFrames
(one calibrated population per agent class)::

    sim = RDFSimulator(n_periods=12)          # Poledna rules by default
    sim.fit(Firm=firms, Household=households, CentralBank=central_bank)

``init_rules`` and ``update_rules`` are ``__init__`` parameters
(``string.Template`` SPARQL — serializable, so ``get_params`` / ``clone``
work), defaulting to the full Poledna rule sets
(``behaviour.params.poledna_params``).  Rule *logic* lives in the templates;
rule *parameters* live in the ``params`` dict, merged over the canonical
Poledna values and substituted into the templates' ``$placeholders`` at fit
time — overriding one number (``params={"total_deposits": 2.5e4}``) never
means re-writing a rule.  The defaults self-scope to the agent kinds actually
passed: at fit time, rules whose referenced classes (``ex:Firm``,
``ex:CentralBank``, ...) are all absent from the populations are filtered out
entirely, so a use-case with only ``Firm=`` and ``Household=`` gets exactly
the firm and household dynamics.

``fit_iter`` is the generator variant of ``fit``: it yields the raw
per-agent state after each tick — the columns derived from the rules
themselves (``skabm.ir``), so no model family has to hand-write an extract.

Lifecycle: an empty maplib ``Model`` is created at ``__init__`` and exposed
as ``model_`` — the *fitted artifact*.  A cold ``fit``/``fit_iter`` rebuilds
it from scratch, maps each population with ``rules.map_df``, applies
``init_rules`` (necessarily *after* mapping), then advances ``n_periods``
ticks of ``update_rules`` upserts.  Population keywords not referenced by any
rule are mapped but trigger a ``UserWarning``.

Rules that depend on the model's own past declare it by naming an
``ex:sig__<agg>__<Class>__<predicate>`` signal (see ``skabm.history``).  When
any rule does, the fit also opens a DuckDB history and re-runs ``history_rules``
over the accumulated series — SPARQL SELECTs whose result columns are upserted
back into ``model_`` as ``def:`` predicates.  That is the whole memory
mechanism, and it is agnostic about what the rules compute: Poledna's
SAC-learned expectations are one query plus one polars UDF
(``behaviour.learning``).  A rule set naming no signal opens no database.

*What* gets recorded is wider than what gets read back.  ``skabm.ir`` derives
every aggregate the rule set implies — each state predicate under the
aggregate its own rules apply to it, and the share of agents sitting at each
binding constraint the rules impose — and a run that opens a history records
all of them, batched one query per agent class.  ``observables_`` is that list
and ``history()`` the long frame of what they did — telemetry, in a shape any
numeric ecosystem takes (``polars.DataFrame.to_jax``, black-it's loss
functions).  Deciding that ``SUM(def:output)`` is worth measuring is derived;
naming it *GDP* is not, and neither is anything distributional.  Both stay with
the caller, computed in polars over ``history()`` or over the per-agent frame
``fit_iter`` yields each tick (see ``examples/gini.py``).

**Two models, on purpose.**  ``model_`` is the world: agents and nothing else.
``meta_`` is the sidecar the simulator keeps about the run — the rule IR, the
behaviour metadata, the signal nodes, the virtualized history, and any UDF a
history rule calls.  The split is forced rather than stylistic: registering a
virtualization silently nulls every graph-local aggregate on the model that
carries it, and the behaviour rules are built out of those aggregates.  It also
keeps ``?s ?p ?o`` over ``model_`` returning agents, so the simulation graph
stays portable.  The price is that maplib has no cross-graph join
(``GRAPH ?g { ... }`` is unimplemented, and a query is scoped to one graph), so
combining the two means two queries and a polars join.

``warm_start=True`` skips the rebuild/map/init phase and keeps ticking the
existing ``model_`` — possibly under *different* update rules, after a
do-calculus style intervention (``model_.update``), or on a model built by
hand.  Passing populations together with ``warm_start=True`` is an error.

The graph's content splits into **structure** (predicates no update rule
DELETEs/INSERTs: links, coefficients, classes — written at fit, immutable
during simulation) and **state** (predicates the update rules upsert: output,
price, wealth, ... — owned by the rules after t=0).  That partition is derived
rather than asserted — ``skabm.ir`` reads it off the rules — and it is what
decides which predicates ``extract()`` projects and which aggregates get
recorded.  Keep interventions on structure between passes.

``model_`` is a regular maplib model post-fit: SPARQL queries, interventions,
``explore()`` visualization, and serialization all work on it directly.

**The IR is machinery, not an interface.**  ``skabm.ir`` is deliberately not
exposed: it decides what ``extract()`` projects and what gets recorded, and
that is the whole of its job.  Three things are the frontend, in the order
anyone actually needs them — ``fit_iter()`` for statistics over per-agent
state, ``history()`` for telemetry a calibrator or JAX takes as-is, and
``model_`` itself for interventions and queries.  See
``notebooks/extraction.ipynb``.

TODO: numerical backends — compile the graph to polars frames
(``Model.query`` -> ``pl.DataFrame`` -> polars expressions, or ``.to_jax()``
for differentiable kernels), step in frame-land, and re-map at observation
points.  The SPARQL path below then becomes the slow, semantically
transparent reference implementation the fast kernels are validated against.
"""

from __future__ import annotations

import warnings
from string import Template
from typing import Callable, Iterator, Sequence

import polars as pl
import polars_random as pr
from maplib import Model
from sklearn.base import BaseEstimator

from skabm.history import TABLE as HISTORY_TABLE
from skabm.history import (
    apply_history_rules,
    attach_history,
    connect,
    record,
    signal_name,
    state_frame,
)
from skabm.ir import analyse, rule_name
from skabm.rules import EX_NS, map_df, register_polars_random, render
from skabm.behaviour.learning import register_sac, sac_learning
from skabm.behaviour.params import poledna_params as _BEHAVIOUR_PARAMS
from skabm.behaviour.household import (
    household_income_init,
    household_income,
    household_wealth_init,
    satisificing_consume,
)
from skabm.behaviour.firm import (
    firm_ownership,
    firm_produce,
    firm_price,
    firm_sales,
)
from skabm.behaviour.macro import (
    government_spend,
    centralbank_rate,
)

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
    ``RDFSimulator._fit_iter``.  Used directly by tests so the block is
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
    n_periods : int
        Number of ticks ``fit`` runs (and ``fit_iter`` yields).
    warm_start : bool
        When True, ``fit``/``fit_iter`` continue ticking the existing
        ``model_`` instead of rebuilding it — no populations may be passed.
    state_extract : callable | None
        ``Model -> pl.DataFrame``, called by ``fit_iter`` after each tick to
        yield raw per-agent state.  ``None`` (the default) uses the IR-derived
        extract: one UNION branch per agent class, projecting exactly the
        predicates the rules touch plus whatever the populations carried.  Pass
        one only when that is not enough — a model whose state hangs off untyped
        link targets, like ``schelling.state_extract`` reaching ``def:x``
        through ``def:location``.
    track : bool | None
        Whether to record the observables the IR derives.  ``None`` (the
        default) records everything when a history is opened anyway — a rule set
        reading an ``ex:sig__`` signal pays for DuckDB already, so the rest of
        the observables cost one extra query per class per tick — and nothing
        otherwise, which keeps rule sets like Schelling on the core install.
        ``True`` forces the history open (needs the ``history`` extra); ``False``
        records only the signals the rules read back.
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
        The measurements actually recorded this run, derived from the rules and
        filtered by the ``track`` policy.  Their names are the signal column of
        ``history()``.
    meta_ : maplib.Model
        The sidecar: rule IR triples, behaviour metadata, signal nodes, the
        virtualized history, and the UDFs a history rule calls.  Never holds
        agents — see "Two models, on purpose" above.
    connection_ : duckdb.DuckDBPyConnection | None
        The DuckDB connection holding the realized series behind the learned
        expectations — one row per tracked signal per tick, virtualized into
        ``model_`` so SPARQL SELECT queries span graph and history at once.
        ``None`` when no rule reads an expectation.  ``history()`` returns the
        whole table as polars.
    """

    def __init__(
        self,
        init_rules: Sequence[Template | str] = DEFAULT_INIT_RULES,
        update_rules: Sequence[Template | str] = DEFAULT_UPDATE_RULES,
        infer: Sequence[Template | str] | None = None,
        params: dict = _BEHAVIOUR_PARAMS,
        n_periods: int = 12,
        warm_start: bool = False,
        state_extract: Callable[[Model], pl.DataFrame] | None = None,
        track: bool | None = None,
        udfs: Sequence[Callable[[Model], None]] = (
            register_polars_random,
            register_sac,
        ),
        random_seed: int | None = None,
        history_rules: Sequence[Template | str] = DEFAULT_HISTORY_RULES,
        duckdb_connection: "str | object | None" = None,
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
        self.observables_ = ()

    def _cold_start(
        self,
        populations: dict[str, pl.DataFrame],
        init_rules: list[str],
        update_rules: list[str],
        infer_rules: list[str] | None,
    ) -> None:
        """Rebuild model_, map populations, inject provenance, apply init rules.

        Runs only on a cold ``fit``/``fit_iter`` (warm_start=False).  The tick
        loop in ``_fit_iter`` is shared by both paths.

        When the rules read expectations, this also opens the DuckDB history,
        virtualizes it into the fresh model, seeds each signal's prior, and
        records the t=0 state — so the first tick already has a base period to
        measure growth against and a forecast to read.
        """
        rules_text = "\n".join((*init_rules, *update_rules, *(infer_rules or ())))
        for kind in populations:
            if f"ex:{kind}" not in rules_text:
                warnings.warn(
                    f"population {kind!r} is not referenced by any init/update/"
                    "infer rule (no 'ex:' + kind pattern): it will be mapped into the "
                    "model but stay inert during simulation.",
                    UserWarning,
                    stacklevel=3,
                )
        self.model_ = Model()
        self.meta_ = Model()
        self.virtualized_ = False
        self._register_udfs()
        self._register_udfs(self.meta_)
        for kind, df in populations.items():
            map_df(self.model_, df, kind)
        all_rules = (*self.init_rules, *self.update_rules, *(self.infer or ()))
        _inject_metadata(self.meta_, all_rules)
        for rule in init_rules:
            self.model_.insert(rule)
        # only now is the graph complete enough to finish the IR: the init rules
        # write links (firm ownership) that no population carried
        self._bind_graph()
        if self.observables_:
            # a cold fit restarts the world, memory included: a fresh in-memory
            # database starts empty, and a file the user pointed us at is
            # cleared of this model's signals rather than silently continued.
            self.connection_ = connect(self.duckdb_connection)
            self.connection_.execute(f"DELETE FROM {HISTORY_TABLE}")
            self._attach_history()
            self._observe(t=0)

    def _fit_iter(self, **populations: pl.DataFrame) -> Iterator[None]:
        """Advance the model one tick per iteration (no extraction)."""
        if self.random_seed is not None:
            pr.set_random_seed(self.random_seed)
        merged = {**_BEHAVIOUR_PARAMS, **self.params}
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
            if populations:
                raise ValueError(
                    "warm_start=True continues the existing model_; do not pass "
                    "populations (assign model_ directly instead)."
                )
            self._register_udfs()
            self._register_udfs(self.meta_)
            self._bind_graph()
            if self.observables_ and self.connection_ is None:
                # a hand-built or externally-assigned model_ gains the history
                # machinery here, continuing from whatever the database holds.
                self.connection_ = connect(self.duckdb_connection)
                self._attach_history()
                self._observe(t=self._next_tick())
        else:
            self._cold_start(populations, init_rules, update_rules, infer_rules)
        for _ in range(self.n_periods):
            for rule in update_rules:
                self.model_.update(rule)
            if self.infer is not None:
                self.model_.infer(infer_rules)  # type: ignore[arg-type]
            if self.observables_:
                # record what the tick produced, then re-run the history rules
                # over everything so far, so the next tick's update rules read
                # values derived from every period to date and not just this one
                self._observe(t=self._next_tick())
            yield

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

    def _next_tick(self) -> int:
        """One past the last tick in the history — where the next row belongs.

        Read from the table rather than counted in Python so a warm start
        continues a run (or a persisted file) at the right period instead of
        overwriting its opening ticks.
        """
        last = self.connection_.execute(
            f"SELECT MAX(t) FROM {HISTORY_TABLE}"
        ).fetchone()[0]
        return 0 if last is None else int(last) + 1

    def _observe(self, t: int) -> None:
        """Record this period, then re-run the history rules over everything so far.

        The two halves of one step, and the only place the simulator touches the
        history: ``record`` writes what actually happened into DuckDB, then each
        ``history_rules`` query reads the accumulated series back through the
        virtualization and its result columns are upserted into ``model_`` for
        the next tick's update rules to read.

        Skipped while the table is still empty — there is nothing to derive from
        no observations, and maplib panics resolving a UDF projection over a
        zero-row frame.
        """
        record(self.connection_, self.model_, self.observables_, t)
        if (
            self.virtualized_
            and self.connection_.execute(
                f"SELECT COUNT(*) FROM {HISTORY_TABLE}"
            ).fetchone()[0]
        ):
            apply_history_rules(
                self.model_,
                self.meta_,
                [
                    render(rule, {**_BEHAVIOUR_PARAMS, **self.params})
                    for rule in self.history_rules
                ],
            )

    def _bind_graph(self) -> None:
        """Finish the IR against the mapped graph and settle what gets recorded.

        Split out because it has to happen after the populations are in — the
        graph is what distinguishes a quantity from a link, and what reveals a
        predicate no rule mentions — but before the history is opened, which
        needs the final observable list.
        """
        self._ir = self._ir.with_graph(self.model_)
        self.observables_ = self._recorded()

    def _recorded(self) -> tuple:
        """Which observables this run records — the ``track`` policy, applied.

        ``track=None`` (the default) records everything the IR implies *if* a
        database is opened anyway, and nothing otherwise: a rule set that reads
        no expectation stays on the core install, and one that already pays for
        DuckDB gets the full observable set for the price of one extra query per
        class per tick.  ``track=True`` forces the database open; ``track=False``
        records only the signals the rules actually read back.
        """
        if self.track is False or not (self._ir.consumed or self.track):
            required = self._ir.consumed
            return tuple(
                o
                for o in self._ir.observables()
                if (o.agg, o.klass, o.name) in required
            )
        return self._ir.observables()

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
        """Every recorded observable, long — ``signal``, ``t``, ``level``."""
        if self.connection_ is None:
            raise RuntimeError(
                "nothing was recorded: this rule set reads no ex:sig__ signal, so "
                "no history was opened. Pass track=True (needs the 'history' extra) "
                "to record the observables the IR derives."
            )
        return state_frame(self.connection_)

    def fit_iter(self, **populations: pl.DataFrame) -> Iterator[pl.DataFrame]:
        """Map the populations, apply init rules, then yield per-agent state
        (``self.state_extract``) after each of the ``n_periods`` ticks.

        Keyword names are agent classes (``Firm=...``, ``Household=...``);
        each value is the population DataFrame mapped via ``rules.map_df``.
        """
        for _ in self._fit_iter(**populations):
            yield self.extract()

    def fit(self, **populations: pl.DataFrame) -> "RDFSimulator":
        """Map the populations, apply init rules, run all ticks; return self."""
        for _ in self._fit_iter(**populations):
            pass
        return self
