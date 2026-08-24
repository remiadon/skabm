"""
State history: record aggregates to DuckDB, read them back through SPARQL.

A SPARQL update rule sees one time slice — the present.  Any behaviour that is
a function of the model's own past (Poledna's expectation equations 6 and 9;
``behaviour.macro.centralbank_rate``'s hand-rolled ``def:prev_output`` lag) has
no way to express that in-graph.  This module gives the graph a memory: named
aggregates are appended to a DuckDB table each tick and exposed back to SPARQL
through maplib's chrontext virtualization, following the chrontext tutorial's
shape — a ``StateDB`` wrapper with a ``query`` method, a ``resource_sql_map`` of
SQLAlchemy ``Select``s, and one OTTR template per resource.

    signal                     t   level
    sig__SUM__Firm__output     0   4500.0
    sig__SUM__Firm__output     1   4514.3
    sig__AVG__Firm__price      0   1.0

**Nothing here knows what the history is used for.**  Recording is driven by
``ex:sig__<agg>__<Class>__<predicate>`` IRIs found in the rule text, and what
gets computed from the recorded series is an ordinary SPARQL SELECT the caller
supplies (``RDFSimulator(history_rules=...)``), with any polars reduction it
needs registered as a UDF through the existing ``udfs`` seam.  Sample-
autocorrelation learning is one such pair of a query and a UDF and lives in
``behaviour.learning``; a moving average, a volatility estimate or a
reinforcement-learning update would be another, with no change to this file.

A history rule is a SELECT whose **first projected variable is a subject IRI**
and whose remaining variables become ``def:`` predicates on it — the same
column-to-predicate convention ``rules.map_df`` uses for populations.
``apply_history_rules`` upserts them into the simulation graph, which is how a
value derived from history reaches the update rules that need it.

Two properties of maplib's virtualization shape all of the above, and both fail
silently, so they are worth stating precisely.

**1. Only ``query`` federates.**

===============================  ==========================================
``Model.query`` (SELECT)         federates — SQL is pushed down, UDFs run
``Model.query`` (CONSTRUCT)      panics: "not implemented: Not supported by
                                 chrontext"
``Model.insert`` (CONSTRUCT)     no pushdown, inserts nothing, no error
``Model.update`` (DELETE/INSERT) no pushdown, matches nothing, no error
===============================  ==========================================

An ABM's behaviour lives in ``update`` rules, so a behaviour rule cannot read
the history itself — its WHERE would quietly match zero rows and the tick would
become a no-op.  Hence the write-back: history rules run as SELECTs and their
results are materialised as triples the update rules read normally.

**2. Registering a virtualization breaks graph-local aggregates on that model.**
After ``add_virtualization``, ``SELECT (SUM(?x) AS ?s) WHERE { ?a a ex:Firm ;
def:output ?x }`` returns ``None`` where it returned 4500.0 before — including
aggregates nested in sub-SELECTs, which ``behaviour.firm.firm_sales`` (four of
them), ``macro.centralbank_rate`` and ``household.household_wealth_init`` are
built out of.  So the virtualization never touches the simulation graph:
``build_learner`` puts it on a small dedicated ``Model`` holding only the signal
nodes, and ``model_`` keeps working aggregates.

Rows reach a UDF in database order, *not* time order, so a history rule that
cares about sequence must say so in SPARQL — ``ORDER BY ?ext ?t`` inside a
sub-SELECT, which chrontext pushes down.  See ``behaviour.learning``.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

import polars as pl

from skabm.rules import DEF_NS, EX_NS, _PREFIXES

CT_NS = "https://github.com/DataTreehouse/chrontext#"

# The DuckDB table and the chrontext resource it is exposed as.  Namespaced so
# pointing the simulator at a database holding the user's own tables is safe.
TABLE = "skabm_state"
RESOURCE = "state"

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

# chrontext indexes on xsd:dateTime, so the integer tick is projected onto one
# day per tick from this epoch; only the ordering carries meaning.
EPOCH = "2000-01-01"

# A signal is (aggregate, agent class, predicate) — what to measure and over
# whom.  The class is part of the key because several kinds share a predicate
# (firms, households and government all carry a price), so a predicate alone
# would aggregate the wrong population.
Signal = tuple[str, str, str]

signal_name = "sig__{}__{}__{}".format
_SIGNAL_RE = re.compile(r"ex:sig__([A-Za-z]+)__([A-Za-z][A-Za-z0-9]*)__([A-Za-z_]\w*)")


def signal_iri(*signal: str) -> str:
    """Full IRI of a signal node; ``ex:`` + the same name used as its DB key."""
    return EX_NS + signal_name(*signal)


def parse_signals(rule_texts: Iterable[str]) -> list[Signal]:
    """The distinct signals a rule set records, read off its ``ex:sig__`` IRIs.

    Purely a function of the rule text — no registry, no decorators — so a rule
    carries its own data requirements wherever it travels.  Sorted, so recording
    order (and therefore the tests) are deterministic.
    """
    return sorted({m.groups() for t in rule_texts for m in _SIGNAL_RE.finditer(t)})


def connect(connection: "str | object | None" = None):
    """Open (or adopt) the DuckDB connection holding the state history.

    ``None`` opens a fresh ``:memory:`` database — history lives as long as the
    simulator does.  A string is a file path, so a long run persists tick by
    tick and outlives the process.  An already-open connection is used as given
    and never closed here, which is how two simulators share one store.  The
    table is created if absent, so pointing at an existing file resumes it.
    """
    try:
        import duckdb
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on install
        raise ModuleNotFoundError(
            "state history needs the optional 'history' extra "
            "(pip install 'skabm[history]' — duckdb + sqlalchemy). It is only "
            "reached by a rule set declaring an ex:sig__ signal; rule sets "
            "without one never open a database."
        ) from exc

    con = (
        duckdb.connect(connection or ":memory:")
        if connection is None or isinstance(connection, str)
        else connection
    )
    con.execute("SET TimeZone = 'UTC'")
    con.execute(
        f"CREATE TABLE IF NOT EXISTS {TABLE} (signal VARCHAR, t INTEGER, level DOUBLE)"
    )
    return con


class StateDB:
    """The database wrapper chrontext calls: ``query(sql) -> pl.DataFrame``.

    Deliberately thin, per the chrontext tutorial — maplib pushes a SQL string
    down and expects Arrow back.  Held read-write, unlike the tutorial's
    read-only sensor archive, because the simulator appends to the same
    connection it reads through.
    """

    def __init__(self, con):
        self.con = con

    def query(self, sql: str) -> pl.DataFrame:
        return self.con.execute(sql).pl()


def build_learner(con, signals: Sequence[Signal]):
    """A dedicated ``Model`` carrying the virtualization and nothing else.

    Holds the signal nodes and their chrontext intermediate nodes — without
    ``ct:hasExternalId`` (matching the SQL ``id`` column) and ``ct:hasResource``
    (matching the ``resource_sql_map`` key) chrontext returns zero rows and no
    error.  History rules are queried against this model; the simulation graph
    never has a virtualization registered on it, for the reason in the module
    docstring (it would silently null every aggregate the behaviour rules use).
    """
    from maplib import (
        Model,
        Parameter,
        Prefix,
        RDFType,
        Template,
        Triple,
        Variable,
        VirtualizedDatabase,
        xsd,
    )
    from sqlalchemy import Column, MetaData, Table, literal_column, select

    ct = Prefix(CT_NS)
    iris = [signal_iri(*s) for s in signals]
    ts_nodes = [iri + "__ts" for iri in iris]

    learner = Model()
    learner.map_triples(
        pl.DataFrame({"subject": iris, "object": [EX_NS + "Signal"] * len(iris)}),
        predicate=RDF_TYPE,
    )
    for subjects, predicate, objects in (
        (iris, "hasTimeseries", ts_nodes),
        (ts_nodes, "hasExternalId", [signal_name(*s) for s in signals]),
        (ts_nodes, "hasResource", [RESOURCE] * len(iris)),
    ):
        learner.map_triples(
            pl.DataFrame({"subject": subjects, "object": objects}),
            predicate=CT_NS + predicate,
        )

    # chrontext requires the projection to produce id / timestamp / value.  The
    # tick is an integer in the table and a timestamp in the graph; DuckDB
    # spells that conversion INTERVAL, which no SQLAlchemy dialect emits, so it
    # goes through literal_column as raw SQL — the same escape hatch the
    # tutorial uses for DuckDB's || concatenation.
    state = Table(TABLE, MetaData(), Column("signal"), Column("t"), Column("level"))
    resource_sql = (
        select(state.c.level.label("value"))
        .select_from(state)
        .add_columns(
            literal_column(f"{TABLE}.signal").label("id"),
            literal_column(f"(TIMESTAMP '{EPOCH}' + INTERVAL 1 DAY * {TABLE}.t)").label(
                "timestamp"
            ),
        )
    )
    # dp, the data-point node, appears in the instances but is not a parameter:
    # chrontext generates it internally.
    id_var, ts_var, value_var, dp_var = (
        Variable("id"),
        Variable("timestamp"),
        Variable("value"),
        Variable("dp"),
    )
    learner.add_virtualization(
        virtualized_database=VirtualizedDatabase(
            database=StateDB(con),
            resource_sql_map={RESOURCE: resource_sql},
            sql_dialect="postgres",  # closest SQLAlchemy dialect to DuckDB
        ),
        resources={
            RESOURCE: Template(
                iri=ct.suf("StateHistory"),
                parameters=[
                    Parameter(variable=id_var, rdf_type=RDFType.Literal(xsd.string)),
                    Parameter(variable=ts_var, rdf_type=RDFType.Literal(xsd.dateTime)),
                    Parameter(variable=value_var, rdf_type=RDFType.Literal(xsd.double)),
                ],
                instances=[
                    Triple(id_var, ct.suf("hasDataPoint"), dp_var),
                    Triple(dp_var, ct.suf("hasValue"), value_var),
                    Triple(dp_var, ct.suf("hasTimestamp"), ts_var),
                ],
            )
        },
    )
    return learner


def record(con, model, signals: Sequence[Signal], t: int) -> None:
    """Measure each signal out of *model* and append it to the table at tick *t*.

    One small SPARQL aggregate per signal, evaluated graph-locally against the
    tick's fresh state.  A signal whose class is absent from the populations
    (household income in a Firm-only run) aggregates to nothing and is skipped,
    so no row is written and the rules referencing it — which anchor on that
    same absent class — stay inert in step.
    """
    rows = []
    for agg, klass, predicate in signals:
        level = model.query(
            _PREFIXES + f"SELECT ({agg}(?x) AS ?level) "
            f"WHERE {{ ?a a ex:{klass} ; def:{predicate} ?x }}"
        )
        if level.height and level["level"][0] is not None:
            rows.append(
                (signal_name(agg, klass, predicate), int(t), float(level["level"][0]))
            )
    if rows:
        con.executemany(f"INSERT INTO {TABLE} VALUES (?, ?, ?)", rows)


def apply_history_rules(model, learner, rules: Sequence[str]) -> None:
    """Run each history rule against *learner* and upsert its results into *model*.

    A rule is a SELECT whose first projected variable is a subject IRI and whose
    remaining variables become ``def:`` predicates on it, exactly as
    ``rules.map_df`` turns a population's columns into predicates.  Each written
    predicate is cleared first, so the graph carries one current value however
    many ticks have run, and null results are dropped rather than written — that
    is how a rule says "not enough history yet" and leaves the consuming
    behaviour rule to fall back to its own default.
    """
    for rule in rules:
        result = learner.query(rule)
        if not result.height:
            continue
        subject, *columns = result.columns
        result = result.with_columns(pl.col(subject).str.strip_chars("<>"))
        for column in columns:
            model.update(
                _PREFIXES
                + f"DELETE {{ ?s def:{column} ?o }} WHERE {{ ?s def:{column} ?o }}"
            )
            values = result.select(subject, column).drop_nulls()
            if values.height:
                model.map_triples(
                    values.rename({subject: "subject", column: "object"}),
                    predicate=DEF_NS + column,
                )


def state_frame(con) -> pl.DataFrame:
    """The whole history as polars (``signal``, ``t``, ``level``) — realized series."""
    return con.execute(f"SELECT signal, t, level FROM {TABLE} ORDER BY signal, t").pl()
