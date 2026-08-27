"""
State history: the graph's memory, and the sidecar that persists it.

``measure`` is the primary path and needs no database — one row of aggregates read
out of the model, which ``RDFSimulator.fit_iter`` yields.  A database opens for the
other two reasons: a rule that reads its own past, and a caller who wants the series
to outlive the process.

A SPARQL update rule sees one time slice, so a behaviour that is a function of the
model's own past has nowhere in-graph to express it.  Named aggregates are appended to
a DuckDB table each tick and exposed back to SPARQL through maplib's chrontext
virtualization, one ``(signal, t, level)`` row per aggregate per tick.  A history rule
is a SELECT whose first projected variable is a subject IRI and whose remaining
variables become ``def:`` predicates on it; those are upserted into the simulation
graph, which is how history reaches the update rules that need it.

Three properties of the virtualization shape all of the above, and all three fail
*silently*:

1. Only ``query`` (SELECT) federates.  CONSTRUCT panics; ``insert`` and ``update`` see
   virtualized data as empty and quietly no-op.  Behaviour lives in ``update`` rules,
   which is why history is read by SELECT and written back as triples.
2. ``add_virtualization`` makes every graph-local aggregate on that model return
   ``None``, nested sub-SELECTs included — and most behaviour rules are built out of
   those.  Hence ``attach_history`` puts it on ``meta_``, never on ``model_``.
3. Rows reach a UDF in database order, not time order, so a rule that cares must say
   ``ORDER BY ?ext ?t`` in a sub-SELECT for chrontext to push down.
"""

from __future__ import annotations

from typing import Sequence

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

# A signal's key is (aggregate, agent class, predicate) — what to measure and
# over whom.  The class is part of it because several kinds share a predicate
# (firms, households and government all carry a price), so a predicate alone
# would aggregate the wrong population.  ``skabm.ir.Observable.signal`` builds
# the same string from the other direction.
signal_name = "sig__{}__{}__{}".format


def signal_iri(*signal: str) -> str:
    """Full IRI of a signal node; ``ex:`` + the same name used as its DB key."""
    return EX_NS + signal_name(*signal)


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


def attach_history(meta, con, signals: Sequence[str]) -> None:
    """Register the virtualized history onto the simulator's sidecar model.

    Adds the signal nodes and their chrontext intermediate nodes — without
    ``ct:hasExternalId`` (matching the SQL ``id`` column) and ``ct:hasResource``
    (matching the ``resource_sql_map`` key) chrontext returns zero rows and no
    error — then registers the virtualization itself.

    *meta* must not be the simulation graph.  Registering a virtualization
    silently nulls every graph-local aggregate on the model that carries it
    (see the module docstring), and the behaviour rules are built out of those
    aggregates.  That constraint is the whole reason ``RDFSimulator`` keeps a
    separate ``meta_`` model, and it is verified in ``tests/test_history.py``.
    """
    from maplib import (
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
    iris = [EX_NS + name for name in signals]
    ts_nodes = [iri + "__ts" for iri in iris]

    meta.map_triples(
        pl.DataFrame({"subject": iris, "object": [EX_NS + "Signal"] * len(iris)}),
        predicate=RDF_TYPE,
    )
    for subjects, predicate, objects in (
        (iris, "hasTimeseries", ts_nodes),
        (ts_nodes, "hasExternalId", list(signals)),
        (ts_nodes, "hasResource", [RESOURCE] * len(iris)),
    ):
        meta.map_triples(
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
    meta.add_virtualization(
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


def measure(model, observables: Sequence) -> dict:
    """Every observable, measured out of *model* — one row of the run, wide.

    One SPARQL query **per agent class**, not per signal: ``ir.ModelIR`` batches
    a class's aggregates into one projection, which is what makes it affordable
    to measure everything a rule set implies rather than only what it reads
    back.  The result frame's columns are already the signal keys.

    A predicate no agent carries comes back ``None`` rather than missing, so the
    row is the same width every tick and a run stacks rectangular.
    """
    from skabm.ir import measure_queries

    measured: dict = {}
    for query in measure_queries(observables).values():
        frame = model.query(query)
        if frame.height:
            measured.update(zip(frame.columns, frame.row(0)))
    return {o.signal: measured.get(o.signal) for o in observables}


def record(con, row: dict, t: int) -> None:
    """Append the measured half of *row* to the table at *t*, long."""
    rows = [
        (signal, int(t), float(level))
        for signal, level in row.items()
        if level is not None
    ]
    if rows:
        con.executemany(f"INSERT INTO {TABLE} VALUES (?, ?, ?)", rows)


def apply_history_rules(model, meta, rules: Sequence[str]) -> None:
    """Run each history rule against *meta* and upsert its results into *model*.

    A rule is a SELECT whose first projected variable is a subject IRI and whose
    remaining variables become ``def:`` predicates on it, exactly as
    ``rules.map_population`` turns a population's columns into predicates.  Each written
    predicate is cleared first, so the graph carries one current value however
    many ticks have run, and null results are dropped rather than written — that
    is how a rule says "not enough history yet" and leaves the consuming
    behaviour rule to fall back to its own default.
    """
    for rule in rules:
        result = meta.query(rule)
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
