"""
Sample-Autocorrelation Learning — Poledna's expectations, as a query and a UDF.

Poledna eq. 6 and 9 do not give agents a constant expected growth rate; they
make expectations a *regression on the model's own past*.  Hommes & Zhu (2014)
call the rule SAC learning: agents do not know the model, they perceive an
aggregate's growth rate as an AR(1) process and estimate its two parameters from
the realized sample —

``a``
    the sample mean (the growth rate's long-run level), and
``b``
    the first-order sample autocorrelation (its persistence),
    ``b = Sum (g_k - a)(g_{k-1} - a) / Sum (g_k - a)^2``, clamped into
    ``(-1, 1)`` so the perceived law of motion stays stationary —

giving the one-step forecast ``a + b * (g_last - a)``.

The whole implementation is the two things ``RDFSimulator`` already accepts:

``register_sac``
    a polars UDF, passed through the existing ``udfs=`` seam exactly like
    ``rules.register_polars_random`` (the Poledna default pairs the two); and
``sac_learning``
    a SPARQL SELECT, passed through ``history_rules=``, that runs the UDF over
    the virtualized history and projects ``?sig ?forecast`` — subject IRI first,
    remaining columns becoming ``def:`` predicates, per ``history``'s contract.

``skabm.history`` knows none of this.  Swapping SAC for a moving average, an
adaptive-expectations rule or a reinforcement-learning update means writing
another query and another UDF, and changing nothing else.

**Ordering is the query's job, not the UDF's.**  Rows reach a UDF in database
order, so ``sac_learning`` sorts inside a sub-SELECT (``ORDER BY ?ext ?t``),
which chrontext pushes down into SQL.  The UDF then trusts that order — it must,
since a UDF returns one value per input row and cannot reorder its output.
"""

from __future__ import annotations

from string import Template

import polars as pl

from skabm.history import CT_NS, signal_name
from skabm.rules import _PREFIXES

SAC_NS = "urn:sac:"


def sac_forecast(df: pl.DataFrame) -> pl.Series:
    """The SAC-learning reduction — polars UDF behind ``sac:forecast``.

    maplib calls this with one row per observation and three columns, in
    argument order: signal key, timestamp, level.  Everything is done with
    ``.over(key)`` window expressions rather than a group-by, because a UDF must
    return exactly one value per input row: the per-signal forecast is broadcast
    across that signal's rows and the caller collapses it with ``SAMPLE(...)``
    under a ``GROUP BY``.  Several signals therefore share one call without
    contaminating each other, which a plain column-wise reduction would not
    survive.

    Rows are assumed to arrive in ``(key, timestamp)`` order — ``sac_learning``
    guarantees it with an ``ORDER BY`` that chrontext pushes down to SQL, and
    sorting here instead would misalign the returned Series with maplib's input
    frame.

    A single growth observation has zero sample autocorrelation, so the formula
    degenerates to naive expectations — next period's growth is this period's —
    which is the right answer rather than a special case.  Only a signal with no
    growth observation at all (its opening tick) returns null, which
    ``history.apply_history_rules`` drops rather than writing.
    """
    key, _, level = df.columns[:3]
    growth = pl.col(level) / pl.col(level).shift(1).over(key) - 1.0
    deviation = pl.col("g") - pl.col("g").mean().over(key)
    stats = df.with_columns(g=growth).with_columns(
        mean=pl.col("g").mean().over(key),
        last=pl.col("g").last().over(key),
        num=(deviation * deviation.shift(1)).sum().over(key),
        den=(deviation * deviation).sum().over(key),
        n=pl.col("g").count().over(key),
    )
    b = (
        pl.when(pl.col("den") > 0)
        .then(pl.col("num") / pl.col("den"))
        .otherwise(0.0)
        .clip(-0.999, 0.999)
    )
    return stats.select(
        out=pl.when(pl.col("n") >= 1)
        .then(pl.col("mean") + b * (pl.col("last") - pl.col("mean")))
        .otherwise(None)
        .cast(pl.Float64)
    )["out"].alias("out")


def register_sac(model) -> None:
    """Expose SAC learning to SPARQL as ``sac:forecast(?key, ?timestamp, ?level)``.

    Three arguments and not four — the prior belongs here conceptually, but a
    second literal-valued argument collides inside maplib's UDF projection
    ("projections contained duplicate output name 'literal'", raised as a Rust
    panic).  The prior stays in the behaviour rule instead, via ``expect``.
    """
    from maplib import xsd

    model.add_udf(
        SAC_NS + "forecast",
        sac_forecast,
        xsd.double,
        [xsd.string, xsd.dateTime, xsd.double],
    )


def expect(agg: str, klass: str, predicate: str, out: str) -> str:
    """A SPARQL fragment binding ``?<out>`` to a learned expectation.

    Splice it into an update rule's WHERE clause::

        ?f a ex:Firm ; def:output ?y0 .
        + expect("SUM", "Firm", "output", out="g_e") +
        BIND(?y0 * (1e0 + ?g_e) AS ?y1)

    and ``?g_e`` is the one-step-ahead forecast of the growth rate of
    ``SUM(def:output)`` over ``ex:Firm``, re-estimated from the model's own
    history after every tick.

    The ``ex:sig__<agg>__<Class>__<predicate>`` IRI is the detection hook:
    ``history.parse_signals`` finds it in the rendered rule text and the
    simulator starts recording that aggregate.  Nothing is registered elsewhere.

    The ``OPTIONAL``/``BOUND`` pair covers exactly one tick.  A forecast exists
    from the first growth observation onward (one observation gives zero sample
    autocorrelation, hence naive expectations — last period's growth), so only
    the opening tick, which has a level but no change to measure, finds nothing.
    It defaults to ``0e0``: with no history there is no information, so no
    expected growth.  That is a structural zero, not a tunable prior — Poledna
    initialises from real series and never faces an empty history, so a
    calibration knob here would be modelling an artefact of starting cold.
    """
    return f"""
    OPTIONAL {{ ex:{signal_name(agg, klass, predicate)} def:forecast ?{out}__learned }}
    BIND(IF(BOUND(?{out}__learned), ?{out}__learned, 0e0) AS ?{out})
"""


# The history rule: one federated SELECT per tick.  The sub-SELECT's ORDER BY is
# load-bearing — it is pushed down into SQL and is what puts each signal's rows
# in time order before the UDF sees them.  SAMPLE collapses the per-row
# broadcast the UDF returns to one row per signal.
#
# Projects ?sig then ?forecast: subject IRI first, remaining columns become
# def: predicates (history.apply_history_rules), so this writes exactly the
# def:forecast triples ``expect`` reads.
sac_learning = Template(
    _PREFIXES
    + f"PREFIX ct:<{CT_NS}>\nPREFIX sac:<{SAC_NS}>\n"
    + """
SELECT ?sig (SAMPLE(?f) AS ?forecast)
WHERE {
    { SELECT ?sig ?ext ?t ?v
      WHERE {
          ?sig ct:hasTimeseries ?tsn .
          ?tsn ct:hasExternalId ?ext ;
               ct:hasDataPoint ?dp .
          ?dp ct:hasTimestamp ?t ;
              ct:hasValue ?v .
      }
      ORDER BY ?ext ?t }
    BIND(sac:forecast(?ext, ?t, ?v) AS ?f)
}
GROUP BY ?sig
"""
)
