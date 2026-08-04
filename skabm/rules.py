"""Graph construction and SPARQL rules for Poledna-style ABM simulation.

``map_df`` lifts a calibrated agent population (a polars DataFrame) into a
maplib model in one call: maplib's ``map_default`` auto-generates the OTTR
template (primary key ``id``, IRI-typed link columns, nullable attributes)
and applies it, and one ``map_triples`` call tags every agent with its
class.  There is no hand-written template registry: **predicates are column
names** under maplib's default namespace (``urn:maplib_default:``), classes
live under ``EX_NS``.  DataFrame conventions: an ``id`` column of full IRIs
(``EX_NS + "firm_0"``); link columns hold the target agent's IRI (or null).

Rules are ``string.Template`` objects: the SPARQL text carries the *logic*,
``$placeholders`` carry the *parameters*, and ``render(rule, params)``
substitutes them as xsd:double literals at fit time.  Default parameter
values live in ``POLEDNA_PARAMS`` (published Table 2, 2010:Q4); overriding
one number means passing ``params={"total_deposits": ...}`` to
``RDFSimulator``, never re-writing a rule.  Two kinds of rules:

* **Initialization rules** (``DEFAULT_INIT_RULES``) — CONSTRUCTs run once
  through ``Model.insert`` after the populations are mapped: initial
  conditions, not laws of motion.
* **Update rules** (``DEFAULT_UPDATE_RULES``) — run every tick through
  ``Model.update``.  Each is an *upsert*: ``DELETE`` the old value (matched
  through ``OPTIONAL`` where needed) then ``INSERT`` the new one — SPARQL
  INSERT alone has set semantics and would accumulate duplicate values.
  The tuple order is the paper's event sequence (Section 3.5).  A rule
  anchored on a class with no mapped agents matches nothing and no-ops, so
  the full default set self-scopes to the kinds actually passed.

``state_extract(model)`` returns per-agent state as a sparse wide frame
with no aggregation; summary logic belongs in polars expressions.

Deliberate simplifications versus the paper, where pure SPARQL hits its
limits (kept on purpose — this architecture is being stress-tested):

* **No search-and-matching** (paper Section 3.1.1): goods demand is
  allocated proportionally to supply shares instead of by random visiting;
  employment links are static.  Random sequential algorithms are not
  expressible declaratively.
* **SAC-learning expectations** (eq. 6/9): expected growth, inflation and
  household-income growth are *re-estimated in-graph* every tick by
  Sample-Autocorrelation Learning (Hommes & Zhu 2014).  A rule calls the
  estimator inline — ``sac("SUM", "Firm", "output", prior="growth_e", out="g_e")``
  splices a ``sac:forecast`` UDF (``register_sac``) over the signal's history —
  and ``Ontology.expand_rules`` auto-generates the measurement + ``PROCESS_INIT``
  / ``PROCESS_ACCOUNTING`` bookkeeping, so no rule set carries history plumbing
  by hand.  The drifts additionally carry ``pr:normal`` innovations
  (``$..._sigma`` params, off by default), so learned drift and stochastic shock
  coexist.
* **No log/exp**: AR(1) laws of motion in log-levels (eq. 51, 77, 81)
  become linear growth factors.

Numeric parameters are injected as xsd:double literals via ``dbl`` — never
decimals, which maplib 0.20.19 mixes into unstable multi-typed columns.
Unbound-variable handling uses explicit BOUND tests: maplib evaluates
comparisons on unbound variables as false instead of erroring, so a
COALESCE chain would bind the wrong arm.
"""

import re
from string import Template

import polars as pl
import polars_random as pr
from maplib import xsd

EX_NS = "http://example.net/skabm#"
DEF_NS = "urn:maplib_default:"
PR_NS = "urn:pr:"  # polars-random UDFs, registered by register_polars_random
SAC_NS = "urn:sac:"  # SAC-learning UDF, registered by register_sac

_PREFIXES = (
    f"PREFIX ex:<{EX_NS}>\n"
    f"PREFIX def:<{DEF_NS}>\n"
    f"PREFIX pr:<{PR_NS}>\n"
    f"PREFIX sac:<{SAC_NS}>\n"
    "PREFIX xsd:<http://www.w3.org/2001/XMLSchema#>\n"
)

_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


def register_polars_random(model) -> None:
    """Expose polars-random draws to SPARQL as UDFs (maplib >= 0.20.26).

    Registers ``pr:uniform(low, high)`` and ``pr:normal(mean, std)`` — callable
    in any rule via ``BIND(pr:uniform(0e0, 1e0) AS ?u)`` — giving the graph the
    seedable RAND plain SPARQL lacks.  Each UDF receives a DataFrame with one
    column per argument (``"0"``, ``"1"``) and returns the ``out`` Series.

    The draw uses polars-random's ``size=`` Series form (the only one that
    honours ``pr.set_random_seed``; the Series-argument form returns a
    non-reproducible expression) and affine-transforms it per row, so bounds
    may vary by row and a run is reproducible whenever the caller fixes the
    seed (see ``RDFSimulator(random_seed=...)``).
    """

    def _uniform(df: pl.DataFrame) -> pl.Series:
        u = pr.uniform(0.0, 1.0, size=len(df))
        return (df["0"] + (df["1"] - df["0"]) * u).alias("out")

    def _normal(df: pl.DataFrame) -> pl.Series:
        z = pr.normal(0.0, 1.0, size=len(df))
        return (df["0"] + df["1"] * z).alias("out")

    model.add_udf(PR_NS + "uniform", _uniform, xsd.double, [xsd.double, xsd.double])
    model.add_udf(PR_NS + "normal", _normal, xsd.double, [xsd.double, xsd.double])


def register_sac(model) -> None:
    """Expose SAC-learning (Sample Autocorrelation Learning) to SPARQL as a UDF.

    ``sac:forecast(?t, ?x)`` is the Hommes & Zhu (2014) behavioral-learning
    forecast the ``sac(...)`` inline helper calls: agents perceive an aggregate
    as an AR(1) process and learn its two parameters from the realized sample —

    * the **sample mean** ``a`` (the variable's long-run level), and
    * the **first-order sample autocorrelation** ``b`` (its persistence,
      ``b = Sum (x_t - a)(x_{t-1} - a) / Sum (x_t - a)^2``, clamped to
      ``(-1, 1)`` for stationarity) —

    returning the one-step forecast ``a + b * (x_last - a)``.

    Called with **one row per history observation** (arg 0 = tick, arg 1 =
    value): it reduces those rows to the scalar forecast and broadcasts it over
    every row, so the caller collapses it with ``SAMPLE(...)`` in a sub-SELECT.
    Fewer than two observations (or a constant series) has no autocorrelation to
    learn, so it falls back to the last value.
    """

    def _sac_forecast(df: pl.DataFrame) -> pl.Series:
        # columns arrive in argument order (tick, value); maplib names an
        # integer-sourced argument "literal" not "0", so index by position.
        tick, val = df.columns[0], df.columns[1]
        dev = pl.col(val) - pl.col(val).mean()
        s = (
            df.sort(tick)
            .select(
                mean=pl.col(val).mean(),
                last=pl.col(val).last(),
                num=(dev * dev.shift(1)).sum(),
                den=(dev * dev).sum(),
                n=pl.len(),
            )
            .row(0, named=True)
        )
        if s["n"] < 2 or not s["den"]:
            fc = s["last"]
        else:
            beta = max(-0.999, min(0.999, s["num"] / s["den"]))
            fc = s["mean"] + beta * (s["last"] - s["mean"])
        # explicit dtype so an empty input (maplib probes the UDF with zero
        # rows) still returns an f64 column rather than a Null-typed one.
        return pl.Series("out", [fc] * s["n"], dtype=pl.Float64)

    model.add_udf(
        SAC_NS + "forecast", _sac_forecast, xsd.double, [xsd.double, xsd.double]
    )


def sac(agg: str, klass: str, predicate: str, prior: str, out: str) -> str:
    """Inline SAC-learning forecast, for splicing into an update rule's WHERE.

    Drop ``+ sac("SUM", "Firm", "output", prior="growth_e", out="g_e") +`` into a
    rule and use ``?g_e``: it binds ``?<out>`` to the one-step SAC-learning
    forecast of the growth rate of ``<agg>`` over ``ex:<klass>``'s
    ``def:<predicate>`` (e.g. total firm output), falling back to the ``$<prior>``
    parameter until two observations exist.  The class scopes the aggregate —
    several agent kinds may share a predicate (firms, households and government
    all carry ``def:price``), so a predicate alone would measure the wrong
    population.

    The signal IRI it references — ``ex:sig__<agg>__<klass>__<predicate>`` — is
    the detection hook: ``Ontology.expand_rules`` scans rule text for it and tops
    up the graph with the matching measurement + history bookkeeping, so the
    end-user never writes ``PROCESS_ACCOUNTING`` / ``MEASURE`` / clocks / obs.

    ``xsd:double(?t)`` casts the integer tick (maplib passes an integer UDF
    argument as an all-null column); every internal variable is namespaced by
    ``out`` so several ``sac(...)`` calls (or the host rule's own variables)
    never collide.
    """
    sig = _sig(agg, klass, predicate)
    return f"""
        {{ SELECT (SAMPLE(?{out}_f) AS ?{out}_raw) (COUNT(?{out}_v) AS ?{out}_n) WHERE {{
            ?{out}_o def:of {sig} ; def:t ?{out}_t ; def:value ?{out}_v .
            BIND(sac:forecast(xsd:double(?{out}_t), ?{out}_v) AS ?{out}_f)
        }} }}
        BIND(IF(?{out}_n >= 2e0, ?{out}_raw, ${prior}) AS ?{out})
"""


def dbl(x: float) -> str:
    """Format a Python float as a SPARQL xsd:double literal."""
    return f"{x:.6e}"


def render(rule: "Template | str", params: dict) -> str:
    """Substitute a rule Template's $-placeholders with xsd:double literals.

    Numeric parameter values go through ``dbl`` so decimal literals can
    never leak into the SPARQL; plain-string rules pass through unchanged.
    Missing placeholders raise ``KeyError`` (loudly, at fit time).
    """
    if isinstance(rule, Template):
        return rule.substitute(
            {k: dbl(v) if isinstance(v, (int, float)) else v for k, v in params.items()}
        )
    return rule


def map_df(model, df: pl.DataFrame, kind: str) -> None:
    """Map an agent population into the model in one call.

    Users pass bare local names everywhere (``"firm_0"``): the ``id``
    column is prefixed with ``EX_NS``, and a *link* column — one whose bare
    values name agents already mapped into the model — is prefixed to the
    same nodes, so ``employer="firm_0"`` resolves to the firm.  Which
    columns are links is read from the graph, not declared: a value is a
    reference iff an agent with that id already exists.  (Map referenced
    populations first; e.g. firms before households.)

    ``map_default`` generates the template from the DataFrame schema (every
    non-``id`` column becomes a ``def:`` predicate; IRI-valued and nullable
    columns are detected) **and applies it to ``df`` in the same call** —
    its return value is the template document for inspection only, and a
    follow-up ``model.map`` would map the rows a second time.

    The generated template emits no class triple, and the class is not
    derivable from the schema (two populations may share identical
    columns), so every agent is tagged ``rdf:type ex:<kind>`` here — that
    is what lets rules anchor on ``?hh a ex:Household``.
    """
    df = df.with_columns(pl.format(EX_NS + "{}", pl.col("id")).alias("id"))
    known = {s.strip("<>") for s in model.query("SELECT ?s WHERE { ?s a ?c }")["s"]}
    for c in df.columns:
        if c == "id" or df.schema[c] != pl.String:
            continue
        vals = {EX_NS + v for v in df[c].drop_nulls().to_list()}
        if vals and vals <= known:
            df = df.with_columns((pl.lit(EX_NS) + pl.col(c)).alias(c))
    model.map_default(df, primary_key_column="id")
    model.map_triples(
        df.select(subject="id").with_columns(object=pl.lit(EX_NS + kind)),
        predicate=_RDF_TYPE,
    )


# ---------------------------------------------------------------------------
# Initialization rules — Model.insert, once, right after mapping
# ---------------------------------------------------------------------------

# A fraction $firm_ownership_ratio of households own firms (Poledna Section
# 3.2: investor households): firm j is owned by household floor(j / ratio),
# spreading owners uniformly across the household index range.  The
# assignment is deterministic by choice (a stable index map is easier to
# reason about than random matching; the pr:uniform UDF could randomise it if
# wanted).  Only firms without a data-defined owner are filled
# (FILTER NOT EXISTS), and users never declare `owns` themselves.  Relies on
# the skabm id convention ``EX_NS + "firm_<j>"`` / ``EX_NS + "hh_<i>"``.
FIRM_OWNERSHIP = Template(
    _PREFIXES
    + """
    CONSTRUCT { ?owner def:owns ?f }
    WHERE {
        ?f a ex:Firm .
        FILTER NOT EXISTS { ?anyone def:owns ?f }
        BIND(xsd:integer(STRAFTER(STR(?f), "#firm_")) AS ?j)
        BIND(xsd:integer(FLOOR(?j / $firm_ownership_ratio)) AS ?i)
        BIND(IRI(CONCAT(\""""
    + EX_NS
    + """hh_", STR(?i))) AS ?owner)
    }
    """
)

# Income by activity status (Poledna eq. 49): wage from the `employer` link;
# else dividend $dividend_ratio * max(0, profit) from the `owns` link; else
# unemployment benefits $benefit_replacement times the average wage.
HOUSEHOLD_INCOME = Template(
    _PREFIXES
    + """
    CONSTRUCT { ?hh def:income ?income }
    WHERE {
        { SELECT (AVG(?any_w) AS ?w_avg) WHERE { ?any_f def:w_bar ?any_w } }
        ?hh a ex:Household .
        OPTIONAL { ?hh def:employer ?f . ?f def:w_bar ?w . }
        OPTIONAL { ?hh def:owns ?g . ?g def:profit ?p . }
        BIND(
            IF(BOUND(?w), ?w,
            IF(BOUND(?p), $dividend_ratio * IF(?p > 0e0, ?p, 0e0),
            $benefit_replacement * ?w_avg)) AS ?income)
    }
    """
)

# Initial wealth D_h(0) = $total_deposits * Y_h(0) / sum Y_h(0)
# (Poledna Section 5.2); run after HOUSEHOLD_INCOME.
HOUSEHOLD_WEALTH = Template(
    _PREFIXES
    + """
    CONSTRUCT { ?hh def:wealth ?wealth }
    WHERE {
        { SELECT (SUM(?any_i) AS ?total) WHERE { ?any_hh def:income ?any_i } }
        ?hh def:income ?income .
        BIND($total_deposits * ?income / ?total AS ?wealth)
    }
    """
)


# ---------------------------------------------------------------------------
# SAC-learning bookkeeping — generated, not authored.  A
# ``sac(agg, klass, pred, ...)`` call in any update rule references
# ``ex:sig__<agg>__<klass>__<pred>``; ``Ontology.expand_rules`` finds those
# signals and tops the graph up with the rules below, so the end-user never
# writes history bookkeeping by hand.
#
# Each tracked signal is an ``ex:Signal`` node carrying ``def:level`` (this
# tick's aggregate) and ``def:prev`` (last tick's); observations accumulate as
# ``?o def:of ?sig ; def:t ?k ; def:value ?rate`` under a shared ``ex:clock``.
# ---------------------------------------------------------------------------

_SIGNAL_RE = re.compile(r"ex:sig__([A-Za-z]+)__([A-Za-z]+)__([A-Za-z_][A-Za-z0-9_]*)")


def parse_signal_specs(rule_texts: "list[str]") -> "list[tuple[str, str, str]]":
    """The distinct ``(agg, klass, predicate)`` signals a rule set forecasts with ``sac()``.

    Reads them straight off the ``ex:sig__<agg>__<klass>__<predicate>`` IRIs the
    ``sac()`` fragments embed — no registry, purely a function of the rule text.
    Sorted so generated rules (and tests) are deterministic.
    """
    specs = {m.groups() for text in rule_texts for m in _SIGNAL_RE.finditer(text)}
    return sorted(specs)


def _sig(agg: str, klass: str, predicate: str) -> str:
    return f"ex:sig__{agg}__{klass}__{predicate}"


def process_init(specs: "list[tuple[str, str, str]]") -> str:
    """The init rule seeding ``ex:clock`` and one ``ex:Signal`` per spec.

    Each signal's ``def:prev`` is primed with the t=0 aggregate (scoped to the
    signal's class) so the first tick already has a lag to measure against.  An
    aggregate over an absent population (e.g. income in a Firm-only run) is
    unbound, so CONSTRUCT skips that ``def:prev`` — the signal stays dormant and
    its ``sac()`` consumer falls back to the prior (pre-learning behavior).
    """
    decls, wheres = [], []
    for i, (agg, klass, pred) in enumerate(specs):
        decls.append(f"{_sig(agg, klass, pred)} a ex:Signal ; def:prev ?v{i} .")
        wheres.append(
            f"{{ SELECT ({agg}(?x{i}) AS ?v{i}) "
            f"WHERE {{ ?a{i} a ex:{klass} ; def:{pred} ?x{i} }} }}"
        )
    return (
        _PREFIXES
        + '\n    CONSTRUCT {\n        ex:clock a ex:Clock ; def:t "0"^^xsd:integer .\n        '
        + "\n        ".join(decls)
        + "\n    }\n    WHERE {\n        "
        + "\n        ".join(wheres)
        + "\n    }\n"
    )


def measure_rule(agg: str, klass: str, predicate: str) -> str:
    """Per-signal upsert materializing ``def:level`` = ``<agg>(ex:<klass> def:<predicate>)``.

    Scoped to the signal's class: several agent kinds may share a predicate, so
    the aggregate must name the class the ``sac()`` call meant.
    """
    sig = _sig(agg, klass, predicate)
    return (
        _PREFIXES
        + f"""
    DELETE {{ {sig} def:level ?old }}
    INSERT {{ {sig} def:level ?v }}
    WHERE {{
        {sig} a ex:Signal .
        OPTIONAL {{ {sig} def:level ?old }}
        {{ SELECT ({agg}(?x) AS ?v) WHERE {{ ?a a ex:{klass} ; def:{predicate} ?x }} }}
    }}
    """
    )


# Generic history accounting (no variable names): for every signal, turn this
# tick's level into a growth-rate observation, roll the lag forward, and bump
# the shared clock.  ``(?level / ?prev) - 1e0`` is parenthesised — maplib
# evaluates additive chains right-associatively (see maplib-gotchas memory).
PROCESS_ACCOUNTING = (
    _PREFIXES
    + """
    DELETE { ?sig def:prev ?prev . ex:clock def:t ?t . }
    INSERT {
        ?obs def:of ?sig ; def:t ?t ; def:value ?rate .
        ?sig def:prev ?level .
        ex:clock def:t ?tnext .
    }
    WHERE {
        ex:clock def:t ?t .
        ?sig a ex:Signal ; def:level ?level ; def:prev ?prev .
        BIND((?level / ?prev) - 1e0 AS ?rate)
        BIND(?t + "1"^^xsd:integer AS ?tnext)
        BIND(IRI(CONCAT(STR(?sig), "_obs_", STR(?t))) AS ?obs)
    }
    """
)


# ---------------------------------------------------------------------------
# Update rules — Model.update, every tick (upsert pattern)
# ---------------------------------------------------------------------------

# Supply choice (eq. 5) capped by labor capacity (eq. 12, Leontief):
# output <- min(output * (1 + g_e + eps), alpha * size), eps an AR(1)
# innovation eps ~ N(0, $growth_sigma) (eq. 6).  Expected growth ?g_e is
# SAC-learned in-graph from the realized SUM(output) history (eq. 6 behavioral
# learning), falling back to the $growth_e prior until two ticks exist.  The
# intermediate-input and capital legs of the Leontief nest are omitted.
FIRM_PRODUCTION = Template(
    _PREFIXES
    + """
    DELETE { ?f def:output ?y0 }
    INSERT { ?f def:output ?y1 }
    WHERE {
        ?f a ex:Firm ; def:output ?y0 ; def:alpha ?alpha ; def:size ?n ."""
    + sac("SUM", "Firm", "output", prior="growth_e", out="g_e")
    + """
        BIND(?y0 * (1e0 + ?g_e + pr:normal(0e0, $growth_sigma)) AS ?y_desired)
        BIND(?alpha * ?n AS ?y_capacity)
        BIND(IF(?y_desired < ?y_capacity, ?y_desired, ?y_capacity) AS ?y1)
    }
    """
)

# Cost-push price setting (eq. 8), reduced to the expected-inflation
# passthrough with an AR(1) innovation eps ~ N(0, $inflation_sigma):
# price <- price * (1 + infl_e + eps).  Expected inflation ?infl_e is SAC-learned
# from the realized AVG(price) history, else the $inflation_e prior.
FIRM_PRICING = Template(
    _PREFIXES
    + """
    DELETE { ?f def:price ?p0 }
    INSERT { ?f def:price ?p1 }
    WHERE {
        ?f a ex:Firm ; def:price ?p0 ."""
    + sac("AVG", "Firm", "price", prior="inflation_e", out="infl_e")
    + """
        BIND(?p0 * (1e0 + ?infl_e + pr:normal(0e0, $inflation_sigma)) AS ?p1)
    }
    """
)

# Per-tick refresh of eq. 49 income from current wages and profits — same
# logic as HOUSEHOLD_INCOME but as an upsert, so dividends track the owned
# firm's evolving profit.
HOUSEHOLD_INCOME_UPDATE = Template(
    _PREFIXES
    + """
    DELETE { ?hh def:income ?i0 }
    INSERT { ?hh def:income ?i1 }
    WHERE {
        { SELECT (AVG(?any_w) AS ?w_avg) WHERE { ?any_f def:w_bar ?any_w } }
        ?hh a ex:Household .
        OPTIONAL { ?hh def:income ?i0 }
        OPTIONAL { ?hh def:employer ?f . ?f def:w_bar ?w . }
        OPTIONAL { ?hh def:owns ?g . ?g def:profit ?p . }
        BIND(
            IF(BOUND(?w), ?w,
            IF(BOUND(?p), $dividend_ratio * IF(?p > 0e0, ?p, 0e0),
            $benefit_replacement * ?w_avg)) AS ?i1)
    }
    """
)

# One household tick: consumption budget (eq. 40) out of *expected* income,
# savings absorb the rest (eq. 50).  Expected income is current income grown by
# ?ig_e, the SAC-learned growth of realized SUM(income) (else the
# $income_growth_e prior): wealth <- wealth + income - psi*income*(1+ig_e)/(1+$vat_rate).
# Run HOUSEHOLD_INCOME_UPDATE earlier in the tick so income is current.
HOUSEHOLD_UPDATE = Template(
    _PREFIXES
    + """
    DELETE { ?hh def:wealth ?w0 }
    INSERT { ?hh def:wealth ?w1 }
    WHERE {
        ?hh a ex:Household ;
            def:wealth ?w0 ;
            def:psi ?psi ;
            def:income ?inc ."""
    + sac("SUM", "Household", "income", prior="income_growth_e", out="ig_e")
    + """
        BIND(?inc * (1e0 + ?ig_e) AS ?exp_income)
        BIND(?psi * ?exp_income / (1e0 + $vat_rate) AS ?consumption)
        BIND((?w0 + ?inc) - ?consumption AS ?w1)
    }
    """
)

# Goods market without search-and-matching (eqs. 1-2, 27, 31 collapsed):
# total nominal demand (households + government + foreign) is allocated to
# firms proportionally to their supply share price*output / sum(price*output)
# — the paper's size-weighted visiting probability without the random
# sequential element — and capped by supply.  profit = margin * revenue;
# liquidity accumulates profit.
FIRM_SALES = Template(
    _PREFIXES
    + """
    DELETE { ?f def:profit ?pi0 . ?f def:liquidity ?d0 }
    INSERT { ?f def:profit ?pi1 . ?f def:liquidity ?d1 }
    WHERE {
        { SELECT (SUM(?psi_h * ?i_h) AS ?c_hh)
          WHERE { ?h a ex:Household ; def:psi ?psi_h ; def:income ?i_h } }
        { SELECT (SUM(?b_j) AS ?c_gov) WHERE { ?j a ex:Government ; def:budget ?b_j } }
        { SELECT (SUM(?d_l) AS ?c_row) WHERE { ?l a ex:ForeignFirm ; def:demand_size ?d_l } }
        { SELECT (SUM(?p_g * ?y_g) AS ?supply)
          WHERE { ?g a ex:Firm ; def:price ?p_g ; def:output ?y_g } }
        ?f a ex:Firm ; def:price ?p ; def:output ?y ; def:margin ?mrg ;
           def:profit ?pi0 ; def:liquidity ?d0 .
        BIND(IF(BOUND(?c_hh), ?c_hh / (1e0 + $vat_rate), 0e0)
             + IF(BOUND(?c_gov), ?c_gov, 0e0)
             + IF(BOUND(?c_row), ?c_row, 0e0) AS ?demand_total)
        BIND(?demand_total * (?p * ?y) / ?supply AS ?demand_f)
        BIND(IF(?demand_f < ?p * ?y, ?demand_f, ?p * ?y) AS ?revenue)
        BIND(?mrg * ?revenue AS ?pi1)
        BIND(?d0 + ?pi1 AS ?d1)
    }
    """
)

# Government consumption AR(1) (eq. 51), the drift carrying an innovation
# eps ~ N(0, $gov_growth_sigma): budget <- budget * (1 + $gov_growth + eps).
# (Still linear rather than the paper's log-AR(1) form.)
GOVERNMENT_CONSUMPTION = Template(
    _PREFIXES
    + """
    DELETE { ?j def:budget ?b0 }
    INSERT { ?j def:budget ?b1 }
    WHERE {
        ?j a ex:Government ; def:budget ?b0 .
        BIND(?b0 * (1e0 + $gov_growth + pr:normal(0e0, $gov_growth_sigma)) AS ?b1)
    }
    """
)

# Generalized Taylor rule (eq. 69, euro-area terms dropped).  Realized
# growth and inflation are measured in-graph against the lagged aggregates
# stored on the CentralBank node, refreshed by the same upsert.
TAYLOR_RULE = Template(
    _PREFIXES
    + """
    DELETE { ?cb def:policy_rate ?r0 . ?cb def:prev_output ?py . ?cb def:prev_price ?pp }
    INSERT { ?cb def:policy_rate ?r1 . ?cb def:prev_output ?y_now . ?cb def:prev_price ?p_now }
    WHERE {
        { SELECT (SUM(?y_f) AS ?y_now) (AVG(?p_f) AS ?p_now)
          WHERE { ?f a ex:Firm ; def:output ?y_f ; def:price ?p_f } }
        ?cb a ex:CentralBank ; def:policy_rate ?r0 ;
            def:prev_output ?py ; def:prev_price ?pp .
        BIND(?y_now / ?py - 1e0 AS ?growth)
        BIND(?p_now / ?pp - 1e0 AS ?inflation)
        BIND($rho * ?r0
             + (1e0 - $rho) * ($r_star + $pi_star
                + $xi_pi * (?inflation - $pi_star)
                + $xi_gamma * ?growth) AS ?r_raw)
        BIND(IF(?r_raw > 0e0, ?r_raw, 0e0) AS ?r1)
    }
    """
)


# ---------------------------------------------------------------------------
# State extract — Model.query, one row per agent, no aggregation
# ---------------------------------------------------------------------------


def state_extract(model) -> pl.DataFrame:
    """Per-agent state as a sparse wide frame — no aggregation in SPARQL.

    One row per firm (price, output, tech_share), household (wealth), and
    central bank (policy_rate); the other columns are null.  Summary logic
    (GDP, price level, ...) belongs in polars expressions on the caller's
    side.  SAC-learned expectations are ephemeral (computed inline in the rules,
    not stored); observe them via the obs history (``def:of``/``def:value``) or
    by running ``sac:forecast`` yourself.
    """
    return model.query(
        _PREFIXES
        + """
    SELECT ?agent ?price ?output ?tech_share ?wealth ?policy_rate
    WHERE {
        { ?agent a ex:Firm ; def:price ?price ; def:output ?output ;
                 def:tech_share ?tech_share }
        UNION { ?agent a ex:Household ; def:wealth ?wealth }
        UNION { ?agent a ex:CentralBank ; def:policy_rate ?policy_rate }
    }
    """
    )


# ---------------------------------------------------------------------------
# Defaults — RDFSimulator's __init__ defaults.  POLEDNA_PARAMS carries the
# published Table 2 values (2010:Q4); user params dicts are merged over it,
# so overriding one number never means re-writing a rule.
# ---------------------------------------------------------------------------

POLEDNA_PARAMS = {
    "firm_ownership_ratio": 0.03,  # investor share of households (one owner per firm: set n_firms / n_households)
    "dividend_ratio": 0.7768,  # theta^DIV
    "benefit_replacement": 0.3586,  # theta^UB
    "vat_rate": 0.1529,  # tau^VAT
    "total_deposits": 222_933.2e6,  # D^H, Austria 2010:Q4 (EUR); rescale for demos
    # Expectation priors (eq. 6/9): the inline sac(...) forecasts fall back to
    # these ($<name> in the fragment) until two history observations exist, so a
    # run this short is unchanged by learning.
    "growth_e": 0.005,  # expected quarterly real growth
    "inflation_e": 0.005,  # expected quarterly inflation
    "income_growth_e": 0.0,  # expected household-income growth (neutral prior)
    "gov_growth": 0.005,  # government consumption drift
    # AR(1) innovation std devs (eq. 6/8/51).  Default 0 => normal(0,0)=0 =>
    # deterministic drifts (backward-compatible); set > 0 with
    # RDFSimulator(random_seed=...) for stochastic runs / Monte-Carlo ensembles.
    "growth_sigma": 0.0,  # firm output growth shock
    "inflation_sigma": 0.0,  # firm price shock
    "gov_growth_sigma": 0.0,  # government consumption shock
    "rho": 0.9263,  # Taylor rule: policy-rate smoothing
    "r_star": -0.0034,  # Taylor rule: real equilibrium rate
    "pi_star": 0.005,  # Taylor rule: inflation target
    "xi_pi": 0.3214,  # Taylor rule: inflation weight
    "xi_gamma": 1.2994,  # Taylor rule: growth weight
}

DEFAULT_INIT_RULES = (FIRM_OWNERSHIP, HOUSEHOLD_INCOME, HOUSEHOLD_WEALTH)

# Only the economic rules — no SAC/history bookkeeping.  FIRM_PRODUCTION,
# FIRM_PRICING and HOUSEHOLD_UPDATE call sac(...) inline; Ontology.expand_rules
# detects those signals and injects the matching PROCESS_INIT / measurement /
# PROCESS_ACCOUNTING at fit time, so the default set stays purely behavioral.
DEFAULT_UPDATE_RULES = (
    FIRM_PRODUCTION,  # (i) supply choice, eq. 5 + 12
    FIRM_PRICING,  # (i) price setting, eq. 8
    HOUSEHOLD_INCOME_UPDATE,  # eq. 49
    HOUSEHOLD_UPDATE,  # (v) consumption + savings, eqs. 40 + 50
    FIRM_SALES,  # (iv) goods market, eqs. 1-2 + 27 + 31
    GOVERNMENT_CONSUMPTION,  # eq. 51
    TAYLOR_RULE,  # eq. 69
)
