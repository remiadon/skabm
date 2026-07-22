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
* **No stochastic shocks**: SPARQL has no seedable RAND, so exogenous AR(1)
  processes degenerate to deterministic drifts and Monte Carlo ensembles
  are out of reach in-graph.
* **No AR(1) re-estimation** (eq. 6/9): expectations are constant
  parameters, not regressions on the model's own history.
* **No log/exp**: AR(1) laws of motion in log-levels (eq. 51, 77, 81)
  become linear growth factors.

Numeric parameters are injected as xsd:double literals via ``dbl`` — never
decimals, which maplib 0.20.19 mixes into unstable multi-typed columns.
Unbound-variable handling uses explicit BOUND tests: maplib evaluates
comparisons on unbound variables as false instead of erroring, so a
COALESCE chain would bind the wrong arm.
"""

from string import Template

import polars as pl

EX_NS = "http://example.net/skabm#"
DEF_NS = "urn:maplib_default:"

_PREFIXES = (
    f"PREFIX ex:<{EX_NS}>\n"
    f"PREFIX def:<{DEF_NS}>\n"
    "PREFIX xsd:<http://www.w3.org/2001/XMLSchema#>\n"
)

_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


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
# assignment is deterministic — SPARQL has no seedable RAND, so truly random
# matching belongs to the calibration layer or a numerical backend.  Only
# firms without a data-defined owner are filled (FILTER NOT EXISTS), and
# users never declare `owns` themselves.  Relies on the skabm id convention
# ``EX_NS + "firm_<j>"`` / ``EX_NS + "hh_<i>"``.
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
# Update rules — Model.update, every tick (upsert pattern)
# ---------------------------------------------------------------------------

# Supply choice (eq. 5) capped by labor capacity (eq. 12, Leontief):
# output <- min(output * (1 + $growth_e), alpha * size).  The
# intermediate-input and capital legs of the Leontief nest are omitted.
FIRM_PRODUCTION = Template(
    _PREFIXES
    + """
    DELETE { ?f def:output ?y0 }
    INSERT { ?f def:output ?y1 }
    WHERE {
        ?f a ex:Firm ; def:output ?y0 ; def:alpha ?alpha ; def:size ?n .
        BIND(?y0 * (1e0 + $growth_e) AS ?y_desired)
        BIND(?alpha * ?n AS ?y_capacity)
        BIND(IF(?y_desired < ?y_capacity, ?y_desired, ?y_capacity) AS ?y1)
    }
    """
)

# Cost-push price setting (eq. 8), reduced to the expected-inflation
# passthrough: price <- price * (1 + $inflation_e).
FIRM_PRICING = Template(
    _PREFIXES
    + """
    DELETE { ?f def:price ?p0 }
    INSERT { ?f def:price ?p1 }
    WHERE {
        ?f a ex:Firm ; def:price ?p0 .
        BIND(?p0 * (1e0 + $inflation_e) AS ?p1)
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

# One household tick: consumption budget (eq. 40) out of stored def:income,
# savings absorb the rest (eq. 50):
# wealth <- wealth + income - psi * income / (1 + $vat_rate).
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
            def:income ?inc .
        BIND(?psi * ?inc / (1e0 + $vat_rate) AS ?consumption)
        BIND(?w0 + ?inc - ?consumption AS ?w1)
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

# Government consumption drift (eq. 51 without the log-AR(1) form and
# without shocks): budget <- budget * (1 + $gov_growth).
GOVERNMENT_CONSUMPTION = Template(
    _PREFIXES
    + """
    DELETE { ?j def:budget ?b0 }
    INSERT { ?j def:budget ?b1 }
    WHERE {
        ?j a ex:Government ; def:budget ?b0 .
        BIND(?b0 * (1e0 + $gov_growth) AS ?b1)
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
    side.
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
    "growth_e": 0.005,  # expected quarterly real growth
    "inflation_e": 0.005,  # expected quarterly inflation
    "gov_growth": 0.005,  # government consumption drift
    "rho": 0.9263,  # Taylor rule: policy-rate smoothing
    "r_star": -0.0034,  # Taylor rule: real equilibrium rate
    "pi_star": 0.005,  # Taylor rule: inflation target
    "xi_pi": 0.3214,  # Taylor rule: inflation weight
    "xi_gamma": 1.2994,  # Taylor rule: growth weight
}

DEFAULT_INIT_RULES = (FIRM_OWNERSHIP, HOUSEHOLD_INCOME, HOUSEHOLD_WEALTH)

DEFAULT_UPDATE_RULES = (
    FIRM_PRODUCTION,  # (i) supply choice, eq. 5 + 12
    FIRM_PRICING,  # (i) price setting, eq. 8
    HOUSEHOLD_INCOME_UPDATE,  # eq. 49
    HOUSEHOLD_UPDATE,  # (v) consumption + savings, eqs. 40 + 50
    FIRM_SALES,  # (iv) goods market, eqs. 1-2 + 27 + 31
    GOVERNMENT_CONSUMPTION,  # eq. 51
    TAYLOR_RULE,  # eq. 69
)
