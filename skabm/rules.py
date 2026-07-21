"""Graph construction and SPARQL rules for Poledna-style ABM simulation.

``map_df`` lifts a calibrated agent population (a polars DataFrame) into a
maplib model in one call: maplib's ``map_default`` auto-generates the OTTR
template (primary key ``id``, IRI-typed link columns, nullable attributes),
and one ``map_triples`` call tags every agent with its class.  There is no
hand-written template registry: **predicates are column names** under
maplib's default namespace (``urn:maplib_default:``), classes live under
``EX_NS``.  DataFrame conventions: an ``id`` column of full IRIs
(``EX_NS + "firm_0"``); link columns hold the target agent's IRI (or null).

The rule factories return the behavioural SPARQL over those predicates:

* **Initialization rules** — run once through ``Model.insert`` while
  constructing the model (``household_income``, ``household_wealth``).
* **Update rules** — run every tick through ``Model.update`` (that is what
  ``RDFSimulator`` does).  Each is an *upsert*: ``DELETE`` the old value
  (matched through ``OPTIONAL`` so the first application works on a freshly
  initialized graph), ``INSERT`` the new one — SPARQL INSERT alone has set
  semantics and would accumulate duplicate values.
* **State extract** — ``state_extract()`` is a plain SELECT of per-agent
  state with no aggregation; ``RDFSimulator`` yields its result each tick
  and summary logic stays in polars expressions on the caller's side.

The recommended per-tick ordering follows the paper's event sequence
(Poledna et al. 2023, Section 3.5): production, pricing, income, consumption
and savings, sales and profits, government consumption, monetary policy.

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

import polars as pl

EX_NS = "http://example.net/skabm#"
DEF_NS = "urn:maplib_default:"

_PREFIXES = f"PREFIX ex:<{EX_NS}>\n    PREFIX def:<{DEF_NS}>"

_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


def dbl(x: float) -> str:
    """Format a Python float as a SPARQL xsd:double literal."""
    return f"{x:.6e}"


def map_df(model, df: pl.DataFrame, kind: str) -> None:
    """Map an agent population into the model in one call.

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
    model.map_default(df, primary_key_column="id")
    model.map_triples(
        df.select(subject="id").with_columns(object=pl.lit(EX_NS + kind)),
        predicate=_RDF_TYPE,
    )


# ---------------------------------------------------------------------------
# Initialization rules — Model.insert, once, at model construction
# ---------------------------------------------------------------------------


def household_income(dividend_ratio: float, benefit_replacement: float) -> str:
    """Income by activity status (Poledna eq. 49), for ``Model.insert``.

    Wage from the ``employer`` link; else dividend
    ``dividend_ratio * max(0, profit)`` from the ``owns`` link; else
    unemployment benefits ``benefit_replacement`` times the average wage.
    """
    return f"""
    {_PREFIXES}
    CONSTRUCT {{ ?hh def:income ?income }}
    WHERE {{
        {{ SELECT (AVG(?any_w) AS ?w_avg) WHERE {{ ?any_f def:w_bar ?any_w }} }}
        ?hh a ex:Household .
        OPTIONAL {{ ?hh def:employer ?f . ?f def:w_bar ?w . }}
        OPTIONAL {{ ?hh def:owns ?g . ?g def:profit ?p . }}
        BIND(
            IF(BOUND(?w), ?w,
            IF(BOUND(?p), {dbl(dividend_ratio)} * IF(?p > 0e0, ?p, 0e0),
            {dbl(benefit_replacement)} * ?w_avg)) AS ?income)
    }}
    """


def household_wealth(total_deposits: float) -> str:
    """Initial wealth D_h(0) = D^H Y_h(0) / sum Y_h(0) (Poledna Section 5.2),
    for ``Model.insert`` after ``household_income``."""
    return f"""
    {_PREFIXES}
    CONSTRUCT {{ ?hh def:wealth ?wealth }}
    WHERE {{
        {{ SELECT (SUM(?any_i) AS ?total) WHERE {{ ?any_hh def:income ?any_i }} }}
        ?hh def:income ?income .
        BIND({dbl(total_deposits)} * ?income / ?total AS ?wealth)
    }}
    """


# ---------------------------------------------------------------------------
# Update rules — Model.update, every tick (upsert pattern)
# ---------------------------------------------------------------------------


def firm_production(growth_e: float) -> str:
    """Supply choice (eq. 5) capped by labor capacity (eq. 12, Leontief).

    output <- min(output * (1 + growth_e), alpha * size).  The
    intermediate-input and capital legs of the Leontief nest are omitted
    (no input stocks in the simplified state).
    """
    return f"""
    {_PREFIXES}
    DELETE {{ ?f def:output ?y0 }}
    INSERT {{ ?f def:output ?y1 }}
    WHERE {{
        ?f a ex:Firm ; def:output ?y0 ; def:alpha ?alpha ; def:size ?n .
        BIND(?y0 * (1e0 + {dbl(growth_e)}) AS ?y_desired)
        BIND(?alpha * ?n AS ?y_capacity)
        BIND(IF(?y_desired < ?y_capacity, ?y_desired, ?y_capacity) AS ?y1)
    }}
    """


def firm_pricing(inflation_e: float) -> str:
    """Cost-push price setting (eq. 8), reduced to the expected-inflation
    passthrough: price <- price * (1 + inflation_e) with a constant real
    cost structure."""
    return f"""
    {_PREFIXES}
    DELETE {{ ?f def:price ?p0 }}
    INSERT {{ ?f def:price ?p1 }}
    WHERE {{
        ?f a ex:Firm ; def:price ?p0 .
        BIND(?p0 * (1e0 + {dbl(inflation_e)}) AS ?p1)
    }}
    """


def household_income_update(dividend_ratio: float, benefit_replacement: float) -> str:
    """Per-tick refresh of eq. 49 income from current wages and profits.

    Same logic as ``household_income`` but as an upsert, so dividends track
    the owned firm's evolving profit.
    """
    return f"""
    {_PREFIXES}
    DELETE {{ ?hh def:income ?i0 }}
    INSERT {{ ?hh def:income ?i1 }}
    WHERE {{
        {{ SELECT (AVG(?any_w) AS ?w_avg) WHERE {{ ?any_f def:w_bar ?any_w }} }}
        ?hh a ex:Household .
        OPTIONAL {{ ?hh def:income ?i0 }}
        OPTIONAL {{ ?hh def:employer ?f . ?f def:w_bar ?w . }}
        OPTIONAL {{ ?hh def:owns ?g . ?g def:profit ?p . }}
        BIND(
            IF(BOUND(?w), ?w,
            IF(BOUND(?p), {dbl(dividend_ratio)} * IF(?p > 0e0, ?p, 0e0),
            {dbl(benefit_replacement)} * ?w_avg)) AS ?i1)
    }}
    """


def household_update(vat_rate: float) -> str:
    """One household tick: consumption budget (eq. 40) out of stored
    ``def:income``, savings absorb the rest (eq. 50).

    wealth <- wealth + income - psi * income / (1 + vat_rate).
    Run ``household_income_update`` earlier in the tick so income is current.
    """
    return f"""
    {_PREFIXES}
    DELETE {{ ?hh def:wealth ?w0 }}
    INSERT {{ ?hh def:wealth ?w1 }}
    WHERE {{
        ?hh a ex:Household ;
            def:wealth ?w0 ;
            def:psi ?psi ;
            def:income ?inc .
        BIND(?psi * ?inc / (1e0 + {dbl(vat_rate)}) AS ?consumption)
        BIND(?w0 + ?inc - ?consumption AS ?w1)
    }}
    """


def firm_sales(vat_rate: float) -> str:
    """Goods market without search-and-matching (eqs. 1-2, 27, 31 collapsed).

    Total nominal demand = household consumption budgets + government
    budgets + foreign demand, allocated to firms proportionally to their
    supply share price*output / sum(price*output) (the paper's
    size-weighted visiting probability without the random sequential
    element) and capped by supply.  profit = margin * revenue; liquidity
    accumulates profit.
    """
    return f"""
    {_PREFIXES}
    DELETE {{ ?f def:profit ?pi0 . ?f def:liquidity ?d0 }}
    INSERT {{ ?f def:profit ?pi1 . ?f def:liquidity ?d1 }}
    WHERE {{
        {{ SELECT (SUM(?psi_h * ?i_h) AS ?c_hh)
           WHERE {{ ?h a ex:Household ; def:psi ?psi_h ; def:income ?i_h }} }}
        {{ SELECT (SUM(?b_j) AS ?c_gov) WHERE {{ ?j a ex:Government ; def:budget ?b_j }} }}
        {{ SELECT (SUM(?d_l) AS ?c_row) WHERE {{ ?l a ex:ForeignFirm ; def:demand_size ?d_l }} }}
        {{ SELECT (SUM(?p_g * ?y_g) AS ?supply)
           WHERE {{ ?g a ex:Firm ; def:price ?p_g ; def:output ?y_g }} }}
        ?f a ex:Firm ; def:price ?p ; def:output ?y ; def:margin ?mrg ;
           def:profit ?pi0 ; def:liquidity ?d0 .
        BIND(IF(BOUND(?c_hh), ?c_hh / (1e0 + {dbl(vat_rate)}), 0e0)
             + IF(BOUND(?c_gov), ?c_gov, 0e0)
             + IF(BOUND(?c_row), ?c_row, 0e0) AS ?demand_total)
        BIND(?demand_total * (?p * ?y) / ?supply AS ?demand_f)
        BIND(IF(?demand_f < ?p * ?y, ?demand_f, ?p * ?y) AS ?revenue)
        BIND(?mrg * ?revenue AS ?pi1)
        BIND(?d0 + ?pi1 AS ?d1)
    }}
    """


def government_consumption(growth: float) -> str:
    """Government consumption drift (eq. 51 without the log-AR(1) form and
    without shocks): budget <- budget * (1 + growth)."""
    return f"""
    {_PREFIXES}
    DELETE {{ ?j def:budget ?b0 }}
    INSERT {{ ?j def:budget ?b1 }}
    WHERE {{
        ?j a ex:Government ; def:budget ?b0 .
        BIND(?b0 * (1e0 + {dbl(growth)}) AS ?b1)
    }}
    """


def taylor_rule(
    rho: float, r_star: float, pi_star: float, xi_pi: float, xi_gamma: float
) -> str:
    """Generalized Taylor rule (eq. 69, euro-area terms dropped).

    Realized growth and inflation are measured in-graph against the lagged
    aggregates stored on the CentralBank node, which are refreshed by the
    same upsert.
    """
    return f"""
    {_PREFIXES}
    DELETE {{ ?cb def:policy_rate ?r0 . ?cb def:prev_output ?py . ?cb def:prev_price ?pp }}
    INSERT {{ ?cb def:policy_rate ?r1 . ?cb def:prev_output ?y_now . ?cb def:prev_price ?p_now }}
    WHERE {{
        {{ SELECT (SUM(?y_f) AS ?y_now) (AVG(?p_f) AS ?p_now)
           WHERE {{ ?f a ex:Firm ; def:output ?y_f ; def:price ?p_f }} }}
        ?cb a ex:CentralBank ; def:policy_rate ?r0 ;
            def:prev_output ?py ; def:prev_price ?pp .
        BIND(?y_now / ?py - 1e0 AS ?growth)
        BIND(?p_now / ?pp - 1e0 AS ?inflation)
        BIND({dbl(rho)} * ?r0
             + (1e0 - {dbl(rho)}) * ({dbl(r_star)} + {dbl(pi_star)}
                + {dbl(xi_pi)} * (?inflation - {dbl(pi_star)})
                + {dbl(xi_gamma)} * ?growth) AS ?r_raw)
        BIND(IF(?r_raw > 0e0, ?r_raw, 0e0) AS ?r1)
    }}
    """


# ---------------------------------------------------------------------------
# State extract — Model.query, one row per agent, no aggregation
# ---------------------------------------------------------------------------


def state_extract() -> str:
    """Per-agent state as a sparse wide frame — no aggregation in SPARQL.

    One row per firm (price, output, tech_share), household (wealth), and
    central bank (policy_rate); the other columns are null.  Summary logic
    (GDP, price level, ...) belongs in polars expressions on the caller's
    side.
    """
    return f"""
    {_PREFIXES}
    SELECT ?agent ?price ?output ?tech_share ?wealth ?policy_rate
    WHERE {{
        {{ ?agent a ex:Firm ; def:price ?price ; def:output ?output ;
                  def:tech_share ?tech_share }}
        UNION {{ ?agent a ex:Household ; def:wealth ?wealth }}
        UNION {{ ?agent a ex:CentralBank ; def:policy_rate ?policy_rate }}
    }}
    """
