"""
Housing market: two behavioural rules that set one price, composed without merging.

The question this model family answers is not spatial, it is *architectural*. A seller
prices a house from the comparables on their street — a rule over the geographical
neighbourhood — and also follows a private heuristic: ask above the market first, and
cut if nobody bites. Both determine ``def:ask``. Writing one rule containing both logics
is the obvious move and the wrong one: the two are separately hypothesised, separately
cited and separately ablatable, and a merged rule can be none of those things.

**Give each rule its own predicate, and let a combinator read them.** Two rules that write
the same predicate do not compose — a SPARQL UPDATE evaluates its WHERE against the
pre-update graph, so the second overwrites the first and the first's effect is gone
within the tick. Two rules that write *different* predicates compose by tick order, which
is what ``update_rules`` already is::

    COMPARABLE_PRICE   ->  def:reference          "what the neighbourhood says"
    TIME_ON_MARKET     ->  def:weeks_on_market    "how long nobody bit"
    MARKUP_HEURISTIC   ->  def:markup             "how far above the market I dare"
    ASK_PRICE          ->  def:ask                reference, markup  ->  price
    SALE               ->  def:sold               the hazard that clears the listing

Nothing here is a housing idiom. It is Poledna's own shape: ``firm_produce`` writes
``def:output`` and ``firm_labor`` reads it to write ``def:size``, in one tick, in list
order. The seam scales to a third pricing motive (a mortgage-rate channel, a seller's
reservation value) by adding a predicate and one term to ``ASK_PRICE`` — never by
reopening ``COMPARABLE_PRICE``.

**When the logics are genuinely simultaneous**, compose the *text* instead: a SPARQL
fragment spliced into a WHERE clause, exactly as ``behaviour.learning.expect`` splices a
learned expectation into ``firm_produce``. ``neighbourhood_mean`` below is that idiom —
the comparables aggregate as a reusable string, so a rule wanting the neighbourhood's
mean ``def:ask`` and its mean ``def:weeks_on_market`` in one BIND calls it twice rather
than growing a second subselect by hand.

**What the decomposition buys.** ``$anchor`` in ``ASK_PRICE`` is the weight on the
comparables against the seller's own last ask: pure comparables at ``1e0``, pure
heuristic at ``0e0``, and anywhere between is a mixture — so *how much of the price each
rule explains* is a number black-it can calibrate (``calibration.parameters``) rather
than a fact buried in rule logic. Ablation is ``clone(sim, update_rules=...)`` with one
rule dropped, and because ``skabm.ir`` derives observables from the rules, the ablated
run measures whatever the surviving rules still imply, with no reporter to edit.

Deliberate simplifications: buyers are a hazard rate, not agents (there is no bid, so no
matching); a sold listing freezes rather than leaving the graph, because SPARQL DELETE of
a whole agent would take its comparables edges with it; every quantity is xsd:double,
coordinates included, for the reason ``schelling`` states — BGPs join on RDF terms, not
values.
"""

from string import Template

import polars as pl

from skabm.rules import _PREFIXES

# ---------------------------------------------------------------------------
# Fragment composition — the within-rule seam
# ---------------------------------------------------------------------------


def neighbourhood_mean(predicate: str, out: str, link: str = "comparable") -> str:
    """A SPARQL fragment binding ``?<out>`` to the mean of *predicate* over unsold comps.

    Splice it into any rule's WHERE clause, the way ``learning.expect`` is spliced::

        ?l a ex:Listing ; def:ask ?own .
        + neighbourhood_mean("ask", out="reference") +
        BIND(IF(BOUND(?reference), ?reference, ?own) AS ?r)

    The aggregate is wrapped in ``OPTIONAL`` and left *unbound* when a listing has no
    unsold comparable: a GROUP BY drops empty groups rather than yielding zero, and a
    zero would be a price. The caller decides the fallback, which is why it is not
    decided here — ``ASK_PRICE`` falls back to the seller's own last ask, a different
    rule might fall back to a district median.

    *link* names the edge the neighbourhood rides on, so the same fragment serves a
    radius-based ``def:comparable``, a shared-street link, or any relation an init rule
    CONSTRUCTs.
    """
    return f"""
    OPTIONAL {{
        {{ SELECT ?l (AVG(?__v) AS ?{out})
           WHERE {{ ?l a ex:Listing ; def:{link} ?__n .
                   ?__n def:{predicate} ?__v ; def:sold ?__s .
                   FILTER(?__s < 1e0) }}
           GROUP BY ?l }}
    }}
"""


# ---------------------------------------------------------------------------
# Init rule — the geographical neighbourhood, CONSTRUCTed once
# ---------------------------------------------------------------------------
# Comparables are the listings within $radius of this one: an O(N^2) pair scan, the
# same shape as schelling.GEO_NEIGHBORHOOD and for the same reason (maplib ships no
# `geof:` functions, so there is no spatial index to lean on).  Paid once at init
# rather than every tick, which is the whole point of materialising the edge.
#
# Squared distance, so no square root is needed; both sides of the FILTER are
# bracketed because maplib's equal-precedence operators associate to the *right*
# (see skabm.rules).  Placeholder: ``radius``.

NEIGHBOURHOOD = Template(
    _PREFIXES
    + """
    CONSTRUCT { ?l def:comparable ?n }
    WHERE {
        ?l a ex:Listing ; def:x ?x1 ; def:y ?y1 .
        ?n a ex:Listing ; def:x ?x2 ; def:y ?y2 .
        FILTER(?l != ?n)
        BIND((?x1 - ?x2) AS ?dx)
        BIND((?y1 - ?y2) AS ?dy)
        FILTER(((?dx * ?dx) + (?dy * ?dy)) < ($radius * $radius))
    }
    """
)

# ---------------------------------------------------------------------------
# Rule 1 — what the neighbourhood says (update, writes def:reference)
# ---------------------------------------------------------------------------
# The market signal, and nothing else: the mean ask of the unsold comparables, or the
# seller's own last ask when the street holds none.  It does not know a markup exists.

COMPARABLE_PRICE = Template(
    _PREFIXES
    + """
    DELETE { ?l def:reference ?r0 }
    INSERT { ?l def:reference ?r1 }
    WHERE {
        ?l a ex:Listing ; def:ask ?own .
        OPTIONAL { ?l def:reference ?r0 }"""
    + neighbourhood_mean("ask", out="comps")
    + """
        BIND(IF(BOUND(?comps), ?comps, ?own) AS ?r1)
    }
    """
)
COMPARABLE_PRICE.metadata = {
    "@id": "comparable_price",
    "@type": "Behaviour",
    "agentClass": "ex:Listing",
    "source": "Comparable-sales appraisal; the social-influence channel of the hedonic literature",
}

# ---------------------------------------------------------------------------
# Rule 2 — how long nobody bit (update, writes def:weeks_on_market)
# ---------------------------------------------------------------------------
# The heuristic's memory, as its own one-line rule.  Per-agent duration is *state*, not
# an aggregate: `skabm.history` is for series the whole model shares (see
# behaviour.learning), and a listing's own clock is a triple.

TIME_ON_MARKET = Template(
    _PREFIXES
    + """
    DELETE { ?l def:weeks_on_market ?w0 }
    INSERT { ?l def:weeks_on_market ?w1 }
    WHERE {
        ?l a ex:Listing ; def:weeks_on_market ?w0 ; def:sold ?sold .
        BIND(?w0 + IF(?sold > 0e0, 0e0, 1e0) AS ?w1)
    }
    """
)

# ---------------------------------------------------------------------------
# Rule 3 — the private heuristic (update, writes def:markup)
# ---------------------------------------------------------------------------
# "Ask above the market first; if it has not sold after $patience weeks, start cutting."
# It reads the clock rule 2 just advanced — staged activation, so list order is the
# dependency order — and knows nothing about the neighbourhood.  The markup floor is
# what stops the cut running away; it is the seller's reservation discount.
# Placeholders: ``patience``, ``cut``, ``floor_markup``.

MARKUP_HEURISTIC = Template(
    _PREFIXES
    + """
    DELETE { ?l def:markup ?m0 }
    INSERT { ?l def:markup ?m1 }
    WHERE {
        ?l a ex:Listing ; def:markup ?m0 ; def:weeks_on_market ?w ; def:sold ?sold .
        BIND(IF(?w > $patience, (?m0 - $cut), ?m0) AS ?cut_markup)
        BIND(IF(?cut_markup > $floor_markup, ?cut_markup, $floor_markup) AS ?m1)
    }
    """
)
MARKUP_HEURISTIC.metadata = {
    "@id": "markup_heuristic",
    "@type": "Behaviour",
    "agentClass": "ex:Listing",
    "source": "List-high-then-reduce; the seller's time-on-market price path",
}

# ---------------------------------------------------------------------------
# Rule 4 — the combinator (update, writes def:ask)
# ---------------------------------------------------------------------------
# The only rule that writes the price, and the only place the two motives meet.
# $anchor weights the neighbourhood reference against the seller's own last ask: 1e0 is
# pure comparables, 0e0 is a seller who never looks outside, and the markup multiplies
# whichever anchor results.  A sold listing keeps its ask — the price it cleared at.
# Placeholder: ``anchor``.

ASK_PRICE = Template(
    _PREFIXES
    + """
    DELETE { ?l def:ask ?a0 }
    INSERT { ?l def:ask ?a1 }
    WHERE {
        ?l a ex:Listing ; def:ask ?a0 ; def:reference ?r ; def:markup ?m ; def:sold ?sold .
        BIND(($anchor * ?r) + ((1e0 - $anchor) * ?a0) AS ?base)
        BIND(?base * (1e0 + ?m) AS ?fresh)
        BIND(IF(?sold > 0e0, ?a0, ?fresh) AS ?a1)
    }
    """
)

# ---------------------------------------------------------------------------
# Rule 5 — the buyer side, as a hazard (update, writes def:sold)
# ---------------------------------------------------------------------------
# Not a matching market: the probability of clearing this week rises as the ask falls
# below the neighbourhood reference, drawn from the pr:uniform UDF.  Enough to close the
# feedback loop — a cut sells the house, the sale leaves the comparables pool, and the
# reference the neighbours read moves.  Placeholder: ``base_hazard``.

SALE = Template(
    _PREFIXES
    + """
    DELETE { ?l def:sold ?s0 }
    INSERT { ?l def:sold ?s1 }
    WHERE {
        ?l a ex:Listing ; def:sold ?s0 ; def:ask ?a ; def:reference ?r .
        BIND($base_hazard * (?r / ?a) AS ?hazard)
        BIND(IF(pr:uniform(0e0, 1e0) < ?hazard, 1e0, 0e0) AS ?clears)
        BIND(IF(?s0 > 0e0, 1e0, ?clears) AS ?s1)
    }
    """
)


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------


def housing_extract(model) -> pl.DataFrame:
    """Per-listing state — one row per listing, unlike the IR's generic extract.

    ``def:comparable`` is a *self*-link, so the derived extract (which keeps link
    columns, since a relational plot wants them) fans each listing out to one row per
    comparable. Projecting the quantities only puts it back to one row per agent; the
    neighbourhood is still queryable from ``model_`` when a plot needs it.
    """
    return model.query(
        _PREFIXES
        + """
    SELECT ?agent ?x ?y ?ask ?reference ?markup ?weeks_on_market ?sold
    WHERE {
        ?agent a ex:Listing ; def:x ?x ; def:y ?y ; def:ask ?ask ;
               def:markup ?markup ; def:weeks_on_market ?weeks_on_market ;
               def:sold ?sold .
        OPTIONAL { ?agent def:reference ?reference }
    }
    """
    )


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

HOUSING_PARAMS = {
    "radius": 1.5,  # comparables cutoff, in grid units
    "anchor": 0.8,  # weight on the neighbourhood reference vs. the own last ask
    "patience": 4.0,  # weeks above the market before the seller starts cutting
    "cut": 0.03,  # markup shaved per week once patience runs out
    "floor_markup": -0.10,  # reservation discount: the cut stops here
    "base_hazard": 0.10,  # weekly clearing probability at ask == reference
}

HOUSING_INIT_RULES = (NEIGHBOURHOOD,)

# Order is the dependency order: reference before the clock, the clock before the
# markup that reads it, both before the ask that combines them, the ask before the
# hazard that judges it.
HOUSING_UPDATE_RULES = (
    COMPARABLE_PRICE,
    TIME_ON_MARKET,
    MARKUP_HEURISTIC,
    ASK_PRICE,
    SALE,
)

# Ablations, for the comparison the decomposition exists to make: drop the heuristic and
# every seller prices at a constant markup over the neighbourhood; drop the comparables
# and each seller walks their own opening ask down alone.
NO_HEURISTIC_RULES = tuple(r for r in HOUSING_UPDATE_RULES if r is not MARKUP_HEURISTIC)
NO_COMPARABLES_RULES = tuple(
    r for r in HOUSING_UPDATE_RULES if r is not COMPARABLE_PRICE
)


def make_listings(side: int = 8, opening_ask: float = 250.0) -> pl.DataFrame:
    """A ``side * side`` block of listings, one per lattice point.

    Opening asks vary across the block so the comparables channel has something to
    average that is not already the answer; everything else starts at the same markup,
    an empty clock and unsold.
    """
    n = side * side
    return pl.DataFrame(
        {
            "id": [f"listing_{i}" for i in range(n)],
            "x": [float(i % side) for i in range(n)],
            "y": [float(i // side) for i in range(n)],
            "ask": [opening_ask + 10.0 * float(i % 5) for i in range(n)],
            "markup": [0.10] * n,
            "weeks_on_market": [0.0] * n,
            "sold": [0.0] * n,
        }
    )
