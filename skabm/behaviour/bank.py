"""
Bank behaviour templates — extensions to Poledna (2023) not present in the paper.

Poledna et al. calibrate 12 Basel III banks (capital_ratio, leverage, deposit_share) but
leave them passive: no interbank market, no contagion, no deposit flight.  This module
adds ``bank_depositors`` (an init CONSTRUCT: which agents hold deposits at which bank),
``bank_capital`` (a per-tick update: the capital ratio responds to deposit outflows) and
``interbank_contagion`` (a recursive CONSTRUCT evaluated by ``Model.infer``: distress
propagates through the depositor network to a fixed point).  Placeholder
defaults sit on each ``DefaultTemplate``; the SPARQL itself never contains a number.

**Licensing.**  ``bank_depositors`` and ``bank_capital`` run on the free maplib core.
Only ``interbank_contagion`` needs the licensed reasoning add-on — free for academic
use, absent from the stock PyPI wheels — so importing this module never requires a
license, but passing that rule to ``RDFSimulator(infer=...)`` does.

**Engine choice.**  The contagion rule is a SPARQL CONSTRUCT, not Datalog: the Datalog
triple-pattern form in maplib 0.20.29 supports neither FILTER in the body nor
aggregation, and recursive CONSTRUCT does.  ``RDFSimulator`` passes the string as-is to
``Model.infer``, which accepts both forms.

**CONSTRUCT/WHERE line break.**  maplib's SPARQL parser (0.20.29) rejects a line break
between the closing ``}`` of CONSTRUCT and the ``WHERE`` keyword, so every CONSTRUCT
rule here keeps ``} WHERE {`` on one line.
"""

from __future__ import annotations

from skabm.behaviour import DefaultTemplate
from skabm.sparql import _PREFIXES, EX_NS

# ---------------------------------------------------------------------------
# bank_depositors - init CONSTRUCT, structural, runs once after mapping
# ---------------------------------------------------------------------------
# For every agent (firm or household) with deposits/wealth, assign them to a
# bank weighted by the bank's ``def:deposit_share``.  The draw uses
# ``pr:uniform`` so it is reproducible under ``RDFSimulator(random_seed=...)``
# - same contract as ``schelling.SETTLE`` and ``firm.firm_ownership``.
# This is a one-time CONSTRUCT, NOT an update rule.

bank_depositors = DefaultTemplate(
    _PREFIXES
    + """CONSTRUCT { ?agent def:holds_at ?bank } WHERE {
    { ?agent a ex:Household ; def:wealth ?w }
    UNION
    { ?agent a ex:Firm ; def:liquidity ?l }
    ?bank a ex:Bank ; def:deposit_share ?ds .
    FILTER(pr:uniform(0e0, 1e0) < ?ds)
}"""
)
bank_depositors.metadata = {
    "@id": "bank_depositors",
    "@type": "Behaviour",
    "agentClass": "ex:Bank",
    "source": "skabm extension - not in Poledna et al. (2023)",
}

# ---------------------------------------------------------------------------
# bank_capital - update rule, state, per-tick
# ---------------------------------------------------------------------------
# Capital ratio deteriorates when depositors move money out (flight-to-safety
# in a contagion round).  The outflow aggregate is optional so a bank with no
# outflows stays put.
#
# ``bank_asset_scale`` is the deposit base one unit of leverage stands for, and
# it is a parameter rather than a constant because it carries the *units* of the
# population: an outflow is in whatever currency the agents hold, so a run
# calibrated 1:1 from national accounts and a 500-agent demo need scales that
# differ by orders of magnitude.  Leave it at its default for a toy population;
# set it from the deposit base (``total_deposits / n_banks``, say) for a
# realistically scaled one, or every flight event drives the ratio to nonsense.

bank_capital = DefaultTemplate(
    _PREFIXES
    + """DELETE { ?b def:capital_ratio ?cr0 }
INSERT { ?b def:capital_ratio ?cr1 }
WHERE {
    ?b a ex:Bank ;
       def:capital_ratio ?cr0 ;
       def:leverage ?lev .
    OPTIONAL {
        SELECT ?b (SUM(?out) AS ?outflow)
        WHERE {
            ?agent def:holds_at ?b ;
                   def:flees_amount ?out .
        } GROUP BY ?b
    }
    BIND(?cr0 - IF(BOUND(?outflow), ?outflow / (?lev * $bank_asset_scale), 0e0) AS ?cr1)
}""",
    {
        "bank_asset_scale": 1e3,  # unsourced: deposit base per unit of leverage; scale with the population
    },
)
bank_capital.metadata = {
    "@id": "bank_capital",
    "@type": "Behaviour",
    "agentClass": "ex:Bank",
    "source": "skabm extension - not in Poledna et al. (2023)",
}

# ---------------------------------------------------------------------------
# interbank_contagion - recursive CONSTRUCT for ``Model.infer``
# ---------------------------------------------------------------------------
# This is the rule that replaces a hand-tuned sub-tick propagation loop.
#
# Mechanics (one fixed-point pass):
#   1. A bank is distressed if its capital_ratio has dropped below the
#      threshold (this state is written by bank_capital before ``infer`` runs).
#   2. An agent holding at a distressed bank flees to any safe (non-distressed)
#      bank.  The amount flown is the agent's full deposit at the distressed
#      bank (wealth for households, liquidity for firms).
#   3. A bank that receives flight from a distressed bank above the flee amount
#      threshold becomes distressed itself in the next round.
#
# The rule is written as a recursive SPARQL CONSTRUCT (not Datalog) because
# maplib 0.20.29's Datalog engine does not support FILTER in the body or
# aggregation; recursive CONSTRUCT does.  ``RDFSimulator`` passes the string
# as-is to ``Model.infer`` - it accepts both forms.
#
# Note on rendering: each clause is a ``string.Template``, so
# ``distress_threshold`` and ``flee_amount_threshold`` are substituted from the
# simulator's ``params`` dict at fit time exactly like every other rule
# parameter.  Both thresholds carry the *units of the population* - a capital
# ratio is dimensionless but a flight amount is currency - so a realistically
# scaled run has to set the second one, and a rule that baked in a literal
# would have silently ignored it.
#
# Each CONSTRUCT clause is on a single line (no line break between } and WHERE)
# because maplib's SPARQL parser rejects it.
#
# ``ex:is_distressed`` is a *boolean-valued* predicate (``?b ex:is_distressed
# true``), not a bare flag: SPARQL has no unary predicates, so ``{ ?b
# ex:is_distressed }`` is a parse error under ``Model.infer``.  Distress is
# therefore carried as a ``true`` object, and every match tests it as
# ``?b ex:is_distressed true``.
interbank_contagion = [
    DefaultTemplate(
        f"PREFIX ex: <{EX_NS}>"
        f"PREFIX def: <urn:maplib_default:>"
        f"CONSTRUCT {{ ?b ex:is_distressed true }} WHERE {{ ?b a ex:Bank ; def:capital_ratio ?cr . FILTER(?cr < $distress_threshold) . }}",
        {"distress_threshold": 0.03},  # ζ, the Basel III minimum capital ratio
    ),
    DefaultTemplate(
        f"PREFIX ex: <{EX_NS}>"
        f"PREFIX def: <urn:maplib_default:>"
        f"CONSTRUCT {{ ?agent def:flees_to ?safe ; def:flees_amount ?amount }} WHERE {{ {{ ?agent a ex:Household ; def:holds_at ?distressed ; def:wealth ?amount }} UNION {{ ?agent a ex:Firm ; def:holds_at ?distressed ; def:liquidity ?amount }} ?distressed ex:is_distressed true . ?safe a ex:Bank . FILTER NOT EXISTS {{ ?safe ex:is_distressed true }} . }}"
    ),
    DefaultTemplate(
        f"PREFIX ex: <{EX_NS}>"
        f"PREFIX def: <urn:maplib_default:>"
        f"CONSTRUCT {{ ?b ex:is_distressed true }} WHERE {{ ?other def:flees_to ?b ; def:flees_amount ?amount . ?b a ex:Bank ; def:capital_ratio ?cr . FILTER(?amount > $flee_amount_threshold) . }}",
        {
            "flee_amount_threshold": 10.0
        },  # unsourced: outflow that destabilises the receiving bank
    ),
]
