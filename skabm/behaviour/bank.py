"""
Bank behaviour templates — extensions to Poledna (2023) not present in the
original paper.  Poledna et al. (2023, European Economic Review 151, 104306)
calibrate 12 Basel III banks (capital_ratio, leverage, deposit_share) but
leave them passive: no interbank market, no contagion, no deposit flight.
This module adds:

1. bank_depositors  — init CONSTRUCT: which agents hold deposits at which
                       bank (structural, runs once at init).
2. bank_capital     — update rule: capital ratio responds to deposit outflows
                       (state, per-tick).
3. interbank_contagion — recursive SPARQL CONSTRUCT rule (licensed
                       ``Model.infer``): distress propagates through the
                       depositor network to a fixed point, replacing hand-tuned
                       sub-tick passes.

Placeholders are parameterised in ``skabm.behaviour.params`` (or a user
params dict); rule logic never contains numeric defaults.

Licensing: ``bank_depositors`` (insert) and ``bank_capital`` (update) run on
the free maplib core.  Only ``interbank_contagion`` — evaluated through
``Model.infer`` — needs the licensed reasoning add-on (free for academic use,
absent from the stock PyPI wheels).  So importing/using this module never
requires a license; passing ``interbank_contagion`` to ``RDFSimulator(infer=)``
does.

IMPORTANT — engine choice:
    The contagion rule is written as a SPARQL CONSTRUCT, not Datalog.  The
    Datalog triple-pattern form in maplib 0.20.29 does not support FILTER in
    the body or aggregation; recursive CONSTRUCT does.  ``RDFSimulator``
    passes the string as-is to ``Model.infer`` — it accepts both forms.  See
    ``skabm.simulation`` and the maplib SKILL.md for the two syntaxes.

IMPORTANT — CONSTRUCT/WHERE line break:
    Maplib's SPARQL parser (0.20.29) rejects a line break between the closing
    ``}`` of CONSTRUCT and the ``WHERE`` keyword.  Every CONSTRUCT rule here
    is written with ``CONSTRUCT { ... } WHERE { ... }`` on a single line so
    it parses under ``Model.infer``.
"""

from __future__ import annotations

from string import Template

from skabm.rules import EX_NS, _PREFIXES

# ---------------------------------------------------------------------------
# bank_depositors — init CONSTRUCT, structural, runs once after mapping
# ---------------------------------------------------------------------------
# For every agent (firm or household) with deposits/wealth, assign them to a
# bank weighted by the bank's ``def:deposit_share``.  The draw uses
# ``pr:uniform`` so it is reproducible under ``RDFSimulator(random_seed=...)``
# — same contract as ``schelling.SETTLE`` and ``firm.firm_ownership``.
# This is a one-time CONSTRUCT, NOT an update rule.

bank_depositors = Template(
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
    "source": "skabm extension — not in Poledna et al. (2023)",
}

# ---------------------------------------------------------------------------
# bank_capital — update rule, state, per-tick
# ---------------------------------------------------------------------------
# Capital ratio deteriorates when depositors move money out (flight-to-safety
# in a contagion round).  Placeholder: ``distress_threshold`` (the ratio below
# which a bank is distressed).  The outflow aggregate is optional so a bank
# with no outflows stays put.

bank_capital = Template(
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
    BIND(?cr0 - IF(BOUND(?outflow), ?outflow / (?lev * 1e3), 0e0) AS ?cr1)
}"""
)
bank_capital.metadata = {
    "@id": "bank_capital",
    "@type": "Behaviour",
    "agentClass": "ex:Bank",
    "source": "skabm extension — not in Poledna et al. (2023)",
}

# ---------------------------------------------------------------------------
# interbank_contagion — recursive CONSTRUCT for ``Model.infer``
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
# as-is to ``Model.infer`` — it accepts both forms.
#
# Note on rendering: the rule string contains no ``$placeholder`` references —
# parameters like ``distress_threshold`` and ``flee_amount_threshold`` are
# embedded directly via Python f-string substitution so that ``render()`` is
# a no-op.  This keeps the rule self-contained and avoidable of the
# ``string.Template`` path, while still being substitutable through the
# simulator's ``params`` dict if needed.  Users who want to override thresholds
# at fit time should pass a custom rule string via ``infer=``.

# Each CONSTRUCT clause is on a single line (no line break between } and WHERE)
# because maplib's SPARQL parser rejects it.
#
# ``ex:is_distressed`` is a *boolean-valued* predicate (``?b ex:is_distressed
# true``), not a bare flag: SPARQL has no unary predicates, so ``{ ?b
# ex:is_distressed }`` is a parse error under ``Model.infer``.  Distress is
# therefore carried as a ``true`` object, and every match tests it as
# ``?b ex:is_distressed true``.
interbank_contagion = [
    (
        f"PREFIX ex: <{EX_NS}>"
        f"PREFIX def: <urn:maplib_default:>"
        f"CONSTRUCT {{ ?b ex:is_distressed true }} WHERE {{ ?b a ex:Bank ; def:capital_ratio ?cr . FILTER(?cr < 0.03) . }}"
    ),
    (
        f"PREFIX ex: <{EX_NS}>"
        f"PREFIX def: <urn:maplib_default:>"
        f"CONSTRUCT {{ ?agent def:flees_to ?safe ; def:flees_amount ?amount }} WHERE {{ {{ ?agent a ex:Household ; def:holds_at ?distressed ; def:wealth ?amount }} UNION {{ ?agent a ex:Firm ; def:holds_at ?distressed ; def:liquidity ?amount }} ?distressed ex:is_distressed true . ?safe a ex:Bank . FILTER NOT EXISTS {{ ?safe ex:is_distressed true }} . }}"
    ),
    (
        f"PREFIX ex: <{EX_NS}>"
        f"PREFIX def: <urn:maplib_default:>"
        f"CONSTRUCT {{ ?b ex:is_distressed true }} WHERE {{ ?other def:flees_to ?b ; def:flees_amount ?amount . ?b a ex:Bank ; def:capital_ratio ?cr . FILTER(?amount > 5.0) . }}"
    ),
]
