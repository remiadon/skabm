"""
HALE — Moon et al. (2026), *LLM-powered reasoning in agent-based modeling*, on skabm.

The paper couples an individual-based SIR model on an activity contact network with an
LLM that decides, each week, whether a group of people still goes out.  Its scalability
claim rests on one reduction: the LLM is asked once per *demographic group*, not once
per agent — 1,552 groups standing in for 1.13M residents of Salt Lake County.  That is
the same reduction ``rules._register_choice`` already performs by memoizing on the
prompt string, which is why HALE lands here as four SPARQL rules and no new machinery.

**The structural idea worth keeping** is the split between deliberate and non-deliberate
contact.  Living in a house is not a decision; going to the office is.  So a contact
carries a ``def:kind``, and only ``"out"`` edges are switched off when a group declines —
an infected agent still infects its household.  Everything else follows: the LLM writes
one triple per group, and the transmission rule reads it through the agents' ``def:group``
link.  ``|N| >> |L|`` is a join, not an architecture.

Populations (see ``notebooks/hale.ipynb`` for a synthetic one):

``Person``   ``group`` link, ``infected``/``recovered`` as 0/1 doubles — susceptible is
             neither, so ``sig__AVG__Person__infected`` is prevalence for free.
``Group``    the demographic cells the prompt is built from (municipality, age band, sex,
             race).  Carries ``prevalence`` and ``goes_out``, both written by rules.
``Edge``     the contact network, reified — ``ottr.edge_template`` already declares
             ``src``/``dst``/``weight``, and ``kind`` widens it.  Contact is symmetric and
             the rules read edges directionally, so supply both directions.

Deliberate differences from the paper, none of them hiding a limitation of the framework:
the activity network is not rebuilt hourly from NHTS schedules (one static weighted graph,
``kind`` carrying what the 22 activities carried); ``Delta t_LLM`` is not a separate clock
(the decision rule runs every tick, and the prompt's *bucketed* prevalence is what makes
the memo collapse a run to a few hundred completions); and the population is synthetic.
"""

from string import Template

from skabm.rules import _PREFIXES

# ---------------------------------------------------------------------------
# Update rules — Model.update, every tick (one tick = one day)
# ---------------------------------------------------------------------------

# County prevalence, copied onto every group so the prompt can read it.  The
# paper's prompt states one county-wide figure ("0.02% people infected by
# COVID-19"), not a neighbourhood one, so this is a single ungrouped aggregate
# broadcast across groups rather than a per-group measure.
PREVALENCE = Template(
    _PREFIXES
    + """
    DELETE { ?g def:prevalence ?p0 }
    INSERT { ?g def:prevalence ?p1 }
    WHERE {
        ?g a ex:Group .
        OPTIONAL { ?g def:prevalence ?p0 }
        { SELECT (AVG(?inf) AS ?p1) WHERE { ?a a ex:Person ; def:infected ?inf } }
    }
    """
)

# The LLM rule, and the paper's prompt verbatim (Section 3.2, sample prompt box).
# `llm:choose` maps yes/no to 1/0 — see rules.register_llm(choices=GOES_OUT).
#
# The prevalence percentage is *bucketed* before it reaches the prompt, and that
# is the whole cost model.  Unrounded, every tick invents a new sentence for
# every group and the memo never hits; at $bucket per mille the run asks
# |groups| x |buckets| distinct questions however long it runs — which is this
# framework's version of the paper's weekly Delta t_LLM, arrived at by making
# the question coarse rather than by adding a second clock.
DECIDE = Template(
    _PREFIXES
    + """
    DELETE { ?g def:goes_out ?o0 }
    INSERT { ?g def:goes_out ?o1 }
    WHERE {
        ?g a ex:Group ;
           def:municipality ?city ;
           def:age_band ?age ;
           def:sex ?sex ;
           def:race ?race ;
           def:prevalence ?prev .
        OPTIONAL { ?g def:goes_out ?o0 }
        BIND(ROUND(?prev * $bucket) AS ?steps)
        BIND(?steps * 1e2 / $bucket AS ?pct)
        BIND(CONCAT(
            "Behavior prediction task\\n",
            "Location: Salt Lake County, UT\\n",
            "Context: ", STR(?pct), "% people infected by COVID-19\\n",
            "Person: ", ?race, " ", ?sex, ", age ", ?age,
            ", lives in ", ?city, ", Salt Lake County, UT\\n",
            "Question: Considering the infected percentage and a person's behavior ",
            "based on demographic factors, will this person go to a public place?"
        ) AS ?q)
        BIND(llm:choose(?q) AS ?o1)
    }
    """
)

# Recovery, I -> R at rate $gamma.  Runs *before* transmission so that an agent
# infected this tick cannot recover in the same tick — the ordering is the
# simultaneity convention, not an accident.
RECOVER = Template(
    _PREFIXES
    + """
    DELETE { ?p def:infected ?i0 . ?p def:recovered ?r0 }
    INSERT { ?p def:infected 0e0 . ?p def:recovered 1e0 }
    WHERE {
        ?p a ex:Person ; def:infected ?i0 ; def:recovered ?r0 .
        FILTER(?i0 > 5e-1)
        FILTER(?r0 < 5e-1)
        BIND(pr:uniform(0e0, 1e0) AS ?u)
        FILTER(?u < $gamma)
    }
    """
)

# Transmission, one Bernoulli draw per *contact* rather than per agent: the
# match runs over (susceptible, infected) edges, so a person with more infected
# contacts gets more draws, and INSERT is set-based so any one success infects.
# That is the individual-based hazard the paper's network model has, without a
# per-agent force-of-infection to accumulate.
#
# The behavioural feedback is the FILTER: a "home" edge always conducts, an
# "out" edge only while *both* endpoints' groups are still going out.  This is
# the one line where the LLM changes the epidemic.
INFECT = Template(
    _PREFIXES
    + """
    DELETE { ?s def:infected ?s0 }
    INSERT { ?s def:infected 1e0 }
    WHERE {
        ?s a ex:Person ; def:infected ?s0 ; def:recovered ?sr ; def:group ?gs .
        FILTER(?s0 < 5e-1)
        FILTER(?sr < 5e-1)
        ?e a ex:Edge ; def:src ?s ; def:dst ?i ; def:weight ?w ; def:kind ?kind .
        ?i def:infected ?ii ; def:group ?gi .
        FILTER(?ii > 5e-1)
        ?gs def:goes_out ?os .
        ?gi def:goes_out ?oi .
        FILTER(?kind = "home" || (?os > 5e-1 && ?oi > 5e-1))
        BIND(pr:uniform(0e0, 1e0) AS ?u)
        FILTER(?u < $beta * ?w)
    }
    """
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# The answer space of the paper's question, and what each answer means for the
# network: yes, this person goes to a public place -> their out-edges conduct.
GOES_OUT = {"yes": 1.0, "no": 0.0}

# Omicron, as the paper parameterises it (Section 3): basic reproduction number
# 9.5 (IQR 7.25-11.88), recovery rate 1/5 per day, and "we adjust the
# transmissibility for each edge according to the average degree" — a mean
# degree of 22.41 in their daily networks. So beta = R0 * gamma / <k>, per
# contact per day, and a heavier edge weight scales it.
HALE_PARAMS = {
    "beta": 9.5 * 0.2 / 22.41,
    "gamma": 0.2,
    "bucket": 1e3,  # prevalence rounded to a tenth of a percent for the prompt
}

# PREVALENCE feeds DECIDE feeds INFECT; RECOVER before INFECT (see above).
HALE_UPDATE_RULES = (PREVALENCE, DECIDE, RECOVER, INFECT)
