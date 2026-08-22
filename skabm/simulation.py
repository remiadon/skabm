"""
RDFSimulator: advance a maplib knowledge-graph ABM with SPARQL update rules.

The sklearn contract, adapted to ABM: X is a set of keyword DataFrames
(one calibrated population per agent class)::

    sim = RDFSimulator(n_periods=12)          # Poledna rules by default
    sim.fit(Firm=firms, Household=households, CentralBank=central_bank)

``init_rules`` and ``update_rules`` are ``__init__`` parameters
(``string.Template`` SPARQL — serializable, so ``get_params`` / ``clone``
work), defaulting to the full Poledna rule sets
(``behaviour.params.poledna_params``).  Rule *logic* lives in the templates;
rule *parameters* live in the ``params`` dict, merged over the canonical
Poledna values and substituted into the templates' ``$placeholders`` at fit
time — overriding one number (``params={"total_deposits": 2.5e4}``) never
means re-writing a rule.  The defaults self-scope to the agent kinds actually
passed: at fit time, rules whose referenced classes (``ex:Firm``,
``ex:CentralBank``, ...) are all absent from the populations are filtered out
entirely, so a use-case with only ``Firm=`` and ``Household=`` gets exactly
the firm and household dynamics.

``fit_iter`` is the generator variant of ``fit``: it yields the raw
per-agent state (``rules.state_extract``) after each tick, and summary
logic stays in polars expressions on the caller's side.

Lifecycle: an empty maplib ``Model`` is created at ``__init__`` and exposed
as ``model_`` — the *fitted artifact*.  A cold ``fit``/``fit_iter`` rebuilds
it from scratch, maps each population with ``rules.map_df``, applies
``init_rules`` (necessarily *after* mapping), then advances ``n_periods``
ticks of ``update_rules`` upserts.  Population keywords not referenced by any
rule are mapped but trigger a ``UserWarning``.

``warm_start=True`` skips the rebuild/map/init phase and keeps ticking the
existing ``model_`` — possibly under *different* update rules, after a
do-calculus style intervention (``model_.update``), or on a model built by
hand.  Passing populations together with ``warm_start=True`` is an error.

The graph's content splits into **structure** (predicates no update rule
DELETEs/INSERTs: links, coefficients, classes — written at fit, immutable
during simulation) and **state** (predicates the update rules upsert: output,
price, wealth, ... — owned by the rules after t=0).  The partition is
derivable from the rule strings themselves; keep interventions on structure
between passes.

``model_`` is a regular maplib model post-fit: SPARQL queries, interventions,
``explore()`` visualization, and serialization all work on it directly.

TODO: numerical backends — compile the graph to polars frames
(``Model.query`` -> ``pl.DataFrame`` -> polars expressions, or ``.to_jax()``
for differentiable kernels), step in frame-land, and re-map at observation
points.  The SPARQL path below then becomes the slow, semantically
transparent reference implementation the fast kernels are validated against.
"""

from __future__ import annotations

import warnings
from string import Template
from typing import Callable, Iterator, Sequence

import polars as pl
import polars_random as pr
from maplib import Model
from sklearn.base import BaseEstimator

from skabm.rules import EX_NS, map_df, render, state_extract, register_polars_random
from skabm.behaviour.params import poledna_params as _BEHAVIOUR_PARAMS
from skabm.behaviour.household import (
    household_income_init,
    household_income,
    household_wealth_init,
    satisificing_consume,
)
from skabm.behaviour.firm import (
    firm_ownership,
    firm_produce,
    firm_price,
    firm_sales,
)
from skabm.behaviour.macro import (
    government_spend,
    centralbank_rate,
)

# Canonical Poledna (2023) rule composition, sourced from behaviour/.
# Users override via __init__(init_rules=..., update_rules=..., params=...).
DEFAULT_INIT_RULES = (firm_ownership, household_income_init, household_wealth_init)
DEFAULT_UPDATE_RULES = (
    firm_produce,  # supply choice, eq. 5 + 12
    firm_price,  # price setting, eq. 8
    household_income,  # income refresh, eq. 49
    satisificing_consume,  # consumption + savings, eqs. 40 + 50
    firm_sales,  # goods market, eqs. 1-2 + 27 + 31
    government_spend,  # AR(1), eq. 51
    centralbank_rate,  # Taylor rule, eq. 69
)


def _inject_metadata(model: Model, rules: Sequence) -> None:
    """Insert behaviour provenance triples into *model* for templates that carry metadata.

    Covers the ``template.metadata`` injection path used by
    ``RDFSimulator._fit_iter``.  Used directly by tests so the block is
    exercised without a full cold-fit.
    """
    for rule in rules:
        if hasattr(rule, "metadata"):
            m = rule.metadata
            kind = m.get("agentClass", "")
            # Accept both ex:Firm and http://example.net/skabm#Firm forms
            if kind.startswith("ex:"):
                kind = EX_NS + kind[3:]
            if kind.startswith(EX_NS):
                kind_name = kind[len(EX_NS) :]
                class_iri = EX_NS + kind_name
                beh_iri = EX_NS + m["@id"]
                model.map_triples(
                    pl.DataFrame(
                        {
                            "subject": [class_iri],
                            "predicate": [EX_NS + "behaviour"],
                            "object": [beh_iri],
                        }
                    )
                )
                model.map_triples(
                    pl.DataFrame(
                        {
                            "subject": [beh_iri],
                            "predicate": [
                                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
                            ],
                            "object": [EX_NS + "Behaviour"],
                        }
                    )
                )
                if "source" in m:
                    from rdflib import Literal

                    model.map_triples(
                        pl.DataFrame(
                            {
                                "subject": [beh_iri],
                                "predicate": ["http://purl.org/dc/terms/source"],
                                "object": [Literal(m["source"])],
                            }
                        )
                    )


class RDFSimulator(BaseEstimator):
    """Advance a maplib knowledge-graph ABM with SPARQL update rules.

    Parameters
    ----------
    init_rules : Sequence[Template | str]
        SPARQL CONSTRUCT rules applied once through ``Model.insert`` right
        after the populations are mapped (e.g. ``behaviour.household.household_income_init``).
        ``string.Template`` rules get their ``$placeholders`` substituted
        from ``params``; plain strings pass through.  Rules anchored on
        unmapped agent classes no-op harmlessly.
    update_rules : Sequence[Template | str]
        SPARQL UPDATE rules (DELETE/INSERT upserts) applied in order within
        each tick — the model's event sequence (Poledna Section 3.5).
    infer : str | Sequence[Template | str] | None
        Optional Datalog or recursive SPARQL CONSTRUCT rules evaluated by
        ``Model.infer`` (licensed feature: maplib commercial add-on, free for
        academic / personal use) once per tick, **after** the update rules have
        advanced state and **before** the next tick begins.  The inference runs
        to a fixed point — derived triples feed back into the rule bodies until
        nothing new is produced — so a single ``infer`` call propagates relational
        effects (contagion, transitive closure, reachability) across the whole
        graph without hand-tuned sub-tick passes.  ``infer=None`` (default) keeps
        the simulator in its original Poledna (2023) configuration where banks
        are passive and no multi-hop propagation exists.  Because the feature is
        licensed, users must ensure their maplib installation carries the add-on;
        calling ``infer`` on a build without it raises at the ``Model.infer`` call.
        Pass a single rule string or a sequence of ``Template``/string rules;
        templates get their ``$placeholders`` substituted from ``params`` exactly
        like ``update_rules``, and plain strings pass through unrendered.
    n_periods : int
        Number of ticks ``fit`` runs (and ``fit_iter`` yields).
    warm_start : bool
        When True, ``fit``/``fit_iter`` continue ticking the existing
        ``model_`` instead of rebuilding it — no populations may be passed.
    state_extract : callable
        ``Model -> pl.DataFrame``, called by ``fit_iter`` after each tick to
        yield raw per-agent state.  Defaults to ``rules.state_extract`` (the
        Poledna columns); a different model family passes its own (e.g.
        ``schelling.state_extract``).  Which predicates are observable is a
        property of the rule set, not of the simulator.
    random_seed : int | None
        When set, ``pr.set_random_seed`` is called at the start of each
        ``fit``/``fit_iter`` so the ``pr:uniform`` / ``pr:normal`` SPARQL UDFs
        (registered on ``model_`` via ``rules.register_polars_random``) draw a
        reproducible sequence.  Leave None for entropy-seeded stochastic runs.

    Attributes
    ----------
    model_ : maplib.Model
        The world state: empty after ``__init__``, populated and evolved by
        ``fit`` / ``fit_iter``.
    """

    def __init__(
        self,
        init_rules: Sequence[Template | str] = DEFAULT_INIT_RULES,
        update_rules: Sequence[Template | str] = DEFAULT_UPDATE_RULES,
        infer: Sequence[Template | str] | None = None,
        params: dict = _BEHAVIOUR_PARAMS,
        n_periods: int = 12,
        warm_start: bool = False,
        state_extract: Callable[[Model], pl.DataFrame] = state_extract,
        random_seed: int | None = None,
    ):
        self.init_rules = init_rules
        self.update_rules = update_rules
        self.infer = infer
        self.params = params
        self.n_periods = n_periods
        self.warm_start = warm_start
        self.state_extract = state_extract
        self.random_seed = random_seed
        self.model_ = Model()

    def _fit_iter(self, **populations: pl.DataFrame) -> Iterator[None]:
        """Advance the model one tick per iteration (no extraction)."""
        if self.random_seed is not None:
            pr.set_random_seed(self.random_seed)
        merged = {**_BEHAVIOUR_PARAMS, **self.params}
        init_rules = [render(rule, merged) for rule in self.init_rules]
        update_rules = [render(rule, merged) for rule in self.update_rules]
        infer_rules: list[str] | None = None
        if self.infer is not None:
            infer_rules = [render(rule, merged) for rule in self.infer]
        if self.warm_start:
            if populations:
                raise ValueError(
                    "warm_start=True continues the existing model_; do not pass "
                    "populations (assign model_ directly instead)."
                )
            register_polars_random(self.model_)
        else:
            rules_text = "\\n".join((*init_rules, *update_rules, *(infer_rules or ())))
            for kind in populations:
                if f"ex:{kind}" not in rules_text:
                    warnings.warn(
                        f"population {kind!r} is not referenced by any init/update/"
                        "infer rule (no 'ex:' + kind pattern): it will be mapped into the "
                        "model but stay inert during simulation.",
                        UserWarning,
                        stacklevel=3,
                    )
            self.model_ = Model()
            register_polars_random(self.model_)
            for kind, df in populations.items():
                map_df(self.model_, df, kind)
            all_rules = (*self.init_rules, *self.update_rules, *(self.infer or ()))
            _inject_metadata(self.model_, all_rules)
            for rule in init_rules:
                self.model_.insert(rule)
        for _ in range(self.n_periods):
            for rule in update_rules:
                self.model_.update(rule)
            if self.infer is not None:
                self.model_.infer(infer_rules)  # type: ignore[arg-type]
            yield

    def fit_iter(self, **populations: pl.DataFrame) -> Iterator[pl.DataFrame]:
        """Map the populations, apply init rules, then yield per-agent state
        (``self.state_extract``) after each of the ``n_periods`` ticks.

        Keyword names are agent classes (``Firm=...``, ``Household=...``);
        each value is the population DataFrame mapped via ``rules.map_df``.
        """
        for _ in self._fit_iter(**populations):
            yield self.state_extract(self.model_)

    def fit(self, **populations: pl.DataFrame) -> "RDFSimulator":
        """Map the populations, apply init rules, run all ticks; return self."""
        for _ in self._fit_iter(**populations):
            pass
        return self
