"""RDFSimulator: run an ABM by applying SPARQL update rules to a maplib model.

The sklearn contract, adapted to ABM: in sklearn ``fit`` takes one array X,
but an ABM needs heterogeneous agent populations laid out as DataFrames of
different sizes.  X is therefore a set of keyword arguments to ``fit`` —
one calibrated DataFrame per agent class::

    sim = RDFSimulator(n_periods=12)          # Poledna rules by default
    sim.fit(Firm=firms, Household=households, CentralBank=central_bank)

Everything else follows sklearn: ``init_rules`` and ``update_rules`` are
``__init__`` parameters (``string.Template`` SPARQL — serializable, so
``get_params`` / ``clone`` work), defaulting to the full Poledna rule sets
(``rules.DEFAULT_INIT_RULES`` / ``rules.DEFAULT_UPDATE_RULES``).  Rule
*logic* lives in the templates; rule *parameters* live in the ``params``
dict, merged over ``rules.POLEDNA_PARAMS`` and substituted into the
templates' ``$placeholders`` at fit time — overriding one number
(``params={"total_deposits": 2.5e4}``) never means re-writing a rule.
The defaults self-scope to the agent kinds actually passed: at fit time,
rules whose referenced classes (``ex:Firm``, ``ex:CentralBank``, ...) are
all absent from the populations are filtered out entirely, so a use-case
with only ``Firm=`` and ``Household=`` gets exactly the firm and household
dynamics.  Declaring a new economic ABM with newer data is just
calibrating new DataFrames.

``fit_iter`` is the generator variant of ``fit``: it yields the raw
per-agent state (``rules.state_extract``) after each tick, and summary
logic stays in polars expressions on the caller's side.

Lifecycle: an empty maplib ``Model`` is created at ``__init__`` and exposed
as ``model_`` — the *fitted artifact*, where data and rules blend into one
ontology-shaped world.  A cold ``fit``/``fit_iter`` rebuilds it from
scratch (sklearn semantics: refitting restarts the world), maps each
population with ``rules.map_df``, applies ``init_rules`` — necessarily
*after* mapping, since init rules are CONSTRUCTs over agent patterns and
insert nothing into an empty graph — then advances ``n_periods`` ticks of
``update_rules`` upserts.  Population keywords not referenced by any rule
are mapped but trigger a ``UserWarning``, since no rule will ever touch
them.

``warm_start=True`` skips the rebuild/map/init phase entirely and keeps
ticking the existing ``model_`` — possibly under *different* update rules,
after a do-calculus style intervention (``model_.update``), or on a model
built by hand (assign ``model_`` yourself).  Passing populations together
with ``warm_start=True`` is an error: the world already exists.

The graph's content splits into **structure** (predicates no update rule
DELETEs/INSERTs: links, coefficients, classes — written at fit, immutable
during simulation, editable only by explicit user intervention) and
**state** (predicates the update rules upsert: output, price, wealth, ...
— owned by the rules after t=0).  The partition is derivable from the rule
strings themselves; keep interventions on structure between passes.

Users never need to know maplib to run a simulation — but ``model_`` is a
regular maplib model they can embrace post-fit: SPARQL queries,
interventions, ``explore()`` visualization, or serialization.

TODO: numerical backends — compile the graph to polars frames
(``Model.query`` -> ``pl.DataFrame`` -> polars expressions, or ``.to_jax()``
for differentiable kernels), step in frame-land, and re-map at observation
points.  The SPARQL path below then becomes the slow, semantically
transparent reference implementation the fast kernels are validated
against.
"""

from __future__ import annotations

import warnings
from string import Template
from typing import Iterator, Sequence

import polars as pl
from maplib import Model
from sklearn.base import BaseEstimator

from skabm.rules import (
    DEFAULT_INIT_RULES,
    DEFAULT_UPDATE_RULES,
    POLEDNA_PARAMS,
    map_df,
    render,
    state_extract,
)


class RDFSimulator(BaseEstimator):
    """Advance a maplib knowledge-graph ABM with SPARQL update rules.

    Parameters
    ----------
    init_rules : Sequence[Template | str]
        SPARQL CONSTRUCT rules applied once through ``Model.insert`` right
        after the populations are mapped (e.g. ``rules.HOUSEHOLD_INCOME``).
        ``string.Template`` rules get their ``$placeholders`` substituted
        from ``params``; plain strings pass through.  Rules anchored on
        unmapped agent classes no-op harmlessly.
    update_rules : Sequence[Template | str]
        SPARQL UPDATE rules (DELETE/INSERT upserts) applied in order within
        each tick — the model's event sequence (Poledna Section 3.5).
    params : dict
        Substitutes for the rule templates' ``$placeholders``, merged over
        ``rules.POLEDNA_PARAMS`` — pass only what differs (e.g.
        ``{"total_deposits": 2.5e4}``).  Numeric values are injected as
        xsd:double literals.
    n_periods : int
        Number of ticks ``fit`` runs (and ``fit_iter`` yields).
    warm_start : bool
        When True, ``fit``/``fit_iter`` continue ticking the existing
        ``model_`` instead of rebuilding it — no populations may be passed.

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
        params: dict = POLEDNA_PARAMS,
        n_periods: int = 12,
        warm_start: bool = False,
    ):
        self.init_rules = init_rules
        self.update_rules = update_rules
        self.params = params
        self.n_periods = n_periods
        self.warm_start = warm_start
        self.model_ = Model()

    def _fit_iter(self, **populations: pl.DataFrame) -> Iterator[None]:
        """Advance the model one tick per iteration (no extraction)."""
        merged = {**POLEDNA_PARAMS, **self.params}
        init_rules = [render(rule, merged) for rule in self.init_rules]
        update_rules = [render(rule, merged) for rule in self.update_rules]
        if self.warm_start:
            if populations:
                raise ValueError(
                    "warm_start=True continues the existing model_; do not pass "
                    "populations (assign model_ directly instead)."
                )
        else:
            rules_text = "\n".join((*init_rules, *update_rules))
            for kind in populations:
                if f"ex:{kind}" not in rules_text:
                    warnings.warn(
                        f"population {kind!r} is not referenced by any init/update "
                        f"rule (no 'ex:{kind}' pattern): it will be mapped into the "
                        "model but stay inert during simulation.",
                        UserWarning,
                        stacklevel=3,
                    )
            self.model_ = Model()
            for kind, df in populations.items():
                map_df(self.model_, df, kind)
            for rule in init_rules:
                self.model_.insert(rule)
        for _ in range(self.n_periods):
            for rule in update_rules:
                self.model_.update(rule)
            yield

    def fit_iter(self, **populations: pl.DataFrame) -> Iterator[pl.DataFrame]:
        """Map the populations, apply init rules, then yield per-agent state
        (``rules.state_extract``) after each of the ``n_periods`` ticks.

        Keyword names are agent classes (``Firm=...``, ``Household=...``);
        each value is the population DataFrame mapped via ``rules.map_df``.
        """
        for _ in self._fit_iter(**populations):
            yield state_extract(self.model_)

    def fit(self, **populations: pl.DataFrame) -> "RDFSimulator":
        """Map the populations, apply init rules, run all ticks; return self."""
        for _ in self._fit_iter(**populations):
            pass
        return self
