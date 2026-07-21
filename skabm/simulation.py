"""RDFSimulator: run an ABM by applying SPARQL update rules to a maplib model.

The division of labour with the rest of skabm:

* **Calibration** (skabm.calibration) produces agent populations as polars
  DataFrames.
* **Model construction** (user code) lifts them into a maplib ``Model``
  with ``rules.map_df`` and applies the initialization rules
  (``Model.insert``, e.g. ``rules.household_income``).
* **RDFSimulator** then advances that model tick by tick.  It is an
  *iterable*: each iteration applies every update rule (a SPARQL
  DELETE/INSERT upsert through ``Model.update`` — no ``Model.map`` calls,
  no DataFrame round-trips) and yields the raw per-agent state from
  ``rules.state_extract()``.  Summary logic stays in polars expressions on
  the caller's side::

      sim = RDFSimulator(update_rules=[...], n_periods=12).fit(model)
      for state in sim:
          print(state.select((pl.col("price") * pl.col("output")).sum()))

The model is mutated in place (also exposed as ``model_``): the graph is
the single evolving world state, which keeps it inspectable with maplib's
``explore()`` at any point.  Iterating again continues from where the last
pass stopped — rebuild the model to restart from initial conditions, since
maplib models cannot be deep-copied.

TODO: numerical backends — compile the graph to polars frames
(``Model.query`` -> ``pl.DataFrame`` -> polars expressions, or ``.to_jax()``
for differentiable kernels), step in frame-land, and re-map at observation
points.  The SPARQL path below then becomes the slow, semantically
transparent reference implementation the fast kernels are validated
against.
"""

from __future__ import annotations

from typing import Iterator, Sequence

import polars as pl
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from skabm.rules import state_extract


class RDFSimulator(BaseEstimator):
    """Advance a maplib knowledge-graph ABM with SPARQL update rules.

    Parameters
    ----------
    update_rules : Sequence[str]
        SPARQL UPDATE strings (DELETE/INSERT upserts, e.g. from
        ``skabm.rules``), applied in order within each tick.  The order is
        the model's event sequence (Poledna Section 3.5): e.g. production,
        pricing, income, consumption, sales, government, monetary policy.
    n_periods : int
        Number of ticks one iteration pass yields.

    Attributes
    ----------
    model_ : maplib.Model
        The evolving model bound by ``fit`` (the same object, mutated).
    """

    def __init__(self, update_rules: Sequence[str] = (), n_periods: int = 12):
        self.update_rules = update_rules
        self.n_periods = n_periods

    def fit(self, model, y=None) -> "RDFSimulator":
        """Bind the constructed model; ticks run on iteration."""
        self.model_ = model
        return self

    def __iter__(self) -> Iterator[pl.DataFrame]:
        """Advance one tick per iteration and yield the raw per-agent state."""
        check_is_fitted(self, "model_")
        for _ in range(self.n_periods):
            for rule in self.update_rules:
                self.model_.update(rule)
            yield self.model_.query(state_extract())
