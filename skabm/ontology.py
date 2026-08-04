"""``Ontology``: a maplib ``Model`` that self-registers UDFs and auto-wires SAC-learning.

The end-user of skabm should never hand-write the history bookkeeping that
SAC-learning needs (clocks, observation nodes, ``PROCESS_ACCOUNTING``).  They
just drop a ``rules.sac(agg, predicate, ...)`` call into one of their own update
rules and use its output variable.  ``Ontology`` closes the loop:

* its ``__init__`` registers the ``pr:`` and ``sac:`` UDFs, so any model built
  through it can call them; and
* ``expand_rules`` reads the ``ex:sig__<agg>__<predicate>`` IRIs the ``sac(...)``
  fragments embed straight off the rule text and **tops the rule set up** with
  the matching measurement rule, a ``PROCESS_INIT`` that seeds the signals and a
  shared clock, and one generic ``PROCESS_ACCOUNTING`` — nothing when no rule
  uses ``sac(...)``.

``maplib.Model`` is a pyo3 type Python cannot subclass (``class Ontology(Model)``
raises ``TypeError: type 'builtins.Model' is not an acceptable base type``), so
``Ontology`` *wraps* a ``Model`` and delegates: the methods ``RDFSimulator`` uses
are forwarded explicitly and everything else (``explore``, ``serialize``,
``validate``, ...) falls through ``__getattr__``, so an ``Ontology`` is a drop-in
for a ``Model`` post-fit.
"""

from __future__ import annotations

from maplib import Model

from skabm.rules import (
    PROCESS_ACCOUNTING,
    measure_rule,
    parse_signal_specs,
    process_init,
    register_polars_random,
    register_sac,
)


class Ontology:
    """A ``maplib.Model`` wrapper that owns the ``pr:``/``sac:`` UDFs and the
    SAC-learning auto-wiring.

    Parameters
    ----------
    model : maplib.Model, optional
        Wrap an existing model (e.g. to upgrade a hand-built one for a
        ``warm_start``); a fresh ``Model`` is created otherwise.  UDF
        registration is idempotent, so wrapping is always safe.
    """

    def __init__(self, model: Model | None = None):
        self._model = model if model is not None else Model()
        register_polars_random(self._model)
        register_sac(self._model)

    # -- Model delegation ---------------------------------------------------
    def query(self, *args, **kwargs):
        return self._model.query(*args, **kwargs)

    def insert(self, *args, **kwargs):
        return self._model.insert(*args, **kwargs)

    def update(self, *args, **kwargs):
        return self._model.update(*args, **kwargs)

    def map_default(self, *args, **kwargs):
        return self._model.map_default(*args, **kwargs)

    def map_triples(self, *args, **kwargs):
        return self._model.map_triples(*args, **kwargs)

    def __getattr__(self, name):
        # only reached for attributes not found on Ontology itself; guard the
        # wrapped model's own name so an access before __init__ can't recurse.
        if name == "_model":
            raise AttributeError(name)
        return getattr(self._model, name)

    # -- SAC-learning auto-wiring ------------------------------------------
    def expand_rules(
        self, init_rules: list[str], update_rules: list[str]
    ) -> tuple[list[str], list[str]]:
        """Return the rule set topped up with the bookkeeping SAC-learning needs.

        Detects the ``(agg, predicate)`` signals the rules forecast with
        ``sac(...)`` (from the ``ex:sig__<agg>__<predicate>`` IRIs) and appends,
        only when at least one exists: a ``PROCESS_INIT`` seeding the clock and
        signals (to ``init_rules``), a measurement rule per signal, and a single
        generic ``PROCESS_ACCOUNTING`` (to ``update_rules``).  With no ``sac(...)``
        in sight the rule set is returned unchanged.

        Rules are the already-rendered SPARQL strings; the generated rules carry
        no ``$`` parameters, so they need no further rendering.
        """
        specs = parse_signal_specs([*init_rules, *update_rules])
        if not specs:
            return init_rules, update_rules
        init = [*init_rules, process_init(specs)]
        update = [
            *update_rules,
            *(measure_rule(a, k, p) for a, k, p in specs),
            PROCESS_ACCOUNTING,
        ]
        return init, update
