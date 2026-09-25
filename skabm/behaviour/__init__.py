"""
The rule library, by model family: ``firm``, ``household``, ``macro``, ``bank``,
``labour``, ``learning``, ``schelling``, ``traffic``.

A module's docstring specifies its rules; the code below it is the rules, as dicts
(``skabm.dsl``), listed in ``RULES`` in the order each step runs them.  Everything
before the first step (links, derived populations, starting values) is data, built by
the module's polars functions before mapping.  Each module's ``PARAMETERS`` gives every parameter its rules read that
has a published value, next to its citation.  ``defaults()`` merges them, which is what
``RDFSimulator`` lays its ``params=`` over.
"""

import importlib
import pkgutil
from functools import cache


@cache
def defaults() -> dict:
    """Every module's ``PARAMETERS``, by name; a name has one value across modules."""
    merged: dict = {}
    for info in pkgutil.iter_modules(__path__):
        module = importlib.import_module(f"{__name__}.{info.name}")
        for symbol, value in getattr(module, "PARAMETERS", {}).items():
            name = str(symbol)
            if merged.setdefault(name, value) != value:
                raise ValueError(
                    f"{name} is {merged[name]} in one module, {value} in {module.__name__}"
                )
    return merged
