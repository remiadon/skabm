"""
The rule library, by agent type: ``firm``, ``household``, ``macro`` (government and
central bank), ``bank``, ``labour``, ``residence``, ``traffic``, and ``learning``, the
expectations they share.

A module is how one type of agent behaves, according to each source it cites. Its
docstring specifies the rules; the code below it is the rules, as dicts (``skabm.dsl``),
named after their source (``firm.poledna_produce``, ``firm.canvas_heuristic``) and listed
per source in the order each step runs them (``firm.poledna_rules``,
``firm.canvas_rules``), so a model takes one source's rules or mixes several.  Everything
before the first step (links, derived populations, starting values) is data, built by
the module's polars functions before mapping.  Each module's ``PARAMETERS`` gives every
parameter its rules read that has a published value, next to its citation.
``defaults()`` merges them, which is what ``RDFSimulator`` lays its ``params=`` over.
"""

import importlib
import pkgutil
from functools import cache


@cache
def defaults() -> dict:
    """Every module's ``PARAMETERS``, by name; a name has one value across modules."""
    cited = {
        (str(symbol), value)
        for info in pkgutil.iter_modules(__path__)
        for symbol, value in getattr(
            importlib.import_module(f"{__name__}.{info.name}"), "PARAMETERS", {}
        ).items()
    }
    names = [name for name, _ in cited]
    assert len(names) == len(set(names)), (
        f"a name with two values: {sorted(c for c in cited if names.count(c[0]) > 1)}"
    )
    return dict(cited)
