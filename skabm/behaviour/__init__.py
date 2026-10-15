"""Agent behaviours, one module per type of agent (``firm``, ``household``, ``person``,
``labour_market``, ``trader``, ``commuter``, ``bank``, ``macro``) and the expectations
they share (``learning``).  A rule is named for the mechanism it implements and cites the
source that introduced it; its ``PARAMETERS`` are that source's values.  The polars
functions that build an agent's opening state sit with its rules.  A model is a list of
rules, composed in a paper's validation test and on its page.

``defaults()`` merges every module's ``PARAMETERS``, which is what ``RDFSimulator`` lays
its ``params=`` over.
"""

import importlib
import pkgutil
from functools import cache


@cache
def defaults() -> dict:
    """Every behaviour's ``PARAMETERS``, by name; a name has one value across modules."""
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
