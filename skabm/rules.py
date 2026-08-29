"""
Infrastructure for SPARQL-based ABM simulation.

Provides the constants, rendering, mapping, and random-UDF registration that
the behaviour templates (``skabm.behaviour.*``) and the simulator
(``skabm.simulation.RDFSimulator``) depend on.

SPARQL rule *logic* lives in ``skabm.behaviour`` (firm.py, household.py,
macro.py).  This module carries only the plumbing: namespaces, ``render()``,
``register_polars_random()``, ``register_math()``, ``register_llm()`` and
``dbl()``.  Mapping is
``skabm.ottr``'s.

There is no ``state_extract`` here any more: what a model's per-agent frame
should contain is derivable from the rules themselves, and ``skabm.ir`` derives
it (``ModelIR.extract``).
"""

from __future__ import annotations

from string import Template

import polars as pl
import polars_random as pr
from maplib import xsd

EX_NS = "http://example.net/skabm#"
DEF_NS = "urn:maplib_default:"
PR_NS = "urn:pr:"  # polars-random UDFs, registered by register_polars_random
MATH_NS = "urn:math:"  # transcendental UDFs, registered by register_math
LLM_NS = "urn:llm:"  # LLM decisions, registered by register_llm

_PREFIXES = (
    f"PREFIX ex:<{EX_NS}>\n"
    f"PREFIX def:<{DEF_NS}>\n"
    f"PREFIX pr:<{PR_NS}>\n"
    f"PREFIX math:<{MATH_NS}>\n"
    f"PREFIX llm:<{LLM_NS}>\n"
    "PREFIX xsd:<http://www.w3.org/2001/XMLSchema#>\n"
)


def register_polars_random(model) -> None:
    """Expose polars-random draws to SPARQL as UDFs (maplib >= 0.20.26).

    Registers ``pr:uniform(low, high)`` and ``pr:normal(mean, std)`` — callable
    in any rule via ``BIND(pr:uniform(0e0, 1e0) AS ?u)`` — giving the graph the
    seedable RAND plain SPARQL lacks.  Each UDF receives a DataFrame with one
    column per argument (``"0"``, ``"1"``) and returns the ``out`` Series.

    The draw uses polars-random's ``size=`` Series form (the only one that
    honours ``pr.set_random_seed``; the Series-argument form returns a
    non-reproducible expression) and affine-transforms it per row, so bounds
    may vary by row and a run is reproducible whenever the caller fixes the
    seed (see ``RDFSimulator(random_seed=...)``).
    """

    def _uniform(df: pl.DataFrame) -> pl.Series:
        u = pr.uniform(0.0, 1.0, size=len(df))
        return (df["0"] + (df["1"] - df["0"]) * u).alias("out")

    def _normal(df: pl.DataFrame) -> pl.Series:
        z = pr.normal(0.0, 1.0, size=len(df))
        return (df["0"] + df["1"] * z).alias("out")

    model.add_udf(PR_NS + "uniform", _uniform, xsd.double, [xsd.double, xsd.double])
    model.add_udf(PR_NS + "normal", _normal, xsd.double, [xsd.double, xsd.double])


def register_math(model) -> None:
    """Expose the transcendental functions SPARQL lacks as UDFs.

    Registers ``math:exp(x)`` and ``math:log(x)`` — callable in any rule via
    ``BIND(math:exp(?x) AS ?y)``.  Plain SPARQL has no ``EXP``/``LOG`` (only
    the four arithmetic operators plus ``ABS``/``ROUND``/``FLOOR``/``CEIL``),
    which is what forces log-level laws of motion into linear growth factors
    in the Poledna rule set.  Models whose *specification* is transcendental —
    the urn-ball matching function and the S-curve technology shock of
    del Rio-Chanona et al. (2021) — need the real thing, so it is registered
    here on the same ``Model.add_udf`` seam as the random draws.

    Pass alongside ``register_polars_random`` via
    ``RDFSimulator(udfs=(register_polars_random, register_math))``.
    """

    def _exp(df: pl.DataFrame) -> pl.Series:
        return df["0"].exp().alias("out")

    def _log(df: pl.DataFrame) -> pl.Series:
        return df["0"].log().alias("out")

    model.add_udf(MATH_NS + "exp", _exp, xsd.double, [xsd.double])
    model.add_udf(MATH_NS + "log", _log, xsd.double, [xsd.double])


def _register_choice(model, batch, choices: dict, ns: str = LLM_NS) -> None:
    """Register ``llm:choose(?prompt)`` over a ``list[str] -> list[str]`` batch.

    The UDF returns ``xsd:double``, not the model's word: the answer space is
    known when the decider is built, so ``choices`` maps it to a number there
    and the graph never holds free text.  That matters twice over — a
    string-valued predicate makes ``skabm.ir`` propose ``AVG`` over it and
    maplib panics inside the aggregate, and parsing the word in SPARQL instead
    (``IF(?a = "yes", 1e0, 0e0)``) trips the IR's regime detector into naming
    two observables the same thing.  A number is also a measurement, so the
    share of groups choosing each branch comes back as an observable nobody
    declared.

    **One call per distinct prompt per run, not per agent per tick.**  Answers
    are memoized on the prompt string for the life of the registrar.  That is
    not an optimisation bolted on: it is the reduction Moon et al. (2026) build
    their scalability claim on — one LLM agent per demographic group rather
    than per person — arrived at here by making the prompt coarse.  Two agents
    whose prompt strings are identical get the same answer, so a rule wanting
    them to differ splices in what differs.

    An answer outside ``choices`` becomes null, and a null object makes the
    INSERT skip that subject rather than write a wrong number.
    """
    seen: dict[str, float | None] = {}

    def _choose(df: pl.DataFrame) -> pl.Series:
        prompts = df["0"]
        todo = [p for p in prompts.unique() if p not in seen]
        if todo:
            seen.update(
                (p, choices.get(str(a).strip().lower()))
                for p, a in zip(todo, batch(todo))
            )
        return prompts.replace_strict(seen, return_dtype=pl.Float64).alias("out")

    model.add_udf(ns + "choose", _choose, xsd.double, [xsd.string])


def register_llm(
    model,
    chat: str = "claude-haiku-4-5-20251001",
    system: str | None = None,
    choices: dict | None = None,
    provider: str = "aanthropic",
    **kwargs,
) -> None:
    """Expose an LLM to SPARQL as ``llm:choose(?prompt)``, needs ``polars-llm``.

    Same seam as the random draws and ``exp``/``log``: a rule builds a prompt
    out of an agent's own triples with ``CONCAT``, calls the UDF in a ``BIND``,
    and the decision becomes a triple like any other value::

        BIND(CONCAT("Context: ", STR(?pct), "% infected ...") AS ?q)
        BIND(llm:choose(?q) AS ?goes_out)

    which is the HALE architecture of Moon et al. (2026) — an LLM standing in
    for a behavioural rule the data cannot supply — with no simulator change:
    the model is a function inside the rule, not a stage around it.  See
    ``skabm.hale``.

    ``provider`` names the ``polars_llm`` method, so an OpenAI-compatible server
    is a base URL rather than new code — which is how to run this without an
    account::

        register_llm(model, provider="aopenai", chat="<model>",
                     base_url="http://localhost:11434/v1", api_key="ollama")

    covering Ollama, vLLM, llama.cpp and LM Studio.  Remaining ``kwargs`` reach
    the LangChain chat constructor (``max_tokens=``, ``base_url=``), which is
    also where request headers go: an identity-linked Anthropic key rejects
    every call with ``400 anthropic-workspace-id is required`` until it is told
    which workspace it acts in::

        register_llm(model, default_headers={"anthropic-workspace-id": "wrkspc_..."})

    Two defaults are flipped on the way, and neither is a preference.
    ``on_error="raise"``, because polars-llm's default of nulling a failed call
    would turn a missing key into a run where nobody ever decided anything.  And
    ``temperature=0``, because the memo caches one answer per prompt: a sampled
    answer is not a function of its prompt, so identical groups would decide
    differently for the rest of the run and the elicited behaviour would read as
    incoherent when the incoherence was the sampler.  (HALE samples deliberately
    at ``temperature=0.2`` to read a *probability* off a group; that is a
    different measurement, and it wants the memo off.)

    Ask for words, not digits.  Given ``choices={"1": 1.0, "0": 0.0}`` a small
    model echoes a digit out of the prompt instead of deciding.
    """
    import polars_llm  # noqa: F401  — registers the `.llm` expression namespace

    def _batch(prompts: list[str]) -> list[str]:
        return (
            pl.DataFrame({"prompt": prompts})
            .select(
                getattr(pl.col("prompt").llm, provider)(
                    model=chat,
                    system=system,
                    **{"on_error": "raise", "temperature": 0, **kwargs},
                )
            )
            .to_series()
            .to_list()
        )

    _register_choice(model, _batch, choices or {"yes": 1.0, "no": 0.0})


# maplib SPARQL gotcha, worth knowing before writing any rule: arithmetic
# operators of equal precedence associate to the *right*, against the SPARQL
# grammar.  ``?a - ?b + ?c`` evaluates as ``?a - (?b + ?c)`` and ``?a / ?b * ?c``
# as ``?a / (?b * ?c)``; both are silently wrong, with no error and no warning.
# Chains that start with ``+`` or ``*`` happen to survive (``a + (b - c)`` and
# ``a * (b / c)`` are algebraically what you meant), which is why the Poledna
# and Schelling rules are unaffected — but a chain led by ``-`` or ``/`` is a
# live bug.  Bracket every mixed chain explicitly.


def dbl(x: float) -> str:
    """Format a Python float as a SPARQL xsd:double literal."""
    return f"{x:.6e}"


def render(rule: Template | str, params: dict) -> str:
    """Substitute a rule Template's $-placeholders with xsd:double literals.

    Numeric parameter values go through ``dbl`` so decimal literals can
    never leak into the SPARQL; plain-string rules pass through unchanged.
    Missing placeholders raise ``KeyError`` (loudly, at fit time).
    """
    if isinstance(rule, Template):
        return rule.substitute(
            {k: dbl(v) if isinstance(v, (int, float)) else v for k, v in params.items()}
        )
    return rule
