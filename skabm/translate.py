"""
Rules from a researcher's description and the OTTR templates of the agents involved.

The description is the logic an individual agent follows, in words (a paper's model
section will do).  The templates are the context: which classes exist, their fields,
where each link lands.  They are also the grammar: ``grammar(templates)`` is the DSL
of ``behaviour/labour.py``, one statement per line, over exactly the fields those
templates declare.  A model generating under that grammar (llguidance) can only write
rules skABM compiles, so its answer runs as the Python it is, with nothing but the DSL
in scope, and the rules it assigns are the result: ``Rule({...})`` is a readable name
for the dict itself.

Any template will do: skABM's own (``ottr.TEMPLATES``), ``Model.get_templates()``, or
one a researcher wrote.  A class ``ottr.agent_template`` built also tells the model
what each field means and where its links go (``ottr.SCHEMA``).
"""

from __future__ import annotations

import re
from functools import cache

import polars as pl
import sympy as sp
from sympy.stats import Normal, Uniform

from skabm import dsl
from skabm.behaviour.learning import expect
from skabm.dsl import Agents, DSLError, _Rule, is_rule, sparql
from skabm.ottr import SCHEMA, Link

# A tiny local model: any ``(prompt, lark grammar) -> text`` callable will do.
MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"  # ~15 GB; the 1.5B fails macro.py

# The DSL as a Lark grammar: every function ends in "(", so no name can pass for one.
# ``rule`` is appended per call: one alternative per class, keyed by its own fields.
# FIELD, LINK, PATH, KEY and the class names come from the templates, per call; an
# alternative naming one the templates leave empty is dropped.
GRAMMAR = r"""
start: stmt+  # a line may break inside brackets, as Python's may
stmt: HELPER "=" expr | RULENAME "=" rule
expr: prod (("+" | "-") prod)*
prod: unary (("*" | "/") unary)*
unary: "-" unary | power
power: atom ("**" unary)?
atom: NUMBER | FIELD | "(" expr ")" | call
    | PARAM | HELPER
call: FUNC expr ")" | MINMAX expr ("," expr)+ ")" | DRAW STRING "," expr "," expr ")"
    | "sp.Piecewise(" (branch ",")* "(" expr "," "True" ")" ")"
    | "expect(" AGG "," CLASSNAME "," STRING ")"
    | "sum_over(" LINK "," expr ")"
    | "sum_over(" PATH "," expr ")"
    | "total_by(" KEY "," expr ")"
    | "running_sum(" FIELD "," expr ")"
    | "running_sum(" FIELD "," expr "," KEY ")"
branch: "(" expr "," cond ")"
cond: comparison | part (("&" | "|") part)+ | "~" part
part: "(" comparison ")" | "(" cond ")" | "~" part | eq
comparison: expr ("<" | ">" | "<=" | ">=") expr | eq
eq: EQ expr "," expr ")"
    | EQ KEY "," KEY ")"
FUNC: "sp.exp(" | "sp.log(" | "sp.sqrt(" | "sp.Abs(" | "total(" | "mean(" | "lag("
MINMAX: "sp.Max(" | "sp.Min(" | "coalesce("
EQ: "sp.Eq(" | "sp.Ne("
DRAW: "Normal(" | "Uniform("
AGG: "\"SUM\"" | "\"AVG\""
RULENAME: /[a-z][a-z0-9_]*/
HELPER: /_[a-z][a-z0-9_]*/
NUMBER: /[0-9]+(\.[0-9]+)?([eE]-?[0-9]+)?/
STRING: /"[a-z][a-z0-9_]*"/
%ignore /[ \t\n]+/
"""

PROMPT = """\
Translate the behaviour described at the end into the rules an agent-based model runs
each step: Python, one statement per line, nothing else.  A rule is
`name = Rule({{Class.field: expression, ...}})`; it updates every agent of one class,
reading the values from before it ran, so the fields an agent updates together go in
one rule.  Rules run in the order written.  A quantity the description names ("where
growth is ...") or several fields share is `_growth = expression`, written before it is
used; it is not a rule.  A field read inside a rule is always its value from before the
rule ran, so a value the same rule computes ("that new profit") is a `_name` quantity.  In an
expression:
- `Class.field` is the agent's field; `Class.link.field` the field of the agent its
  link points to.
- `sum_over(Other.link, e)`: e summed over the Other agents whose link points to this
  one, e read on them.  `sum_over(Class.link, e)`: e summed over the agents this
  agent's many-valued link reaches, their fields read as `Target.field`.
- `total(e)`, `mean(e)`: over every agent of e's class.  `total_by(Class.key, e)`: over
  the agents with the same key.  `running_sum(Class.order, e, Class.key)`: over those
  with the same key whose order is at most this one's.
- `sp.Max(a, b)`, `sp.Min(a, b)`; `sp.Piecewise((value, condition), ..., (value,
  True))`, each comparison bracketed when joined with & | ~, as in
  `sp.Piecewise((a / b, (a > 0) & (b > 0)), (0, True))`; equality is `sp.Eq(a, b)`.
- `coalesce(a, b, ...)`: the first of its arguments that exists, the only way to ask
  whether something exists: `coalesce(Class.link.field, e)` is e for an agent without
  that link, and `coalesce(Class.field, e)` keeps a value the agent already has.
- `lag(e)`: e one step ago.  `expect("SUM", "Class", "field")`: the expected growth of
  the total of a field, "AVG" of its mean.  `Normal("eps", 0, sigma)`,
  `Uniform("u", 0, 1)`: a fresh draw per agent, written where it is used.
- `Class.link: coalesce(pick(Other, condition, order), Class.link)` moves the link to
  the Other agent meeting the condition with the least order.
- "the parameter x" in the description is the name `x`; there are no others.
- Each field is written at most once in a rule.

For example:
clock_tick = Rule({{Clock.t: Clock.t + 1}})
_seekers = Edge.weight * Edge.src.unemployment
applications = Rule({{Occupation.applications: Occupation.vacancies * sum_over(Edge.dst, _seekers)}})

The agents:
{agents}

The parameters: {parameters}

The behaviour to translate:
{description}
"""


def _classes(templates) -> dict:
    """``{class: {field: (kind, meaning)}}``, kinds from ``SCHEMA`` when it has them."""
    classes = {}
    for template in templates:
        klass = re.split(r"[#/]", template.iri.iri)[-1]
        known = SCHEMA.get(klass, {})
        classes[klass] = {
            p.variable.name: known.get(p.variable.name)
            or (Link() if "IRI" in str(p.rdf_type) else str(p.rdf_type), "")
            for p in template.parameters
            if p.variable.name != "id"
        }
    return classes


def _number(kind) -> bool:
    """A polars float (a template skABM built) or an xsd:double (anyone's)."""
    return "double" in kind if isinstance(kind, str) else kind.is_float()


def _string(kind) -> bool:
    return "string" in kind if isinstance(kind, str) else kind == pl.String


def grammar(templates, description: str = "") -> str:
    """The DSL over the templates' classes: every field, up to two links away, and as
    parameters the names *description* introduces as "the parameter x" (any name,
    without a description)."""
    classes = _classes(templates)
    terminals = {"FIELD": [], "LINK": [], "PATH": [], "KEY": []}

    def walk(path: str, klass: str, hops: int):
        for name, (kind, _) in classes[klass].items():
            here = f"{path}.{name}"
            if isinstance(kind, Link) and str(kind) in classes:
                if hops < 2:
                    terminals["LINK" if not hops else "PATH"].append(here)
                    walk(here, str(kind), hops + 1)
            elif _number(kind):
                terminals["FIELD"].append(here)
            elif _string(kind) and not hops:
                terminals["KEY"].append(here)

    for klass in classes:
        walk(klass, klass, 0)
    terminals["CLASS"] = list(classes)
    terminals["CLASSNAME"] = [f'\\"{k}\\"' for k in classes]
    terminals["PARAM"] = _parameters(description) if description else ["/[a-z]\\w*/"]
    empty = [name for name, found in terminals.items() if not found]
    kept = [
        line
        for line in GRAMMAR.splitlines(keepends=True)
        if not any(re.search(rf"\b{name}\b", line) for name in empty)
    ]
    return (
        "".join(kept)
        + _statements(classes)
        + "".join(
            f"{name}: "
            + " | ".join(t if t[0] == "/" else f'"{t}"' for t in found)
            + "\n"
            for name, found in terminals.items()
            if found
        )
    )


def _parameters(description: str) -> list:
    """The names *description* introduces as "the parameter x"."""
    return sorted(set(re.findall(r"\bparameters?\s+([a-z][a-z0-9_]*)", description)))


def _statements(classes) -> str:
    """``rule``: a rule over any one class, writing that class's fields."""
    writable = []
    for klass, kinds in classes.items():
        own = [f"{klass}.{f}" for f, (k, _) in kinds.items() if _number(k)]
        links = [
            f"{klass}.{f}"
            for f, (k, _) in kinds.items()
            if isinstance(k, Link) and str(k) in classes
        ]
        if own or links:
            writable.append((own, links))
    lines = []
    for i, (own, links) in enumerate(writable):
        pick = f'LINK{i} ":" "coalesce(pick(" CLASS "," cond "," expr ")," LINK{i} ")"'
        writes = [f'OWN{i} ":" expr'] * bool(own) + [pick] * bool(links)
        rule = f'"Rule({{" write{i} ("," write{i})* "}})"'
        lines += [f"rule{i}: {rule}", f"write{i}: " + " | ".join(writes)]
        lines += [f"OWN{i}: " + " | ".join(f'"{f}"' for f in own)] * bool(own)
        lines += [f"LINK{i}: " + " | ".join(f'"{f}"' for f in links)] * bool(links)
    lines.append("rule: " + " | ".join(f"rule{i}" for i in range(len(writable))))
    return "\n".join(lines) + "\n"


def _accepts(text: str, lark: str) -> bool:
    """Whether *text* is in the grammar: llguidance, fed one byte at a time."""
    import llguidance as llg

    class Bytes:
        tokens = [bytes([i]) for i in range(256)] + [b"<eos>"]
        eos_token_id, bos_token_id = 256, None

        def __call__(self, text: bytes) -> list:
            return list(text)

    tokenizer = llg.LLTokenizer(llg.TokenizerWrapper(Bytes()))
    matcher = llg.LLMatcher(tokenizer, llg.grammar_from("lark", lark), log_level=0)
    return all(map(matcher.consume_token, text.encode())) and matcher.is_accepting()


@cache
def local(name: str = MODEL):  # pragma: no cover - downloads the model on first use
    """A Hugging Face model, greedy, under a Lark grammar: transformers with an llguidance mask."""
    import llguidance as llg
    import llguidance.hf
    import llguidance.numpy as mask_np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    lm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(device)
    vocab = llguidance.hf.from_tokenizer(tokenizer, n_vocab=lm.config.vocab_size)

    def generate(prompt: str, lark: str) -> str:
        matcher = llg.LLMatcher(vocab, llg.grammar_from("lark", lark), log_level=0)
        mask = mask_np.allocate_token_bitmask(1, vocab.vocab_size)
        chat = [{"role": "user", "content": prompt}]
        ids = tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )["input_ids"].to(lm.device)

        def constrain(seen, scores):
            if seen.shape[1] > ids.shape[1]:
                matcher.consume_token(int(seen[0, -1]))
            mask_np.fill_next_token_bitmask(matcher, mask)
            logits = scores.float().cpu().numpy()
            mask_np.apply_token_bitmask_inplace(logits, mask)
            return torch.from_numpy(logits).to(scores.device)

        out = lm.generate(
            ids, max_new_tokens=1024, do_sample=False, logits_processor=[constrain]
        )
        return tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True)

    return generate


class _Scope(dict):
    """What generated code sees: the DSL and the classes; any other name is a parameter,
    except a ``_shared`` quantity, which must be assigned before it is read."""

    def __missing__(self, name: str) -> sp.Symbol:
        if name.startswith("_"):
            raise NameError(f"{name} is read before it is assigned")
        return sp.Symbol(name)


def rules(description: str, templates, model=None) -> dict:
    """``{name: rule}``: the rules a description asks for, in order, over the classes
    *templates* declare.

    *model* is ``(prompt, lark grammar) -> text``, the tiny local ``MODEL`` by
    default.  It generates under ``grammar(templates, description)``, and the answer is checked
    against the same grammar before it runs, since a model that ignores the constraint
    must not run anything.
    """
    lark = grammar(templates, description)
    text = (model or local())(prompt(description, templates), lark).strip() + "\n"
    if not _accepts(text, lark):
        raise DSLError(f"the model's answer is not in the rule language:\n{text}")
    for line in text.splitlines():  # a dict keeps the last of two equal keys, silently
        keys = re.findall(r"([A-Z]\w*\.\w+)\s*:", line)
        if len(keys) != len(set(keys)):
            raise DSLError(f"a rule writes the same field twice: {line}")
    classes = _classes(templates)
    scope = _Scope(
        Rule=dict,
        sp=sp,
        expect=expect,
        Normal=Normal,
        Uniform=Uniform,
        **{op.__name__: op for op in (*dsl.OPERATORS, dsl.lag, dsl.coalesce)},
        **{k: Agents(k, " ".join(fields)) for k, fields in classes.items()},
    )
    try:
        exec(text, {"__builtins__": {}}, scope)  # noqa: S102 - checked: DSL only
    except Exception as error:
        raise DSLError(f"the generated rules do not compile: {error}") from error
    found = {name: v for name, v in scope.items() if is_rule(v)}
    for rule in found.values():  # a dict is checked when it compiles: compile it now
        compiled = _Rule(rule)
        sparql(rule, dict.fromkeys(compiled.parameters(), 1.0))
    return found


def prompt(description: str, templates) -> str:
    """What the model is asked: the rule language, the agents, the behaviour."""
    classes = _classes(templates)
    agents = "\n".join(
        f"{klass}: "
        + ", ".join(
            f"{name} (link to {kind or 'another agent'}{': ' + doc if doc else ''})"
            if isinstance(kind, Link)
            else f"{name} ({doc or 'number'})"
            for name, (kind, doc) in kinds.items()
            if isinstance(kind, Link) or _number(kind)
        )
        for klass, kinds in classes.items()
    )
    # a docstring's line breaks are layout: one line per paragraph
    description = "\n\n".join(" ".join(p.split()) for p in description.split("\n\n"))
    parameters = ", ".join(_parameters(description)) or "none"
    return PROMPT.format(agents=agents, parameters=parameters, description=description)
