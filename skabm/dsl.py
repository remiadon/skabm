"""
Rules as SymPy: written once, compiled to SPARQL for maplib and to JAX for gradients.

A rule is a dict, ``{field: expression}``, over one agent class, applied to every agent
of it at once.  Every read sees the state before the rule ran, which is the
DELETE/INSERT upsert SPARQL spells out.  Fields come from the class's OTTR template,
``Occupation = Agents("Occupation")`` then ``Occupation.employment``, so a typo raises.
Any other symbol is a parameter, replaced by its value (``.subs``) when the rule is
compiled: ``sparql(rule, params)``, or passed in, traced, to ``jax_tick``.

The graph needs what arithmetic cannot say:

* a link is followed by name: ``Edge.src.unemployment`` is the unemployment of the
  occupation an edge starts from;
* ``sum_over(link, e)`` sums ``e`` over the agents a link relates to this one;
* ``total(e)`` / ``mean(e)`` run over every agent of ``e``'s class;
* ``total_by``, ``running_sum`` and ``pick`` sum over the agents sharing a key and up to
  a rank, and choose an agent: SPARQL only, so far.

``coalesce(a, b, ...)`` is the first of its arguments that exists: inside it a link
may be missing and a field unset.  ``node(subject, field)`` is a field of one named
node: a rule whose keys are such fields updates that node alone (``learning.sac``).
``lag(e)`` is ``e`` as the rule saw it one run ago, and ``sympy.stats`` ``Normal`` /
``Uniform`` draw once per agent.

The SPARQL printer brackets every operation (maplib groups ``a - b + c`` from the
right), writes every number as a double, defaults every sum with ``COALESCE``
(``IF(BOUND)`` turns into a struct column on an empty result), and binds each draw once
(an inline draw re-draws on every row of a join).  JAX is ``sympy.lambdify``, handed the
graph functions.  SPARQL text passed to the simulator has no JAX form.
"""

from __future__ import annotations

import difflib
import re
import zlib
from functools import reduce

import sympy as sp
from sympy.core.symbol import Str
from sympy.stats import Normal, Uniform
from sympy.stats.crv_types import NormalDistribution, UniformDistribution
from sympy.stats.rv import RandomSymbol

from skabm.ottr import SCHEMA, Link
from skabm.sparql import _PREFIXES, EX_NS


class DSLError(ValueError):
    """A rule the compiler cannot place; the message says where and why."""


class gather(sp.Function):
    """What ``Edge.src.unemployment`` builds: a field read across a link."""

    nargs = 2


class total(sp.Function):
    """``total(Firm.output)``: the sum over every agent of the class."""

    nargs = 1


class mean(sp.Function):
    """``mean(Firm.price)``: the mean over every agent of the class."""

    nargs = 1


class lag(sp.Function):
    """``lag(e)``: ``e`` one run of the rule ago, itself on the first run."""

    nargs = 1


class coalesce(sp.Function):
    """``coalesce(Household.employer.w_bar, benefit)``: the first argument that exists."""

    @classmethod
    def eval(cls, *args):  # coalesce(a, coalesce(b, c)) is coalesce(a, b, c)
        if any(isinstance(a, coalesce) for a in args):
            return cls(
                *(
                    b
                    for a in args
                    for b in (a.args if isinstance(a, coalesce) else (a,))
                )
            )


def node(subject: str, field: str) -> sp.Symbol:
    """``field`` of the named node ``ex:<subject>``, missing until something writes it."""
    return sp.Symbol(f"@{subject}.{field}")


class sum_over(sp.Function):
    """``sum_over(link, e)``: the sum of ``e`` over the agents a link relates to this one.

    Another class's link sums over its agents that point here: ``sum_over(Edge.dst,
    Edge.flow)`` is an occupation's inflow.  The agent's own link, or a path from it,
    sums over the agents it reaches: ``sum_over(Route.via, Link.t0)``, where
    ``Route.via.t0`` reads the same.
    """

    nargs = 2


class total_by(sp.Function):
    """``total_by(Option.od, e)``: the sum of ``e`` over the agents sharing this ``od``."""

    nargs = 2


class running_sum(sp.Function):
    """``running_sum(Route.rank, e, Route.od)``: the sum of ``e`` over the agents
    (sharing this ``od``, if given) whose ``rank`` is at most this one's."""

    nargs = (2, 3)


class pick(sp.Function):
    """``pick(Route, where, order)``: the agent of that class meeting ``where`` with
    the smallest ``order``, missing if none does; ``where`` reads both agents."""

    nargs = 3


OPERATORS = (gather, total, mean, sum_over, total_by, running_sum, pick)
RELATIONAL = (sum_over, total_by, running_sum, pick)


def owner(symbol) -> tuple[str, str] | None:
    """``("Occupation", "employment")`` for a field symbol, None for a parameter."""
    klass, dot, name = symbol.name.partition(".")
    return (klass, name) if dot else None


def _field(klass: str, name: str, fields):
    """``klass.name``: a field symbol, or a link to read on through."""
    if name.startswith("_") or name not in fields:
        close = difflib.get_close_matches(name, list(fields), n=1)
        hint = f"; did you mean {close[0]!r}?" if close else ""
        raise AttributeError(f"{klass} has no field {name!r}{hint}")
    symbol = sp.Symbol(f"{klass}.{name}")
    kind = SCHEMA.get(klass, {}).get(name, (None,))[0]
    return _Link((symbol,), str(kind)) if isinstance(kind, Link) and kind else symbol


class _Link:
    """``Edge.src``: its attributes are the target's fields, read across the link."""

    def __init__(self, links: tuple, target: str):
        self._links, self._target = links, target

    def _sympy_(self):  # the link itself: its value, or what an operator runs over
        return self._links[0] if len(self._links) == 1 else sp.Tuple(*self._links)

    def __getattr__(self, name: str):
        far = _field(self._target, name, SCHEMA[self._target])
        if isinstance(far, _Link):
            return _Link(self._links + far._links, far._target)
        return reduce(
            lambda value, link: gather(link, value), reversed(self._links), far
        )


class Agents:
    """A class's fields: ``Occupation = Agents("Occupation")``, then ``Occupation.employment``.

    The fields come from the class's OTTR template (``ottr.SCHEMA``); a link field
    reads on through its target, ``Edge.src.unemployment``.  A class without a
    template names its own fields: ``Agents("Signal", "k s1 s2")``.
    """

    def __init__(self, klass: str, fields: str | None = None):
        self._klass = klass
        self._fields = fields.split() if fields is not None else list(SCHEMA[klass])

    def __getattr__(self, name: str):
        return _field(self._klass, name, self._fields)

    def _sympy_(self):  # pick(Route, ...) names a class
        return Str(self._klass)


def is_rule(value) -> bool:
    """Whether *value* is a rule: a non-empty dict keyed by fields."""
    return (
        isinstance(value, dict)
        and bool(value)
        and all(
            isinstance(k, _Link) or (isinstance(k, sp.Symbol) and owner(k))
            for k in value
        )
    )


class _Rule:
    """A rule dict, desugared: the class it updates (or the named node, ``subject``),
    the fields it writes, and the reads that may be missing (``maybe``)."""

    def __init__(self, rule: dict):
        keys = [sp.sympify(s) for s in rule]  # a link field is written like any other
        owners = [owner(s) if isinstance(s, sp.Symbol) else None for s in keys]
        if None in owners or len({k for k, _ in owners}) != 1:
            raise DSLError(
                f"a rule writes fields of one class, got {sorted(map(str, keys))}"
            )
        self.klass = owners[0][0]
        self.subject = self.klass[1:] if self.klass[0] == "@" else None
        self.writes = {}
        for (_, field), e in zip(owners, rule.values()):
            self.writes[field] = self._desugar(sp.sympify(e))
        # what appears only inside a coalesce may be missing: both backends allow it
        inside, outside = set(), set()
        for e in self.writes.values():
            gaps = e.atoms(coalesce)
            for c in gaps:  # an operator's own reads are its own business
                ops = c.atoms(total, mean, *RELATIONAL)
                c = c.xreplace({op: sp.Dummy() for op in ops})
                inside |= c.atoms(gather, sp.Symbol)
            outside |= e.xreplace({c: sp.Dummy() for c in gaps}).atoms(
                gather, sp.Symbol
            )
        self.maybe = inside - outside

    def parameters(self) -> set:
        """The names of the parameters the rule reads."""
        drawn = {
            s
            for e in self.writes.values()
            for d in e.atoms(RandomSymbol)
            for s in d.atoms(sp.Symbol)
        }
        return {
            s.name
            for e in self.writes.values()
            for s in e.atoms(sp.Symbol) - drawn
            if owner(s) is None
        }

    def check(self, params: dict) -> _Rule:
        """This rule, if *params* give every parameter it reads."""
        missing = sorted(self.parameters() - set(params))
        if missing:
            raise KeyError(
                f"rule needs parameters {missing}: pass them in params= "
                "(skabm.behaviour's own are in each module's PARAMETERS)"
            )
        return self

    def _far(self, path, inner):
        """*inner* with the bare fields of the class *path* reaches read through it."""
        links = path.args if isinstance(path, sp.Tuple) else (path,)
        if owner(links[0])[0] != self.klass:
            return inner  # another class's link: its agents are the ones read
        target = str(SCHEMA[owner(links[-1])[0]][owner(links[-1])[1]][0])
        if target == self.klass:
            return inner  # a link to its own class: a bare field is the agent's own

        def walk(e):
            if isinstance(e, gather):
                return e
            if e.is_Symbol and owner(e) and owner(e)[0] == target:
                return reduce(
                    lambda value, link: gather(link, value), reversed(links), e
                )
            return e.func(*map(walk, e.args)) if e.args else e

        return walk(inner)

    def _reach(self, e):
        """*e* with another class's fields, read in the rule's own scope, read through
        the one link the rule's class has to it: ``Option.share`` in a Route rule is
        ``Route.option.share``.  With two links (an employer, a firm owned), the read
        stays ambiguous and the printer refuses it."""
        links: dict = {}
        for field, (kind, _) in SCHEMA.get(self.klass, {}).items():
            if isinstance(kind, Link) and kind and not kind.many:
                links.setdefault(str(kind), []).append(
                    sp.Symbol(f"{self.klass}.{field}")
                )
        one = {k: v[0] for k, v in links.items() if len(v) == 1 and k != self.klass}

        def walk(e):
            if isinstance(e, sum_over) and e.args[0].is_Symbol:
                link = e.args[0]  # another class's link, not pointing here: from here
                klass, name = owner(link)
                if klass in one and str(SCHEMA[klass][name][0]) != self.klass:
                    return sum_over(sp.Tuple(one[klass], link), e.args[1])
            if isinstance(e, (*OPERATORS, RandomSymbol)) or not e.args:
                if e.is_Symbol and owner(e) and owner(e)[0] in one:
                    return gather(one[owner(e)[0]], e)
                return e
            return e.func(*map(walk, e.args))

        return walk(e)

    def _desugar(self, e):
        """``lag`` and draws, in the one form each backend knows.

        ``lag(x)`` becomes a hidden field written with ``x`` and read back next run;
        a ``Normal``/``Uniform`` becomes an affine map of a standard draw; a far
        agent's field named bare is read through the link that reaches it
        (``_reach``, ``_far``).
        """
        e = self._reach(e).replace(
            sum_over, lambda path, inner: sum_over(path, self._far(path, inner))
        )
        for op in e.atoms(*OPERATORS):
            if op.atoms(lag, RandomSymbol):
                raise DSLError(f"{op}: lag and draws belong on the rule's own agents")
        swaps = {}
        for memory in e.atoms(lag):
            field = f"lag_{zlib.crc32(sp.srepr(memory.args[0]).encode()):08x}"
            self.writes[field] = memory.args[0]
            swaps[memory] = coalesce(sp.Symbol(f"{self.klass}.{field}"), memory.args[0])
        for draw in e.atoms(RandomSymbol):
            law = draw.pspace.distribution
            if isinstance(law, NormalDistribution):
                swaps[draw] = law.mean + law.std * Normal(f"{draw}_", 0, 1)
            elif isinstance(law, UniformDistribution):
                spread = law.right - law.left
                swaps[draw] = law.left + spread * Uniform(f"{draw}_", 0, 1)
            else:
                raise DSLError(f"{draw}: only Normal and Uniform have a UDF")
        return e.xreplace(swaps)


# ---------------------------------------------------------------------------
# Scope: which class an expression reads, and where a link lands
# ---------------------------------------------------------------------------


def _classes(e) -> set:
    """Classes whose fields *e* reads in its own scope, not through a nested operator."""
    if isinstance(e, sp.Symbol):
        return {owner(e)[0]} if owner(e) and owner(e)[0][0] != "@" else set()
    if isinstance(e, gather):
        return {owner(e.args[0])[0]}  # the link is read here; its target is elsewhere
    if isinstance(e, (total, mean, RandomSymbol, Str, *RELATIONAL)):
        return set()
    return set().union(set(), *map(_classes, e.args))


def _target(link, klass: str) -> tuple[str, str]:
    """``(field, target class)`` of a link followed from *klass*."""
    src, field = owner(link)
    if src != klass:
        raise DSLError(f"{link} belongs to {src}, but the agent here is a {klass}")
    return field, str(SCHEMA[src][field][0])


def _many(link) -> bool:
    kind = SCHEMA.get(owner(link)[0], {}).get(owner(link)[1], (None,))[0]
    return isinstance(kind, Link) and kind.many


def _aggregated(e) -> str:
    classes = _classes(e.args[0])
    if len(classes) != 1:
        raise DSLError(
            f"{e} must read fields of exactly one class, got {sorted(classes)}"
        )
    return classes.pop()


# ---------------------------------------------------------------------------
# SPARQL
# ---------------------------------------------------------------------------


def _num(x) -> str:
    """A number as an xsd:double literal, exactly: ``52.0e0``, ``1e-05``, ``(1.0e0 / 3.0e0)``."""
    if x.is_Rational and not x.is_Integer:
        return f"({_num(sp.Integer(x.p))} / {_num(sp.Integer(x.q))})"
    text = repr(float(x))
    return text if "e" in text else text + "e0"


def _pair(op: str):
    return lambda a, b: f"({a} {op} {b})"


def _arithmetic(e, p, bind) -> str:  # noqa: PLR0911, PLR0912 - one branch per SymPy node
    """Print a non-operator node, *p* printing its children in the same scope.

    *bind* names a printed value, for a node that repeats its arguments.
    """
    if e.is_Number or e.is_NumberSymbol:
        return _num(e if e.is_Number else sp.Float(e))
    if isinstance(e, sp.Add):
        plus = [a for a in e.args if not a.could_extract_minus_sign()]
        minus = [-a for a in e.args if a.could_extract_minus_sign()]
        head = reduce(_pair("+"), map(p, plus)) if plus else "0.0e0"
        return reduce(_pair("-"), map(p, minus), head)
    if isinstance(e, sp.Mul):
        over = [a for a in e.args if a.is_Pow and a.exp.is_Integer and a.exp < 0]
        top = [p(a) for a in e.args if a not in over] or ["1.0e0"]
        text = reduce(_pair("*"), top)
        if over:
            text = f"({text} / {reduce(_pair('*'), (p(a.base**-a.exp) for a in over))})"
        return text
    if isinstance(e, sp.Pow):
        base, exp = e.args
        if exp.is_Integer and exp > 0:
            return reduce(_pair("*"), [p(base)] * int(exp))
        if exp.is_Integer:
            return f"(1.0e0 / {p(base**-exp)})"
        return f"math:exp(({p(exp)} * math:log({p(base)})))"
    function = {
        sp.exp: "math:exp",
        sp.log: "math:log",
        sp.Abs: "ABS",
        coalesce: "COALESCE",
    }.get(type(e))
    if function:
        return f"{function}({', '.join(map(p, e.args))})"
    if isinstance(e, (sp.Max, sp.Min)):
        cmp = ">" if isinstance(e, sp.Max) else "<"
        return reduce(
            lambda a, b: bind(f"IF(({a} {cmp} {b}), {a}, {b})"),
            (bind(p(a)) for a in e.args),
        )
    if isinstance(e, sp.Piecewise):
        *branches, (last, cond) = e.args
        if cond != sp.true:
            raise DSLError(
                f"{e} needs a last (value, True) branch: what if nothing holds?"
            )
        text = p(last)
        for value, test in reversed(branches):
            text = f"IF({p(test)}, {p(value)}, {text})"
        return text
    if isinstance(e, sp.core.relational.Relational):
        return f"({p(e.lhs)} {'=' if e.rel_op == '==' else e.rel_op} {p(e.rhs)})"
    if isinstance(e, (sp.And, sp.Or)):
        return (
            "("
            + (" && " if isinstance(e, sp.And) else " || ").join(map(p, e.args))
            + ")"
        )
    if isinstance(e, sp.Not):
        return f"(!{p(e.args[0])})"
    if isinstance(e, sp.ITE):  # SymPy's spelling of a comparison on a Piecewise
        return f"IF({p(e.args[0])}, {p(e.args[1])}, {p(e.args[2])})"
    raise DSLError(f"{type(e).__name__} has no SPARQL form: {e}")


def _hide(e, hidden: dict):
    """*e* with each outermost operator and draw swapped for a placeholder.

    ``cse`` then only ever sees one scope: a subexpression inside a gather reads the
    agent at the far end of the link, and must not be merged with the same text read
    at the rule's own agent.
    """
    if isinstance(e, (*OPERATORS, RandomSymbol)):
        return hidden.setdefault(e, sp.Dummy("op"))
    return e.func(*(_hide(a, hidden) for a in e.args)) if e.args else e


class _Where:
    """The patterns and BINDs of one agent variable: the rule's own, or a subquery's.

    *alias* maps another class to the variable of its agent, for a scope that reads
    two agents at once (``pick``'s candidate and the rule's own agent).
    """

    def __init__(self, query: _Sparql, agent: str, alias=None, top=False):
        self.query, self.agent = query, agent
        self.alias, self.top = dict(alias or {}), top
        self.required, self.optional, self.binds = [], [], []
        self.reads, self.links, self.names, self.sums = {}, {}, {}, {}

    def fresh(self, stem: str) -> str:
        self.query.count += 1
        return f"?{stem}{self.query.count}"

    def bind(self, text: str) -> str:
        if re.fullmatch(r"\?\w+|[-+.\de]+", text):  # a variable or a number already
            return text
        var = self.fresh("v")
        self.binds.append(f"BIND({text} AS {var})")
        return var

    def body(self) -> str:
        return "\n".join(self.required + self.optional + self.binds)

    def read(self, agent: str, field: str, maybe: bool = False) -> str:
        if (agent, field) not in self.reads:
            stem = agent.rsplit("#", 1)[-1].strip("?>")  # ?g3, or <...#sig__...>
            var = f"?{'this' if agent == self.query.this else stem}__{field}"
            pattern = f"{agent} def:{field} {var}"
            if maybe:  # unbound when missing, which coalesce skips
                self.optional.append(f"OPTIONAL {{ {pattern} }}")
            else:
                self.required.append(pattern + " .")
            if self.top and agent == self.query.this:  # what DELETE removes
                self.query.old.setdefault(field, var)
            self.reads[agent, field] = var
        return self.reads[agent, field]

    def expr(self, e, agent: str, klass: str) -> str:  # noqa: PLR0911, PLR0912
        if e in self.names:
            return self.names[e]
        maybe = e in self.query.rule.maybe
        if isinstance(e, RandomSymbol):
            return self.draw(e)
        if isinstance(e, sp.Symbol):
            if owner(e) is None:
                return _num(sp.Float(self.query.params[e.name]))
            other, field = owner(e)
            if other[0] == "@":
                return self.read(f"<{EX_NS}{other[1:]}>", field, maybe=True)
            if other != klass and other in self.alias:
                return self.read(self.alias[other], field, maybe)
            if other != klass:
                raise DSLError(
                    f"{e} is read where the agent is a {klass}: reach it by a link"
                )
            return self.read(agent, field, maybe)
        if isinstance(e, gather):
            src = owner(e.args[0])[0]
            if src != klass and src in self.alias:
                agent, klass = self.alias[src], src
            field, target = _target(e.args[0], klass)
            if (agent, field) not in self.links:
                if _many(e.args[0]):
                    raise DSLError(f"{e.args[0]} reaches many agents: use sum_over")
                if maybe:
                    return self.path(e, agent, klass)
                self.links[agent, field] = self.fresh("g")
                self.required.append(
                    f"{agent} def:{field} {self.links[agent, field]} ."
                )
            return self.expr(e.args[1], self.links[agent, field], target)
        # one subquery per sum, however often it is read
        if isinstance(e, OPERATORS):
            if agent.startswith("<") and not isinstance(e, (total, mean)):
                raise DSLError(f"{e}: a named node is no link's target")
            if (e, agent) not in self.sums:
                build = {
                    sum_over: self.over,
                    total_by: self.by,
                    running_sum: self.running,
                    pick: self.pick,
                }.get(type(e))
                self.sums[e, agent] = (
                    build(e, agent, klass) if build else self.aggregate(e)
                )
            return self.sums[e, agent]
        return _arithmetic(e, lambda a: self.expr(a, agent, klass), self.bind)

    def path(self, e, agent: str, klass: str) -> str:
        """A link path that may be missing: one OPTIONAL, unbound if any hop is."""
        patterns = []
        while isinstance(e, gather):
            field, klass = _target(e.args[0], klass)
            hop = self.fresh("o")
            patterns.append(f"{agent} def:{field} {hop} .")
            agent, e = hop, e.args[1]
        var = f"{agent}__{owner(e)[1]}"
        self.optional.append(
            f"OPTIONAL {{ {' '.join(patterns)} {agent} def:{owner(e)[1]} {var} }}"
        )
        return var

    def subquery(self, sub: _Where, value: str, group: str = "") -> str:
        """``SUM(value)`` over *sub*'s patterns, per *group*: the default 0 bound."""
        out, by = self.fresh("sum"), f" GROUP BY {group}" if group else ""
        self.optional.append(
            f"OPTIONAL {{ SELECT {group} (SUM({value}) AS {out}) WHERE {{\n"
            f"{sub.body()}\n}}{by} }}"
        )
        return self.bind(f"COALESCE({out}, 0.0e0)")

    def scatter(self, e, agent: str, klass: str) -> str:
        link, inner = e.args
        if isinstance(link, sp.Tuple):
            raise DSLError(f"{e}: another class's agents are summed over one link")
        src, field = owner(link)
        target = SCHEMA[src][field][0]
        if str(target) != klass:
            raise DSLError(
                f"{link} points to {target}, but the agent here is a {klass}"
            )
        sub = _Where(self.query, self.fresh("s"))
        sub.required.append(f"{sub.agent} a ex:{src} ; def:{field} {agent} .")
        return self.subquery(sub, sub.expr(inner, sub.agent, src), agent)

    def over(self, e, agent: str, klass: str) -> str:
        """``sum_over``: the rule's agent, joined to every agent its path reaches, or,
        over another class's link, every agent of it pointing here."""
        path, inner = e.args
        if owner(path.args[0] if isinstance(path, sp.Tuple) else path)[0] != klass:
            return self.scatter(e, agent, klass)
        sub, here, cls = _Where(self.query, agent, alias=self.alias), agent, klass
        for link in path.args if isinstance(path, sp.Tuple) else (path,):
            field, target = _target(link, cls)
            sub.links[here, field] = hop = self.fresh("p")
            sub.required.append(f"{here} def:{field} {hop} .")
            here, cls = hop, target
        return self.subquery(sub, sub.expr(inner, agent, klass), agent)

    def by(self, e, agent: str, klass: str) -> str:
        """``total_by``: grouped on the key, joined back on this agent's value of it."""
        key, inner = e.args
        mine = self.read(agent, owner(key)[1])
        sub = _Where(self.query, self.fresh("k"))
        sub.required.append(f"{sub.agent} a ex:{klass} ; def:{owner(key)[1]} {mine} .")
        return self.subquery(sub, sub.expr(inner, sub.agent, klass), mine)

    def running(self, e, agent: str, klass: str) -> str:
        """``running_sum``: a self-join on the order, within the key if there is one."""
        order, inner, *key = e.args
        sub = _Where(self.query, self.fresh("k"))
        mine, theirs = self.fresh("r"), self.fresh("r")
        sub.required.append(f"{agent} a ex:{klass} ; def:{owner(order)[1]} {mine} .")
        sub.required.append(
            f"{sub.agent} a ex:{klass} ; def:{owner(order)[1]} {theirs} ."
        )
        if key:
            shared = self.fresh("key")
            for who in (agent, sub.agent):
                sub.required.append(f"{who} def:{owner(key[0])[1]} {shared} .")
        sub.binds.append(f"FILTER({theirs} <= {mine})")
        return self.subquery(sub, sub.expr(inner, sub.agent, klass), agent)

    def pick(self, e, agent: str, klass: str) -> str:
        """``pick``: the least *order* among candidates meeting *where*, then the
        candidate holding it.  *where* reads both agents, told apart by class."""
        cls, where, order = str(e.args[0]), e.args[1], e.args[2]
        if cls == klass:
            raise DSLError(f"{e}: pick a {cls} for a {klass}, not for its own class")
        alias = {**self.alias, klass: agent}

        def candidates(stem: str, rank: str) -> _Where:
            sub = _Where(self.query, self.fresh(stem), alias=alias)
            sub.required.append(f"{agent} a ex:{klass} . {sub.agent} a ex:{cls} .")
            for test in where.args if isinstance(where, sp.And) else (where,):
                fields = [a for a in test.args if a.is_Symbol and owner(a)]
                if isinstance(test, sp.Eq) and len(fields) == 2:
                    # a join on a shared variable, not a filter on every pair
                    sides = sorted(fields, key=lambda s: owner(s)[0] == cls)
                    field = owner(sides[1])[1]
                    sub.reads[sub.agent, field] = sub.expr(sides[0], agent, klass)
                    sub.required.append(
                        f"{sub.agent} def:{field} {sub.reads[sub.agent, field]} ."
                    )
            sub.binds.append(f"FILTER({sub.expr(where, sub.agent, cls)})")
            sub.binds.append(f"BIND({sub.expr(order, sub.agent, cls)} AS {rank})")
            return sub

        # maplib's OPTIONAL filters cannot see outside it: join on the best order
        best, rank = self.fresh("best"), self.fresh("r")
        first, chosen = candidates("c", rank), candidates("pick", best)
        self.optional.append(
            f"OPTIONAL {{ {{ SELECT {agent} (MIN({rank}) AS {best}) WHERE {{\n"
            f"{first.body()}\n}} GROUP BY {agent} }}\n"
            f"{{ SELECT {agent} {chosen.agent} {best} WHERE {{\n{chosen.body()}\n}} }} }}"
        )
        return chosen.agent

    def aggregate(self, e, agent: str = "", klass: str = "") -> str:
        klass = _aggregated(e)
        sub = _Where(self.query, self.fresh("a"))
        sub.required.append(f"{sub.agent} a ex:{klass} .")
        value, out = sub.expr(e.args[0], sub.agent, klass), self.fresh("agg")
        function = "SUM" if isinstance(e, total) else "AVG"
        self.optional.append(
            f"OPTIONAL {{ SELECT ({function}({value}) AS {out}) WHERE {{\n"
            f"{sub.body()}\n}} }}"
        )
        return self.bind(f"COALESCE({out}, 0.0e0)")

    def draw(self, e) -> str:
        """A standard draw (``_Rule`` desugared the rest), bound once per agent."""
        normal = isinstance(e.pspace.distribution, NormalDistribution)
        self.names[e] = self.fresh("draw")
        udf = "pr:normal" if normal else "pr:uniform"
        self.binds.append(f"BIND({udf}(0.0e0, 1.0e0) AS {self.names[e]})")
        return self.names[e]


class _Sparql:
    """One rule's DELETE/INSERT text."""

    def __init__(self, rule: _Rule, params: dict):
        # a parameter prints as its value: substituted earlier, SymPy would fold it
        # into constants (exp(k (t - s)) as exp(-k s) exp(k t), which overflows)
        self.rule, self.params, self.count, self.old = rule, params, 0, {}
        self.this = f"<{EX_NS}{rule.subject}>" if rule.subject else "?this"
        top = _Where(self, self.this, top=True)
        if not rule.subject:
            top.required.append(f"?this a ex:{rule.klass} .")
        hidden: dict = {}
        pure = [_hide(e, hidden) for e in rule.writes.values()]
        for op, placeholder in hidden.items():
            top.names[placeholder] = top.bind(top.expr(op, self.this, rule.klass))
        shared, pure = sp.cse(pure, symbols=sp.numbered_symbols("_cse", sp.Dummy))
        # a shared 1/x would turn every x/y into y * (1/x), which is not the same double
        inline = {}
        for name, e in shared:
            if e.is_Pow and e.exp.is_negative:
                inline[name] = e.xreplace(inline)
        shared = [(n, e.xreplace(inline)) for n, e in shared if n not in inline]
        pure = [e.xreplace(inline) for e in pure]
        for name, e in shared:
            top.names[name] = top.bind(top.expr(e, self.this, rule.klass))
        for field, e in zip(rule.writes, pure):
            top.binds.append(
                f"BIND({top.expr(e, self.this, rule.klass)} AS ?new__{field})"
            )
        for field in rule.writes:
            if field not in self.old:
                self.old[field] = f"?old__{field}"
                top.optional.append(
                    f"OPTIONAL {{ {self.this} def:{field} ?old__{field} }}"
                )
        self.text = (
            _PREFIXES
            + "DELETE { "
            + " . ".join(f"{self.this} def:{f} {self.old[f]}" for f in rule.writes)
            + " }\nINSERT { "
            + " . ".join(f"{self.this} def:{f} ?new__{f}" for f in rule.writes)
            + " }\nWHERE {\n"
            + top.body()
            + "\n}\n"
        )


def sparql(rule: dict, params: dict | None = None) -> str:
    """*rule*'s DELETE/INSERT, its parameters replaced by *params* (by name)."""
    params = params or {}
    return _Sparql(_Rule(rule).check(params), params).text


# ---------------------------------------------------------------------------
# JAX: sympy.lambdify, handed the graph functions
# ---------------------------------------------------------------------------


def _size(state: dict, agent: str) -> int:
    return len(next(iter(state[agent].values())))


def jax_tick(rules):
    """``tick(state, params=None, key=None) -> state``: the rules in order, in JAX.

    *state* is ``{class: {field: array}}`` (``arrays`` builds it from frames), with
    link fields as row indices into their target class.  *params* are every parameter
    the rules read, by name, and may be traced, which is what makes ``jax.grad`` with
    respect to a behavioural parameter one line.  *key* feeds the draws.  A state
    from ``arrays(frames, rules)`` keeps its structure, so a tick drops into
    ``jax.lax.scan`` as is.
    """
    import jax
    import jax.numpy as jnp

    plain = [r for r in rules if not is_rule(r)]
    if plain:
        raise TypeError(
            f"{len(plain)} rule(s) are SPARQL text, not a rule dict: JAX has nothing "
            "to compile"
        )
    graph = {  # a link arrives as (row indices, size of the class it points to)
        "gather": lambda link, value: jnp.where(link[0] >= 0, value[link[0]], jnp.nan),
        "sum_over": lambda link, value: jax.ops.segment_sum(
            value, link[0], num_segments=link[1]
        ),
        "total": jnp.sum,
        "mean": lambda value: jnp.mean(value) if value.size else 0.0,
        "coalesce": lambda *values: reduce(
            lambda first, then: jnp.where(jnp.isnan(first), then, first), values
        ),
    }
    compiled = []
    for rule in map(_Rule, rules):
        exprs = list(rule.writes.values())
        for e in exprs:
            for op in e.atoms(*RELATIONAL):
                link = op.args[0]
                inflow = isinstance(op, sum_over) and link.is_Symbol
                if not inflow or owner(link)[0] == rule.klass or _many(link):
                    raise DSLError(f"{op} has no JAX form yet: it stays SPARQL")
        draws = sorted(set().union(*(e.atoms(RandomSymbol) for e in exprs)), key=str)
        exprs = [e.xreplace({d: sp.Symbol(f"_{d}") for d in draws}) for e in exprs]
        args = sorted(set().union(*(e.free_symbols for e in exprs)), key=str)
        function = sp.lambdify(args, exprs, modules=[graph, "jax"])
        compiled.append((rule, args, draws, function))

    def value(state, rule, arg):
        """A field's array (a link's with its target's size), NaN if it may be missing."""
        klass, field = owner(arg)
        if klass[0] == "@":  # a named node: NaN until something writes it
            return state.get(klass[1:], {}).get(field, jnp.nan)
        agent = (rule.subject or rule.klass) if klass == rule.klass else klass
        if agent not in state and klass != rule.klass:  # nobody to sum, as in SPARQL
            return jnp.zeros(0)
        if field in state.get(agent, {}):
            kind = SCHEMA.get(klass, {}).get(field, (None,))[0]
            if isinstance(kind, Link) and kind:
                return state[agent][field], _size(state, kind)
            return state[agent][field]
        if arg in rule.maybe:
            return jnp.nan
        raise KeyError(f"{arg}: the state has no such array")

    def tick(state: dict, params: dict | None = None, key=None) -> dict:
        for i, (rule, args, draws, function) in enumerate(compiled):
            agent = rule.subject or rule.klass
            if agent not in state and not rule.subject:
                continue  # nobody to update, as in SPARQL
            n = 1 if rule.subject else _size(state, agent)
            values = dict(params or {})
            for j, draw in enumerate(draws):
                if key is None:
                    raise ValueError(f"{draw} is random: call tick(state, params, key)")
                normal = isinstance(draw.pspace.distribution, NormalDistribution)
                sample = jax.random.normal if normal else jax.random.uniform
                folded = jax.random.fold_in(jax.random.fold_in(key, i), j)
                values[f"_{draw}"] = sample(folded, (n,))
            for arg in args:
                if arg.name in values:
                    continue
                if owner(arg) is None:
                    raise KeyError(
                        f"rule needs parameter {arg.name!r}: pass it in params="
                    )
                values[arg.name] = value(state, rule, arg)
            out = function(*(values[a.name] for a in args))
            written = {f: jnp.broadcast_to(o, (n,)) for f, o in zip(rule.writes, out)}
            state = {**state, agent: {**state.get(agent, {}), **written}}
        return state

    return tick


def arrays(frames: dict, rules=()) -> dict:
    """``{class: polars frame}`` to the state ``jax_tick`` steps.

    Float columns become arrays; a link column the template types (``ottr.Link``)
    becomes row indices into its target's frame, in that frame's order.  Anything
    else (strings, untyped links) is left behind, since no expression can read it.
    Every field *rules* write that the frames lack is added as NaN, so the first tick
    changes no structure.
    """
    import jax.numpy as jnp
    import polars as pl

    rows = {k: dict(zip(df["id"], range(df.height))) for k, df in frames.items()}
    state = {}
    for klass, df in frames.items():
        state[klass] = {}
        for name, dtype in df.schema.items():
            kind = SCHEMA.get(klass, {}).get(name, (None,))[0]
            if isinstance(kind, Link) and kind:  # -1 where the link is missing
                rows_of = df[name].replace_strict(rows[kind], return_dtype=pl.Int64)
                state[klass][name] = jnp.asarray(rows_of.fill_null(-1).to_numpy())
            elif dtype.is_float():
                state[klass][name] = jnp.asarray(df[name].to_numpy())
    for rule in map(_Rule, rules):
        agent = rule.subject or rule.klass
        if agent not in state and not rule.subject:
            continue  # a rule over a class the world lacks changes nothing
        own = state.setdefault(agent, {})
        n = 1 if rule.subject else _size(state, agent)
        for field in rule.writes:
            own.setdefault(field, jnp.full(n, jnp.nan))
    return state
