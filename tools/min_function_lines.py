"""Fail on functions under MINIMUM lines of code, and on long module docstrings.

A function that is one call under three lines of prose is a name, not an
abstraction: it hides the thing it wraps and costs a file jump to read.  The
count is executable lines only — the docstring, comments and blank lines do not
pay for a function's existence.

No linter ships the first rule: ruff, pylint and the flake8 plugins all bound
function length from above (``PLR0915``, ``C901``, ``MFL000``) and none from
below, because "keep functions small" is the consensus this inverts.  Nested
functions are skipped: a callback handed to an external API has to be a
function whatever its length.

The second is the same argument at file scope.  Good code reads as itself, so a
module docstring is for what the code cannot say — a library's silent failure
modes, a constraint an external engine imposes — and MAX_DOCSTRING lines is
enough for that.  Everything longer was a design note that belongs in the README
or in the docstring of the thing it describes.
"""

import ast
import sys
from pathlib import Path

MINIMUM = 4
MAX_DOCSTRING = 30
ROOTS = ("skabm", "tools")
BASELINE = Path(__file__).with_name("short-functions.baseline")
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def code_lines(node, source: list[str]) -> int:
    """Executable lines in *node*'s body: no docstring, comments or blanks."""
    body = node.body
    first = getattr(body[0], "value", None) if body else None
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        body = body[1:]
    if not body:
        return 0
    span = range(body[0].lineno, body[-1].end_lineno + 1)
    text = (source[i - 1] for i in span)
    return sum(1 for line in text if line.strip() and not line.lstrip().startswith("#"))


def main() -> int:
    """Report every non-nested function under MINIMUM lines that is not grandfathered."""
    allowed = {line.split("#")[0].strip() for line in BASELINE.read_text().splitlines()}
    found = 0
    for path in sorted(p for root in ROOTS for p in Path(root).rglob("*.py")):
        source = path.read_text().splitlines()
        tree = ast.parse("\n".join(source))
        if ast.get_docstring(tree) is not None:
            node = tree.body[0]
            length = node.end_lineno - node.lineno + 1
            if length > MAX_DOCSTRING:
                print(f"{path}:1: module docstring is {length} lines")
                found += 1
        nested = {
            id(inner)
            for outer in ast.walk(tree)
            if isinstance(outer, FUNCTIONS)
            for inner in ast.walk(outer)
            if isinstance(inner, FUNCTIONS) and inner is not outer
        }
        for node in ast.walk(tree):
            if not isinstance(node, FUNCTIONS) or id(node) in nested:
                continue
            length = code_lines(node, source)
            if length < MINIMUM and f"{path}:{node.name}" not in allowed:
                print(f"{path}:{node.lineno}: {node.name}() is {length} lines of code")
                found += 1
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
