"""Charts for a simulation — as values, not as side effects.

Everything here is a pure function returning an ``alt.Chart``: no renderer, no
backend loaded at import, no comm opened, nothing displayed.  That separation is
the whole point.  A chart is built once and *some context* decides what to do
with it — a notebook cell renders it by being the last expression, ``animate``
re-renders it in place through IPython's display handle, and
``streamlit run streamlit_app.py`` hands the same object to ``st.altair_chart``.

Which chart a per-agent frame gets is not decided here either; it comes from the
rules, through ``RDFSimulator.get_altair`` (see there).  ``agent_chart`` only
draws what it is told.

ponytail: Altair serialises the data into the spec and refuses past 5000 rows.
Fine for the lattices here (~2k agents); past that, enable ``vegafusion`` or call
``alt.data_transformers.disable_max_rows()`` — a global switch, so it is the
caller's to throw, not a library's.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator, Sequence

import altair as alt
import polars as pl

# "POINT(x y)" — the WKT a GeoSPARQL population carries under ``def:geometry``.
_WKT = r"\(\s*([^\s)]+)\s+([^\s)]+)"


def _numeric(frame: pl.DataFrame, exclude: Sequence[str] = ()) -> list[str]:
    return [
        c
        for c, dt in zip(frame.columns, frame.dtypes)
        if dt.is_numeric() and c not in exclude
    ]


def telemetry_chart(
    rows: Sequence[dict], y: Sequence[str] | None = None, columns: int = 2
) -> alt.Chart:
    """The observables so far, one faceted line per signal.

    *y* narrows the signals (default: every numeric key but ``t``).  Faceted with
    independent y scales because a rule set's observables are levels and shares
    at once, and one axis would flatten the shares to a line.
    """
    frame = pl.DataFrame(rows)
    long = frame.unpivot(
        index="t",
        on=y or _numeric(frame, exclude=("t",)),
        variable_name="signal",
        value_name="level",
    )
    return (
        alt.Chart(long)
        .mark_line(point=True)
        .encode(
            x=alt.X("t:Q", title="tick"),
            y=alt.Y("level:Q", title=None).scale(zero=False),
        )
        .properties(width=240, height=150)
        .facet("signal:N", columns=columns)
        .resolve_scale(y="independent")
    )


def agent_chart(
    frame: pl.DataFrame, predicates: set, c: str | None = None
) -> alt.Chart:
    """Draw *frame* the way *predicates* — the rule set's — say it should be.

    ``def:x``/``def:y`` is a lattice and gets a rect grid; ``def:geometry`` is a
    set of WKT points and gets a scatter, its coordinates parsed here rather than
    by a hand-written extract; neither means there is no space to draw, so the
    agents show as a histogram instead.  *c* is the predicate the agents are
    coloured by, defaulting to the first numeric non-coordinate one.
    """
    if "geometry" in predicates:
        frame = frame.drop_nulls("geometry").with_columns(
            pl.col("geometry").str.extract(_WKT, 1).cast(pl.Float64).alias("x"),
            pl.col("geometry").str.extract(_WKT, 2).cast(pl.Float64).alias("y"),
        )
        spatial = "points"
    elif {"x", "y"} <= predicates:
        frame = frame.drop_nulls(["x", "y"])
        spatial = "lattice"
    else:
        spatial = None
    value = c or _numeric(frame, exclude=("x", "y"))[0]
    if spatial is None:
        return (
            alt.Chart(frame.drop_nulls(value).select(value))
            .mark_bar()
            .encode(x=alt.X(f"{value}:Q").bin(maxbins=30), y=alt.Y("count()"))
            .properties(width=340, height=220)
        )
    frame = frame.drop_nulls(value).select("x", "y", value)
    colour = alt.Color(f"{value}:Q", title=value).scale(scheme="viridis")
    if spatial == "lattice":
        return (
            alt.Chart(frame)
            .mark_rect()
            .encode(x=alt.X("x:O", axis=None), y=alt.Y("y:O", axis=None), color=colour)
            .properties(width=360, height=360)
        )
    return (
        alt.Chart(frame)
        .mark_circle(size=40)
        .encode(x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None), color=colour)
        .properties(width=360, height=360)
    )


def animate(
    rows: Iterable[dict], chart: Callable[[list], alt.Chart], pause: float = 0.0
) -> Iterator[dict]:
    """The notebook context: yield *rows* unchanged, re-rendering *chart* in place.

    The one thing in this module that renders, and it does so through IPython's
    display handle — the spec is replaced wholesale, which is exactly why it is
    reliable: there is no JS backend to have been loaded and no comm to have been
    attached.  It is also why *pause* matters: each update re-mounts the view, so
    a tick faster than the eye reads as a flicker.  ``streamlit_app.py`` is the same
    charts under Streamlit, where the redraw is the framework's job.
    """
    from IPython.display import display

    handle = None
    seen: list[dict] = []
    for row in rows:
        seen.append(row)
        drawn = chart(seen)
        if handle is None:
            handle = display(drawn, display_id=True)
        else:
            handle.update(drawn)
        if pause:
            time.sleep(pause)
        yield row
