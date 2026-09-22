"""Bayonne's old town closed to cars: the traffic ABM, one morning at a time.

    uv run --no-sync streamlit run notebooks/bayonne/app.py

The rules settle each working day's peak, day after day, as commuters learn the network.
Pick a day here and the map replays its morning — 06:00 to the peak — with the jam
forming.  "Today" is a scenario like any other: run it to see the city as it stands.
"""

import json
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl
import pydeck as pdk
import streamlit as st

import world as W

st.set_page_config(page_title="Bayonne traffic ABM", page_icon=":material/traffic:", layout="wide")

DARK = st.context.theme.type == "dark"
# dataviz reference palette: the blue sequential ramp, the blue <-> red diverging pair
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
LESS, MIDDLE, MORE = ("#3987e5", "#383835", "#e66767") if DARK else ("#2a78d6", "#f0efec", "#e34948")
BEFORE = "before"  # the key world.run_once files the pre-closure day under
WAS = "before the closure"
# The three mornings worth looking at.  Everything between them is the solver settling
# the assignment, not days of Bayonne, so it stays under the hood.
STATES = {"Before the closure": "before", "The first morning after": "first", "Once drivers have adjusted": "last"}


def colour(t: np.ndarray, stops: list[str]) -> list[list[int]]:
    """Piecewise-linear colour for t in [0, 1] along *stops*."""
    table = np.array([[int(h[i : i + 2], 16) for i in (1, 3, 5)] for h in stops], dtype=float)
    at = np.linspace(0, 1, len(stops))
    return np.stack([np.interp(np.clip(t, 0, 1), at, table[:, k]) for k in range(3)], 1).round().astype(int).tolist()


@st.cache_resource(show_spinner="Loading roads, communes and census flows…")
def load(sample: float) -> tuple[dict, pl.DataFrame]:
    parts = W.frames(sample=sample)
    paths = parts["links"].select("id", "name", "capacity", "t0", path=W.paths(parts["links"]))
    return parts, paths.with_columns(pl.col("name").fill_null("unnamed road"))


@st.cache_data(max_entries=8, show_spinner=False)
def run(scenario: str, period: str, sample: float, knobs: tuple) -> dict:
    """The settled peaks, computed in a subprocess so the graph never outlives a rerun.

    maplib models built in one of Streamlit's script threads and dropped in the next
    take the server down with them (exit 139, no traceback); a child process that ends
    with the run cannot.  It costs a python start and a rebuild of the frames per setting.

    Plainly ``world.py`` as a script, not multiprocessing: Streamlit runs this file as
    ``__main__`` through runpy, and both spawn and forkserver refuse to start a child
    from a main module that is still executing.
    """
    out = Path(tempfile.gettempdir()) / f"bayonne_{abs(hash((scenario, period, sample, knobs)))}.pkl"
    done = subprocess.run([sys.executable, W.__file__, scenario, period, str(sample),
                           json.dumps(dict(knobs)), str(out)], capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f"the run died ({done.returncode}): {done.stderr[-800:]}")
    try:
        return pickle.loads(out.read_bytes())
    finally:
        out.unlink(missing_ok=True)


def deck(shown: pl.DataFrame) -> pdk.Deck:
    return pdk.Deck(
        layers=[pdk.Layer("PathLayer", shown.select("name", "path", "color", "width", "label").to_dicts(),
                          get_path="path", get_color="color", get_width="width",
                          width_units=pdk.types.String("pixels"), width_min_pixels=1,
                          cap_rounded=True, pickable=True, auto_highlight=True)],
        initial_view_state=pdk.ViewState(latitude=43.4895, longitude=-1.478, zoom=13.2),
        map_style="dark" if DARK else "light",
        tooltip={"html": "<b>{name}</b><br/>{label}"},
    )


def idle(frame: pl.DataFrame) -> pdk.Deck:
    """The network with nothing on it yet — drawn before the model is asked anything."""
    return deck(frame.with_columns(
        color=pl.Series([[120, 120, 118]] * frame.height),
        width=0.6 + 2 * (pl.col("capacity") / 2000).clip(0, 1),
        label=pl.lit("waiting for the model"),
    ))


def jam(frame: pl.DataFrame) -> pdk.Deck:
    """Congestion as it stands: volume over capacity, light to dark."""
    shown = frame.filter(pl.col("flow") >= 50)
    return deck(shown.with_columns(
        color=pl.Series(colour(shown["ratio"].to_numpy() / 1.25, RAMP)),
        width=1 + 5 * (pl.col("flow") / 3000).clip(0, 1),
        label=pl.format("{} veh/h, {}% of capacity, {} min",
                        pl.col("flow").round(0), (100 * pl.col("ratio")).round(0), (pl.col("time") / 60).round(1)),
    ))


def moved(frame: pl.DataFrame) -> pdk.Deck:
    """What the closure moved: peak flows against the day before."""
    shown = frame.with_columns(delta=pl.col("flow") - pl.col("was")).filter(pl.col("delta").abs() >= 25)
    return deck(shown.with_columns(
        color=pl.Series(colour((shown["delta"].to_numpy() / 1500 + 1) / 2, [LESS, MIDDLE, MORE])),
        width=1.5 + 6 * (pl.col("delta").abs() / 1500).clip(0, 1),
        label=pl.format("{} → {} veh/h", pl.col("was").round(0), pl.col("flow").round(0)),
    ))


with st.sidebar.form("setup"):
    scenario = st.selectbox("Scenario", list(W.SCENARIOS), index=1)
    period = st.selectbox("Period of the year", list(W.PERIODS), help="Commuting volume relative to a school-term weekday.")
    with st.expander("Behaviour and calibration", icon=":material/tune:"):
        mu = st.slider("Mode sensitivity μ", 0.05, 1.0, W.PARAMS["mu"],
                       help="How strongly mode shares answer a change in travel time (nest scale ratio).")
        replan = st.slider("Commuters reconsidering each day", 0.02, 0.5, W.PARAMS["replan"], format="%.2f")
        peak = st.slider("Peak-hour vehicles per car commuter", 0.3, 1.2, W.PARAMS["peak_factor"],
                         help="Share of commutes in the peak hour, times the uplift for non-commute traffic. The knob traffic counts calibrate.")
        sample = st.segmented_control("Agent sample", [0.05, 0.1, 0.2], default=0.1, format_func=lambda v: f"{v:.0%}") or 0.1
    st.form_submit_button("Run the model", type="primary", icon=":material/play_arrow:", width="stretch")

areas, streets = W.SCENARIOS[scenario]
closes = ", ".join(streets) or ", ".join(f"all of {a}" for a in areas) or "nothing — the city as it stands"
st.title("Bayonne with a car-free old town")
st.caption(
    "The morning commute across the Bayonne–Anglet–Biarritz area and its hinterland: census 2022 home–work "
    "flows, OpenStreetMap roads, day-to-day route and mode choice in SPARQL on a knowledge graph "
    f"([skabm](https://github.com/remiadon/skabm)). Closed to cars: {closes}. "
    "Not calibrated on traffic counts yet: read the differences, not the levels."
)

knobs = (("mu", mu), ("replan", replan), ("peak_factor", peak))
pick, view = st.columns([2, 3])
choice = pick.selectbox("Morning", list(STATES), index=2,
                        help="The first morning after is the transient: the streets are shut and nobody has "
                             "changed route or mode yet.")
seen = view.segmented_control("Map", ["The morning, hour by hour", "What the closure moved"],
                              default="The morning, hour by hour") or "The morning, hour by hour"

parts, paths = load(sample)
geo = paths.select("id", "name", "path")
numbers = st.container()  # filled once the model has a day to show
with st.container(border=True):
    dial = st.container(horizontal=True)
    canvas, caption = st.empty(), st.empty()
clock, jammed = dial.empty(), dial.empty()
clock.metric("Time of day", "—", border=True)
jammed.metric("Roads over capacity", "—", border=True)
canvas.pydeck_chart(idle(paths.select("id", "name", "capacity", "path")), height=520)
caption.caption("The network, before anything moves.")

with st.spinner("Settling the assignment: commuters try routes and modes until it stops moving…", show_time=True):
    result = run(scenario, period, sample, knobs)
days, flows = result["days"], result["flows"]
after = days.filter(pl.col("phase") != "warm-up")["day"].to_list()
day = {"before": BEFORE, "first": after[0], "last": after[-1]}[STATES[choice]]

TIME, CAR, BUS = "sig__AVG__Commuter__time", "sig__AVG__Commuter__car", "sig__AVG__Commuter__bus"
SHOWN = {"Mean commute": (TIME, 1 / 60, "{:.1f} min", "inverse"),
         "Commuting by car": (CAR, 100, "{:.1f}%", "inverse"),
         "By public transport": (BUS, 100, "{:.1f}%", "normal"),
         "Cars in Grand Bayonne": ("vkt", 1, "{:,.0f} veh·km/h", "inverse")}
base = days.filter(pl.col("phase") == "warm-up").tail(1).to_dicts()[0]
now = base if day == BEFORE else days.filter(pl.col("day") == day).to_dicts()[0]
with numbers.container(horizontal=True):
    for label, (column, scale, form, sense) in SHOWN.items():
        value = now[column] * scale
        gap = None if day == BEFORE else value - base[column] * scale
        delta = None if gap is None else ("+" if gap > 0 else "") + form.format(gap)
        st.metric(label, form.format(value), delta, delta_color=sense, border=True)

chosen = flows[day]
if seen == "What the closure moved":
    clock.metric("Time of day", "08:00", "the peak", delta_color="off", border=True)
    at_peak = chosen.join(flows[BEFORE].rename({"flow": "was"}), on="id").join(geo, on="id")
    jammed.metric("Roads over capacity", f"{at_peak.filter(pl.col('flow') > pl.col('capacity')).height:,}", border=True)
    canvas.pydeck_chart(moved(at_peak), height=520)
    caption.caption(f":blue[less traffic] · :red[more traffic] than {WAS}, at the peak; under 25 veh/h hidden")
else:
    for at, share in W.MORNING.items():  # one tick per half hour: the settled peak, scaled back
        frame = W.at_hour(chosen, paths, share, dict(knobs)).join(geo, on="id")
        clock.metric("Time of day", at, f"{share:.0%} of the peak's cars", delta_color="off", border=True)
        jammed.metric("Roads over capacity", f"{frame.filter(pl.col('flow') > pl.col('capacity')).height:,}", border=True)
        canvas.pydeck_chart(jam(frame), height=520)
        caption.caption(f"{choice}. Volume over capacity, light (free-flowing) to dark (saturated); "
                        "width follows the flow.")
        time.sleep(1.4)

change = (chosen.join(flows[BEFORE].rename({"flow": "was"}), on="id").join(geo, on="id")
          .with_columns(delta=pl.col("flow") - pl.col("was"))
          .filter(pl.col("name") != "unnamed road").sort(pl.col("delta").abs(), descending=True)
          .unique("name", maintain_order=True).head(12).select("name", "was", "flow", "delta"))
with st.container(border=True):
    st.markdown(f"**Streets whose peak traffic changes most** — {choice.lower()}, against {WAS}")
    st.dataframe(change, hide_index=True, width="stretch", column_config={
        "name": "Street", "was": st.column_config.NumberColumn("Before (veh/h)", format="%d"),
        "flow": st.column_config.NumberColumn("That day (veh/h)", format="%d"),
        "delta": st.column_config.NumberColumn("Change", format="%+d")})
st.line_chart(days.select(**{"pass": pl.col("day"), "mean commute (min)": pl.col(TIME) / 60}),
              x="pass", y="mean commute (min)", color=MORE, height=200)
st.caption(f"The solver settling: each pass is commuters retrying routes and modes, not a day of Bayonne. "
           f"The streets close after pass {W.WARMUP}.")

with st.expander("What this model is, and is not", icon=":material/info:"):
    st.markdown(f"""
- **Agents**: a {sample:.0%} sample of the {parts["flows"]["workers"].sum():,.0f} commuters living or working in the 13 communes
  of the Bayonne–Anglet–Biarritz core (INSEE census 2022, flows by mode, via data.gouv.fr), in {parts["zones"].height:,} zones.
- **Network**: {parts["links"].height:,} directed road links from OpenStreetMap — every street of the core, the main roads
  from Hendaye to Dax — with speed limits, lanes and bus lanes as mapped.
- **How a morning is computed**: commuters retry routes and modes until the assignment stops moving — nested logit
  on the previous pass's travel times (Horowitz 1984; Cascetta 1989), mode shares pivoting on the census
  (Koppelman 1983), links congesting by BPR. Those passes are how the equilibrium is *solved*, not days of
  Bayonne; only three mornings mean anything, and they are the ones you can pick.
- **The morning replay** scales that settled peak by an assumed profile ({", ".join(f"{k} {v:.0%}" for k, v in W.MORNING.items())})
  and re-applies the same delay curve: flows fall linearly, delay does not. No queue carries from one half hour to
  the next, and an hourly count profile is the first thing that should replace it.
- **Not modelled**: departure-time shifts, destination changes and trips given up (so traffic "evaporation" is only
  mode shift), queues spilling back, parking capacity, through traffic on the A63 that neither starts nor ends in
  the area, tourists, deliveries, residents' permits.
- **Noise**: at a 10% sample, mode-share changes under ~0.3 points are seed noise, not signal.
- **Periods** scale commuting volume by assumption ({", ".join(f"{k}: {v:.0%}" for k, v in W.PERIODS.items())}) until
  seasonal counts calibrate them; the fêtes de Bayonne are the natural experiment that can calibrate the rest.
""")
