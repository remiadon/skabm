"""Run, watch, replay: the controls every live simulation page shares.

A page (a Voilà notebook) builds its own model controls and drawing, and hands
``Player`` two functions: *play()*, which reads those controls and returns the run as
an iterator of steps, and *show(history, k)*, which draws step k of the steps so far.
``Player`` adds the rest: nothing runs until ``button`` is clicked, each step is drawn
as it comes and stays at least ``pace`` seconds, ``slider`` replays the run afterwards,
and ``errors`` shows what a callback raised, which Voilà would otherwise swallow.
"""

import time
from collections.abc import Callable, Iterator

import ipywidgets as w


class Player:
    def __init__(
        self,
        play: Callable[[], Iterator],
        show: Callable[[list, int], None],
        *,
        pace: float = 0.25,
        unit: str = "tick",
        done: Callable[[list], None] | None = None,
    ):
        self.play, self.show, self.done = play, show, done
        self.history: list = []
        self.pace = w.FloatSlider(
            pace,
            min=0.0,
            max=max(2.0, 3 * pace),
            step=0.05,
            description=f"Seconds per {unit}",
            style={"description_width": "11em"},
            layout=w.Layout(width="100%"),
        )
        self.button = w.Button(
            description="Run",
            button_style="danger",
            icon="play",
            layout=w.Layout(width="100%"),
        )
        self.slider = w.IntSlider(
            0,
            min=0,
            max=0,
            description=unit.capitalize(),
            disabled=True,
            continuous_update=False,
        )
        self.errors = w.Output()
        self.button.on_click(self.errors.capture(clear_output=True)(self.run))
        self._replay = self.errors.capture()(self.replay)
        self.slider.observe(self._replay, names="value")

    def run(self, _=None) -> None:
        """Play the run step by step, then hand it to *done* and the slider."""
        self.button.disabled, self.slider.disabled = True, True
        self.button.description = "Running…"
        self.history, shown = [], 0.0
        try:
            for step in self.play():
                self.history.append(step)
                time.sleep(max(0.0, shown + self.pace.value - time.monotonic()))
                self.show(self.history, len(self.history) - 1)
                shown = time.monotonic()
        finally:
            self.slider.unobserve(self._replay, names="value")  # the last step is drawn
            self.slider.max = self.slider.value = max(len(self.history) - 1, 0)
            self.slider.observe(self._replay, names="value")
            self.slider.disabled = self.slider.max == 0  # one step: nothing to replay
            self.button.disabled, self.button.description = False, "Run"
        if self.done:
            self.done(self.history)

    def replay(self, change: dict) -> None:
        if self.history:  # a run that failed before its first step leaves nothing
            self.show(self.history, change["new"])
