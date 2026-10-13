"""``skabm.player`` — engine tests: the Run / watch / replay contract, no model behind it."""

from skabm.player import Player


def test_a_run_draws_every_step_then_the_slider_replays_it():
    drawn, finished = [], []
    player = Player(
        lambda: iter("abc"),
        lambda history, k: drawn.append(history[k]),
        pace=0.0,
        done=finished.append,
    )
    player.button.click()
    assert drawn == ["a", "b", "c"] and finished == [["a", "b", "c"]]
    assert (player.slider.max, player.slider.value, player.slider.disabled) == (
        2,
        2,
        False,
    )
    assert not player.button.disabled and player.button.description == "Run"
    player.slider.value = 0
    assert drawn[-1] == "a"


def test_one_step_leaves_nothing_to_replay():
    player = Player(lambda: iter("a"), lambda history, k: None, pace=0.0)
    player.button.click()
    assert player.slider.disabled and player.slider.max == 0


def test_a_failing_run_hands_the_controls_back():
    def play():
        yield "a"
        raise RuntimeError("the model died")

    drawn = []
    player = Player(play, lambda history, k: drawn.append(k), pace=0.0)
    player.button.click()  # outside a kernel the dispatcher logs the error
    assert (
        player.history == ["a"]
        and not player.button.disabled
        and player.slider.disabled
    )
    player.history = []
    player.slider.max, player.slider.value = 1, 1
    assert drawn == [0]  # a replay with no run behind it draws nothing
