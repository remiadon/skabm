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


def test_fetch_reads_only_the_members_asked_for(tmp_path, monkeypatch):
    """``fetch`` opens a remote zip by range requests and keeps what it already has."""
    import io
    import zipfile

    import httpx

    from skabm import player

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("deep/dir/wanted.csv", "a,b\n1,2\n")
        z.writestr("other.csv", "x" * 10_000)
    body, ranges = archive.getvalue(), []

    def serve(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            return httpx.Response(200, headers={"content-length": str(len(body))})
        start, end = map(int, request.headers["range"][6:].split("-"))
        ranges.append((start, end))
        return httpx.Response(206, content=body[start : end + 1])

    real = httpx.Client
    monkeypatch.setattr(
        player.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(serve))
    )
    got = player.fetch("https://example.org/a.zip", ["wanted.csv"], tmp_path / "data")
    assert got["wanted.csv"].read_text() == "a,b\n1,2\n"
    assert ranges and not (tmp_path / "data" / "other.csv").exists()
    ranges.clear()
    player.fetch("https://example.org/a.zip", ["wanted.csv"], tmp_path / "data")
    assert ranges == []  # already there
