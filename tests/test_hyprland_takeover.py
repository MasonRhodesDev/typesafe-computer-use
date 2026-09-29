import pytest

from typesafe_computer_use.hyprland import remote
from typesafe_computer_use.hyprland.remote import ESC_WINDOW, SETTLE_SECONDS, Takeover


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def seat(clock):
    t = Takeover(tolerance=64, clock=clock)
    t.baseline((500, 500))
    return t


def test_the_mouse_within_tolerance_is_still_jevs(seat):
    seat.cursor_seen((540, 530))
    assert seat.reason is None


def test_a_hand_on_the_mouse_latches_a_takeover(seat):
    seat.cursor_seen((900, 500))
    assert "mouse moved 400px" in seat.reason
    seat.cursor_seen((500, 500))  # back where it was: still taken over
    assert seat.reason is not None


def test_jevs_own_moves_and_their_settle_are_not_a_takeover(seat, clock):
    seat.begin()
    seat.cursor_seen((1500, 900))  # hypruse carrying the pointer to the target
    seat.end((1500, 900))
    seat.cursor_seen((1520, 910))
    clock.now += SETTLE_SECONDS + 0.1
    seat.cursor_seen((1510, 905))
    assert seat.reason is None
    seat.cursor_seen((1500, 1100))
    assert seat.reason is not None


def test_one_escape_is_use_a_burst_is_a_takeover(seat, clock):
    seat.escape()
    clock.now += ESC_WINDOW + 0.1
    seat.escape()
    clock.now += 0.2
    seat.escape()
    assert seat.reason is None
    clock.now += 0.2
    seat.escape()
    assert "Escape pressed 3 times" in seat.reason


def test_jevs_own_escape_does_not_count(seat, clock):
    for _ in range(3):
        seat.begin()
        seat.end(None)
        seat.escape()  # the bind's echo of jev's press_escape, inside the settle window
        clock.now += 0.1
    assert seat.reason is None


def test_focus_leaving_jevs_monitor_is_a_takeover(seat):
    seat.focused_monitor("HDMI-A-1", "HDMI-A-1")
    assert seat.reason is None
    seat.focused_monitor("DP-2", "HDMI-A-1")
    assert seat.reason == "owner took over: focus moved to DP-2"


def test_a_new_run_starts_clean(seat):
    seat.cursor_seen((0, 0))
    seat.baseline((10, 10))
    assert seat.reason is None


def test_input_is_refused_once_taken_over_and_reads_still_work(monkeypatch, seat):
    monkeypatch.setattr(remote, "TAKEOVER", seat)
    monkeypatch.setattr(remote, "OPS", {"click": lambda **a: {"clicked": a}, "state": lambda: {"read": True}})
    monkeypatch.setattr(remote, "global_cursor", lambda: (500, 500))
    assert remote.handle({"op": "click", "x": 1, "y": 2}) == {"clicked": {"x": 1, "y": 2}}
    seat.reason = "owner took over: test"
    with pytest.raises(RuntimeError, match="owner took over: test"):
        remote.handle({"op": "click", "x": 1, "y": 2})
    assert remote.handle({"op": "state"}) == {"read": True}
