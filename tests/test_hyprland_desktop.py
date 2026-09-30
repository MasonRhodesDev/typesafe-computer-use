"""The Hyprland adapter's side of hyprhands' wire, and how its replies reach jev, with a fake
hyprhands on the other end of the pipe instead of ssh."""

import io
import json
import struct

import pytest
from PIL import Image

from typesafe_computer_use import perception
from typesafe_computer_use.hyprland import desktop as hypr
from typesafe_computer_use.models import Abort, Screen


def frame(header: dict, blob: bytes | None = None) -> bytes:
    if blob is not None:
        header = {**header, "blob": len(blob)}
    body = json.dumps(header).encode()
    return struct.pack(">I", len(body)) + body + (blob or b"")


def test_a_request_is_length_prefixed_json():
    out = io.BytesIO()
    hypr.write_request(out, {"op": "tree", "text": True})
    wire = out.getvalue()
    (n,) = struct.unpack(">I", wire[:4])
    assert json.loads(wire[4 : 4 + n]) == {"op": "tree", "text": True} and len(wire) == 4 + n


def test_a_blob_follows_its_header_and_the_next_reply_follows_the_blob():
    r = io.BytesIO(frame({"ok": True, "format": "qoi"}, b"PIXELS") + frame({"ok": True}))
    assert hypr.read_reply(r) == ({"ok": True, "format": "qoi", "blob": 6}, b"PIXELS")
    assert hypr.read_reply(r) == ({"ok": True}, None)
    with pytest.raises(EOFError):
        hypr.read_reply(r)


def test_the_serve_command_finds_hyprhands_outside_a_login_path(monkeypatch):
    monkeypatch.setattr(hypr, "HYPRHANDS", None)
    cmd = hypr.serve_command("HDMI-A-1")
    assert '$HOME/.local/bin/hyprhands' in cmd and cmd.endswith('exec "$h" serve --monitor HDMI-A-1')
    monkeypatch.setattr(hypr, "HYPRHANDS", "/opt/hh")
    assert hypr.serve_command(None) == "exec /opt/hh serve"


@pytest.mark.parametrize(
    "error, why",
    [
        ("click: refused: owner took over: mouse moved 66px from where hyprhands left it", "owner took over: mouse moved 66px from where hyprhands left it"),
        ("key: refused: stopped: the panic file exists", "stopped: the panic file exists"),
        ("key: SUPER+Q is the owner's compositor bind (exec_cmd: Apps: Terminal); it would not reach the app. Pass allow_bind to trigger it.", "SUPER+Q is the owner's compositor bind (exec_cmd: Apps: Terminal); it would not reach the app. Pass allow_bind to trigger it."),
        ("click: no monitor", None),
    ],
)
def test_refusals_end_the_run_and_other_failures_stay_errors(error, why):
    assert hypr._refusal(error) == why


class FakeHyprhands:
    """hyprhands' end of the pipe: each request written gets the next scripted reply to read."""

    def __init__(self, replies):
        self.replies, self.seen, self.pending = list(replies), [], b""
        self.stdin, self.stdout = self, self

    def poll(self):
        return None

    def write(self, wire):
        (n,) = struct.unpack(">I", wire[:4])
        self.seen.append(json.loads(wire[4 : 4 + n]))
        self.pending += self.replies.pop(0)

    def flush(self):
        pass

    def read(self, n):
        out, self.pending = self.pending[:n], self.pending[n:]
        return out


def desk(replies):
    d = hypr.HyprlandDesktop.__new__(hypr.HyprlandDesktop)
    d.host, d._obs, d.atspi = "desk", None, True
    d.monitor = {"name": "HDMI-A-1", "width": 1080.0, "height": 1920.0, "scale": 1.0}
    d._proc = FakeHyprhands(replies)
    return d, d._proc


WINDOW = {"address": "0x1", "class": "chromium", "title": "t", "pid": 7, "frame": [9.0, 49.0, 1062, 922]}


def test_one_observation_reads_the_tree_with_its_text_once():
    tree = {"ok": True, "xml": "<desktop-frame><application name='Chromium' /></desktop-frame>", "capped": False, "lines": [{"text": "Hello", "box": [1, 2, 3, 4]}]}
    d, fake = desk([frame(tree), frame({"ok": True, "window": WINDOW, "cursor": [5, 5], "takeover": None})])
    assert d.text_lines(None) == [("Hello", 1.0, (1, 2, 3, 4))]
    assert d.frontmost_app_and_pid() == ("Chromium", 7)
    assert [r["op"] for r in fake.seen] == ["tree", "state"] and fake.seen[0]["text"] is True


def test_a_thin_tree_offers_no_text():
    d, _ = desk([frame({"ok": True, "xml": "<desktop-frame />", "capped": False, "lines": None}), frame({"ok": True, "window": None})])
    assert d.text_lines(None) is None


def test_scroll_is_one_positioned_op_over_the_window_centre():
    tree = {"ok": True, "xml": "<desktop-frame />", "lines": None}
    d, fake = desk([frame({"ok": True, "takeover": None, "cursor": [500, 500]}), frame(tree), frame({"ok": True, "window": WINDOW}), frame({"ok": True})])
    d.scroll(6)
    assert fake.seen[-1] == {"op": "scroll", "x": 9.0 + 531, "y": 49.0 + 461, "notches": -2}


def test_the_panic_file_and_a_takeover_abort_the_run():
    d, _ = desk([frame({"ok": True, "takeover": "stopped: the panic file exists", "cursor": None})])
    with pytest.raises(Abort, match="panic file"):
        d.check_abort()
    d, _ = desk([frame({"ok": True, "takeover": None, "cursor": [500, 500]}), frame({"ok": False, "error": "refused: owner took over: focus moved to DP-2"})])
    with pytest.raises(Abort, match="focus moved to DP-2"):
        d.click_at((10, 10))


def test_a_capture_arrives_as_qoi():
    qoi = pytest.importorskip("qoi")
    import numpy as np

    pixels = np.zeros((4, 6, 3), dtype=np.uint8)
    pixels[1, 2] = (255, 0, 0)
    d, _ = desk([frame({"ok": True, "format": "qoi", "width": 6, "height": 4}, qoi.encode(pixels))])
    img = d.screenshot()
    assert img.size == (6, 4) and img.getpixel((2, 1)) == (255, 0, 0)


def screen(pid):
    return Screen(image=Image.new("RGB", (100, 100)), scale=1.0, app="Google Chrome", field=None, url=None, pid=pid)


def test_ocr_takes_the_adapters_tree_text_when_it_offers_some(monkeypatch):
    offered = [("$83.99 USD", 1.0, (1.0, 2.0, 50.0, 12.0))]
    monkeypatch.setattr(perception.desktop, "text_lines", lambda s: offered, raising=False)
    monkeypatch.setattr(perception, "ocr_crop", lambda *a: (_ for _ in ()).throw(AssertionError("read pixels")))
    assert perception.ocr_lines(screen(pid=42)) == (offered, 0.0, 0)


def test_ocr_reads_pixels_when_the_tree_is_thin_or_the_capture_is_a_replay(monkeypatch):
    read = [("pixels", 0.9, (0.0, 0.0, 10.0, 10.0))]
    monkeypatch.setattr(perception, "ocr_crop", lambda *a: read)
    monkeypatch.setattr(perception.desktop, "text_lines", lambda s: None, raising=False)
    assert perception.ocr_lines(screen(pid=42))[0] == read
    monkeypatch.setattr(perception.desktop, "text_lines", lambda s: [("tree", 1.0, (0, 0, 1, 1))], raising=False)
    assert perception.ocr_lines(screen(pid=None))[0] == read
