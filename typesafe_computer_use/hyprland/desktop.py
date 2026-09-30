"""The `Desktop` surface over ssh to a Hyprland machine, so jev's loop drives it unchanged.

jev runs here, with its keys; the screen, the mouse and the keyboard are on another machine, where
`hyprhands serve` (https://github.com/MasonRhodesDev/hyprhands, installed there) holds one session
for the whole run. It speaks framed JSON on stdio, so it rides the ssh session unchanged: captures
come back as raw QOI bytes, the active window's accessibility tree as the OSWorld XML jev's
`osworld/a11y.py` reads, and the screen's text from that tree when there is enough of it. OCR runs
here, on RapidOCR, only when there is not.

jev knows one display with its origin at 0,0. That is one monitor of the Hyprland machine, the
focused one unless named, and hyprhands does the translation both ways, so every coordinate on
this side is in that monitor's logical space.

Reads are cached per observation, as in `osworld/desktop.py`: the first read after an input
fetches the window and the tree once, and every read up to the next input reuses them.

hyprhands watches for the owner taking the seat back (the cursor moving off where it left it, or
focus leaving the monitor) and refuses input from then on; that refusal ends the run as an Abort.
Nothing is pressed through the tree yet, so the `ax_*` actions refuse and every click is a real
pointer click.
"""

from __future__ import annotations

import json
import os
import shlex
import struct
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import BinaryIO

from PIL import Image

from ..config import ABORT_CORNER_PX
from ..models import Abort, AxNode, Field
from ..osworld import a11y
from ..platform_adapter import OcrLine

ABORT_POLL_SECONDS = 0.25
SCROLL_LINES_PER_NOTCH = 3  # jev scrolls in lines (a Mac wheel unit); a wheel notch is about three
LAUNCH_SETTLE_SECONDS = 1.0
# Where hyprhands is on the Hyprland machine: a non-interactive ssh login's PATH leaves out
# ~/.local/bin, so it is tried first. JEV_HYPRHANDS names another path.
HYPRHANDS = os.environ.get("JEV_HYPRHANDS")
MAX_FRAME = 64 << 20

# Window classes onto the app names jev compares with CLICKER_BROWSER, and back.
APP_NAMES = {"google-chrome": "Google Chrome", "chromium": "Chromium"}
APP_CLASSES = {name: klass for klass, name in APP_NAMES.items()}

# The browser jev drives: the owner's own, logins and all (a URL opens as a new tab in the running
# instance), unless JEV_BROWSER_PROFILE names a separate profile directory on the Hyprland machine.
# A browser jev starts gets its accessibility tree on (Chromium exposes none to AT-SPI without the
# flag unless an assistive technology asked for it).
BROWSER_ARGV = {
    "Google Chrome": ["google-chrome-stable"],
    "Chromium": ["chromium"],
}
BROWSER_FLAGS = [
    *([f"--user-data-dir={os.environ['JEV_BROWSER_PROFILE']}"] if os.environ.get("JEV_BROWSER_PROFILE") else []),
    "--force-renderer-accessibility",
    "--no-first-run",
    "--no-default-browser-check",
]

# jev's key names are the Mac's; the Mac's delete erases backwards.
KEYS = {"return": "enter", "escape": "esc", "delete": "backspace", "tab": "tab"}


class RemoteError(RuntimeError):
    """The Hyprland machine refused or failed a request."""


def write_request(w: BinaryIO, req: dict) -> None:
    """One request: a u32 big-endian length, then that much JSON."""
    body = json.dumps(req).encode()
    w.write(struct.pack(">I", len(body)) + body)
    w.flush()


def _read_exact(r: BinaryIO, n: int) -> bytes:
    out = b""
    while len(out) < n:
        chunk = r.read(n - len(out))
        if not chunk:
            raise EOFError
        out += chunk
    return out


def read_reply(r: BinaryIO) -> tuple[dict, bytes | None]:
    """One reply: a length-prefixed JSON header, then exactly `blob` raw bytes when it names some."""
    (n,) = struct.unpack(">I", _read_exact(r, 4))
    if n > MAX_FRAME:
        raise RemoteError(f"a {n}-byte reply header is not hyprhands talking")
    header = json.loads(_read_exact(r, n))
    blob = _read_exact(r, header["blob"]) if "blob" in header else None
    return header, blob


def serve_command(monitor: str | None) -> str:
    """The shell command that starts hyprhands' session on the Hyprland machine."""
    args = " serve" + (f" --monitor {shlex.quote(monitor)}" if monitor else "")
    if HYPRHANDS:
        return f"exec {shlex.quote(HYPRHANDS)}{args}"
    return f'h="$HOME/.local/bin/hyprhands"; [ -x "$h" ] || h=hyprhands; exec "$h"{args}'


def _refusal(message: str) -> str | None:
    """Why hyprhands refused an input, when the refusal should end the run rather than crash it:
    the owner took the seat back, the panic file exists, or the key is one of the owner's
    compositor binds (sending it anyway would run the owner's shortcut)."""
    if "refused: " in message:
        return message.split("refused: ", 1)[1]
    if "compositor bind" in message:
        return message.split(": ", 1)[-1]
    return None


class HyprlandDesktop:
    """One ssh session to a Hyprland machine, and the observation read over it."""

    def __init__(self, host: str, monitor: str | None, recognize_text, *, ssh: list[str] | None = None) -> None:
        self.host = host
        self._recognize = recognize_text
        self._proc = subprocess.Popen(
            [*(ssh or ["ssh", "-o", "BatchMode=yes"]), host, serve_command(monitor)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        hello = self._call("hello")
        self.monitor = hello["monitor"]
        self.atspi = hello["a11y"]
        self._obs: dict | None = None

    def __repr__(self) -> str:
        return f"<HyprlandDesktop {self.host}:{self.monitor['name']}>"

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.stdin.close()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    # ----- the wire --------------------------------------------------------------------------

    def _call_blob(self, op: str, **args) -> tuple[dict, bytes | None]:
        if self._proc.poll() is not None:
            raise RemoteError(f"the ssh session to {self.host} has ended")
        try:
            write_request(self._proc.stdin, {"op": op, **args})
            reply, blob = read_reply(self._proc.stdout)
        except (EOFError, BrokenPipeError) as e:
            raise RemoteError(f"the ssh session to {self.host} closed during {op} (is hyprhands installed there?)") from e
        if not reply.pop("ok"):
            raise RemoteError(f"{op}: {reply.get('error')}")
        return reply, blob

    def _call(self, op: str, **args) -> dict:
        return self._call_blob(op, **args)[0]

    def _input(self, op: str, **args) -> dict:
        self._obs = None  # anything read before this input is stale after it
        try:
            return self._call(op, **args)
        except RemoteError as e:
            if (why := _refusal(str(e))) is not None:
                raise Abort(why) from e
            raise

    def _now(self) -> dict:
        """The window and the tree as they are now, fetched once per observation."""
        if self._obs is None:
            tree = self._call("tree", text=True)
            window = self._call("state")["window"]
            root = a11y.parse(tree["xml"])
            name = APP_NAMES.get(window["class"], "") if window else ""
            if root is not None and name:
                for app in root:
                    app.set("name", name)  # the name jev compares with CLICKER_BROWSER, whatever AT-SPI calls it
            lines = tree.get("lines")
            self._obs = {
                "window": window,
                "root": root,
                "capped": tree.get("capped", False),
                "lines": None if lines is None else [(l["text"], 1.0, tuple(l["box"])) for l in lines],
            }
        return self._obs

    # ----- the escape hatch ------------------------------------------------------------------

    def check_abort(self) -> None:
        state = self._call("state")
        if state.get("takeover"):  # the owner at the controls, or the panic file
            raise Abort(state["takeover"])
        if state.get("cursor") is None:
            return
        x, y = state["cursor"]
        if 0 <= x <= ABORT_CORNER_PX and 0 <= y <= ABORT_CORNER_PX:
            raise Abort(f"mouse in the top-left corner of {self.monitor['name']}")

    def abort_hint(self) -> str:
        return (
            f"take the seat on {self.host}: move the mouse or focus another monitor; "
            "or touch $XDG_RUNTIME_DIR/hyprhands-stop there, or Ctrl-C here"
        )

    def sleep_watching(self, seconds: float) -> None:
        self._obs = None  # the screen moves on while we wait
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.check_abort()
            time.sleep(min(ABORT_POLL_SECONDS, max(0.0, end - time.monotonic())))

    def accessibility_trusted(self) -> bool:
        return True

    # ----- input -----------------------------------------------------------------------------

    def click_at(self, point: tuple[float, float]) -> None:
        self.check_abort()
        self._input("click", x=point[0], y=point[1])

    def press(self, key: str, command: bool = False) -> None:
        self.check_abort()
        if command and key == "[":
            combo = "alt+Left"  # Back, where Command-[ is Back on a Mac
        elif command:
            combo = f"ctrl+{KEYS.get(key, key)}"
        else:
            combo = KEYS.get(key, key)
        self._input("key", combo=combo)

    def type_text(self, text: str) -> None:
        self.check_abort()
        self._input("type", text=text)

    def clear_field(self) -> None:
        self.press("a", command=True)
        self.press("delete")

    def scroll(self, lines: int) -> None:
        """Scroll over the active window's centre, as the Mac adapter does. jev's positive lines
        scroll up; hyprhands' positive notches scroll down."""
        self.check_abort()
        notches = -round(lines / SCROLL_LINES_PER_NOTCH) or (-1 if lines > 0 else 1)
        bounds = self.frontmost_window_bounds()
        if bounds is None:
            self._input("scroll", notches=notches)
        else:
            x, y, w, h = bounds
            self._input("scroll", x=x + w / 2, y=y + h / 2, notches=notches)

    # ----- apps and windows ------------------------------------------------------------------

    def frontmost_app_and_pid(self) -> tuple[str, int]:
        window = self._now()["window"]
        if window is None:
            return "", 0
        return APP_NAMES.get(window["class"], window["class"]), int(window["pid"])

    def _front_class(self) -> str:
        window = self._now()["window"]
        return window["class"] if window else ""

    def activate(self, app: str, timeout: float = 3.0) -> bool:
        """Bring `app` forward; the browser is started when it has no window."""
        self.check_abort()
        klass = APP_CLASSES.get(app, app)
        found = self._call("find_window", **{"class": klass})["address"]
        if found is None:
            if app not in BROWSER_ARGV:
                return False
            found = self._input("launch", argv=[*BROWSER_ARGV[app], *BROWSER_FLAGS])["address"]
            if found is None:
                found = self._call("find_window", **{"class": klass})["address"]
            if found is None:
                return False
        self._input("bring", address=found)
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self._obs = None
            if self._front_class() == klass:
                return True
            time.sleep(0.1)
        return False

    def open_url(self, browser: str, url: str) -> bool:
        """A new tab in jev's browser when it runs (the browser hands the URL to its running
        instance), or a new browser on the driven monitor when it does not."""
        self.check_abort()
        argv = [*BROWSER_ARGV.get(browser, BROWSER_ARGV["Google Chrome"]), *BROWSER_FLAGS, url]
        klass = APP_CLASSES.get(browser, "google-chrome")
        if self._call("find_window", **{"class": klass})["address"] is None:
            self._input("launch", argv=argv)
        else:
            self._input("spawn", argv=argv)
            time.sleep(LAUNCH_SETTLE_SECONDS)
        return self.activate(browser)

    def browser_url(self, browser: str) -> str | None:
        return a11y.browser_url(self._now()["root"], browser or None)

    def open_path(self, path: Path, as_text: bool = False) -> None:
        print(f"  (on this machine) {path}")

    def frontmost_window_bounds(self, pid: int | None = None) -> tuple[float, float, float, float] | None:
        window = self._now()["window"]
        return tuple(window["frame"]) if window else None

    # ----- capture, OCR, and accessibility ---------------------------------------------------

    def screenshot(self) -> Image.Image:
        self._obs = None  # a fresh capture starts a fresh observation
        import qoi  # the `hyprland` extra; imported here so the rest of the package never needs it

        _, blob = self._call_blob("screenshot")
        return Image.fromarray(qoi.decode(blob)).convert("RGB")

    def display_scale(self, image: Image.Image) -> float:
        return image.width / self.monitor["width"]

    def recognize_text(self, image: Image.Image) -> list[OcrLine]:
        return self._recognize(image)

    def text_lines(self, screen) -> list[OcrLine] | None:
        """The screen's text from the accessibility tree, in capture pixels, or None to read pixels.

        hyprhands reads it beside the tree, and offers none when the walk was cut short (text in
        view may be missing) or the tree holds too little text to stand in for OCR.
        """
        return self._now()["lines"]

    def focused_field(self) -> Field | None:
        return a11y.focused_field(self._now()["root"])

    def actionable_elements(self, pid: int, display_w_pt: float, display_h_pt: float) -> tuple[list[AxNode], list[AxNode], bool]:
        obs = self._now()
        found, hidden, capped = a11y.walk(a11y.active_app(obs["root"]), display_w_pt, display_h_pt)
        return found, hidden, capped or obs["capped"]

    # ----- acting on an element: nothing crosses ssh -----------------------------------------

    def ax_press(self, ref) -> bool:
        return False

    def ax_focus(self, ref) -> bool:
        return False

    def ax_set_value(self, ref, text: str) -> bool:
        return False

    def ax_value(self, ref) -> str | None:
        return None


def tree_xml(root: ET.Element | None) -> str:
    """The tree as text, for a run folder or a bug report."""
    return "" if root is None else ET.tostring(root, encoding="unicode")
