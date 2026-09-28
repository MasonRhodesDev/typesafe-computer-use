"""The `Desktop` surface over ssh to a Hyprland machine, so jev's loop drives it unchanged.

jev runs here, with its keys; the screen, the mouse and the keyboard are on another machine. One
ssh session carries `remote.py` there and then its JSON-lines requests: a capture with `grim`, the
active window from `hyprctl`, the active app's accessibility tree through AT-SPI, and every input
through `hypruse`. OCR runs here, on RapidOCR, the same backend the OSWorld runs use on Linux.

jev knows one display with its origin at 0,0. That is one monitor of the Hyprland machine, the
focused one unless named, and `remote.py` does the translation both ways, so every coordinate on
this side is in that monitor's logical space.

Reads are cached per observation, as in `osworld/desktop.py`: the first read after an input
fetches the window and the tree once, and every read up to the next input reuses them.

Nothing can be pressed through the tree from here (AT-SPI objects do not cross ssh), so the
`ax_*` actions refuse and every click is a real pointer click, delivered by hypruse.
"""

from __future__ import annotations

import base64
import json
import shlex
import subprocess
import time
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path

from PIL import Image

from ..config import ABORT_CORNER_PX
from ..models import Abort, AxNode, Field
from ..osworld import a11y
from ..platform_adapter import OcrLine
from . import tree_text

REMOTE_SCRIPT = Path(__file__).with_name("remote.py")
ABORT_POLL_SECONDS = 0.25
SCROLL_LINES_PER_NOTCH = 3  # jev scrolls in lines (a Mac wheel unit); a wheel notch is about three
LAUNCH_SETTLE_SECONDS = 1.0

# Window classes onto the app names jev compares with CLICKER_BROWSER, and back.
APP_NAMES = {"google-chrome": "Google Chrome", "chromium": "Chromium"}
APP_CLASSES = {name: klass for klass, name in APP_NAMES.items()}

# The browser jev drives: its own profile, never the owner's logins, with the accessibility tree
# on (Chromium exposes none to AT-SPI without the flag).
BROWSER_ARGV = {
    "Google Chrome": ["google-chrome-stable"],
    "Chromium": ["chromium"],
}
BROWSER_FLAGS = [
    "--user-data-dir=$HOME/.cache/jev-browser",
    "--force-renderer-accessibility",
    "--no-first-run",
    "--no-default-browser-check",
]

# jev's key names are the Mac's; the Mac's delete erases backwards.
KEYS = {"return": "enter", "escape": "esc", "delete": "backspace", "tab": "tab"}


class RemoteError(RuntimeError):
    """The Hyprland machine refused or failed a request."""


class HyprlandDesktop:
    """One ssh session to a Hyprland machine, and the observation read over it."""

    def __init__(self, host: str, monitor: str | None, recognize_text, *, ssh: list[str] | None = None) -> None:
        self.host = host
        self._recognize = recognize_text
        script = base64.b64encode(REMOTE_SCRIPT.read_bytes()).decode()
        remote_cmd = f"python3 -u -c {shlex.quote(f'import base64;exec(base64.b64decode({script!r}))')}"
        self._proc = subprocess.Popen(
            [*(ssh or ["ssh", "-o", "BatchMode=yes"]), host, remote_cmd],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        hello = self._call("hello", monitor=monitor)
        self.monitor = hello["monitor"]
        self.atspi = hello["atspi"]
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

    def _call(self, op: str, **args) -> dict:
        if self._proc.poll() is not None:
            raise RemoteError(f"the ssh session to {self.host} has ended")
        self._proc.stdin.write(json.dumps({"op": op, **args}) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise RemoteError(f"the ssh session to {self.host} closed during {op}")
        reply = json.loads(line)
        if not reply.pop("ok"):
            raise RemoteError(f"{op}: {reply.get('error')}")
        return reply

    def _input(self, op: str, **args) -> dict:
        self._obs = None  # anything read before this input is stale after it
        return self._call(op, **args)

    def _now(self) -> dict:
        """The window and the tree as they are now, fetched once per observation."""
        if self._obs is None:
            state = self._call("state")
            tree = self._call("tree")
            root = a11y.parse(tree["xml"])
            window = state["window"]
            name = APP_NAMES.get(window["class"], "") if window else ""
            if root is not None and name:
                for app in root:
                    app.set("name", name)  # the name jev compares with CLICKER_BROWSER, whatever AT-SPI calls it
            self._obs = {"window": window, "root": root, "capped": tree.get("capped", False)}
        return self._obs

    # ----- the escape hatch ------------------------------------------------------------------

    def check_abort(self) -> None:
        state = self._call("state")
        if state["stop"]:
            raise Abort(f"panic key on {self.host}")
        x, y = state["cursor"]
        if 0 <= x <= ABORT_CORNER_PX and 0 <= y <= ABORT_CORNER_PX:
            raise Abort(f"mouse in the top-left corner of {self.monitor['name']}")

    def abort_hint(self) -> str:
        return f"Ctrl-C here, the panic key (SUPER+SHIFT+BackSpace) on {self.host}, or the mouse in {self.monitor['name']}'s top-left corner"

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
        """Scroll under the pointer, parked over the active window's centre first, as the Mac adapter
        does. jev's positive lines scroll up; hypruse's positive notches scroll down."""
        self.check_abort()
        bounds = self.frontmost_window_bounds()
        if bounds is not None:
            x, y, w, h = bounds
            self._input("move", x=x + w / 2, y=y + h / 2)
        notches = -round(lines / SCROLL_LINES_PER_NOTCH) or (-1 if lines > 0 else 1)
        self._input("scroll", notches=notches)

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
        found = self._call("find_window", klass=klass)["address"]
        if found is None:
            if app not in BROWSER_ARGV:
                return False
            self._input("launch", argv=[*BROWSER_ARGV[app], *BROWSER_FLAGS])
            found = self._call("find_window", klass=klass)["address"]
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
        if self._call("find_window", klass=klass)["address"] is None:
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
        png = base64.b64decode(self._call("screenshot")["png"])
        return Image.open(BytesIO(png)).convert("RGB")

    def display_scale(self, image: Image.Image) -> float:
        return image.width / self.monitor["width"]

    def recognize_text(self, image: Image.Image) -> list[OcrLine]:
        return self._recognize(image)

    def text_lines(self, screen) -> list[OcrLine] | None:
        """The screen's text from the accessibility tree, in capture pixels, or None to read pixels.

        None when the walk was cut short (text in view may be missing) or the tree holds too little
        text to stand in for OCR.
        """
        obs = self._now()
        if obs["root"] is None or obs["capped"]:
            return None
        app = a11y.active_app(obs["root"])
        lines = tree_text.tree_lines(app, screen.scale, self.monitor["width"], self.monitor["height"])
        return lines if len(lines) >= tree_text.MIN_LINES else None

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
