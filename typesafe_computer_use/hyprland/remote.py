"""The half of the Hyprland adapter that runs on the Hyprland machine, sent over ssh as a script.

It is standard library plus the system's AT-SPI bindings (`gi`, `Atspi`), so nothing is installed
there: `desktop.py` hands this file to `python3 -c` (base64 on the command line, so stdin stays
free) and speaks JSON lines with it on the same ssh session, one request per line, one reply per
line. Keeping one process alive for the whole run is
the point: a tree walk is thousands of AT-SPI calls, which cost microseconds in-process and would
cost a subprocess each through `busctl`.

Input goes through `hypruse` verbs, so its always-on guards (no input into a lock screen, none
under a launcher) and its activity beacon apply to every click and keystroke jev sends.

Geometry: every coordinate this script returns is in the DRIVEN monitor's own logical space, with
its top-left at 0,0, which is the single display jev believes it has. Every coordinate it receives
is in that space too; it adds the monitor's origin before handing a point to the compositor.

The tree comes back as XML in the shape OSWorld's server writes (tag = role name with hyphens,
`st:<state>`, `cp:screencoord`/`cp:size`, `act:<name>_desc`), so jev's `osworld/a11y.py` reads it
unchanged.
"""

import base64
import glob
import json
import os
import shlex
import subprocess
import sys
import time
import warnings
import xml.etree.ElementTree as ET

warnings.filterwarnings("ignore", category=DeprecationWarning)  # Atspi marks the static interface calls too

NS_STATE = "https://accessibility.ubuntu.example.org/ns/state"
NS_ATTRIBUTES = "https://accessibility.ubuntu.example.org/ns/attributes"
NS_COMPONENT = "https://accessibility.ubuntu.example.org/ns/component"
NS_ACTION = "https://accessibility.ubuntu.example.org/ns/action"

NODE_CAP = 3000
TIME_CAP = 1.5
TEXT_CHARS = 500
EXTENT_SANITY = 20000
# uvx runs the current release (the shell verbs arrived after 0.9); an older `uv tool` copy may lack them.
HYPRUSE = shlex.split(os.environ.get("JEV_HYPRUSE") or "uvx hypruse")


def _discover_session():
    runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    os.environ["XDG_RUNTIME_DIR"] = runtime
    if not os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
        sockets = sorted(glob.glob(f"{runtime}/hypr/*/.socket.sock"), key=os.path.getmtime)
        if sockets:
            os.environ["HYPRLAND_INSTANCE_SIGNATURE"] = os.path.basename(os.path.dirname(sockets[-1]))
    if not os.environ.get("WAYLAND_DISPLAY"):
        displays = sorted(p for p in glob.glob(f"{runtime}/wayland-*") if not p.endswith(".lock"))
        if displays:
            os.environ["WAYLAND_DISPLAY"] = os.path.basename(displays[0])
    os.environ.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={runtime}/bus")
    # Over ssh libatspi can settle on a stale bus socket; the session bus's broker knows the live one.
    try:
        out = subprocess.run(
            ["busctl", "--user", "--json=short", "call", "org.a11y.Bus", "/org/a11y/bus", "org.a11y.Bus", "GetAddress"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        address = (json.loads(out.stdout).get("data") or [""])[0] if out.returncode == 0 else ""
        if address:
            os.environ["AT_SPI_BUS_ADDRESS"] = address
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass


_discover_session()

# The panic switch: anything that creates this file (a keybind) stops the run at its next check.
STOP_FILE = os.path.join(os.environ["XDG_RUNTIME_DIR"], "jev-stop")


def run(argv, stdin=None, timeout=30):
    proc = subprocess.run(argv, input=stdin, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{argv[0]} {argv[1] if len(argv) > 1 else ''} failed: {proc.stderr.decode(errors='replace').strip()[:300]}"
        )
    return proc.stdout


def hyprctl(*args):
    return json.loads(run(["hyprctl", "-j", *args]))


def hypruse(*args, stdin=None):
    return run([*HYPRUSE, *args], stdin=stdin).decode(errors="replace").strip()


# ------------------------------------------------------------------ monitor and windows

MONITOR = {}  # name, x, y, width, height (logical), scale, workspace


def select_monitor(name):
    monitors = hyprctl("monitors")
    pick = (
        next((m for m in monitors if m["name"] == name), None)
        if name
        else next((m for m in monitors if m.get("focused")), monitors[0])
    )
    if pick is None:
        raise RuntimeError(f"no monitor named {name!r}; have {[m['name'] for m in monitors]}")
    scale = float(pick["scale"])
    w, h = pick["width"] / scale, pick["height"] / scale
    if pick.get("transform", 0) % 2 == 1:
        w, h = h, w  # rotated: the logical box is the other way round
    MONITOR.update(
        name=pick["name"], x=pick["x"], y=pick["y"], width=w, height=h, scale=scale, workspace=pick["activeWorkspace"]["id"]
    )
    return dict(MONITOR)


def to_local(x, y):
    return x - MONITOR["x"], y - MONITOR["y"]


def to_global(x, y):
    return round(MONITOR["x"] + x), round(MONITOR["y"] + y)


def active_window():
    """jev's window: the most recently focused one on the driven monitor's current workspace.

    Not Hyprland's active window, which follows the owner around the other monitors; jev's world is
    its own monitor, so what it reads and where it types stay there while the owner works elsewhere.
    """
    ws = next((m["activeWorkspace"]["id"] for m in hyprctl("monitors") if m["name"] == MONITOR["name"]), None)
    MONITOR["workspace"] = ws
    here = [c for c in hyprctl("clients") if c["workspace"]["id"] == ws and c.get("mapped", True) and not c.get("hidden")]
    if not here:
        return None
    win = min(here, key=lambda c: c.get("focusHistoryID", 1 << 30))
    x, y = to_local(*win["at"])
    return {
        "address": win["address"],
        "class": win.get("class", ""),
        "title": win.get("title", ""),
        "pid": win.get("pid", 0),
        "frame": [x, y, win["size"][0], win["size"][1]],
        "at": win["at"],
    }


def cursor():
    pos = hyprctl("cursorpos")
    return list(to_local(pos["x"], pos["y"]))


# ------------------------------------------------------------------ accessibility tree

try:
    import gi

    gi.require_version("Atspi", "2.0")
    from gi.repository import Atspi

    ATSPI = True
except (ImportError, ValueError):
    ATSPI = False


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _app_for_pid(pid):
    desktop = Atspi.get_desktop(0)
    for i in range(_safe(desktop.get_child_count, 0)):
        app = _safe(lambda i=i: desktop.get_child_at_index(i))
        if app is not None and _safe(app.get_process_id) == pid:
            return app
    return None


def _window_of(app, title):
    frames = [_safe(lambda i=i: app.get_child_at_index(i)) for i in range(_safe(app.get_child_count, 0))]
    frames = [f for f in frames if f is not None]
    for f in frames:
        if _safe(f.get_name, "") == title:
            return f
    for f in frames:
        states = _safe(f.get_state_set)
        if states is not None and states.contains(Atspi.StateType.ACTIVE):
            return f
    return frames[0] if frames else None


def _states(node):
    states = _safe(node.get_state_set)
    if states is None:
        return set()
    return {s.value_nick.replace("-", "_") for s in _safe(states.get_states, [])}


# GTK 4 and newer Chromium call a push button "button"; OSWorld's GNOME tree, which `a11y` maps, says
# "push-button".
ROLE_ALIASES = {"button": "push-button"}


def _role(node):
    role = (_safe(node.get_role_name, "") or "unknown").replace(" ", "-")
    return ROLE_ALIASES.get(role, role)


def _element(node, parent, win_origin, keep_text):
    role = _role(node)
    el = ET.SubElement(parent, role if role.replace("-", "").isalnum() else "unknown")
    name = _safe(node.get_name, "") or ""
    if name:
        el.set("name", name)
    states = _states(node)
    for s in states:
        el.set(f"{{{NS_STATE}}}{s}", "true")
    if "showing" in states and "visible" in states:
        ext = _safe(lambda: node.get_extents(Atspi.CoordType.WINDOW))
        if ext is not None and ext.width > 0 and ext.height > 0 and abs(ext.x) < EXTENT_SANITY and abs(ext.y) < EXTENT_SANITY:
            gx, gy = win_origin[0] + ext.x, win_origin[1] + ext.y
            lx, ly = to_local(gx, gy)
            el.set(f"{{{NS_COMPONENT}}}screencoord", f"({lx:g}, {ly:g})")
            el.set(f"{{{NS_COMPONENT}}}size", f"({ext.width}, {ext.height})")
    # The interface methods through the static Atspi.Action / Atspi.Text functions: the same names
    # on the Accessible itself are deprecated, and Accessible.get_text answers something else.
    if _safe(node.get_action_iface) is not None:
        for i in range(_safe(lambda: Atspi.Action.get_n_actions(node), 0)):
            act = _safe(lambda i=i: Atspi.Action.get_action_name(node, i), "")
            if act and act.replace("_", "").isalnum():
                desc = _safe(lambda i=i: Atspi.Action.get_action_description(node, i), "")
                el.set(f"{{{NS_ACTION}}}{act}_desc", desc or act)
    attrs = _safe(node.get_attributes, {}) or {}
    for key in ("placeholder", "placeholder-text"):
        if attrs.get(key):
            el.set(f"{{{NS_ATTRIBUTES}}}{key}", attrs[key])
    if keep_text and role != "password-text" and _safe(node.get_text_iface) is not None:
        n = _safe(lambda: Atspi.Text.get_character_count(node), 0)
        content = _safe(lambda: Atspi.Text.get_text(node, 0, min(n, TEXT_CHARS)), "") if n else ""
        if content:
            el.text = content
    return el, states


TEXT_TAGS = {"entry", "text", "static", "label", "paragraph", "heading", "link", "list-item", "table-cell"}


def tree():
    """The active window's app, walked breadth-first under a node and time budget."""
    win = active_window()
    root = ET.Element("desktop-frame")
    info = {"capped": False, "nodes": 0}
    if win is None or not ATSPI:
        return ET.tostring(root, encoding="unicode"), info
    app = _app_for_pid(win["pid"])
    if app is None:
        info["no_app"] = True
        return ET.tostring(root, encoding="unicode"), info
    app_el = ET.SubElement(root, "application", name=_safe(app.get_name, "") or win["class"])
    window = _window_of(app, win["title"])
    if window is None:
        return ET.tostring(root, encoding="unicode"), info
    win_el, _ = _element(window, app_el, win["at"], keep_text=False)
    # Hyprland knows which window has the keyboard; AT-SPI on Wayland often does not say.
    win_el.set(f"{{{NS_STATE}}}active", "true")
    deadline = time.monotonic() + TIME_CAP
    queue = [(window, win_el)]
    head = 0
    while head < len(queue):
        node, el = queue[head]
        head += 1
        if info["nodes"] >= NODE_CAP or time.monotonic() > deadline:
            info["capped"] = True
            break
        for i in range(_safe(node.get_child_count, 0)):
            kid = _safe(lambda i=i, node=node: node.get_child_at_index(i))
            if kid is None:
                continue
            info["nodes"] += 1
            role = _role(kid)
            kid_el, states = _element(kid, el, win["at"], keep_text=role in TEXT_TAGS)
            if "showing" in states or "focused" in states:
                queue.append((kid, kid_el))
    return ET.tostring(root, encoding="unicode"), info


# ------------------------------------------------------------------ requests


def op_hello(monitor=None):
    if os.path.exists(STOP_FILE):
        os.remove(STOP_FILE)  # a press from an earlier run does not stop this one
    return {"monitor": select_monitor(monitor), "atspi": ATSPI}


def op_state():
    return {"window": active_window(), "cursor": cursor(), "stop": os.path.exists(STOP_FILE)}


def op_screenshot():
    png = run(["grim", "-o", MONITOR["name"], "-"], timeout=15)
    return {"png": base64.b64encode(png).decode()}


def op_tree():
    xml, info = tree()
    return {"xml": xml, **info}


def op_click(x, y):
    gx, gy = to_global(x, y)
    return {"out": hypruse("pointer", "click", str(gx), str(gy))}


def op_move(x, y):
    gx, gy = to_global(x, y)
    return {"out": hypruse("pointer", "move", str(gx), str(gy))}


def op_scroll(notches):
    return {"out": hypruse("pointer", "scroll", str(notches))}


def _focus_args():
    """Keys go to the keyboard focus, which may have followed the owner elsewhere: take it back first."""
    win = active_window()
    return ["--window", win["address"]] if win else []


def op_key(combo):
    return {"out": hypruse("keyboard", "key", combo, *_focus_args())}


def op_type(text):
    return {"out": hypruse("keyboard", "type", "-", *_focus_args(), stdin=text.encode())}


def op_focus(address):
    return {"out": hypruse("hypr", "focus_window", address)}


def op_bring(address):
    """Move a window onto the driven monitor's workspace when it is elsewhere, then focus it."""
    active_window()  # refreshes MONITOR["workspace"]
    ws = next((c["workspace"]["id"] for c in hyprctl("clients") if c["address"] == address), None)
    if ws is not None and ws != MONITOR["workspace"]:
        hypruse("hypr", "move_window", address, str(MONITOR["workspace"]))
    return {"out": hypruse("hypr", "focus_window", address)}


def op_find_window(klass):
    for c in hyprctl("clients"):
        if c.get("class") == klass and c.get("mapped", True):
            return {"address": c["address"], "workspace": c["workspace"]["id"]}
    return {"address": None}


def op_launch(argv):
    argv = [os.path.expandvars(a) for a in argv]
    return {"out": hypruse("launch", "--workspace", str(MONITOR["workspace"]), "--", *argv)}


def op_spawn(argv):
    argv = [os.path.expandvars(a) for a in argv]
    subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return {}


OPS = {name[3:]: fn for name, fn in globals().items() if name.startswith("op_")}


def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            reply = {"ok": True, **OPS[req.pop("op")](**req)}
        except Exception as exc:
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
