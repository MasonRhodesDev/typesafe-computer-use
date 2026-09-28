"""`clicker-hypr`: `clicker` and `clicker-inspect`, driving a Hyprland machine over ssh.

    clicker-hypr --host my-desktop "open the Playground"            # dry run, one step
    clicker-hypr --host my-desktop --monitor DP-2 "log in" --act
    clicker-hypr --host my-desktop --inspect "any goal"

Every flag after the host options is `clicker`'s own (or `clicker-inspect`'s, with --inspect).
"""

from __future__ import annotations

import argparse
import os
import sys

from .. import cli
from ..osworld import ocr
from ..platform_adapter import using
from .desktop import HyprlandDesktop


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="clicker-hypr", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default=os.environ.get("JEV_HYPR_HOST"), help="ssh destination of the Hyprland machine")
    parser.add_argument("--monitor", default=os.environ.get("JEV_HYPR_MONITOR"), help="the monitor jev sees (default: focused)")
    parser.add_argument("--inspect", action="store_true", help="run clicker-inspect instead of clicker")
    args, rest = parser.parse_known_args(argv)
    if not args.host:
        sys.exit("clicker-hypr: --host (or JEV_HYPR_HOST) is required")
    try:
        recognize = ocr.backend("rapidocr")
    except (ValueError, ocr.Unavailable) as e:
        sys.exit(str(e))
    os.environ.setdefault("CLICKER_BROWSER", "Google Chrome")
    remote = HyprlandDesktop(args.host, args.monitor, recognize)
    m = remote.monitor
    print(
        f"driving {args.host} monitor {m['name']} ({m['width']:g}x{m['height']:g} at scale {m['scale']:g}); atspi={remote.atspi}"
    )
    try:
        with using(remote):
            if args.inspect:
                cli.inspect([*rest, "--no-open"] if "--no-open" not in rest else rest)
            else:
                cli.main(rest)
    finally:
        remote.close()


if __name__ == "__main__":
    main()
