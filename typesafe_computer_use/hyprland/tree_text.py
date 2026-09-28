"""The screen's text read from the accessibility tree instead of from pixels.

A browser page with its tree on (and most GTK and Qt apps) already publishes every visible string
with its exact frame: headings, paragraphs, prices, link and button labels. Reading those takes no
time and misreads nothing, where RapidOCR on this machine costs seconds a capture. So when the tree
carries enough text, its strings stand in for OCR lines; when it does not (a terminal, a canvas,
an app with a thin tree), the adapter declines and jev reads pixels as before.

Only the deepest text is taken: a paragraph whose static-text children already said everything is
not said again as one giant block. Field contents are never read (an entry's value is what the
user typed, and a password field is skipped outright), as the browser backend never reads them.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from ..osworld import a11y

Line = tuple[str, float, tuple[float, float, float, float]]

# Roles that are text, not controls. A control (button, tab, link, checkbox) already reaches jev as
# an accessibility item with its label; offering its label as a text line too lets jev's block
# merging, tuned for OCR paragraphs, glue stacked controls into one block aimed between them. A
# link's visible words still arrive: Chromium keeps them in static-text children.
TEXT_ROLES = {"static", "label", "text", "heading", "paragraph", "caption", "list-item", "table-cell"}
NEVER = {"entry", "password-text"}  # field contents stay unread
MAX_HEIGHT_SHARE = 0.4  # a "line" taller than this share of the display is a container, not text
MIN_SIDE = 4.0  # thinner than this is visually hidden screen-reader text (sites clip it to 1px), never drawn
# Visually hidden text that is not clipped to a pixel is laid out in a sliver instead, one or two
# letters a row: a column far taller than wide. Real horizontal text is never shaped like that.
COLUMN_MIN_HEIGHT = 60.0
COLUMN_RATIO = 2.0
MIN_LINES = 8  # fewer strings than this and the tree is too thin to stand in for OCR


def _text(el: ET.Element) -> str:
    return a11y.name(el) or " ".join(a11y.text(el).split())


# Controls that sit on top of text they overlap: a sticky header's links cover the page scrolling
# under it, and AT-SPI still reports the covered text as showing.
COVER_ROLES = {"push-button", "toggle-button", "link", "entry", "combo-box", "page-tab", "menu-item", "check-box", "radio-button"}
VIEWPORT_ROLES = {"document-web", "document-frame"}  # a web page: its text is only visible inside its frame
MIN_COVER = 0.5  # share of a text box a control must cover to hide it


Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in points


def _box(frame) -> Box:
    x, y, w, h = frame
    return (x, y, x + w, y + h)


def _clip(a: Box, b: Box) -> Box | None:
    box = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    return box if box[2] > box[0] and box[3] > box[1] else None


def _area(b: Box) -> float:
    return (b[2] - b[0]) * (b[3] - b[1])


def tree_lines(app: ET.Element | None, scale: float, display_w: float, display_h: float) -> list[Line]:
    """Visible leaf strings under `app`, as OCR lines in capture pixels (points times `scale`).

    Each string is clipped to the display and to the web page's viewport, since Chromium reports a
    heading scrolled half out of view with the frame it would have unclipped, reaching up under
    the toolbar. A string mostly covered by a control that is not its own ancestor is dropped: the
    control is what is drawn there, and what a click there would hit.
    """
    if app is None:
        return []
    display: Box = (0.0, 0.0, display_w, display_h)
    covers: list[tuple[Box, ET.Element]] = []
    for el in app.iter():
        frame = a11y.frame(el)
        if el.tag in COVER_ROLES and frame is not None and frame[2] > 0 and frame[3] > 0:
            covers.append((_box(frame), el))
    found: list[tuple[str, Box, frozenset]] = []

    def visit(el: ET.Element, view: Box, ancestors: frozenset) -> bool:
        """Emit the deepest text; True when this subtree emitted anything."""
        if el.tag in NEVER:
            return False
        frame = a11y.frame(el)
        if el.tag in VIEWPORT_ROLES and frame is not None:
            view = _clip(view, _box(frame)) or view
        inner = ancestors | {id(el)}
        below = [visit(kid, view, inner) for kid in el]
        if any(below):
            return True
        if el.tag not in TEXT_ROLES or frame is None:
            return False
        text = _text(el)
        box = _clip(_box(frame), view)
        if not text or box is None or min(box[2] - box[0], box[3] - box[1]) < MIN_SIDE:
            return False
        w, h = box[2] - box[0], box[3] - box[1]
        if h > MAX_HEIGHT_SHARE * display_h or (h > COLUMN_MIN_HEIGHT and h > COLUMN_RATIO * w):
            return False
        found.append((text, box, ancestors))
        return True

    visit(app, display, frozenset())
    lines: list[Line] = []
    for text, box, ancestors in found:
        covered = any(
            id(el) not in ancestors and (hit := _clip(box, cover)) is not None and _area(hit) >= MIN_COVER * _area(box)
            for cover, el in covers
        )
        if not covered:
            lines.append((text, 1.0, (box[0] * scale, box[1] * scale, box[2] * scale, box[3] * scale)))
    return lines
