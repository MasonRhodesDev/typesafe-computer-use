import xml.etree.ElementTree as ET

from PIL import Image

from typesafe_computer_use import perception
from typesafe_computer_use.hyprland.tree_text import tree_lines
from typesafe_computer_use.models import Screen

ST = "https://accessibility.ubuntu.example.org/ns/state"
CP = "https://accessibility.ubuntu.example.org/ns/component"


def el(tag, name="", frame=None, text=None, kids=()):
    e = ET.Element(tag)
    if name:
        e.set("name", name)
    if frame is not None:
        x, y, w, h = frame
        e.set(f"{{{CP}}}screencoord", f"({x}, {y})")
        e.set(f"{{{CP}}}size", f"({w}, {h})")
    if text is not None:
        e.text = text
    e.extend(kids)
    return e


def texts(app, scale=1.0):
    return [t for t, _, _ in tree_lines(app, scale, 1000, 1000)]


def test_the_deepest_text_speaks_once_and_boxes_scale_to_capture_pixels():
    app = el("application", kids=[
        el("paragraph", text="Price $83.99 USD", frame=(10, 10, 300, 40), kids=[
            el("static", text="Price", frame=(10, 10, 50, 20)),
            el("static", text="$83.99 USD", frame=(70, 10, 100, 20)),
        ]),
    ])  # fmt: skip
    lines = tree_lines(app, 2.0, 1000, 1000)
    assert [t for t, _, _ in lines] == ["Price", "$83.99 USD"]
    assert lines[1][2] == (140.0, 20.0, 340.0, 60.0)


def test_controls_are_not_text_lines_but_the_words_inside_a_link_are():
    app = el("application", kids=[
        el("push-button", name="Reload", frame=(0, 0, 30, 30)),
        el("link", name="Moes", frame=(100, 100, 80, 20), kids=[el("static", text="Moes", frame=(100, 100, 80, 20))]),
    ])  # fmt: skip
    assert texts(app) == ["Moes"]


def test_text_is_clipped_to_the_web_page_viewport():
    app = el("application", kids=[
        el("document-web", frame=(0, 136, 1000, 800), kids=[el("heading", text="Title", frame=(10, 100, 300, 60))]),
    ])  # fmt: skip
    ((_, _, box),) = tree_lines(app, 1.0, 1000, 1000)
    assert box == (10.0, 136.0, 310.0, 160.0)


def test_text_under_a_sticky_control_is_dropped_but_not_under_its_own_ancestor():
    app = el("application", kids=[
        el("link", name="Products", frame=(0, 200, 400, 30)),
        el("static", text="SKU: hidden", frame=(10, 205, 100, 20)),
        el("link", name="Card", frame=(0, 400, 400, 100), kids=[el("static", text="In the card", frame=(10, 410, 100, 20))]),
    ])  # fmt: skip
    assert texts(app) == ["In the card"]


def test_hidden_screen_reader_text_and_field_contents_are_never_read():
    app = el("application", kids=[
        el("static", text="sr-only pixel", frame=(0, 0, 1, 1)),
        el("static", text="Variant sold out", frame=(800, 500, 10, 490)),
        el("entry", name="Search", frame=(0, 50, 300, 30), kids=[el("static", text="typed secret", frame=(0, 50, 100, 30))]),
        el("password-text", frame=(0, 90, 300, 30), text="hunter2"),
        el("static", text="Visible", frame=(0, 150, 100, 20)),
    ])  # fmt: skip
    assert texts(app) == ["Visible"]


def test_no_app_reads_nothing():
    assert tree_lines(None, 1.0, 100, 100) == []


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
