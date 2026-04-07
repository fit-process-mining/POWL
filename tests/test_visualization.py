from pathlib import Path

import pytest

import powl
from powl.objects.tagged_powl.activity import Activity
from powl.objects.tagged_powl.builders import sequence
from powl.visualization.powl import visualizer as powl_visualizer


def test_powl_visualizer_apply_returns_svg(running_example_model):
    svg = powl_visualizer.apply(running_example_model)

    assert "<svg" in svg
    assert "register request" in svg


def test_save_visualization_creates_svg_file(running_example_model, tmp_path):
    output_file = tmp_path / "model.svg"

    powl.save_visualization(running_example_model, output_file)

    assert output_file.exists()
    assert "<svg" in output_file.read_text(encoding="utf-8")


def test_save_visualization_net_creates_svg_file(running_example_model, tmp_path):
    output_file = tmp_path / "model-net.svg"

    powl.save_visualization_net(running_example_model, output_file)

    assert output_file.exists()
    assert "<svg" in output_file.read_text(encoding="utf-8")


def test_visualizer_save_rejects_non_svg_output():
    with pytest.raises(Exception, match="Unsupported format"):
        powl_visualizer.save("<svg />", "diagram.png")


def test_inline_images_and_svgs_embeds_child_svg(tmp_path):
    child_svg = tmp_path / "child.svg"
    child_svg.write_text(
        '<svg viewBox="0 0 10 10"><rect width="10" height="10" /></svg>',
        encoding="utf-8",
    )

    parent_svg = (
        f'<svg xmlns:xlink="http://www.w3.org/1999/xlink">'
        f'<image x="1" y="2" width="20" height="30" xlink:href="{child_svg}" />'
        f"</svg>"
    )

    inlined = powl_visualizer.inline_images_and_svgs(parent_svg)

    assert "xlink:href" not in inlined
    assert '<g transform="translate(1.0,2.0) scale(2.0,3.0)">' in inlined


def test_visualizer_view_uses_browser(monkeypatch):
    opened = {}

    def fake_open(url):
        opened["url"] = url
        return True

    monkeypatch.setattr(powl_visualizer.webbrowser, "open", fake_open)

    assert powl_visualizer.view("<svg />") is True
    assert opened["url"].startswith("file://")

    svg_path = Path(opened["url"][7:])
    assert svg_path.exists()
    svg_path.unlink()


def test_public_view_wrappers_delegate_to_visualizer(monkeypatch):
    model = sequence([Activity("A"), Activity("B")])
    captured = {"variants": []}

    def fake_apply(powl_model, variant=None, frequency_tags=True):
        captured["variants"].append(variant)
        assert powl_model is model
        assert frequency_tags is True
        return "<svg />"

    def fake_view(svg_content):
        captured["svg"] = svg_content
        return True

    monkeypatch.setattr(powl_visualizer, "apply", fake_apply)
    monkeypatch.setattr(powl_visualizer, "view", fake_view)

    powl.view(model)
    powl.view_net(model)

    assert captured["svg"] == "<svg />"
    assert len(captured["variants"]) == 2
