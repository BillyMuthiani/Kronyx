"""Tests for SVG primitives, style, theme, and layout computation."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from kronyx.viz.graph import Edge, Graph, Node
from kronyx.viz.layout.vertical import VerticalLayout, _calculate_node_size
from kronyx.viz.primitives import (
    add_marker,
    create_canvas,
    create_empty_canvas,
    draw_arrow,
    draw_border,
    draw_connection,
    draw_label,
    draw_marker,
    draw_rectangle,
    draw_rounded_rectangle,
    draw_text,
    measure_text,
    serialize,
)
from kronyx.viz.renderers.svg import SvgRenderer
from kronyx.viz.style import DefaultStyle, Style
from kronyx.viz.styling.default import DefaultSceneStyler
from kronyx.viz.themes import DefaultTheme, Theme


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _findall(root: ET.Element, local_name: str) -> list[ET.Element]:
    return [el for el in root.iter() if _local(el.tag) == local_name]


# ---------------------------------------------------------------------------
# Canvas tests
# ---------------------------------------------------------------------------


class TestSvgCanvas:
    def test_create_canvas_attributes(self):
        canvas = create_canvas(400, 300, "#fff")
        assert canvas.tag == "svg"
        assert canvas.get("width") == "400"
        assert canvas.get("height") == "300"
        assert canvas.get("viewBox") == "0 0 400 300"
        assert canvas.get("xmlns") == "http://www.w3.org/2000/svg"

    def test_create_canvas_default_background(self):
        canvas = create_canvas(100, 100)
        assert canvas.get("width") == "100"
        assert canvas.get("height") == "100"

    def test_empty_canvas_dimensions(self):
        canvas = create_empty_canvas()
        root = ET.fromstring(serialize(canvas))
        assert root.get("width") == "200"
        assert root.get("height") == "100"

    def test_serialize_returns_string(self):
        canvas = create_canvas(100, 100)
        result = serialize(canvas)
        assert isinstance(result, str)
        assert result.startswith("<svg")
        assert result.strip().endswith("/>") or result.strip().endswith("</svg>")

    def test_add_marker_creates_defs(self):
        canvas = create_canvas(100, 100)
        add_marker(canvas, "arrow", 12, "#000")
        root = ET.fromstring(serialize(canvas))
        markers = root.findall(".//{http://www.w3.org/2000/svg}marker")
        assert len(markers) == 1
        assert markers[0].get("id") == "arrow"
        assert markers[0].get("markerWidth") == "12"

    def test_add_marker_adds_path(self):
        canvas = create_canvas(100, 100)
        add_marker(canvas, "arrow", 12, "#000")
        root = ET.fromstring(serialize(canvas))
        paths = root.findall(".//{http://www.w3.org/2000/svg}path")
        assert len(paths) == 1
        assert paths[0].get("fill") == "#000"


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


class TestSvgShapes:
    def test_draw_rounded_rectangle_attributes(self):
        canvas = create_canvas(200, 200)
        rect = draw_rounded_rectangle(
            canvas, 10, 20, 100, 50, 10, "#fff", "#000", 2
        )
        assert rect.tag == "rect"
        assert rect.get("x") == "10"
        assert rect.get("y") == "20"
        assert rect.get("width") == "100"
        assert rect.get("height") == "50"
        assert rect.get("rx") == "10"
        assert rect.get("ry") == "10"
        assert rect.get("fill") == "#fff"
        assert rect.get("stroke") == "#000"
        assert rect.get("stroke-width") == "2"

    def test_draw_rectangle_attributes(self):
        canvas = create_canvas(200, 200)
        rect = draw_rectangle(canvas, 0, 0, 80, 40, "#fff", "#000", 1)
        assert rect.get("rx") is None
        assert rect.get("fill") == "#fff"
        assert rect.get("stroke-width") == "1"

    def test_draw_border_no_fill(self):
        canvas = create_canvas(200, 200)
        rect = draw_border(canvas, 5, 5, 100, 100, 8, "#000", 2)
        assert rect.get("fill") == "none"
        assert rect.get("stroke") == "#000"
        assert rect.get("rx") == "8"


# ---------------------------------------------------------------------------
# Text tests
# ---------------------------------------------------------------------------


class TestSvgText:
    def test_measure_text_large_font(self):
        assert measure_text("Hello", 16) == int(5 * 8.5)

    def test_measure_text_medium_font(self):
        assert measure_text("Hello", 13) == int(5 * 7.0)

    def test_measure_text_small_font(self):
        assert measure_text("Hello", 12) == int(5 * 6.5)

    def test_draw_text_attributes(self):
        canvas = create_canvas(200, 200)
        text_el = draw_text(
            canvas, 10, 20, "Hello", "Arial", 14, "bold", "#000"
        )
        assert text_el.tag == "text"
        assert text_el.get("x") == "10"
        assert text_el.get("y") == "20"
        assert text_el.get("font-family") == "Arial"
        assert text_el.get("font-size") == "14"
        assert text_el.get("font-weight") == "bold"
        assert text_el.get("fill") == "#000"
        assert text_el.text == "Hello"

    def test_draw_label_vertical_centering(self):
        canvas = create_canvas(200, 200)
        text_el = draw_label(
            canvas, 10, 20, "Hello", "Arial", 14, "normal", "#000", line_height=18
        )
        assert text_el.get("y") == "29"
        assert text_el.text == "Hello"


# ---------------------------------------------------------------------------
# Edge tests
# ---------------------------------------------------------------------------


class TestSvgEdges:
    def test_draw_marker_attributes(self):
        canvas = create_canvas(100, 100)
        marker = draw_marker(canvas, "arrow", 12, "#000")
        assert marker.get("id") == "arrow"
        assert marker.get("markerWidth") == "12"
        assert marker.get("markerHeight") == "6"
        paths = canvas.findall(".//path")
        assert len(paths) == 1

    def test_draw_connection_attributes(self):
        canvas = create_canvas(100, 100)
        line = draw_connection(canvas, 0, 0, 100, 100, "arrow", "#000", 2)
        assert line.tag == "line"
        assert line.get("x1") == "0"
        assert line.get("y1") == "0"
        assert line.get("x2") == "100"
        assert line.get("y2") == "100"
        assert line.get("stroke") == "#000"
        assert line.get("stroke-width") == "2"
        assert line.get("marker-end") == "url(#arrow)"

    def test_draw_arrow_defaults(self):
        canvas = create_canvas(100, 100)
        line = draw_arrow(canvas, 0, 0, 50, 50)
        assert line.get("stroke") == "#000"
        assert line.get("stroke-width") == "2"
        assert line.get("marker-end") == "url(#arrow)"


# ---------------------------------------------------------------------------
# Style tests
# ---------------------------------------------------------------------------


class TestStyle:
    def test_default_style_values(self):
        assert DefaultStyle.node_padding == 12
        assert DefaultStyle.border_radius == 10
        assert DefaultStyle.border_width == 2
        assert DefaultStyle.title_font_size == 16
        assert DefaultStyle.subtitle_font_size == 13
        assert DefaultStyle.body_font_size == 12
        assert DefaultStyle.arrow_size == 12
        assert DefaultStyle.vertical_spacing == 80
        assert DefaultStyle.horizontal_spacing == 0
        assert DefaultStyle.canvas_margin == 40

    def test_custom_style(self):
        style = Style(node_padding=20, border_radius=5, vertical_spacing=100)
        assert style.node_padding == 20
        assert style.border_radius == 5
        assert style.vertical_spacing == 100
        assert style.title_font_size == 16

    def test_style_is_dataclass(self):
        style = Style()
        assert hasattr(style, "node_padding")
        assert hasattr(style, "canvas_margin")

    def test_default_style_singleton(self):
        assert DefaultStyle == Style()


# ---------------------------------------------------------------------------
# Theme tests
# ---------------------------------------------------------------------------


class TestTheme:
    def test_default_theme_values(self):
        assert DefaultTheme.background == "#ffffff"
        assert DefaultTheme.node_fill == "#ffffff"
        assert DefaultTheme.node_border == "#000000"
        assert DefaultTheme.title_color == "#000000"
        assert DefaultTheme.body_color == "#000000"
        assert DefaultTheme.edge_color == "#000000"

    def test_custom_theme(self):
        theme = Theme(node_fill="#f0f0f0", edge_color="#333")
        assert theme.node_fill == "#f0f0f0"
        assert theme.edge_color == "#333"
        assert theme.background == "#ffffff"

    def test_theme_is_dataclass(self):
        theme = Theme()
        assert hasattr(theme, "background")
        assert hasattr(theme, "node_fill")
        assert hasattr(theme, "node_border")


# ---------------------------------------------------------------------------
# Layout computation tests
# ---------------------------------------------------------------------------


class TestComputePositions:
    def _build_simple_graph(self):
        input_node = Node(
            id="input",
            name="Input",
            layer_type="Input",
            params=0,
            output_shape="?",
            tags={"input"},
        )
        dense_node = Node(
            id="dense",
            name="Dense",
            layer_type="Dense",
            params=48,
            output_shape="(None, 16)",
            metadata={"units": 16},
            tags={"trainable", "dense"},
        )
        output_node = Node(
            id="output",
            name="Prediction",
            layer_type="Output",
            params=0,
            output_shape="?",
            tags={"output"},
        )
        edges = [
            Edge(source="input", target="dense"),
            Edge(source="dense", target="output"),
        ]
        return Graph(nodes=[input_node, dense_node, output_node], edges=edges)

    def test_returns_positions_and_dimensions(self):
        graph = self._build_simple_graph()
        layout = VerticalLayout()
        scene = layout.layout(graph)
        positions = [(pnode.x, pnode.y) for pnode in scene.nodes]
        width = scene.canvas_width
        height = scene.canvas_height
        assert len(positions) == 3
        assert width > 0
        assert height > 0

    def test_positions_are_centered(self):
        graph = self._build_simple_graph()
        layout = VerticalLayout()
        scene = layout.layout(graph)
        max_w = max(_calculate_node_size(n, DefaultStyle)[0] for n in graph.nodes)
        expected_center = DefaultStyle.canvas_margin + max_w // 2
        for pnode in scene.nodes:
            assert pnode.x + pnode.width // 2 == expected_center

    def test_canvas_height_accounts_for_all_nodes(self):
        graph = self._build_simple_graph()
        layout = VerticalLayout()
        scene = layout.layout(graph)
        assert scene.canvas_height >= scene.nodes[-1].y

    def test_custom_style_changes_spacing(self):
        graph = self._build_simple_graph()
        style = Style(vertical_spacing=20, canvas_margin=10)
        layout = VerticalLayout(style=style)
        scene = layout.layout(graph)
        assert scene.canvas_width > 0
        assert scene.canvas_height > 0

    def test_single_node(self):
        only_node = Node(
            id="only",
            name="Only",
            layer_type="Dense",
            params=10,
            output_shape="(None, 4)",
            metadata={"units": 4},
        )
        graph = Graph(nodes=[only_node], edges=[])
        layout = VerticalLayout()
        scene = layout.layout(graph)
        assert len(scene.nodes) == 1
        assert scene.canvas_width > 0
        assert scene.canvas_height > 0

    def test_empty_graph(self):
        graph = Graph(nodes=[], edges=[])
        layout = VerticalLayout()
        scene = layout.layout(graph)
        assert scene.nodes == []
        assert scene.canvas_width == 0
        assert scene.canvas_height == 0


# ---------------------------------------------------------------------------
# Renderer output equivalence tests
# ---------------------------------------------------------------------------


class TestRendererOutputEquivalence:
    def _build_simple_graph(self):
        input_node = Node(
            id="input",
            name="Input",
            layer_type="Input",
            params=0,
            output_shape="?",
            tags={"input"},
        )
        dense_node = Node(
            id="dense",
            name="Dense",
            layer_type="Dense",
            params=48,
            output_shape="(None, 16)",
            metadata={"units": 16},
            tags={"trainable", "dense"},
        )
        output_node = Node(
            id="output",
            name="Prediction",
            layer_type="Output",
            params=0,
            output_shape="?",
            tags={"output"},
        )
        edges = [
            Edge(source="input", target="dense"),
            Edge(source="dense", target="output"),
        ]
        return Graph(nodes=[input_node, dense_node, output_node], edges=edges)

    def test_renderer_accepts_default_style_and_theme(self):
        VerticalLayout().layout(self._build_simple_graph())
        renderer = SvgRenderer()
        assert renderer.style is None
        assert renderer.theme is None

    def test_renderer_accepts_custom_style(self):
        style = Style(node_padding=20, vertical_spacing=100)
        VerticalLayout(style=style).layout(self._build_simple_graph())
        renderer = SvgRenderer(style=style)
        assert renderer.style == style
        assert renderer.theme is None

    def test_renderer_accepts_custom_theme(self):
        theme = Theme(node_fill="#f0f0f0", edge_color="#333")
        VerticalLayout().layout(self._build_simple_graph())
        renderer = SvgRenderer(theme=theme)
        assert renderer.theme == theme
        assert renderer.style is None

    def _build_styled_scene(self, graph=None):
        target_graph = graph or self._build_simple_graph()
        layout = VerticalLayout()
        scene = layout.layout(target_graph)
        styler = DefaultSceneStyler()
        return styler.apply(scene)

    def test_renderer_output_contains_svg_root(self):
        scene = self._build_styled_scene()
        result = SvgRenderer().render(scene)
        assert result.startswith("<svg")
        assert result.endswith("</svg>")

    def test_renderer_output_is_valid_xml(self):
        scene = self._build_styled_scene()
        result = SvgRenderer().render(scene)
        ET.fromstring(result)

    def test_renderer_output_contains_nodes(self):
        scene = self._build_styled_scene()
        result = SvgRenderer().render(scene)
        root = ET.fromstring(result)
        rects = _findall(root, "rect")
        assert len(rects) >= 3

    def test_renderer_output_contains_edges(self):
        scene = self._build_styled_scene()
        result = SvgRenderer().render(scene)
        root = ET.fromstring(result)
        lines = _findall(root, "line")
        assert len(lines) == 2

    def test_renderer_output_contains_labels(self):
        scene = self._build_styled_scene()
        result = SvgRenderer().render(scene)
        root = ET.fromstring(result)
        text_elements = _findall(root, "text")
        texts = [el.text for el in text_elements if el.text]
        assert any("Dense" in t for t in texts)
        assert any("Input" in t for t in texts)
        assert any("Prediction" in t for t in texts)

