"""Tests for the styling layer."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from kronyx.viz.engine import VisualizationEngine
from kronyx.viz.graph import Edge, Graph, Node, Scene
from kronyx.viz.layout.vertical import VerticalLayout
from kronyx.viz.renderers.svg import SvgRenderer
from kronyx.viz.style import Style
from kronyx.viz.styling.default import DefaultSceneStyler
from kronyx.viz.styling.scene import StyledEdge, StyledNode, StyledScene
from kronyx.viz.styling.styler import SceneStyler
from kronyx.viz.themes import Theme


class TestStyledNode:
    def test_defaults(self):
        node = StyledNode(
            node_id="n1",
            x=10.0,
            y=20.0,
            width=100.0,
            height=80.0,
            fill_color="#ffffff",
            stroke_color="#000000",
            stroke_width=2,
            border_radius=10,
            title="A",
            subtitle="Dense Layer",
            metadata_lines=["Params: 10"],
            font_family="Arial",
            title_font_size=16,
            subtitle_font_size=13,
            body_font_size=12,
            text_color="#000000",
            padding=12,
        )
        assert node.node_id == "n1"
        assert node.x == 10.0
        assert node.shadow is False
        assert node.icon is None
        assert node.badges == []
        assert node.tags == set()

    def test_optional_fields(self):
        node = StyledNode(
            node_id="n1",
            x=0.0,
            y=0.0,
            width=10.0,
            height=10.0,
            fill_color="#fff",
            stroke_color="#000",
            stroke_width=1,
            border_radius=0,
            title="T",
            subtitle="S",
            metadata_lines=[],
            font_family="Arial",
            title_font_size=14,
            subtitle_font_size=12,
            body_font_size=11,
            text_color="#000",
            padding=8,
            shadow=True,
            icon="dense",
            badges=["trainable"],
            tags={"a"},
        )
        assert node.shadow is True
        assert node.icon == "dense"
        assert node.badges == ["trainable"]
        assert node.tags == {"a"}


class TestStyledEdge:
    def test_defaults(self):
        edge = StyledEdge(
            source_id="a",
            target_id="b",
            x1=0.0,
            y1=0.0,
            x2=10.0,
            y2=10.0,
            stroke_color="#000000",
            stroke_width=2,
            arrow_size=12,
            arrow_color="#000000",
        )
        assert edge.source_id == "a"
        assert edge.style == "solid"

    def test_custom_style(self):
        edge = StyledEdge(
            source_id="a",
            target_id="b",
            x1=0.0,
            y1=0.0,
            x2=10.0,
            y2=10.0,
            stroke_color="#333",
            stroke_width=1,
            arrow_size=8,
            arrow_color="#333",
            style="dashed",
        )
        assert edge.style == "dashed"
        assert edge.arrow_size == 8


class TestStyledScene:
    def test_empty_scene(self):
        scene = StyledScene(graph=Graph(nodes=[], edges=[]))
        assert scene.nodes == []
        assert scene.edges == []
        assert scene.background == "#ffffff"
        assert scene.canvas_width == 0.0
        assert scene.canvas_height == 0.0

    def test_populated_scene(self):
        graph = Graph(
            nodes=[Node(id="n1", name="A", layer_type="Dense", params=1)],
            edges=[Edge(source="n1", target="n2")],
        )
        node = StyledNode(
            node_id="n1",
            x=0.0,
            y=0.0,
            width=10.0,
            height=10.0,
            fill_color="#fff",
            stroke_color="#000",
            stroke_width=1,
            border_radius=0,
            title="A",
            subtitle="Dense Layer",
            metadata_lines=[],
            font_family="Arial",
            title_font_size=14,
            subtitle_font_size=12,
            body_font_size=11,
            text_color="#000",
            padding=8,
        )
        edge = StyledEdge(
            source_id="n1",
            target_id="n2",
            x1=0.0,
            y1=0.0,
            x2=10.0,
            y2=10.0,
            stroke_color="#000",
            stroke_width=2,
            arrow_size=12,
            arrow_color="#000",
        )
        scene = StyledScene(
            graph=graph,
            nodes=[node],
            edges=[edge],
            canvas_width=100.0,
            canvas_height=200.0,
            background="#fff",
        )
        assert len(scene.nodes) == 1
        assert len(scene.edges) == 1
        assert scene.canvas_width == 100.0
        assert scene.background == "#fff"


class TestSceneStylerAbstract:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            SceneStyler()  # type: ignore[abstract]


class TestDefaultSceneStyler:
    def _build_scene(self) -> Scene:
        nodes = [
            Node(
                id="input",
                name="Input",
                layer_type="Input",
                params=0,
                output_shape="?",
                tags={"input"},
            ),
            Node(
                id="dense",
                name="Dense",
                layer_type="Dense",
                params=48,
                output_shape="(None, 16)",
                metadata={"units": 16},
                tags={"trainable", "dense"},
            ),
            Node(
                id="output",
                name="Prediction",
                layer_type="Output",
                params=0,
                output_shape="?",
                tags={"output"},
            ),
        ]
        edges = [
            Edge(source="input", target="dense"),
            Edge(source="dense", target="output"),
        ]
        graph = Graph(nodes=nodes, edges=edges)
        return VerticalLayout().layout(graph)

    def test_default_styling(self):
        scene = self._build_scene()
        styler = DefaultSceneStyler()
        styled = styler.apply(scene)
        assert len(styled.nodes) == 3
        assert len(styled.edges) == 2
        assert styled.background == "#ffffff"

    def test_custom_style(self):
        scene = self._build_scene()
        style = Style(node_padding=20, subtitle_font_size=14)
        styler = DefaultSceneStyler(style=style)
        styled = styler.apply(scene)
        assert styled.nodes[0].padding == 20
        assert styled.nodes[0].subtitle_font_size == 14

    def test_custom_theme(self):
        scene = self._build_scene()
        theme = Theme(
            node_fill="#f0f0f0",
            edge_color="#333",
            canvas_background="#eee",
            arrow_color="#333",
        )
        styler = DefaultSceneStyler(theme=theme)
        styled = styler.apply(scene)
        assert styled.nodes[0].fill_color == "#f0f0f0"
        assert styled.edges[0].stroke_color == "#333"
        assert styled.edges[0].arrow_color == "#333"
        assert styled.background == "#eee"

    def test_node_presentation_fields(self):
        scene = self._build_scene()
        styler = DefaultSceneStyler()
        styled = styler.apply(scene)
        node = styled.nodes[1]
        assert node.node_id == "dense"
        assert node.title == "Dense"
        assert node.subtitle == "Dense Layer"
        assert "Params: 48" in node.metadata_lines
        assert "Units: 16" in node.metadata_lines
        assert node.font_family == "Arial"
        assert node.text_color == "#000000"
        assert node.stroke_width == 2
        assert node.border_radius == 10

    def test_tags_propagated(self):
        scene = self._build_scene()
        styler = DefaultSceneStyler()
        styled = styler.apply(scene)
        assert "trainable" in styled.nodes[1].tags
        assert "input" in styled.nodes[0].tags

    def test_empty_scene(self):
        graph = Graph(nodes=[], edges=[])
        scene = VerticalLayout().layout(graph)
        styler = DefaultSceneStyler()
        styled = styler.apply(scene)
        assert styled.nodes == []
        assert styled.edges == []
        assert styled.graph is scene.graph

    def test_renderer_independence(self):
        scene = self._build_scene()
        styler = DefaultSceneStyler()
        styled = styler.apply(scene)
        renderer = SvgRenderer()
        result = renderer.render(styled)
        assert result.startswith("<svg")
        assert result.endswith("</svg>")
        root = ET.fromstring(result)
        rects = [el for el in root.iter() if el.tag.split("}", 1)[-1] == "rect"]
        assert len(rects) >= 3


class TestEngineStylingIntegration:
    def test_full_pipeline_with_default_styler(self):
        nodes = [
            Node(id="input", name="Input", layer_type="Input", params=0, output_shape="?"),
            Node(
                id="dense",
                name="Dense",
                layer_type="Dense",
                params=48,
                output_shape="(None, 16)",
                metadata={"units": 16},
            ),
            Node(id="output", name="Prediction", layer_type="Output", params=0, output_shape="?"),
        ]
        edges = [
            Edge(source="input", target="dense"),
            Edge(source="dense", target="output"),
        ]
        graph = Graph(nodes=nodes, edges=edges)
        engine = VisualizationEngine()
        result = engine.visualize(graph, renderer_name="svg")
        assert result.startswith("<svg")
        assert result.endswith("</svg>")

    def test_renderer_output_equivalence(self):
        graph = Graph(
            nodes=[
                Node(
                    id="input",
                    name="Input",
                    layer_type="Input",
                    params=0,
                    output_shape="?",
                ),
                Node(
                    id="dense",
                    name="Dense",
                    layer_type="Dense",
                    params=48,
                    output_shape="(None, 16)",
                    metadata={"units": 16},
                ),
                Node(
                    id="output",
                    name="Prediction",
                    layer_type="Output",
                    params=0,
                    output_shape="?",
                ),
            ],
            edges=[
                Edge(source="input", target="dense"),
                Edge(source="dense", target="output"),
            ],
        )
        layout = VerticalLayout()
        scene = layout.layout(graph)
        styler = DefaultSceneStyler()
        styled_scene = styler.apply(scene)
        renderer = SvgRenderer()
        direct_result = renderer.render(styled_scene)

        engine = VisualizationEngine()
        engine_result = engine.visualize(graph, renderer_name="svg")

        assert direct_result.startswith("<svg")
        assert engine_result.startswith("<svg")
        assert direct_result.endswith("</svg>")
        assert engine_result.endswith("</svg>")
        direct_root = ET.fromstring(direct_result)
        engine_root = ET.fromstring(engine_result)
        assert len([el for el in direct_root.iter() if el.tag.split("}", 1)[-1] == "rect"]) > 0
        assert len([el for el in engine_root.iter() if el.tag.split("}", 1)[-1] == "rect"]) > 0

    def test_backward_compatibility(self):
        graph = Graph(
            nodes=[
                Node(id="input", name="Input", layer_type="Input", params=0, output_shape="?"),
                Node(
                    id="dense",
                    name="Dense",
                    layer_type="Dense",
                    params=48,
                    output_shape="(None, 16)",
                    metadata={"units": 16},
                ),
                Node(
                    id="output",
                    name="Prediction",
                    layer_type="Output",
                    params=0,
                    output_shape="?",
                ),
            ],
            edges=[
                Edge(source="input", target="dense"),
                Edge(source="dense", target="output"),
            ],
        )
        engine = VisualizationEngine()
        result = engine.visualize(graph, renderer_name="svg")
        assert isinstance(result, str)
        assert len(result) > 0
        assert result.startswith("<svg")
        assert result.endswith("</svg>")

    def test_custom_styler_in_engine(self):
        graph = Graph(
            nodes=[
                Node(id="input", name="Input", layer_type="Input", params=0, output_shape="?"),
                Node(
                    id="dense",
                    name="Dense",
                    layer_type="Dense",
                    params=48,
                    output_shape="(None, 16)",
                    metadata={"units": 16},
                ),
            ],
            edges=[Edge(source="input", target="dense")],
        )
        style = Style(node_padding=20)
        theme = Theme(node_fill="#f0f0f0")
        styler = DefaultSceneStyler(style=style, theme=theme)
        engine = VisualizationEngine(styler=styler)
        result = engine.visualize(graph, renderer_name="svg")
        assert result.startswith("<svg")
        assert "#f0f0f0" in result


class TestRendererIndependenceFromStyling:
    def test_renderer_draws_styled_scene(self):
        graph = Graph(
            nodes=[
                Node(
                    id="n1",
                    name="A",
                    layer_type="Dense",
                    params=1,
                    output_shape="(None, 2)",
                    metadata={"units": 2},
                ),
            ],
            edges=[],
        )
        scene = VerticalLayout().layout(graph)
        styler = DefaultSceneStyler()
        styled = styler.apply(scene)
        renderer = SvgRenderer()
        result = renderer.render(styled)
        assert "Arial" in result
        assert "#ffffff" in result
        assert "#000000" in result

    def test_different_stylers_produce_different_output(self):
        graph = Graph(
            nodes=[
                Node(
                    id="n1",
                    name="A",
                    layer_type="Dense",
                    params=1,
                    output_shape="(None, 2)",
                    metadata={"units": 2},
                ),
            ],
            edges=[],
        )
        scene = VerticalLayout().layout(graph)

        styler1 = DefaultSceneStyler(theme=Theme(node_fill="#fff", body_color="#111"))
        styled1 = styler1.apply(scene)
        result1 = SvgRenderer().render(styled1)

        styler2 = DefaultSceneStyler(theme=Theme(node_fill="#000", body_color="#222"))
        styled2 = styler2.apply(scene)
        result2 = SvgRenderer().render(styled2)

        assert result1 != result2
        assert 'fill="#fff"' in result1
        assert 'fill="#000"' in result2
