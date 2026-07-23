"""Tests for the visualization renderers."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from kronyx.viz.graph import Edge, Graph, Node
from kronyx.viz.layout.vertical import VerticalLayout
from kronyx.viz.renderers.base import Renderer
from kronyx.viz.renderers.svg import SvgRenderer
from kronyx.viz.styling.default import DefaultSceneStyler
from kronyx.viz.styling.scene import StyledScene

_SVG_NS = "http://www.w3.org/2000/svg"


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _findall(root: ET.Element, local_name: str) -> list[ET.Element]:
    return [el for el in root.iter() if _local(el.tag) == local_name]


class TestRendererBase:
    def test_renderer_is_abstract(self):
        with pytest.raises(TypeError):
            Renderer()  # type: ignore[abstract]


class TestSvgRendererRegistration:
    def test_renderer_registers_in_registry(self):
        from kronyx.viz.registry import RendererRegistry

        registry = RendererRegistry()
        registry.register("svg", SvgRenderer)
        assert registry.get("svg") is SvgRenderer

    def test_renderer_name_is_svg(self):
        from kronyx.viz.registry import RendererRegistry

        registry = RendererRegistry()
        registry.register("svg", SvgRenderer)
        assert "svg" in registry.available()


class TestSvgRendererOutput:
    def _build_simple_graph(self):
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
        return Graph(nodes=nodes, edges=edges)

    def _build_scene(self, graph: Graph | None = None) -> StyledScene:
        target_graph = graph or self._build_simple_graph()
        layout = VerticalLayout()
        scene = layout.layout(target_graph)
        styler = DefaultSceneStyler()
        return styler.apply(scene)

    def test_returns_svg_string(self):
        scene = self._build_scene()
        result = SvgRenderer().render(scene)
        assert isinstance(result, str)
        assert result.startswith("<svg")
        assert result.endswith("</svg>")

    def test_valid_xml(self):
        scene = self._build_scene()
        result = SvgRenderer().render(scene)
        ET.fromstring(result)

    def test_correct_node_count(self):
        scene = self._build_scene()
        result = SvgRenderer().render(scene)
        root = ET.fromstring(result)
        rects = _findall(root, "rect")
        assert len(rects) >= 3

    def test_correct_edge_count(self):
        scene = self._build_scene()
        result = SvgRenderer().render(scene)
        root = ET.fromstring(result)
        lines = _findall(root, "line")
        assert len(lines) == 2

    def test_labels_rendered(self):
        scene = self._build_scene()
        result = SvgRenderer().render(scene)
        root = ET.fromstring(result)
        texts = [el.text for el in _findall(root, "text") if el.text]
        assert any("Dense" in t for t in texts)
        assert any("Input" in t for t in texts)
        assert any("Prediction" in t for t in texts)

    def test_parameter_counts_rendered(self):
        scene = self._build_scene()
        result = SvgRenderer().render(scene)
        root = ET.fromstring(result)
        texts = [el.text for el in _findall(root, "text") if el.text]
        assert any("Params: 48" in t for t in texts)

    def test_output_shapes_rendered(self):
        scene = self._build_scene()
        result = SvgRenderer().render(scene)
        root = ET.fromstring(result)
        texts = [el.text for el in _findall(root, "text") if el.text]
        assert any("(None, 16)" in t for t in texts)

    def test_metadata_rendered(self):
        scene = self._build_scene()
        result = SvgRenderer().render(scene)
        root = ET.fromstring(result)
        texts = [el.text for el in _findall(root, "text") if el.text]
        assert any("Units: 16" in t for t in texts)

    def test_arrow_markers_defined(self):
        scene = self._build_scene()
        result = SvgRenderer().render(scene)
        root = ET.fromstring(result)
        markers = _findall(root, "marker")
        assert len(markers) == 1

    def test_empty_model(self):
        graph = Graph(nodes=[], edges=[])
        scene = self._build_scene(graph)
        result = SvgRenderer().render(scene)
        assert "(Empty model)" in result
        root = ET.fromstring(result)
        assert _local(root.tag) == "svg"

    def test_file_export(self, tmp_path):
        scene = self._build_scene()
        renderer = SvgRenderer()
        svg = renderer.render(scene)
        filepath = tmp_path / "model_architecture.svg"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(svg)
        assert filepath.exists()
        content = filepath.read_text(encoding="utf-8")
        assert content.startswith("<svg")
        assert "Dense" in content

    def test_multiple_dense_layers(self):
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
                id="dense_1",
                name="Dense_1",
                layer_type="Dense",
                params=80,
                output_shape="(None, 10)",
                metadata={"units": 10},
                tags={"trainable", "dense"},
            ),
            Node(
                id="relu",
                name="ReLU",
                layer_type="ReLU",
                params=0,
                output_shape="(None, 10)",
                tags={"activation"},
            ),
            Node(
                id="dense_2",
                name="Dense_2",
                layer_type="Dense",
                params=110,
                output_shape="(None, 11)",
                metadata={"units": 11},
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
            Edge(source="input", target="dense_1"),
            Edge(source="dense_1", target="relu"),
            Edge(source="relu", target="dense_2"),
            Edge(source="dense_2", target="output"),
        ]
        graph = Graph(nodes=nodes, edges=edges)
        scene = self._build_scene(graph)
        result = SvgRenderer().render(scene)
        root = ET.fromstring(result)
        rects = _findall(root, "rect")
        assert len(rects) >= 5
        lines = _findall(root, "line")
        assert len(lines) == 4

    def test_conv2d_metadata(self):
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
                id="conv",
                name="Conv2D",
                layer_type="Conv2D",
                params=448,
                output_shape="(None, 32, 32, 16)",
                metadata={
                    "filters": 16,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": "valid",
                },
                tags={"trainable", "convolution"},
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
            Edge(source="input", target="conv"),
            Edge(source="conv", target="output"),
        ]
        graph = Graph(nodes=nodes, edges=edges)
        scene = self._build_scene(graph)
        result = SvgRenderer().render(scene)
        root = ET.fromstring(result)
        texts = [el.text for el in _findall(root, "text") if el.text]
        assert any("Filters: 16" in t for t in texts)
        assert any("Kernel: 3" in t for t in texts)
        assert any("Stride: 1" in t for t in texts)
        assert any("Padding: valid" in t for t in texts)

    def test_dropout_metadata(self):
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
                id="dropout",
                name="Dropout",
                layer_type="Dropout",
                params=0,
                output_shape="(None, 16)",
                metadata={"rate": 0.25},
                tags={"regularization"},
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
            Edge(source="dense", target="dropout"),
            Edge(source="dropout", target="output"),
        ]
        graph = Graph(nodes=nodes, edges=edges)
        scene = self._build_scene(graph)
        result = SvgRenderer().render(scene)
        root = ET.fromstring(result)
        texts = [el.text for el in _findall(root, "text") if el.text]
        assert any("Rate: 0.25" in t for t in texts)

    def test_batchnorm_metadata(self):
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
                id="bn",
                name="BatchNorm",
                layer_type="BatchNormalization",
                params=64,
                output_shape="(None, 32)",
                metadata={"momentum": 0.99, "epsilon": 0.001},
                tags={"trainable", "normalization"},
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
            Edge(source="input", target="bn"),
            Edge(source="bn", target="output"),
        ]
        graph = Graph(nodes=nodes, edges=edges)
        scene = self._build_scene(graph)
        result = SvgRenderer().render(scene)
        root = ET.fromstring(result)
        texts = [el.text for el in _findall(root, "text") if el.text]
        assert any("Momentum: 0.99" in t for t in texts)
        assert any("Epsilon: 0.001" in t for t in texts)

    def test_graph_consistency(self):
        """Renderer should not mutate the input scene."""
        nodes = [
            Node(
                id="n1",
                name="A",
                layer_type="Dense",
                params=1,
                output_shape="(None, 2)",
                metadata={"units": 2},
            ),
            Node(id="n2", name="B", layer_type="ReLU", params=0, output_shape="(None, 2)"),
        ]
        edges = [Edge(source="n1", target="n2")]
        graph = Graph(nodes=nodes, edges=edges)
        scene = self._build_scene(graph)
        original_nodes = list(scene.nodes)
        original_edges = list(scene.edges)
        SvgRenderer().render(scene)
        assert scene.nodes == original_nodes
        assert scene.edges == original_edges

    def test_font_family_arial(self):
        scene = self._build_scene()
        result = SvgRenderer().render(scene)
        root = ET.fromstring(result)
        texts = _findall(root, "text")
        assert len(texts) > 0
        for t in texts:
            assert t.get("font-family") == "Arial"

    def test_vertical_layout(self):
        nodes = [
            Node(
                id="n1",
                name="A",
                layer_type="Dense",
                params=1,
                output_shape="(None, 2)",
                metadata={"units": 2},
            ),
            Node(id="n2", name="B", layer_type="ReLU", params=0, output_shape="(None, 2)"),
            Node(
                id="n3",
                name="C",
                layer_type="Dense",
                params=2,
                output_shape="(None, 2)",
                metadata={"units": 2},
            ),
        ]
        edges = [
            Edge(source="n1", target="n2"),
            Edge(source="n2", target="n3"),
        ]
        graph = Graph(nodes=nodes, edges=edges)
        scene = self._build_scene(graph)
        result = SvgRenderer().render(scene)
        root = ET.fromstring(result)
        lines = _findall(root, "line")
        assert len(lines) == 2
        for line in lines:
            x1 = float(line.get("x1"))
            x2 = float(line.get("x2"))
            assert abs(x1 - x2) < 1.0

    def test_activation_layer_type_label(self):
        nodes = [
            Node(id="relu", name="ReLU", layer_type="ReLU", params=0, output_shape="(None, 16)"),
            Node(
                id="sigmoid",
                name="Sigmoid",
                layer_type="Sigmoid",
                params=0,
                output_shape="(None, 16)",
            ),
            Node(id="tanh", name="Tanh", layer_type="Tanh", params=0, output_shape="(None, 16)"),
            Node(
                id="softmax",
                name="Softmax",
                layer_type="Softmax",
                params=0,
                output_shape="(None, 10)",
            ),
        ]
        edges = [
            Edge(source="relu", target="sigmoid"),
            Edge(source="sigmoid", target="tanh"),
            Edge(source="tanh", target="softmax"),
        ]
        graph = Graph(nodes=nodes, edges=edges)
        scene = self._build_scene(graph)
        result = SvgRenderer().render(scene)
        root = ET.fromstring(result)
        texts = [el.text for el in _findall(root, "text") if el.text]
        assert any("ReLU Activation" in t for t in texts)
        assert any("Sigmoid Activation" in t for t in texts)
        assert any("Tanh Activation" in t for t in texts)
        assert any("Softmax Activation" in t for t in texts)

    def test_batchnorm_layer_type_label(self):
        nodes = [
            Node(
                id="bn",
                name="BatchNorm",
                layer_type="BatchNormalization",
                params=64,
                output_shape="(None, 32)",
            ),
        ]
        edges = []
        graph = Graph(nodes=nodes, edges=edges)
        scene = self._build_scene(graph)
        result = SvgRenderer().render(scene)
        root = ET.fromstring(result)
        texts = [el.text for el in _findall(root, "text") if el.text]
        assert any("Normalization Layer" in t for t in texts)

    def test_conv2d_layer_type_label(self):
        nodes = [
            Node(
                id="conv",
                name="Conv2D",
                layer_type="Conv2D",
                params=448,
                output_shape="(None, 32, 32, 16)",
            ),
        ]
        edges = []
        graph = Graph(nodes=nodes, edges=edges)
        scene = self._build_scene(graph)
        result = SvgRenderer().render(scene)
        root = ET.fromstring(result)
        texts = [el.text for el in _findall(root, "text") if el.text]
        assert any("Convolutional Layer" in t for t in texts)
