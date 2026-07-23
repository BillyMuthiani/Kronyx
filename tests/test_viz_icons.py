"""Tests for the visualization icon system."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from kronyx.viz.engine import VisualizationEngine
from kronyx.viz.graph import Edge, Graph, Node
from kronyx.viz.icons.base import Icon
from kronyx.viz.icons.registry import IconRegistry
from kronyx.viz.layout.vertical import VerticalLayout
from kronyx.viz.renderers.svg import SvgRenderer
from kronyx.viz.styling.default import DefaultSceneStyler
from kronyx.viz.themes import DefaultTheme


class TestIconDataclass:
    def test_default_values(self):
        icon = Icon(id="dense")
        assert icon.id == "dense"
        assert icon.label == ""
        assert icon.category == "general"

    def test_custom_values(self):
        icon = Icon(id="relu", label="ReLU", category="activation")
        assert icon.id == "relu"
        assert icon.label == "ReLU"
        assert icon.category == "activation"

    def test_icon_equality(self):
        icon1 = Icon(id="dense")
        icon2 = Icon(id="dense")
        assert icon1 == icon2

    def test_icon_inequality(self):
        icon1 = Icon(id="dense")
        icon2 = Icon(id="relu")
        assert icon1 != icon2

    def test_icon_is_frozen(self):
        icon = Icon(id="dense")
        with pytest.raises(AttributeError):
            icon.id = "relu"  # type: ignore[misc]


class TestIconRegistry:
    def test_register_and_get(self):
        registry = IconRegistry()
        registry.register("Dense", Icon(id="dense"))
        assert registry.get("Dense") == Icon(id="dense")

    def test_register_duplicate_overwrites(self):
        registry = IconRegistry()
        registry.register("Dense", Icon(id="dense"))
        registry.register("Dense", Icon(id="relu"))
        assert registry.get("Dense").id == "relu"

    def test_unregister_removes_icon(self):
        registry = IconRegistry()
        registry.register("Dense", Icon(id="dense"))
        registry.unregister("Dense")
        assert "Dense" not in registry.available()

    def test_unregister_nonexistent_is_safe(self):
        registry = IconRegistry()
        registry.unregister("nonexistent")

    def test_get_raises_for_unknown_icon(self):
        registry = IconRegistry()
        registry.register("Dense", Icon(id="dense"))
        with pytest.raises(KeyError, match="No icon registered under name: ReLU"):
            registry.get("ReLU")
        with pytest.raises(KeyError, match="Available icons: Dense"):
            registry.get("ReLU")

    def test_available_returns_sorted_names(self):
        registry = IconRegistry()
        registry.register("Dense", Icon(id="dense"))
        registry.register("ReLU", Icon(id="relu"))
        assert registry.available() == ["Dense", "ReLU"]

    def test_available_empty_on_fresh_registry(self):
        registry = IconRegistry()
        assert registry.available() == []

    def test_clear_removes_all(self):
        registry = IconRegistry()
        registry.register("Dense", Icon(id="dense"))
        registry.register("ReLU", Icon(id="relu"))
        registry.clear()
        assert registry.available() == []


class TestBuiltinIcons:
    def test_builtin_registry_populated(self):
        from kronyx.viz.icons.builtin import _create_builtin_registry

        registry = _create_builtin_registry()
        assert len(registry.available()) > 0

    def test_required_icons_registered(self):
        from kronyx.viz.icons.builtin import _create_builtin_registry

        registry = _create_builtin_registry()
        required = [
            "Input",
            "Output",
            "Dense",
            "Conv2D",
            "Conv1D",
            "Flatten",
            "Dropout",
            "BatchNormalization",
            "ReLU",
            "Sigmoid",
            "Softmax",
            "Tanh",
            "LeakyReLU",
            "MaxPooling2D",
            "AveragePooling2D",
            "UnknownLayer",
        ]
        for name in required:
            assert name in registry.available(), f"Missing built-in icon: {name}"

    def test_unknown_layer_icon_exists(self):
        from kronyx.viz.icons.builtin import _create_builtin_registry

        registry = _create_builtin_registry()
        icon = registry.get("UnknownLayer")
        assert icon.id == "unknown"
        assert icon.category == "unknown"

    def test_icon_ids_are_unique(self):
        from kronyx.viz.icons.builtin import _create_builtin_registry

        registry = _create_builtin_registry()
        ids = [registry.get(name).id for name in registry.available()]
        assert len(ids) == len(set(ids))

    def test_engine_registers_builtin_icons(self):
        engine = VisualizationEngine()
        from kronyx.viz.icons.builtin import _create_builtin_registry

        builtin = _create_builtin_registry()
        for name in builtin.available():
            assert name in engine.icon_registry.available()


class TestSceneStylerIcons:
    def test_styler_attaches_icon_for_known_layer(self):
        graph = Graph(
            nodes=[
                Node(
                    id="dense",
                    name="Dense",
                    layer_type="Dense",
                    params=48,
                    output_shape="(None, 16)",
                    metadata={"units": 16},
                ),
            ],
            edges=[],
        )
        scene = VerticalLayout().layout(graph)
        styler = DefaultSceneStyler()
        styled = styler.apply(scene)
        assert styled.nodes[0].icon == "dense"

    def test_styler_attaches_icon_for_unknown_layer(self):
        graph = Graph(
            nodes=[
                Node(
                    id="unknown",
                    name="Custom",
                    layer_type="CustomLayer",
                    params=0,
                    output_shape="?",
                ),
            ],
            edges=[],
        )
        scene = VerticalLayout().layout(graph)
        styler = DefaultSceneStyler()
        styled = styler.apply(scene)
        assert styled.nodes[0].icon == "unknown"

    def test_styler_attaches_correct_activation_icons(self):
        for layer_type, expected_icon in [
            ("ReLU", "relu"),
            ("Sigmoid", "sigmoid"),
            ("Tanh", "tanh"),
            ("Softmax", "softmax"),
        ]:
            graph = Graph(
                nodes=[
                    Node(
                        id="act",
                        name="Activation",
                        layer_type=layer_type,
                        params=0,
                        output_shape="?",
                    ),
                ],
                edges=[],
            )
            scene = VerticalLayout().layout(graph)
            styler = DefaultSceneStyler()
            styled = styler.apply(scene)
            assert styled.nodes[0].icon == expected_icon

    def test_styler_without_icon_registry(self):
        graph = Graph(
            nodes=[
                Node(
                    id="dense",
                    name="Dense",
                    layer_type="Dense",
                    params=1,
                    output_shape="(None, 2)",
                ),
            ],
            edges=[],
        )
        scene = VerticalLayout().layout(graph)
        styler = DefaultSceneStyler(icon_registry=IconRegistry())
        styled = styler.apply(scene)
        assert styled.nodes[0].icon is None

    def test_styler_uses_custom_icon_registry(self):
        graph = Graph(
            nodes=[
                Node(
                    id="dense",
                    name="Dense",
                    layer_type="Dense",
                    params=1,
                    output_shape="(None, 2)",
                ),
            ],
            edges=[],
        )
        scene = VerticalLayout().layout(graph)
        registry = IconRegistry()
        registry.register("Dense", Icon(id="custom_dense", label="Custom"))
        styler = DefaultSceneStyler(icon_registry=registry)
        styled = styler.apply(scene)
        assert styled.nodes[0].icon == "custom_dense"


class TestRendererIconRendering:
    def test_dense_icon_renders_rect(self):
        graph = Graph(
            nodes=[
                Node(
                    id="dense",
                    name="Dense",
                    layer_type="Dense",
                    params=1,
                    output_shape="(None, 2)",
                ),
            ],
            edges=[],
        )
        scene = VerticalLayout().layout(graph)
        styler = DefaultSceneStyler()
        styled = styler.apply(scene)
        result = SvgRenderer().render(styled)
        root = ET.fromstring(result)
        rects = [el for el in root.iter() if el.tag.split("}", 1)[-1] == "rect"]
        assert len(rects) >= 2  # node rect + icon rect

    def test_input_icon_renders_circle(self):
        graph = Graph(
            nodes=[
                Node(
                    id="input",
                    name="Input",
                    layer_type="Input",
                    params=0,
                    output_shape="?",
                ),
            ],
            edges=[],
        )
        scene = VerticalLayout().layout(graph)
        styler = DefaultSceneStyler()
        styled = styler.apply(scene)
        result = SvgRenderer().render(styled)
        root = ET.fromstring(result)
        circles = [
            el for el in root.iter() if el.tag.split("}", 1)[-1] == "circle"
        ]
        assert len(circles) >= 1

    def test_unknown_layer_icon_renders_diamond(self):
        graph = Graph(
            nodes=[
                Node(
                    id="custom",
                    name="Custom",
                    layer_type="CustomLayer",
                    params=0,
                    output_shape="?",
                ),
            ],
            edges=[],
        )
        scene = VerticalLayout().layout(graph)
        styler = DefaultSceneStyler()
        styled = styler.apply(scene)
        result = SvgRenderer().render(styled)
        root = ET.fromstring(result)
        polygons = [
            el for el in root.iter() if el.tag.split("}", 1)[-1] == "polygon"
        ]
        assert len(polygons) >= 1

    def test_relu_icon_renders_polyline(self):
        graph = Graph(
            nodes=[
                Node(
                    id="relu",
                    name="ReLU",
                    layer_type="ReLU",
                    params=0,
                    output_shape="?",
                ),
            ],
            edges=[],
        )
        scene = VerticalLayout().layout(graph)
        styler = DefaultSceneStyler()
        styled = styler.apply(scene)
        result = SvgRenderer().render(styled)
        root = ET.fromstring(result)
        polylines = [
            el for el in root.iter() if el.tag.split("}", 1)[-1] == "polyline"
        ]
        assert len(polylines) >= 1

    def test_icons_use_text_color(self):
        graph = Graph(
            nodes=[
                Node(
                    id="dense",
                    name="Dense",
                    layer_type="Dense",
                    params=1,
                    output_shape="(None, 2)",
                ),
            ],
            edges=[],
        )
        scene = VerticalLayout().layout(graph)
        theme = DefaultTheme
        styler = DefaultSceneStyler(theme=theme)
        styled = styler.apply(scene)
        result = SvgRenderer().render(styled)
        assert "#000000" in result

    def test_renderer_does_not_choose_icon(self):
        graph = Graph(
            nodes=[
                Node(
                    id="n1",
                    name="A",
                    layer_type="Dense",
                    params=1,
                    output_shape="(None, 2)",
                ),
                Node(
                    id="n2",
                    name="B",
                    layer_type="ReLU",
                    params=0,
                    output_shape="(None, 2)",
                ),
            ],
            edges=[Edge(source="n1", target="n2")],
        )
        scene = VerticalLayout().layout(graph)
        styler = DefaultSceneStyler()
        styled = styler.apply(scene)
        assert styled.nodes[0].icon == "dense"
        assert styled.nodes[1].icon == "relu"
        result = SvgRenderer().render(styled)
        assert "<polyline" in result
        assert "<rect" in result


class TestEngineIconIntegration:
    def test_default_engine_has_icon_registry(self):
        engine = VisualizationEngine()
        assert hasattr(engine, "icon_registry")
        assert isinstance(engine.icon_registry, IconRegistry)

    def test_engine_visualize_with_icons(self):
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
                    id="relu",
                    name="ReLU",
                    layer_type="ReLU",
                    params=0,
                    output_shape="(None, 16)",
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
                Edge(source="dense", target="relu"),
                Edge(source="relu", target="output"),
            ],
        )
        scene = VerticalLayout().layout(graph)
        styler = DefaultSceneStyler()
        styled = styler.apply(scene)
        result = SvgRenderer().render(styled)
        assert result.startswith("<svg")
        assert result.endswith("</svg>")
        root = ET.fromstring(result)
        circles = [
            el for el in root.iter() if el.tag.split("}", 1)[-1] == "circle"
        ]
        rects = [el for el in root.iter() if el.tag.split("}", 1)[-1] == "rect"]
        polylines = [
            el for el in root.iter() if el.tag.split("}", 1)[-1] == "polyline"
        ]
        assert len(circles) >= 2
        assert len(rects) >= 4
        assert len(polylines) >= 1

    def test_backward_compatible_visualize(self):
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
        assert result.startswith("<svg")
        assert result.endswith("</svg>")
