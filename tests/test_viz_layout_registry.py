"""Tests for the VisualizationEngine."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from kronyx.viz.engine import VisualizationEngine
from kronyx.viz.graph import Edge, Graph, Node, Scene
from kronyx.viz.layout.base import LayoutEngine
from kronyx.viz.layout.registry import LayoutRegistry
from kronyx.viz.layout.vertical import VerticalLayout

# ---------------------------------------------------------------------------
# LayoutRegistry tests
# ---------------------------------------------------------------------------


class TestLayoutRegistry:
    def test_register_and_get(self):
        registry = LayoutRegistry()
        registry.register("vertical", VerticalLayout)
        assert registry.get("vertical") is VerticalLayout

    def test_register_duplicate_overwrites(self):
        registry = LayoutRegistry()
        registry.register("vertical", VerticalLayout)
        registry.register("vertical", VerticalLayout)
        assert registry.get("vertical") is VerticalLayout

    def test_unregister_removes_layout(self):
        registry = LayoutRegistry()
        registry.register("vertical", VerticalLayout)
        registry.unregister("vertical")
        assert "vertical" not in registry.available()

    def test_unregister_nonexistent_is_safe(self):
        registry = LayoutRegistry()
        registry.unregister("nonexistent")

    def test_get_raises_for_unknown_layout(self):
        registry = LayoutRegistry()
        registry.register("vertical", VerticalLayout)
        with pytest.raises(KeyError, match="No layout registered under name: horizontal"):
            registry.get("horizontal")
        with pytest.raises(KeyError, match="Available layouts: vertical"):
            registry.get("horizontal")

    def test_available_returns_sorted_names(self):
        registry = LayoutRegistry()
        registry.register("vertical", VerticalLayout)
        registry.register("alpha", VerticalLayout)
        assert registry.available() == ["alpha", "vertical"]

    def test_available_empty_on_fresh_registry(self):
        registry = LayoutRegistry()
        assert registry.available() == []

    def test_clear_removes_all(self):
        registry = LayoutRegistry()
        registry.register("vertical", VerticalLayout)
        registry.register("alpha", VerticalLayout)
        registry.clear()
        assert registry.available() == []

    def test_multiple_layout_types(self):
        registry = LayoutRegistry()

        class DummyLayout(LayoutEngine):
            def layout(self, graph, style=None):
                return Scene(graph=graph)

        registry.register("vertical", VerticalLayout)
        registry.register("dummy", DummyLayout)
        assert registry.get("vertical") is VerticalLayout
        assert registry.get("dummy") is DummyLayout


# ---------------------------------------------------------------------------
# VisualizationEngine integration tests
# ---------------------------------------------------------------------------


class TestEngineLayoutIntegration:
    def test_default_layout_is_vertical(self):
        engine = VisualizationEngine()
        assert "vertical" in engine.layout_registry.available()

    def test_custom_layout_registry(self):
        registry = LayoutRegistry()
        engine = VisualizationEngine(layout_registry=registry)
        assert "vertical" in engine.layout_registry.available()

    def test_engine_looks_up_layout_from_registry(self):
        registry = LayoutRegistry()
        registry.register("vertical", VerticalLayout)
        engine = VisualizationEngine(layout_registry=registry)
        assert engine.layout_registry.get("vertical") is VerticalLayout

    def test_engine_raises_for_unknown_layout(self):
        engine = VisualizationEngine()
        graph = Graph(
            nodes=[Node(id="n1", name="A", layer_type="Dense", params=1)],
            edges=[Edge(source="n1", target="n2") for n2 in ("n2",)],
        )
        with pytest.raises(KeyError, match="No layout registered under name"):
            engine.visualize(graph, renderer_name="svg", layout_name="nonexistent")

    def test_backwards_compatible_default_layout(self):
        engine = VisualizationEngine()
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
        result = engine.visualize(graph, renderer_name="svg")
        assert result.startswith("<svg")
        assert result.endswith("</svg>")

    def test_renderer_and_layout_cooperation(self):
        engine = VisualizationEngine()
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
                Node(id="n2", name="B", layer_type="ReLU", params=0, output_shape="(None, 2)"),
            ],
            edges=[Edge(source="n1", target="n2")],
        )
        result = engine.visualize(graph, renderer_name="svg")
        root = ET.fromstring(result)
        rects = [el for el in root.iter() if el.tag.split("}", 1)[-1] == "rect"]
        assert len(rects) == 2
