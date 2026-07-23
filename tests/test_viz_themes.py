"""Tests for the visualization theme system."""

from __future__ import annotations

import pytest

from kronyx.viz.engine import VisualizationEngine
from kronyx.viz.graph import Edge, Graph, Node, Scene
from kronyx.viz.layout.vertical import VerticalLayout
from kronyx.viz.renderers.svg import SvgRenderer
from kronyx.viz.styling.default import DefaultSceneStyler
from kronyx.viz.themes import (
    BlueprintTheme,
    DarkTheme,
    DefaultTheme,
    LightTheme,
    NeonTheme,
    TerminalTheme,
    Theme,
    ThemeRegistry,
)


class TestThemeDataclass:
    def test_default_theme_values(self):
        theme = Theme()
        assert theme.background_color == "#ffffff"
        assert theme.text_color == "#000000"
        assert theme.secondary_text_color == "#666666"
        assert theme.node_fill == "#ffffff"
        assert theme.node_border == "#000000"
        assert theme.node_border_width == 2
        assert theme.node_radius == 10
        assert theme.edge_color == "#000000"
        assert theme.edge_width == 2
        assert theme.arrow_color == "#000000"
        assert theme.title_font_family == "Arial"
        assert theme.body_font_family == "Arial"
        assert theme.title_font_size == 16
        assert theme.body_font_size == 12
        assert theme.metadata_color == "#666666"
        assert theme.canvas_background == "#ffffff"
        assert theme.shadow_color is None
        assert theme.grid_color is None
        assert theme.selection_color is None

    def test_backward_compatible_aliases(self):
        theme = Theme()
        assert theme.background == "#ffffff"
        assert theme.title_color == "#000000"
        assert theme.body_color == "#000000"

    def test_custom_theme(self):
        theme = Theme(
            background_color="#000000",
            text_color="#ffffff",
            node_fill="#111111",
            node_border="#ff00ff",
        )
        assert theme.background_color == "#000000"
        assert theme.text_color == "#ffffff"
        assert theme.node_fill == "#111111"
        assert theme.node_border == "#ff00ff"

    def test_theme_equality(self):
        theme1 = Theme()
        theme2 = Theme()
        assert theme1 == theme2

    def test_theme_inequality(self):
        theme1 = Theme(background_color="#ffffff")
        theme2 = Theme(background_color="#000000")
        assert theme1 != theme2

    def test_theme_is_frozen(self):
        theme = Theme()
        with pytest.raises(AttributeError):
            theme.background_color = "#000000"  # type: ignore[misc]

    def test_default_theme_singleton(self):
        assert DefaultTheme == Theme()

    def test_light_theme_matches_default(self):
        assert LightTheme == DefaultTheme


class TestBuiltInThemes:
    def test_dark_theme_values(self):
        assert DarkTheme.background_color == "#1e1e1e"
        assert DarkTheme.text_color == "#e0e0e0"
        assert DarkTheme.node_fill == "#2d2d2d"
        assert DarkTheme.node_border == "#555555"
        assert DarkTheme.edge_color == "#777777"
        assert DarkTheme.arrow_color == "#777777"

    def test_blueprint_theme_values(self):
        assert BlueprintTheme.background_color == "#1a3a5c"
        assert BlueprintTheme.text_color == "#e0f0ff"
        assert BlueprintTheme.node_fill == "#234b6e"
        assert BlueprintTheme.node_border == "#4a90d9"
        assert BlueprintTheme.edge_color == "#4a90d9"

    def test_terminal_theme_values(self):
        assert TerminalTheme.background_color == "#0c0c0c"
        assert TerminalTheme.text_color == "#00ff00"
        assert TerminalTheme.node_fill == "#0c0c0c"
        assert TerminalTheme.node_border == "#00ff00"
        assert TerminalTheme.title_font_family == "Courier New"
        assert TerminalTheme.body_font_family == "Courier New"

    def test_neon_theme_values(self):
        assert NeonTheme.background_color == "#0a0a0a"
        assert NeonTheme.text_color == "#ffffff"
        assert NeonTheme.node_border == "#ff00ff"
        assert NeonTheme.edge_color == "#00ffff"
        assert NeonTheme.arrow_color == "#00ffff"

    def test_themes_are_distinct(self):
        themes = [LightTheme, DarkTheme, BlueprintTheme, TerminalTheme, NeonTheme]
        assert len(set(themes)) == len(themes)
        assert DefaultTheme == LightTheme

    def test_themes_are_immutable(self):
        for theme in [DefaultTheme, DarkTheme, BlueprintTheme]:
            with pytest.raises(AttributeError):
                theme.node_fill = "#000"  # type: ignore[misc]


class TestThemeRegistry:
    def test_register_and_get(self):
        registry = ThemeRegistry()
        registry.register("light", LightTheme)
        assert registry.get("light") is LightTheme

    def test_register_duplicate_overwrites(self):
        registry = ThemeRegistry()
        registry.register("light", LightTheme)
        registry.register("light", DarkTheme)
        assert registry.get("light") is DarkTheme

    def test_unregister_removes_theme(self):
        registry = ThemeRegistry()
        registry.register("light", LightTheme)
        registry.unregister("light")
        assert "light" not in registry.available()

    def test_unregister_nonexistent_is_safe(self):
        registry = ThemeRegistry()
        registry.unregister("nonexistent")

    def test_get_raises_for_unknown_theme(self):
        registry = ThemeRegistry()
        registry.register("light", LightTheme)
        with pytest.raises(KeyError, match="No theme registered under name: dark"):
            registry.get("dark")
        with pytest.raises(KeyError, match="Available themes: light"):
            registry.get("dark")

    def test_available_returns_sorted_names(self):
        registry = ThemeRegistry()
        registry.register("dark", DarkTheme)
        registry.register("light", LightTheme)
        assert registry.available() == ["dark", "light"]

    def test_available_empty_on_fresh_registry(self):
        registry = ThemeRegistry()
        assert registry.available() == []

    def test_clear_removes_all(self):
        registry = ThemeRegistry()
        registry.register("light", LightTheme)
        registry.register("dark", DarkTheme)
        registry.clear()
        assert registry.available() == []


class TestEngineThemeIntegration:
    def test_default_theme_is_light(self):
        engine = VisualizationEngine()
        assert "light" in engine.theme_registry.available()

    def test_custom_theme_registry(self):
        registry = ThemeRegistry()
        engine = VisualizationEngine(theme_registry=registry)
        assert "light" in engine.theme_registry.available()

    def test_builtin_themes_registered(self):
        engine = VisualizationEngine()
        for name in ["light", "dark", "blueprint", "terminal", "neon", "default"]:
            assert name in engine.theme_registry.available()

    def test_theme_name_parameter(self):
        engine = VisualizationEngine(theme_name="dark")
        theme = engine.theme_registry.get("dark")
        assert theme.background_color == "#1e1e1e"

    def test_raise_for_unknown_theme(self):
        engine = VisualizationEngine()
        with pytest.raises(KeyError, match="No theme registered under name"):
            engine.theme_registry.get("nonexistent")

    def test_custom_styler_overrides_theme(self):
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
                    id="output",
                    name="Prediction",
                    layer_type="Output",
                    params=0,
                    output_shape="?",
                ),
            ],
            edges=[Edge(source="input", target="output")],
        )
        custom_theme = Theme(node_fill="#ff0000")
        styler = DefaultSceneStyler(theme=custom_theme)
        engine = VisualizationEngine(styler=styler)
        result = engine.visualize(graph, renderer_name="svg")
        assert "#ff0000" in result

    def test_theme_switching(self):
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
                    id="output",
                    name="Prediction",
                    layer_type="Output",
                    params=0,
                    output_shape="?",
                ),
            ],
            edges=[Edge(source="input", target="output")],
        )

        engine_light = VisualizationEngine(theme_name="light")
        result_light = engine_light.visualize(graph, renderer_name="svg")

        engine_dark = VisualizationEngine(theme_name="dark")
        result_dark = engine_dark.visualize(graph, renderer_name="svg")

        assert result_light != result_dark
        assert "#ffffff" in result_light
        assert "#2d2d2d" in result_dark
        assert "#e0e0e0" in result_dark

    def test_backward_compatible_visualize(self):
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
                    id="output",
                    name="Prediction",
                    layer_type="Output",
                    params=0,
                    output_shape="?",
                ),
            ],
            edges=[Edge(source="input", target="output")],
        )
        engine = VisualizationEngine()
        result = engine.visualize(graph, renderer_name="svg")
        assert isinstance(result, str)
        assert result.startswith("<svg")
        assert result.endswith("</svg>")


class TestSceneStylerConsumesTheme:
    def test_styler_reads_colors_from_theme(self):
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
                Node(
                    id="n2",
                    name="B",
                    layer_type="Dense",
                    params=1,
                    output_shape="(None, 2)",
                ),
            ],
            edges=[Edge(source="n1", target="n2")],
        )
        scene = VerticalLayout().layout(graph)
        theme = Theme(
            node_fill="#ff0000",
            node_border="#00ff00",
            text_color="#0000ff",
            edge_color="#ffff00",
            arrow_color="#ffff00",
            canvas_background="#111111",
        )
        styler = DefaultSceneStyler(theme=theme)
        styled = styler.apply(scene)

        assert styled.nodes[0].fill_color == "#ff0000"
        assert styled.nodes[0].stroke_color == "#00ff00"
        assert styled.nodes[0].text_color == "#0000ff"
        assert styled.edges[0].stroke_color == "#ffff00"
        assert styled.edges[0].arrow_color == "#ffff00"
        assert styled.background == "#111111"

    def test_styler_reads_fonts_from_theme(self):
        graph = Graph(
            nodes=[
                Node(
                    id="n1",
                    name="A",
                    layer_type="Dense",
                    params=1,
                    output_shape="(None, 2)",
                ),
            ],
            edges=[],
        )
        scene = VerticalLayout().layout(graph)
        theme = Theme(
            title_font_family="Courier New",
            body_font_family="Courier New",
            title_font_size=18,
            body_font_size=14,
        )
        styler = DefaultSceneStyler(theme=theme)
        styled = styler.apply(scene)

        assert styled.nodes[0].font_family == "Courier New"
        assert styled.nodes[0].title_font_size == 18
        assert styled.nodes[0].body_font_size == 14

    def test_styler_reads_borders_from_theme(self):
        graph = Graph(
            nodes=[
                Node(
                    id="n1",
                    name="A",
                    layer_type="Dense",
                    params=1,
                    output_shape="(None, 2)",
                ),
            ],
            edges=[],
        )
        scene = VerticalLayout().layout(graph)
        theme = Theme(node_border_width=4, node_radius=20)
        styler = DefaultSceneStyler(theme=theme)
        styled = styler.apply(scene)

        assert styled.nodes[0].stroke_width == 4
        assert styled.nodes[0].border_radius == 20

    def test_styler_propagates_tags(self):
        graph = Graph(
            nodes=[
                Node(
                    id="n1",
                    name="A",
                    layer_type="Dense",
                    params=1,
                    output_shape="(None, 2)",
                    tags={"trainable", "dense"},
                ),
            ],
            edges=[],
        )
        scene = VerticalLayout().layout(graph)
        styler = DefaultSceneStyler()
        styled = styler.apply(scene)

        assert "trainable" in styled.nodes[0].tags
        assert "dense" in styled.nodes[0].tags

    def test_styler_preserves_graph_reference(self):
        graph = Graph(
            nodes=[
                Node(id="n1", name="A", layer_type="Dense", params=1),
            ],
            edges=[],
        )
        scene = VerticalLayout().layout(graph)
        styler = DefaultSceneStyler()
        styled = styler.apply(scene)

        assert styled.graph is scene.graph


class TestRendererOutputWithThemes:
    def test_light_theme_renders(self):
        graph = Graph(
            nodes=[
                Node(id="n1", name="A", layer_type="Dense", params=1, output_shape="(None, 2)"),
            ],
            edges=[],
        )
        engine = VisualizationEngine(theme_name="light")
        result = engine.visualize(graph, renderer_name="svg")
        assert result.startswith("<svg")
        assert "#ffffff" in result

    def test_dark_theme_renders(self):
        graph = Graph(
            nodes=[
                Node(id="n1", name="A", layer_type="Dense", params=1, output_shape="(None, 2)"),
            ],
            edges=[],
        )
        engine = VisualizationEngine(theme_name="dark")
        result = engine.visualize(graph, renderer_name="svg")
        assert result.startswith("<svg")
        assert "#2d2d2d" in result
        assert "#e0e0e0" in result

    def test_terminal_theme_renders(self):
        graph = Graph(
            nodes=[
                Node(id="n1", name="A", layer_type="Dense", params=1, output_shape="(None, 2)"),
            ],
            edges=[],
        )
        engine = VisualizationEngine(theme_name="terminal")
        result = engine.visualize(graph, renderer_name="svg")
        assert result.startswith("<svg")
        assert "#00ff00" in result

    def test_renderer_agnostic_to_theme_source(self):
        graph = Graph(
            nodes=[
                Node(id="n1", name="A", layer_type="Dense", params=1, output_shape="(None, 2)"),
            ],
            edges=[],
        )
        scene = VerticalLayout().layout(graph)
        theme = Theme(node_fill="#123456")
        styler = DefaultSceneStyler(theme=theme)
        styled = styler.apply(scene)
        result = SvgRenderer().render(styled)

        assert "#123456" in result
        assert result.startswith("<svg")

    def test_empty_model_with_theme(self):
        scene = Scene(graph=Graph(nodes=[], edges=[]))
        theme = DarkTheme
        styler = DefaultSceneStyler(theme=theme)
        styled = styler.apply(scene)
        renderer = SvgRenderer()
        result = renderer.render(styled)
        assert "(Empty model)" in result
        assert result.startswith("<svg")
