"""Visualization engine pipeline."""

from __future__ import annotations

from typing import cast

from kronyx.viz.builder import GraphBuilder
from kronyx.viz.icons.registry import IconRegistry
from kronyx.viz.layout.registry import LayoutRegistry
from kronyx.viz.layout.vertical import VerticalLayout
from kronyx.viz.registry import RendererRegistry
from kronyx.viz.renderers.base import Renderer
from kronyx.viz.styling.default import DefaultSceneStyler
from kronyx.viz.themes import ThemeRegistry


class VisualizationEngine:
    """Coordinates the visualization pipeline.

    Pipeline:
        Sequential model -> GraphBuilder -> Graph -> LayoutRegistry
        -> LayoutEngine -> Scene -> SceneStyler -> StyledScene
        -> RendererRegistry -> Renderer
    """

    def __init__(
        self,
        registry: RendererRegistry | None = None,
        layout_registry: LayoutRegistry | None = None,
        styler: DefaultSceneStyler | None = None,
        theme_registry: ThemeRegistry | None = None,
        theme_name: str = "light",
        icon_registry: IconRegistry | None = None,
    ) -> None:
        """Initialize the engine.

        Args:
            registry: Optional renderer registry. Uses a default if not provided.
            layout_registry: Optional layout registry. Uses a default if not provided.
            styler: Optional scene styler. Uses DefaultSceneStyler if not provided.
            theme_registry: Optional theme registry. Uses a default if not provided.
            theme_name: Name of the registered theme to use. Defaults to "light".
            icon_registry: Optional icon registry. Uses a default if not provided.
        """
        self.registry = registry or RendererRegistry()
        self.layout_registry = layout_registry or LayoutRegistry()
        self.theme_registry = theme_registry or ThemeRegistry()
        self.icon_registry = icon_registry or IconRegistry()
        self.builder = GraphBuilder()
        self._register_defaults()

        if styler is not None:
            self.styler = styler
        else:
            theme = self.theme_registry.get(theme_name)
            self.styler = DefaultSceneStyler(
                theme=theme, icon_registry=self.icon_registry
            )

    def _register_defaults(self) -> None:
        """Register built-in layouts, renderers, and themes."""
        self.layout_registry.register("vertical", VerticalLayout)

        from kronyx.viz.renderers.svg import SvgRenderer

        self.registry.register("svg", SvgRenderer)

        from kronyx.viz.themes import (
            BlueprintTheme,
            DarkTheme,
            DefaultTheme,
            LightTheme,
            NeonTheme,
            TerminalTheme,
        )

        self.theme_registry.register("light", LightTheme)
        self.theme_registry.register("dark", DarkTheme)
        self.theme_registry.register("blueprint", BlueprintTheme)
        self.theme_registry.register("terminal", TerminalTheme)
        self.theme_registry.register("neon", NeonTheme)
        self.theme_registry.register("default", DefaultTheme)

        from kronyx.viz.icons.builtin import register_builtin_icons

        register_builtin_icons(self.icon_registry)

    def visualize(
        self,
        model: object,
        renderer_name: str = "ascii",
        layout_name: str = "vertical",
        **kwargs: object,
    ) -> str:
        """Run the full visualization pipeline.

        Args:
            model: A Sequential model instance.
            renderer_name: Name of the registered renderer to use.
            layout_name: Name of the registered layout to use.
            **kwargs: Additional arguments passed to the renderer.

        Returns:
            Rendered output string.

        Raises:
            KeyError: If the renderer or layout name is not registered.
        """
        graph = self.builder.build(model)
        layout_cls = self.layout_registry.get(layout_name)
        layout = layout_cls()
        scene = layout.layout(graph)
        styled_scene = self.styler.apply(scene)
        renderer_cls = self.registry.get(renderer_name)
        renderer = cast("type[Renderer]", renderer_cls)()
        return renderer.render(styled_scene, **kwargs)

    def export(
        self,
        model: object,
        renderer_name: str,
        filepath: str,
        layout_name: str = "vertical",
        **kwargs: object,
    ) -> str:
        """Render the model and write the output to a file.

        Args:
            model: A Sequential model instance.
            renderer_name: Name of the registered renderer to use.
            filepath: Destination file path.
            layout_name: Name of the registered layout to use.
            **kwargs: Additional arguments passed to the renderer.

        Returns:
            The filepath that was written.
        """
        rendered = self.visualize(model, renderer_name, layout_name=layout_name, **kwargs)
        with open(filepath, "w", encoding="utf-8") as handle:
            handle.write(rendered)
        return filepath
