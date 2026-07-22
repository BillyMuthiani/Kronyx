"""Visualization engine pipeline."""

from __future__ import annotations

from kronyx.viz.builder import GraphBuilder
from kronyx.viz.registry import RendererRegistry


class VisualizationEngine:
    """Coordinates the visualization pipeline.

    Pipeline:
        Sequential model -> GraphBuilder -> Graph -> RendererRegistry -> Renderer
    """

    def __init__(self, registry: RendererRegistry | None = None) -> None:
        """Initialize the engine.

        Args:
            registry: Optional renderer registry. Uses a default if not provided.
        """
        self.registry = registry or RendererRegistry()
        self.builder = GraphBuilder()

    def visualize(self, model: object, renderer_name: str = "ascii", **kwargs: object) -> str:
        """Run the full visualization pipeline.

        Args:
            model: A Sequential model instance.
            renderer_name: Name of the registered renderer to use.
            **kwargs: Additional arguments passed to the renderer.

        Returns:
            Rendered output string.

        Raises:
            KeyError: If the renderer name is not registered.
        """
        graph = self.builder.build(model)
        renderer_cls = self.registry.get(renderer_name)
        return renderer_cls.render(graph, **kwargs)  # type: ignore[attr-defined, no-any-return]
