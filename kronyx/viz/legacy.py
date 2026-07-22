"""Legacy visualizer adapter for backwards compatibility."""

from __future__ import annotations

from typing import Any

from kronyx.viz.engine import VisualizationEngine


class LegacyVisualizer:
    """Adapter that preserves the legacy model.visualize() behavior.

    This is a bridge for backwards compatibility during the migration
    to the new visualization engine. It should not be used directly
    by new code.
    """

    def __init__(self) -> None:
        self._engine = VisualizationEngine()

    def visualize(self, model: object, **kwargs: Any) -> None:
        """Visualize a model using the legacy ASCII renderer.

        Args:
            model: A Sequential model instance.
            **kwargs: Ignored for backwards compatibility.
        """
        output = self._engine.visualize(model, renderer_name="ascii")
        print(output)
