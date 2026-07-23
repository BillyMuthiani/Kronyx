"""Abstract base class for layout engines."""

from __future__ import annotations

from abc import ABC, abstractmethod

from kronyx.viz.graph import Graph, Scene
from kronyx.viz.style import Style


class LayoutEngine(ABC):
    """Abstract base class for layout engines.

    A layout engine converts a Graph into a Scene by computing
    node positions, edge paths, and canvas dimensions.
    """

    @abstractmethod
    def layout(self, graph: Graph, style: Style) -> Scene:
        """Compute layout for a model graph.

        Args:
            graph: Model architecture graph.
            style: Layout and typography constants.

        Returns:
            Populated Scene with positioned nodes and edges.
        """
        ...
