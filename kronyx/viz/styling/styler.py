"""Styler base class."""

from __future__ import annotations

from abc import ABC, abstractmethod

from kronyx.viz.graph import Scene
from kronyx.viz.styling.scene import StyledScene


class SceneStyler(ABC):
    """Abstract base class for scene stylers.

    A scene styler transforms a geometric Scene into a StyledScene
    by applying presentation properties: colors, fonts, borders,
    spacing, and other visual attributes.
    """

    @abstractmethod
    def apply(self, scene: Scene) -> StyledScene:
        """Apply presentation properties to a scene.

        Args:
            scene: Layout output from a layout engine.

        Returns:
            Styled scene ready for rendering.
        """
        ...
