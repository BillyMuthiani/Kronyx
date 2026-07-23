"""Abstract base class for visualization renderers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from kronyx.viz.styling.scene import StyledScene


class Renderer(ABC):
    """Abstract base class for all visualization renderers."""

    @abstractmethod
    def render(self, scene: StyledScene, **kwargs: Any) -> str:
        """Render a styled scene into a string representation.

        Args:
            scene: The styled scene to render.
            **kwargs: Renderer-specific options.

        Returns:
            Rendered output as a string.
        """
        ...
