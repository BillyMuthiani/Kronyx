"""Renderer registry for the visualization engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class RendererRegistry:
    """Registry for visualization renderers."""

    def __init__(self) -> None:
        self._renderers: dict[str, type] = {}

    def register(self, name: str, renderer_cls: type) -> None:
        """Register a renderer class.

        Args:
            name: Renderer name.
            renderer_cls: Renderer class.
        """
        self._renderers[name] = renderer_cls

    def unregister(self, name: str) -> None:
        """Remove a renderer from the registry.

        Args:
            name: Renderer name.
        """
        self._renderers.pop(name, None)

    def get(self, name: str) -> type:
        """Get a renderer class by name.

        Args:
            name: Renderer name.

        Returns:
            Renderer class.

        Raises:
            KeyError: If the renderer is not registered.
        """
        if name not in self._renderers:
            raise KeyError(f"No renderer registered under name: {name}")
        return self._renderers[name]

    def available(self) -> list[str]:
        """List all registered renderer names.

        Returns:
            Sorted list of renderer names.
        """
        return sorted(self._renderers.keys())
