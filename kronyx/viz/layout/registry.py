"""Layout registry for the visualization engine."""

from __future__ import annotations


class LayoutRegistry:
    """Registry for visualization layout engines.

    Provides a central catalog for layout engines, allowing
    runtime lookup by name. Mirrors the RendererRegistry API.
    """

    def __init__(self) -> None:
        self._layouts: dict[str, type] = {}

    def register(self, name: str, layout_cls: type) -> None:
        """Register a layout engine class.

        Args:
            name: Layout name.
            layout_cls: Layout engine class.
        """
        self._layouts[name] = layout_cls

    def unregister(self, name: str) -> None:
        """Remove a layout from the registry.

        Args:
            name: Layout name.
        """
        self._layouts.pop(name, None)

    def get(self, name: str) -> type:
        """Get a layout engine class by name.

        Args:
            name: Layout name.

        Returns:
            Layout engine class.

        Raises:
            KeyError: If the layout is not registered.
        """
        if name not in self._layouts:
            available = ", ".join(sorted(self._layouts.keys())) or "(none)"
            raise KeyError(
                f"No layout registered under name: {name}. "
                f"Available layouts: {available}"
            )
        return self._layouts[name]

    def available(self) -> list[str]:
        """List all registered layout names.

        Returns:
            Sorted list of layout names.
        """
        return sorted(self._layouts.keys())

    def clear(self) -> None:
        """Remove all registered layouts."""
        self._layouts.clear()
