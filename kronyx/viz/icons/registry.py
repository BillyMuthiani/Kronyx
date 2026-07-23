"""Icon registry for the visualization engine."""

from __future__ import annotations

from kronyx.viz.icons.base import Icon


class IconRegistry:
    """Registry for visualization icons.

    Provides a central catalog for layer type → icon mappings,
    allowing runtime lookup by name. Mirrors the RendererRegistry,
    LayoutRegistry, and ThemeRegistry API.
    """

    def __init__(self) -> None:
        self._icons: dict[str, Icon] = {}

    def register(self, name: str, icon: Icon) -> None:
        """Register an icon.

        Args:
            name: Layer type name or alias.
            icon: Icon instance.
        """
        self._icons[name] = icon

    def unregister(self, name: str) -> None:
        """Remove an icon from the registry.

        Args:
            name: Layer type name or alias.
        """
        self._icons.pop(name, None)

    def get(self, name: str) -> Icon:
        """Get an icon by layer type name.

        Args:
            name: Layer type name.

        Returns:
            Icon instance.

        Raises:
            KeyError: If the icon is not registered.
        """
        if name not in self._icons:
            available = ", ".join(sorted(self._icons.keys())) or "(none)"
            raise KeyError(
                f"No icon registered under name: {name}. "
                f"Available icons: {available}"
            )
        return self._icons[name]

    def available(self) -> list[str]:
        """List all registered icon names.

        Returns:
            Sorted list of icon names.
        """
        return sorted(self._icons.keys())

    def clear(self) -> None:
        """Remove all registered icons."""
        self._icons.clear()
