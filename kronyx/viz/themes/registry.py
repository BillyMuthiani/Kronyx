"""Theme registry for the visualization engine."""

from __future__ import annotations

from kronyx.viz.themes.base import Theme


class ThemeRegistry:
    """Registry for visualization themes.

    Provides a central catalog for theme instances, allowing
    runtime lookup by name. Mirrors the RendererRegistry and
    LayoutRegistry API.
    """

    def __init__(self) -> None:
        self._themes: dict[str, Theme] = {}

    def register(self, name: str, theme: Theme) -> None:
        """Register a theme instance.

        Args:
            name: Theme name.
            theme: Theme instance.
        """
        self._themes[name] = theme

    def unregister(self, name: str) -> None:
        """Remove a theme from the registry.

        Args:
            name: Theme name.
        """
        self._themes.pop(name, None)

    def get(self, name: str) -> Theme:
        """Get a theme instance by name.

        Args:
            name: Theme name.

        Returns:
            Theme instance.

        Raises:
            KeyError: If the theme is not registered.
        """
        if name not in self._themes:
            available = ", ".join(sorted(self._themes.keys())) or "(none)"
            raise KeyError(
                f"No theme registered under name: {name}. "
                f"Available themes: {available}"
            )
        return self._themes[name]

    def available(self) -> list[str]:
        """List all registered theme names.

        Returns:
            Sorted list of theme names.
        """
        return sorted(self._themes.keys())

    def clear(self) -> None:
        """Remove all registered themes."""
        self._themes.clear()
