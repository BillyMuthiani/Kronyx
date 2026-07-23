"""Icon base classes for the visualization engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Icon:
    """Semantic icon definition for a model layer.

    Icons are renderer-independent. They carry no colors, no SVG paths,
    and no XML. Renderers decide how to paint an icon.
    """

    id: str
    label: str = ""
    category: str = "general"
