"""Base theme abstraction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    """Immutable theme dataclass for visualization appearance.

    Contains only appearance data. No renderer logic, no XML,
    no SVG. Future renderers (SVG, PNG, PDF, HTML, Interactive)
    should all consume this same Theme abstraction.
    """

    background_color: str = "#ffffff"
    text_color: str = "#000000"
    secondary_text_color: str = "#666666"

    node_fill: str = "#ffffff"
    node_border: str = "#000000"
    node_border_width: int = 2
    node_radius: int = 10

    edge_color: str = "#000000"
    edge_width: int = 2
    arrow_color: str = "#000000"

    title_font_family: str = "Arial"
    body_font_family: str = "Arial"
    title_font_size: int = 16
    body_font_size: int = 12

    metadata_color: str = "#666666"

    canvas_background: str = "#ffffff"

    shadow_color: str | None = None
    grid_color: str | None = None
    selection_color: str | None = None

    background: str = "#ffffff"
    title_color: str = "#000000"
    body_color: str = "#000000"
