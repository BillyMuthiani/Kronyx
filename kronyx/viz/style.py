"""Visualization style constants."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Style:
    """Layout and typography constants for SVG rendering.

    Attributes:
        node_padding: Internal padding inside each node rectangle.
        border_radius: Corner radius for node rectangles.
        border_width: Width of node border strokes.
        title_font_size: Font size for layer names.
        subtitle_font_size: Font size for layer type labels.
        body_font_size: Font size for metadata fields.
        arrow_size: Length of arrowhead markers.
        vertical_spacing: Space between consecutive nodes.
        horizontal_spacing: Horizontal space between columns (reserved for future layouts).
        canvas_margin: Margin around the outermost node.
    """

    node_padding: int = 12
    border_radius: int = 10
    border_width: int = 2
    title_font_size: int = 16
    subtitle_font_size: int = 13
    body_font_size: int = 12
    arrow_size: int = 12
    vertical_spacing: int = 80
    horizontal_spacing: int = 0
    canvas_margin: int = 40


DefaultStyle = Style()
