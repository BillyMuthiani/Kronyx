"""Default theme implementation."""

from __future__ import annotations

from kronyx.viz.themes.base import Theme

DefaultTheme = Theme(
    background_color="#ffffff",
    text_color="#000000",
    secondary_text_color="#666666",
    node_fill="#ffffff",
    node_border="#000000",
    node_border_width=2,
    node_radius=10,
    edge_color="#000000",
    edge_width=2,
    arrow_color="#000000",
    title_font_family="Arial",
    body_font_family="Arial",
    title_font_size=16,
    body_font_size=12,
    metadata_color="#666666",
    canvas_background="#ffffff",
)
