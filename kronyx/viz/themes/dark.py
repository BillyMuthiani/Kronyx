"""Dark theme implementation."""

from __future__ import annotations

from kronyx.viz.themes.base import Theme

DarkTheme = Theme(
    background_color="#1e1e1e",
    text_color="#e0e0e0",
    secondary_text_color="#a0a0a0",
    node_fill="#2d2d2d",
    node_border="#555555",
    node_border_width=2,
    node_radius=10,
    edge_color="#777777",
    edge_width=2,
    arrow_color="#777777",
    title_font_family="Arial",
    body_font_family="Arial",
    title_font_size=16,
    body_font_size=12,
    metadata_color="#a0a0a0",
    canvas_background="#1e1e1e",
)
