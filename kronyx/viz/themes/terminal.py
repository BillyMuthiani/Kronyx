"""Terminal theme implementation."""

from __future__ import annotations

from kronyx.viz.themes.base import Theme

TerminalTheme = Theme(
    background_color="#0c0c0c",
    text_color="#00ff00",
    secondary_text_color="#00aa00",
    node_fill="#0c0c0c",
    node_border="#00ff00",
    node_border_width=2,
    node_radius=0,
    edge_color="#00ff00",
    edge_width=2,
    arrow_color="#00ff00",
    title_font_family="Courier New",
    body_font_family="Courier New",
    title_font_size=16,
    body_font_size=12,
    metadata_color="#00aa00",
    canvas_background="#0c0c0c",
)
