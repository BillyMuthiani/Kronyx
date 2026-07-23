"""Neon theme implementation."""

from __future__ import annotations

from kronyx.viz.themes.base import Theme

NeonTheme = Theme(
    background_color="#0a0a0a",
    text_color="#ffffff",
    secondary_text_color="#aaaaaa",
    node_fill="#111111",
    node_border="#ff00ff",
    node_border_width=2,
    node_radius=10,
    edge_color="#00ffff",
    edge_width=2,
    arrow_color="#00ffff",
    title_font_family="Arial",
    body_font_family="Arial",
    title_font_size=16,
    body_font_size=12,
    metadata_color="#aaaaaa",
    canvas_background="#0a0a0a",
)
