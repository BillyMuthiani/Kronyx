"""Blueprint theme implementation."""

from __future__ import annotations

from kronyx.viz.themes.base import Theme

BlueprintTheme = Theme(
    background_color="#1a3a5c",
    text_color="#e0f0ff",
    secondary_text_color="#a0c0e0",
    node_fill="#234b6e",
    node_border="#4a90d9",
    node_border_width=2,
    node_radius=10,
    edge_color="#4a90d9",
    edge_width=2,
    arrow_color="#4a90d9",
    title_font_family="Arial",
    body_font_family="Arial",
    title_font_size=16,
    body_font_size=12,
    metadata_color="#a0c0e0",
    canvas_background="#1a3a5c",
)
