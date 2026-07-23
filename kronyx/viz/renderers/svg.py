"""SVG renderer for the visualization engine."""

from __future__ import annotations

from typing import Any

from kronyx.viz.primitives.svg_canvas import add_marker, create_canvas, serialize
from kronyx.viz.primitives.svg_edges import draw_arrow
from kronyx.viz.primitives.svg_shapes import draw_rounded_rectangle
from kronyx.viz.primitives.svg_text import draw_label
from kronyx.viz.renderers.base import Renderer
from kronyx.viz.styling.scene import StyledScene


class SvgRenderer(Renderer):
    """Renders a styled scene as an SVG diagram.

    This renderer is a pure painter. It has no styling logic,
    no theme logic, and no defaults. It simply draws whatever
    is described by the StyledScene.
    """

    def __init__(self, style=None, theme=None, **kwargs) -> None:
        """Initialize the renderer.

        Args:
            style: Deprecated. Accepted for backward compatibility only.
            theme: Deprecated. Accepted for backward compatibility only.
        """
        self.style = style
        self.theme = theme

    def render(self, scene: StyledScene, **kwargs: Any) -> str:
        """Render the styled scene as an SVG string.

        Args:
            scene: Styled scene produced by a SceneStyler.
            **kwargs: Ignored. Accepted for interface compatibility.

        Returns:
            SVG document as a string.
        """
        if not scene.nodes:
            canvas = create_canvas(200, 100, scene.background)
            text_el = _draw_text(
                canvas,
                100,
                55,
                "(Empty model)",
                "Arial",
                14,
                "normal",
                scene.text_color,
            )
            text_el.set("text-anchor", "middle")
            return serialize(canvas)

        canvas = create_canvas(
            int(scene.canvas_width),
            int(scene.canvas_height),
            scene.background,
        )

        if scene.edges:
            add_marker(
                canvas, "arrow", scene.edges[0].arrow_size, scene.edges[0].arrow_color
            )

        _draw_edges(canvas, scene)
        _draw_nodes(canvas, scene)

        return serialize(canvas)


def _draw_edges(canvas: Any, scene: StyledScene) -> None:
    """Draw arrow connections between nodes.

    Args:
        canvas: Root SVG element.
        scene: Styled scene.
    """
    for edge in scene.edges:
        draw_arrow(
            canvas,
            edge.x1,
            edge.y1,
            edge.x2,
            edge.y2,
            marker_id="arrow",
            color=edge.arrow_color,
            stroke_width=edge.stroke_width,
        )


def _draw_nodes(canvas: Any, scene: StyledScene) -> None:
    """Draw all nodes and their labels.

    Args:
        canvas: Root SVG element.
        scene: Styled scene.
    """
    for snode in scene.nodes:
        g = draw_rounded_rectangle(
            canvas,
            snode.x,
            snode.y,
            snode.width,
            snode.height,
            snode.border_radius,
            snode.fill_color,
            snode.stroke_color,
            snode.stroke_width,
        )

        if snode.icon:
            from kronyx.viz.primitives.svg_icons import draw_icon

            icon_x = snode.x + snode.width - snode.padding - 16
            icon_y = snode.y + snode.padding
            draw_icon(g, snode.icon, icon_x, icon_y, 16, snode.text_color)

        text_y = snode.y + snode.padding
        text_x = snode.x + snode.padding

        lines = [
            (snode.title, snode.title_font_size, "bold", snode.text_color),
            (snode.subtitle, snode.subtitle_font_size, "normal", snode.text_color),
        ]
        lines.extend(
            (meta, snode.body_font_size, "normal", snode.text_color)
            for meta in snode.metadata_lines
        )

        for text, font_size, weight, color in lines:
            line_height = _line_height(font_size)
            draw_label(
                g,
                text_x,
                text_y,
                text,
                snode.font_family,
                font_size,
                weight,
                color,
                line_height,
            )
            text_y += line_height


def _line_height(font_size: int) -> int:
    """Compute vertical spacing for a given font size."""
    if font_size == 16:
        return 22
    if font_size == 13:
        return 18
    return 16


def _draw_text(
    parent: Any,
    x: float,
    y: float,
    text: str,
    font_family: str,
    font_size: int,
    font_weight: str,
    color: str,
) -> Any:
    """Draw a single text element."""
    from kronyx.viz.primitives.svg_text import draw_text
    return draw_text(parent, x, y, text, font_family, font_size, font_weight, color)
