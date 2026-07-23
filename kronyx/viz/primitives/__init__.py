"""Low-level SVG drawing primitives.

These helpers know nothing about Kronyx models, graphs, or layers.
They operate purely on SVG elements and geometry.
"""

from kronyx.viz.primitives.svg_canvas import (
    add_marker,
    create_canvas,
    create_empty_canvas,
    serialize,
)
from kronyx.viz.primitives.svg_edges import (
    draw_arrow,
    draw_connection,
    draw_marker,
)
from kronyx.viz.primitives.svg_shapes import (
    draw_border,
    draw_rectangle,
    draw_rounded_rectangle,
)
from kronyx.viz.primitives.svg_text import (
    draw_label,
    draw_text,
    measure_text,
)

__all__ = [
    "create_canvas",
    "create_empty_canvas",
    "add_marker",
    "serialize",
    "draw_rectangle",
    "draw_rounded_rectangle",
    "draw_border",
    "draw_text",
    "draw_label",
    "measure_text",
    "draw_marker",
    "draw_connection",
    "draw_arrow",
]
