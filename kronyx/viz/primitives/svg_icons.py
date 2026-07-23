"""SVG icon rendering primitives."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any


def draw_icon(
    parent: Any,
    icon_id: str,
    x: float,
    y: float,
    size: int,
    color: str,
) -> None:
    """Draw a semantic icon inside a parent SVG element.

    Args:
        parent: Parent SVG element.
        icon_id: Icon identifier (e.g. 'dense', 'relu', 'input').
        x: Left coordinate.
        y: Top coordinate.
        size: Icon size in pixels.
        color: Icon stroke/fill color.
    """
    if icon_id == "input":
        _draw_circle(parent, x, y, size, color, fill=False)
    elif icon_id == "output":
        _draw_circle(parent, x, y, size, color, fill=True)
    elif icon_id == "dense":
        _draw_rect(parent, x, y, size, size, color, radius=2)
    elif icon_id == "conv2d":
        _draw_grid(parent, x, y, size, color)
    elif icon_id == "conv1d":
        _draw_rect(parent, x, y, size, size, color, radius=2)
        _draw_line(parent, x, y + size // 2, x + size, y + size // 2, color)
    elif icon_id == "flatten":
        _draw_rect(parent, x, y, size, size, color, radius=2)
        _draw_arrow_right(parent, x + size // 2, y + size // 2, size // 2, color)
    elif icon_id == "dropout":
        _draw_rect(parent, x, y, size, size, color, radius=2)
        _draw_arrow_down(parent, x + size // 2, y + size // 4, size // 2, color)
    elif icon_id == "batch_norm":
        _draw_circle(parent, x, y, size, color, fill=True)
        _draw_circle(
            parent, x + size // 4, y + size // 4, size // 8, color, fill=True
        )
    elif icon_id in ("relu", "sigmoid", "softmax", "tanh", "leaky_relu"):
        _draw_lightning(parent, x, y, size, color)
    elif icon_id == "max_pool":
        _draw_rect(parent, x, y + 1, size, size // 2 - 1, color, radius=1)
        _draw_rect(
            parent, x, y + size // 2 + 1, size, size // 2 - 1, color, radius=1
        )
    elif icon_id == "avg_pool":
        _draw_rect(parent, x, y + 1, size, size // 2 - 1, color, radius=1)
        _draw_rect(
            parent, x, y + size // 2 + 1, size, size // 2 - 1, color, radius=1
        )
        _draw_line(parent, x, y + size // 2, x + size, y + size // 2, color)
    elif icon_id == "unknown":
        _draw_diamond(parent, x, y, size, color)
    else:
        _draw_diamond(parent, x, y, size, color)


def _draw_circle(
    parent: Any, x: float, y: float, size: int, color: str, fill: bool
) -> None:
    cx = x + size / 2
    cy = y + size / 2
    r = size / 2 - 1
    attrs = {"cx": str(cx), "cy": str(cy), "r": str(r), "stroke": color}
    if fill:
        attrs["fill"] = color
    else:
        attrs["fill"] = "none"
    attrs["stroke-width"] = "1.5"
    ET.SubElement(parent, "circle", attrs)


def _draw_rect(
    parent: Any,
    x: float,
    y: float,
    w: int,
    h: int,
    color: str,
    radius: int = 0,
) -> None:
    ET.SubElement(
        parent,
        "rect",
        {
            "x": str(x),
            "y": str(y),
            "width": str(w),
            "height": str(h),
            "stroke": color,
            "stroke-width": "1.5",
            "fill": "none",
            "rx": str(radius),
            "ry": str(radius),
        },
    )


def _draw_line(
    parent: Any, x1: float, y1: float, x2: float, y2: float, color: str
) -> None:
    ET.SubElement(
        parent,
        "line",
        {
            "x1": str(x1),
            "y1": str(y1),
            "x2": str(x2),
            "y2": str(y2),
            "stroke": color,
            "stroke-width": "1.5",
        },
    )


def _draw_grid(parent: Any, x: float, y: float, size: int, color: str) -> None:
    _draw_rect(parent, x, y, size, size, color, radius=1)
    step = size // 3
    for i in (step, 2 * step):
        _draw_line(parent, x + i, y, x + i, y + size, color)
        _draw_line(parent, x, y + i, x + size, y + i, color)


def _draw_arrow_right(
    parent: Any, cx: float, cy: float, half_size: int, color: str
) -> None:
    _draw_line(
        parent,
        cx - half_size // 2,
        cy,
        cx + half_size // 2,
        cy,
        color,
    )
    _draw_line(
        parent,
        cx + half_size // 2,
        cy - half_size // 3,
        cx + half_size // 2 + 3,
        cy,
        color,
    )
    _draw_line(
        parent,
        cx + half_size // 2,
        cy + half_size // 3,
        cx + half_size // 2 + 3,
        cy,
        color,
    )


def _draw_arrow_down(
    parent: Any, cx: float, cy: float, half_size: int, color: str
) -> None:
    _draw_line(
        parent,
        cx,
        cy - half_size // 2,
        cx,
        cy + half_size // 2,
        color,
    )
    _draw_line(
        parent,
        cx - half_size // 3,
        cy + half_size // 2,
        cx,
        cy + half_size // 2 + 3,
        color,
    )
    _draw_line(
        parent,
        cx + half_size // 3,
        cy + half_size // 2,
        cx,
        cy + half_size // 2 + 3,
        color,
    )


def _draw_lightning(parent: Any, x: float, y: float, size: int, color: str) -> None:
    points = (
        f"{x + size * 0.4},{y + 2} "
        f"{x + size * 0.1},{y + size * 0.5} "
        f"{x + size * 0.4},{y + size * 0.5} "
        f"{x + size * 0.6},{y + size - 2} "
        f"{x + size * 0.9},{y + size * 0.5} "
        f"{x + size * 0.6},{y + size * 0.5}"
    )
    ET.SubElement(
        parent,
        "polyline",
        {
            "points": points,
            "stroke": color,
            "stroke-width": "1.5",
            "fill": "none",
            "stroke-linejoin": "round",
        },
    )


def _draw_diamond(parent: Any, x: float, y: float, size: int, color: str) -> None:
    cx = x + size / 2
    cy = y + size / 2
    points = (
        f"{cx},{y + 1} "
        f"{x + size - 1},{cy} "
        f"{cx},{y + size - 1} "
        f"{x + 1},{cy}"
    )
    ET.SubElement(
        parent,
        "polygon",
        {
            "points": points,
            "stroke": color,
            "stroke-width": "1.5",
            "fill": "none",
        },
    )
