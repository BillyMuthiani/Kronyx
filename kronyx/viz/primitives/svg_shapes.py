"""SVG shape drawing helpers."""

from __future__ import annotations

import xml.etree.ElementTree as ET


def draw_rectangle(
    parent: ET.Element,
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str,
    stroke: str,
    border_width: int,
) -> ET.Element:
    """Draw a plain rectangle.

    Args:
        parent: Parent SVG element.
        x: Left coordinate.
        y: Top coordinate.
        width: Rectangle width.
        height: Rectangle height.
        fill: Fill color.
        stroke: Border color.
        border_width: Border width in pixels.

    Returns:
        The created rect Element.
    """
    return ET.SubElement(
        parent,
        "rect",
        {
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height),
            "fill": fill,
            "stroke": stroke,
            "stroke-width": str(border_width),
        },
    )


def draw_rounded_rectangle(
    parent: ET.Element,
    x: float,
    y: float,
    width: float,
    height: float,
    radius: int,
    fill: str,
    stroke: str,
    border_width: int,
) -> ET.Element:
    """Draw a rectangle with rounded corners.

    Args:
        parent: Parent SVG element.
        x: Left coordinate.
        y: Top coordinate.
        width: Rectangle width.
        height: Rectangle height.
        radius: Corner radius in pixels.
        fill: Fill color.
        stroke: Border color.
        border_width: Border width in pixels.

    Returns:
        The created rect Element.
    """
    return ET.SubElement(
        parent,
        "rect",
        {
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height),
            "rx": str(radius),
            "ry": str(radius),
            "fill": fill,
            "stroke": stroke,
            "stroke-width": str(border_width),
        },
    )


def draw_border(
    parent: ET.Element,
    x: float,
    y: float,
    width: float,
    height: float,
    radius: int,
    stroke: str,
    border_width: int,
) -> ET.Element:
    """Draw a border-only rounded rectangle (no fill).

    Args:
        parent: Parent SVG element.
        x: Left coordinate.
        y: Top coordinate.
        width: Rectangle width.
        height: Rectangle height.
        radius: Corner radius in pixels.
        stroke: Border color.
        border_width: Border width in pixels.

    Returns:
        The created rect Element.
    """
    return ET.SubElement(
        parent,
        "rect",
        {
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height),
            "rx": str(radius),
            "ry": str(radius),
            "fill": "none",
            "stroke": stroke,
            "stroke-width": str(border_width),
        },
    )
