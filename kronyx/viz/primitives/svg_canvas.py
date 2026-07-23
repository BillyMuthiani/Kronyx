"""SVG canvas creation and serialization."""

from __future__ import annotations

import xml.etree.ElementTree as ET


def create_canvas(width: int, height: int, background: str = "#ffffff") -> ET.Element:
    """Create the root SVG element with namespace and dimensions.

    Args:
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        background: Background fill color.

    Returns:
        Root SVG Element.
    """
    svg = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": str(width),
            "height": str(height),
            "viewBox": f"0 0 {width} {height}",
        },
    )
    return svg


def create_empty_canvas(background: str = "#ffffff") -> ET.Element:
    """Create a small empty-state canvas.

    Args:
        background: Background fill color.

    Returns:
        Root SVG Element for an empty model.
    """
    return create_canvas(200, 100, background)


def add_marker(
    canvas: ET.Element,
    marker_id: str,
    size: int,
    color: str,
) -> None:
    """Add an arrow marker definition to the canvas.

    Args:
        canvas: Root SVG element.
        marker_id: Unique identifier for the marker.
        size: Arrow size in pixels.
        color: Arrow fill color.
    """
    defs = ET.SubElement(canvas, "defs")
    marker = ET.SubElement(
        defs,
        "marker",
        {
            "id": marker_id,
            "markerWidth": str(size),
            "markerHeight": str(size // 2),
            "refX": str(size - 1),
            "refY": str(size // 4),
            "orient": "auto",
            "markerUnits": "userSpaceOnUse",
        },
    )
    path_d = f"M 0 0 L {size} {size // 4} L 0 {size // 2} z"
    ET.SubElement(marker, "path", {"d": path_d, "fill": color})


def serialize(canvas: ET.Element) -> str:
    """Serialize an SVG element to a Unicode string.

    Args:
        canvas: Root SVG element.

    Returns:
        SVG document as a string.
    """
    return ET.tostring(canvas, encoding="unicode")
