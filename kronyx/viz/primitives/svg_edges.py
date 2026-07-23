"""SVG edge and arrow helpers."""

from __future__ import annotations

import xml.etree.ElementTree as ET


def draw_marker(
    parent: ET.Element,
    marker_id: str,
    size: int,
    color: str,
) -> ET.Element:
    """Define an arrowhead marker inside a <defs> block.

    Args:
        parent: Parent <defs> element.
        marker_id: Unique identifier for referencing the marker.
        size: Arrow size in pixels.
        color: Arrow fill color.

    Returns:
        The created marker Element.
    """
    marker = ET.SubElement(
        parent,
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
    return marker


def draw_connection(
    parent: ET.Element,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    marker_id: str,
    color: str,
    stroke_width: int,
) -> ET.Element:
    """Draw a directed connection line between two points.

    Args:
        parent: Parent SVG element.
        x1: Source x coordinate.
        y1: Source y coordinate.
        x2: Target x coordinate.
        y2: Target y coordinate.
        marker_id: Marker reference for the arrowhead.
        color: Line stroke color.
        stroke_width: Line width in pixels.

    Returns:
        The created line Element.
    """
    return ET.SubElement(
        parent,
        "line",
        {
            "x1": str(x1),
            "y1": str(y1),
            "x2": str(x2),
            "y2": str(y2),
            "stroke": color,
            "stroke-width": str(stroke_width),
            "marker-end": f"url(#{marker_id})",
        },
    )


def draw_arrow(
    parent: ET.Element,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    marker_id: str = "arrow",
    color: str = "#000",
    stroke_width: int = 2,
) -> ET.Element:
    """Draw a simple arrow between two points.

    Convenience wrapper around draw_connection with default values.

    Args:
        parent: Parent SVG element.
        x1: Source x coordinate.
        y1: Source y coordinate.
        x2: Target x coordinate.
        y2: Target y coordinate.
        marker_id: Marker reference for the arrowhead.
        color: Line stroke color.
        stroke_width: Line width in pixels.

    Returns:
        The created line Element.
    """
    return draw_connection(parent, x1, y1, x2, y2, marker_id, color, stroke_width)
