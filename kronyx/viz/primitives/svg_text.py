"""SVG text helpers."""

from __future__ import annotations

import xml.etree.ElementTree as ET


def measure_text(text: str, font_size: int) -> int:
    """Estimate the pixel width of a text string.

    Uses a simple character-width approximation.

    Args:
        text: Text to measure.
        font_size: Font size in pixels.

    Returns:
        Estimated width in pixels.
    """
    if font_size >= 16:
        return int(len(text) * 8.5)
    if font_size >= 13:
        return int(len(text) * 7.0)
    return int(len(text) * 6.5)


def draw_text(
    parent: ET.Element,
    x: float,
    y: float,
    text: str,
    font_family: str,
    font_size: int,
    font_weight: str,
    color: str,
) -> ET.Element:
    """Draw a single text element.

    Args:
        parent: Parent SVG element.
        x: Left coordinate.
        y: Baseline coordinate.
        text: Text content.
        font_family: Font family name.
        font_size: Font size in pixels.
        font_weight: Font weight (e.g. "normal", "bold").
        color: Text color.

    Returns:
        The created text Element.
    """
    text_el = ET.SubElement(
        parent,
        "text",
        {
            "x": str(x),
            "y": str(y),
            "font-family": font_family,
            "font-size": str(font_size),
            "font-weight": font_weight,
            "fill": color,
            "dominant-baseline": "middle",
        },
    )
    text_el.text = text
    return text_el


def draw_label(
    parent: ET.Element,
    x: float,
    y: float,
    text: str,
    font_family: str,
    font_size: int,
    font_weight: str,
    color: str,
    line_height: int,
) -> ET.Element:
    """Draw a text label at a computed vertical position.

    Convenience wrapper that centers the text vertically within
    its line height using dominant-baseline.

    Args:
        parent: Parent SVG element.
        x: Left coordinate.
        y: Top coordinate for the line.
        text: Text content.
        font_family: Font family name.
        font_size: Font size in pixels.
        font_weight: Font weight.
        color: Text color.
        line_height: Height of the line in pixels.

    Returns:
        The created text Element.
    """
    return draw_text(
        parent,
        x,
        y + line_height // 2,
        text,
        font_family,
        font_size,
        font_weight,
        color,
    )
