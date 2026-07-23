"""Scene dataclasses for the styling layer."""

from __future__ import annotations

from dataclasses import dataclass, field

from kronyx.viz.graph import Graph


@dataclass
class StyledNode:
    """A node with presentation properties applied.

    Attributes:
        node_id: Node identifier.
        x: Left coordinate in pixels.
        y: Top coordinate in pixels.
        width: Node width in pixels.
        height: Node height in pixels.
        fill_color: Node background color.
        stroke_color: Node border color.
        stroke_width: Node border width in pixels.
        border_radius: Node corner radius in pixels.
        title: Primary text (layer name).
        subtitle: Secondary text (layer type).
        metadata_lines: Additional metadata text lines.
        font_family: Font family for all text.
        title_font_size: Font size for title text.
        subtitle_font_size: Font size for subtitle text.
        body_font_size: Font size for metadata text.
        text_color: Text color for all labels.
        padding: Internal padding inside node rectangle.
        tags: Styling and classification hooks.
        shadow: Whether to render a drop shadow.
        icon: Optional icon identifier.
        badges: Optional badge text list.
    """

    node_id: str
    x: float
    y: float
    width: float
    height: float
    fill_color: str
    stroke_color: str
    stroke_width: int
    border_radius: int
    title: str
    subtitle: str
    metadata_lines: list[str]
    font_family: str
    title_font_size: int
    subtitle_font_size: int
    body_font_size: int
    text_color: str
    padding: int
    tags: set[str] = field(default_factory=set)
    shadow: bool = False
    icon: str | None = None
    badges: list[str] = field(default_factory=list)


@dataclass
class StyledEdge:
    """An edge with presentation properties applied.

    Attributes:
        source_id: Source node identifier.
        target_id: Target node identifier.
        x1: Source x coordinate.
        y1: Source y coordinate.
        x2: Target x coordinate.
        y2: Target y coordinate.
        stroke_color: Edge line color.
        stroke_width: Edge line width in pixels.
        arrow_size: Arrowhead marker size in pixels.
        arrow_color: Arrowhead fill color.
        style: Line style (solid, dashed, etc.).
    """

    source_id: str
    target_id: str
    x1: float
    y1: float
    x2: float
    y2: float
    stroke_color: str
    stroke_width: int
    arrow_size: int
    arrow_color: str
    style: str = "solid"


@dataclass
class StyledScene:
    """Presentation output consumed by renderers.

    Attributes:
        graph: Source model architecture graph.
        nodes: Styled nodes in render order.
        edges: Styled edges in render order.
        canvas_width: Total canvas width in pixels.
        canvas_height: Total canvas height in pixels.
        background: Canvas background color.
        text_color: Default text color for fallback rendering.
    """

    graph: Graph
    nodes: list[StyledNode] = field(default_factory=list)
    edges: list[StyledEdge] = field(default_factory=list)
    canvas_width: float = 0.0
    canvas_height: float = 0.0
    background: str = "#ffffff"
    text_color: str = "#000000"
