"""Vertical layout engine for the visualization pipeline."""

from __future__ import annotations

from typing import Any

from kronyx.viz.graph import Graph, Node, PositionedEdge, PositionedNode, Scene
from kronyx.viz.layout.base import LayoutEngine
from kronyx.viz.style import Style


def _text_width(text: str, font_size: int) -> int:
    """Estimate pixel width of a text string.

    Args:
        text: Text to measure.
        font_size: Font size in pixels.

    Returns:
        Estimated width in pixels.
    """
    if font_size == 16:
        return int(len(text) * 8.5)
    if font_size == 13:
        return int(len(text) * 7.0)
    return int(len(text) * 6.5)


def _line_height(font_size: int) -> int:
    """Compute vertical spacing for a given font size.

    Args:
        font_size: Font size in pixels.

    Returns:
        Line height in pixels.
    """
    if font_size == 16:
        return 22
    if font_size == 13:
        return 18
    return 16


def _format_layer_type(layer_type: str) -> str:
    """Convert a layer class name into a human-readable type label.

    Args:
        layer_type: Original Kronyx layer class name.

    Returns:
        Human-readable layer type string.
    """
    if layer_type in ("ReLU", "Sigmoid", "Tanh", "Softmax"):
        return f"{layer_type} Activation"
    if layer_type == "BatchNormalization":
        return "Normalization Layer"
    if layer_type == "Conv2D":
        return "Convolutional Layer"
    if layer_type == "Input":
        return "Input Layer"
    if layer_type == "Output":
        return "Output Layer"
    return f"{layer_type} Layer"


def _format_metadata(metadata: dict[str, Any]) -> list[str]:
    """Extract display lines from layer metadata.

    Args:
        metadata: Layer configuration dictionary.

    Returns:
        List of formatted metadata strings.
    """
    lines = []
    for key, value in metadata.items():
        if key == "units":
            lines.append(f"Units: {value}")
        elif key == "filters":
            lines.append(f"Filters: {value}")
        elif key == "kernel_size":
            lines.append(f"Kernel: {value}")
        elif key == "stride":
            lines.append(f"Stride: {value}")
        elif key == "padding":
            lines.append(f"Padding: {value}")
        elif key == "rate":
            lines.append(f"Rate: {value}")
        elif key == "momentum":
            lines.append(f"Momentum: {value}")
        elif key == "epsilon":
            lines.append(f"Epsilon: {value}")
        elif key == "activation":
            lines.append(f"Type: {value}")
    return lines


def _calculate_node_size(node: Node, style: Style) -> tuple[int, int]:
    """Measure the pixel dimensions of a node.

    Args:
        node: Graph node.
        style: Layout and typography constants.

    Returns:
        Tuple of (width, height).
    """
    lines = [
        (node.name, style.title_font_size),
        (_format_layer_type(node.layer_type), style.subtitle_font_size),
    ]
    if node.output_shape != "?":
        lines.append((f"Shape: {node.output_shape}", style.body_font_size))
    lines.append((f"Params: {node.params}", style.body_font_size))
    for meta_line in _format_metadata(node.metadata):
        lines.append((meta_line, style.body_font_size))

    max_width = max(_text_width(text, size) for text, size in lines)
    width = max_width + 2 * style.node_padding
    height = sum(_line_height(size) for _, size in lines) + 2 * style.node_padding
    return width, height


class VerticalLayout(LayoutEngine):
    """Arranges nodes in a single vertical column.

    Computes node sizes, positions, edge endpoints, and canvas
    dimensions. Produces a Scene suitable for any renderer.
    """

    def __init__(self, style: Style | None = None) -> None:
        """Initialize the layout engine.

        Args:
            style: Optional style overrides. Uses DefaultStyle when omitted.
        """
        self.style = style or Style()

    def layout(self, graph: Graph, style: Style | None = None) -> Scene:
        """Compute layout for a model graph.

        Args:
            graph: Model architecture graph.
            style: Optional style overrides. Uses the instance style when omitted.

        Returns:
            Populated Scene with positioned nodes and edges.
        """
        active_style = style or self.style

        if not graph.nodes:
            return Scene(graph=graph)

        node_sizes = {node.id: _calculate_node_size(node, active_style) for node in graph.nodes}
        max_width = max(w for w, _ in node_sizes.values())
        center_x = active_style.canvas_margin + max_width // 2

        positions: dict[str, tuple[float, float]] = {}
        y = active_style.canvas_margin
        for node in graph.nodes:
            w, h = node_sizes[node.id]
            positions[node.id] = (center_x - w // 2, y)
            y += h + active_style.vertical_spacing

        canvas_width = max_width + 2 * active_style.canvas_margin
        canvas_height = float(y - active_style.vertical_spacing + active_style.canvas_margin)

        positioned_nodes = [
            PositionedNode(
                id=node.id,
                x=positions[node.id][0],
                y=positions[node.id][1],
                width=node_sizes[node.id][0],
                height=node_sizes[node.id][1],
                node=node,
            )
            for node in graph.nodes
        ]

        positioned_edges = []
        for edge in graph.edges:
            src_x, src_y = positions[edge.source]
            tgt_x, tgt_y = positions[edge.target]
            src_w, src_h = node_sizes[edge.source]
            tgt_w, _tgt_h = node_sizes[edge.target]

            positioned_edges.append(
                PositionedEdge(
                    source_id=edge.source,
                    target_id=edge.target,
                    x1=src_x + src_w / 2,
                    y1=src_y + src_h,
                    x2=tgt_x + tgt_w / 2,
                    y2=tgt_y,
                    edge=edge,
                )
            )

        return Scene(
            graph=graph,
            nodes=positioned_nodes,
            edges=positioned_edges,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )
