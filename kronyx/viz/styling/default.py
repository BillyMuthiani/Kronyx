"""Default scene styler implementation."""

from __future__ import annotations

from typing import Any

from kronyx.viz.graph import Scene
from kronyx.viz.icons.registry import IconRegistry
from kronyx.viz.style import DefaultStyle
from kronyx.viz.styling.scene import StyledEdge, StyledNode, StyledScene
from kronyx.viz.styling.styler import SceneStyler
from kronyx.viz.themes import DefaultTheme


def _format_layer_type(layer_type: str) -> str:
    """Convert a layer class name into a human-readable type label."""
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
    """Extract display lines from layer metadata."""
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


class DefaultSceneStyler(SceneStyler):
    """Default scene styler applying standard Kronyx presentation.

    Applies colors, fonts, borders, padding, and text formatting
    from Theme and Style dataclasses.
    """

    def __init__(self, style=None, theme=None, icon_registry=None) -> None:
        """Initialize the styler.

        Args:
            style: Optional style overrides. Uses DefaultStyle when omitted.
            theme: Optional theme overrides. Uses DefaultTheme when omitted.
            icon_registry: Optional icon registry. Uses built-in icons when omitted.
        """
        self._style = style or DefaultStyle
        self._theme = theme or DefaultTheme
        self._icon_registry = icon_registry or _create_default_icon_registry()

    def apply(self, scene: Scene) -> StyledScene:
        """Apply presentation properties to a scene.

        Args:
            scene: Layout output from a layout engine.

        Returns:
            Styled scene ready for rendering.
        """
        styled_nodes = [
            self._style_node(pnode) for pnode in scene.nodes
        ]

        styled_edges = [
            StyledEdge(
                source_id=edge.source_id,
                target_id=edge.target_id,
                x1=edge.x1,
                y1=edge.y1,
                x2=edge.x2,
                y2=edge.y2,
                stroke_color=self._theme.edge_color,
                stroke_width=self._theme.edge_width,
                arrow_size=self._style.arrow_size,
                arrow_color=self._theme.arrow_color,
            )
            for edge in scene.edges
        ]

        return StyledScene(
            graph=scene.graph,
            nodes=styled_nodes,
            edges=styled_edges,
            canvas_width=scene.canvas_width,
            canvas_height=scene.canvas_height,
            background=self._theme.canvas_background,
            text_color=self._theme.text_color,
        )

    def _style_node(self, pnode: Any) -> StyledNode:
        """Apply presentation properties to a single node."""
        title = pnode.node.name
        subtitle = _format_layer_type(pnode.node.layer_type)
        metadata_lines = []
        if pnode.node.output_shape != "?":
            metadata_lines.append(f"Shape: {pnode.node.output_shape}")
        metadata_lines.append(f"Params: {pnode.node.params}")
        metadata_lines.extend(_format_metadata(pnode.node.metadata))

        return StyledNode(
            node_id=pnode.id,
            x=pnode.x,
            y=pnode.y,
            width=pnode.width,
            height=pnode.height,
            fill_color=self._theme.node_fill,
            stroke_color=self._theme.node_border,
            stroke_width=self._theme.node_border_width,
            border_radius=self._theme.node_radius,
            title=title,
            subtitle=subtitle,
            metadata_lines=metadata_lines,
            font_family=self._theme.title_font_family,
            title_font_size=self._theme.title_font_size,
            subtitle_font_size=self._style.subtitle_font_size,
            body_font_size=self._theme.body_font_size,
            text_color=self._theme.text_color,
            padding=self._style.node_padding,
            tags=pnode.node.tags,
            icon=_resolve_icon(pnode.node.layer_type, self._icon_registry),
        )


def _resolve_icon(layer_type: str, registry: IconRegistry) -> str | None:
    """Resolve the icon id for a layer type.

    Args:
        layer_type: Layer class name.
        registry: Icon registry.

    Returns:
        Icon id string, or None if unavailable.
    """
    try:
        return registry.get(layer_type).id
    except KeyError:
        if "UnknownLayer" in registry.available():
            return registry.get("UnknownLayer").id
        return None


def _create_default_icon_registry() -> IconRegistry:
    """Create an icon registry with built-in icons."""
    from kronyx.viz.icons.builtin import register_builtin_icons

    registry = IconRegistry()
    register_builtin_icons(registry)
    return registry
