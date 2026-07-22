"""Graph data structures for the visualization engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    """Represents a single layer within a model graph.

    Attributes:
        id: Unique node identifier.
        name: Human-readable display name.
        layer_type: Original Kronyx layer class name.
        params: Total trainable parameter count.
        param_breakdown: Optional human-readable breakdown
            (e.g. "32W + 16B"), or "0", or "?".
        output_shape: Inferred output shape string.
        metadata: Type-specific layer configuration.
        tags: Styling and classification hooks.
    """

    id: str
    name: str
    layer_type: str
    params: int
    param_breakdown: str | None = None
    output_shape: str = "?"
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: set[str] = field(default_factory=set)


@dataclass
class Edge:
    """Directed connection between two nodes.

    Attributes:
        source: Source node id.
        target: Target node id.
    """

    source: str
    target: str


@dataclass
class Graph:
    """Immutable container for a model architecture graph.

    Attributes:
        nodes: Ordered list of nodes (input first, output last).
        edges: Directed edges connecting consecutive nodes.
        metadata: Optional graph-level statistics and metadata.
    """

    nodes: list[Node]
    edges: list[Edge]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return a concise representation suitable for large graphs."""
        parts = [f"nodes={len(self.nodes)}", f"edges={len(self.edges)}"]
        if self.metadata:
            parts.append(f"metadata={self.metadata}")
        return f"Graph({', '.join(parts)})"

    @property
    def num_nodes(self) -> int:
        """Total number of nodes in the graph."""
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        """Total number of edges in the graph."""
        return len(self.edges)

    @property
    def num_trainable_layers(self) -> int:
        """Number of nodes that contain trainable parameters."""
        return sum(1 for node in self.nodes if "trainable" in node.tags)

    @property
    def num_activation_layers(self) -> int:
        """Number of activation layer nodes."""
        return sum(1 for node in self.nodes if "activation" in node.tags)

    @property
    def total_params(self) -> int:
        """Sum of trainable parameters across all nodes."""
        return sum(node.params for node in self.nodes)


@dataclass
class Scene:
    """Placeholder for future layout and positioning data.

    Attributes:
        graph: Source graph.
        node_positions: Mapping of node id to (x, y) coordinates.
        edge_paths: Mapping of edge id to ordered path points.
        canvas_size: Width and height of the rendered canvas.
    """

    graph: Graph
    node_positions: dict[str, tuple[float, float]] = field(default_factory=dict)
    edge_paths: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    canvas_size: tuple[float, float] = (0.0, 0.0)
