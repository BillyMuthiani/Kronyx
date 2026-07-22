"""Graph builder for converting Sequential models to Graph objects."""

from __future__ import annotations

from typing import Any

from kronyx.viz.graph import Edge, Graph, Node


class GraphBuilder:
    """Builds a Graph from a Sequential model.

    Reuses existing Sequential helper methods for introspection.
    Does not compute coordinates or render output.
    """

    def build(self, model: object) -> Graph:
        """Build a Graph from a model.

        Args:
            model: A Sequential model instance.

        Returns:
            Graph representing the model architecture.
        """
        layers = getattr(model, "layers", [])
        nodes: list[Node] = []
        edges: list[Edge] = []

        input_node = Node(
            id="input",
            name="Input",
            layer_type="Input",
            params=0,
            output_shape="?",
            tags={"input"},
        )
        nodes.append(input_node)

        class_counts: dict[str, int] = {}
        previous_shape: tuple | None = None
        prev_id = "input"

        for i, layer in enumerate(layers):
            layer_type_name = type(layer).__name__

            base_name = self._get_layer_type_name(model, layer)
            count = class_counts.get(base_name, 0)
            class_counts[base_name] = count + 1
            display_name = base_name if count == 0 else f"{base_name}_{count}"

            params = self._get_layer_param_count(model, layer)
            breakdown = self._get_layer_param_breakdown(model, layer, previous_shape)
            output_shape = self._infer_output_shape(model, layer, previous_shape)

            tags = self._build_tags(layer_type_name, params)
            metadata = self._build_metadata(layer)

            node_id = f"layer_{i}"
            node = Node(
                id=node_id,
                name=display_name,
                layer_type=layer_type_name,
                params=params,
                param_breakdown=breakdown if breakdown not in ("0", "?") else None,
                output_shape=output_shape,
                metadata=metadata,
                tags=tags,
            )
            nodes.append(node)

            edges.append(Edge(source=prev_id, target=node_id))
            prev_id = node_id

            if output_shape != "?":
                parsed = self._parse_shape_for_next_layer(model, output_shape)
                if parsed is not None:
                    previous_shape = parsed

        output_node = Node(
            id="output",
            name="Prediction",
            layer_type="Output",
            params=0,
            output_shape="?",
            tags={"output"},
        )
        nodes.append(output_node)
        edges.append(Edge(source=prev_id, target="output"))

        return Graph(
            nodes=nodes,
            edges=edges,
            metadata={},
        )

    def _build_tags(self, layer_type_name: str, params: int) -> set[str]:
        """Build the tag set for a layer node.

        Args:
            layer_type_name: Kronyx layer class name.
            params: Parameter count for the layer.

        Returns:
            Set of tag strings.
        """
        tags: set[str] = set()
        if layer_type_name in ("ReLU", "Sigmoid", "Tanh", "Softmax"):
            tags.add("activation")
        if layer_type_name == "BatchNormalization":
            tags.add("normalization")
        if layer_type_name in ("Dropout",):
            tags.add("regularization")
        if layer_type_name == "Conv2D":
            tags.add("convolution")
        if layer_type_name == "Dense":
            tags.add("dense")
        if layer_type_name == "Flatten":
            tags.add("flatten")
        if params > 0:
            tags.add("trainable")
        return tags

    def _build_metadata(self, layer: Any) -> dict[str, Any]:
        """Extract layer metadata without forward pass.

        Args:
            layer: A Kronyx layer instance.

        Returns:
            Dictionary of metadata fields available from the layer.
        """
        layer_type = type(layer).__name__
        metadata: dict[str, Any] = {}

        if layer_type == "Dense":
            if hasattr(layer, "weights") and layer.weights is not None:
                metadata["units"] = int(layer.weights.shape[1])
            elif hasattr(layer, "biases") and layer.biases is not None:
                metadata["units"] = int(layer.biases.shape[1])

        elif layer_type == "Conv2D":
            metadata["filters"] = getattr(layer, "filters", None)
            metadata["kernel_size"] = getattr(layer, "kernel_size", None)
            metadata["stride"] = getattr(layer, "stride", None)
            metadata["padding"] = getattr(layer, "padding", None)

        elif layer_type == "Dropout":
            metadata["rate"] = getattr(layer, "rate", None)

        elif layer_type == "BatchNormalization":
            metadata["momentum"] = getattr(layer, "momentum", None)
            metadata["epsilon"] = getattr(layer, "epsilon", None)

        elif layer_type in ("ReLU", "Sigmoid", "Tanh", "Softmax"):
            metadata["activation"] = layer_type

        return metadata

    def _get_layer_type_name(self, model: Any, layer: Any) -> str:
        """Get display type name for a layer."""
        method = getattr(model, "_get_layer_type_name", None)
        if callable(method):
            return method(layer)  # type: ignore[no-any-return]
        layer_class = type(layer).__name__
        if layer_class == "BatchNormalization":
            return "BatchNorm"
        return layer_class

    def _get_layer_param_count(self, model: Any, layer: Any) -> int:
        """Count parameters for a layer."""
        method = getattr(model, "_get_layer_param_count", None)
        if callable(method):
            return method(layer)  # type: ignore[no-any-return]
        if hasattr(layer, "weights") and layer.weights is not None:
            return int(layer.weights.size + layer.biases.size)
        if hasattr(layer, "kernels") and layer.kernels is not None:
            return int(layer.kernels.size + layer.biases.size)
        return 0

    def _get_layer_param_breakdown(
        self, model: Any, layer: Any, logical_shape: tuple | None
    ) -> str:
        """Get parameter breakdown string for a layer.

        Returns only the breakdown portion (e.g. "32W + 16B"),
        not the total count. The total count is stored separately
        in Node.params.
        """
        method = getattr(model, "_get_layer_param_breakdown", None)
        if callable(method):
            raw = method(layer, logical_shape)  # type: ignore[no-any-return]
            if raw.startswith("?") or raw == "0":
                return raw  # type: ignore[no-any-return]
            if "(" in raw and raw.endswith(")"):
                return raw[raw.index("(") + 1:-1]  # type: ignore[no-any-return]
            return raw  # type: ignore[no-any-return]
        layer_type = type(layer).__name__
        if layer_type == "Dense":
            if hasattr(layer, "weights") and layer.weights is not None:
                w_count = int(layer.weights.size)
                b_count = int(layer.biases.size)
                return f"{w_count}W + {b_count}B"
            return "0"
        if layer_type == "Conv2D":
            if getattr(layer, "kernels", None) is not None:
                kh, kw, in_ch, filters = layer.kernels.shape
                w_count = kh * kw * in_ch * filters
                b_count = int(layer.biases.size)
                return f"{w_count}W + {b_count}B"
            if logical_shape is not None and len(logical_shape) >= 3:
                in_ch = logical_shape[2]
                kernel_size = getattr(layer, "kernel_size", 0)
                filters = getattr(layer, "filters", 0)
                if isinstance(kernel_size, int):
                    kernel_h = kernel_w = kernel_size
                else:
                    kernel_h, kernel_w = kernel_size
                w_count = kernel_h * kernel_w * in_ch * filters
                b_count = filters
                return f"{w_count}W + {b_count}B"
            return "?"
        if layer_type == "BatchNormalization":
            if hasattr(layer, "gamma") and layer.gamma is not None:
                num_features = int(layer.gamma.size)
                return f"{num_features}\u03b3 + {num_features}\u03b2"
            if logical_shape is not None and len(logical_shape) >= 1:
                num_features = logical_shape[-1]
                return f"{num_features}\u03b3 + {num_features}\u03b2"
            return "0"
        return "0"

    def _infer_output_shape(self, model: Any, layer: Any, logical_shape: tuple | None) -> str:
        """Infer output shape for a layer without forward pass."""
        method = getattr(model, "_infer_output_shape", None)
        if callable(method):
            return method(layer, logical_shape)  # type: ignore[no-any-return]
        if logical_shape is None:
            return "?"
        layer_type = type(layer).__name__
        if layer_type == "Dense":
            if hasattr(layer, "biases") and layer.biases is not None:
                return f"(None, {layer.biases.shape[1]})"
            if hasattr(layer, "weights") and layer.weights is not None:
                return f"(None, {layer.weights.shape[1]})"
            return "?"
        if layer_type == "Flatten":
            if len(logical_shape) >= 1:
                features = 1
                for dim in logical_shape:
                    features *= dim
                return f"(None, {features})"
            return "?"
        if layer_type in ("Dropout", "BatchNormalization", "ReLU", "Sigmoid", "Tanh", "Softmax"):
            parts = ", ".join(str(dim) for dim in logical_shape)
            return f"(None, {parts})"
        return "?"

    def _parse_shape_for_next_layer(self, model: Any, shape_str: str) -> tuple | None:
        """Parse a display shape string to get the logical shape for next layer."""
        method = getattr(model, "_parse_shape_for_next_layer", None)
        if callable(method):
            return method(shape_str)  # type: ignore[no-any-return]
        if shape_str == "?" or not shape_str.startswith("(None, "):
            return None
        inner = shape_str[len("(None, "):-1]
        try:
            dims = tuple(int(d.strip()) for d in inner.split(","))
        except ValueError:
            return None
        return dims
