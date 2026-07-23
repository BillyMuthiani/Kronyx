"""Built-in icon definitions."""

from __future__ import annotations

from kronyx.viz.icons.base import Icon
from kronyx.viz.icons.registry import IconRegistry


def _create_builtin_registry() -> IconRegistry:
    """Create a registry pre-populated with built-in icons."""
    registry = IconRegistry()

    registry.register("Input", Icon(id="input", label="Input", category="io"))
    registry.register("Output", Icon(id="output", label="Output", category="io"))
    registry.register("Dense", Icon(id="dense", label="Dense", category="layer"))
    registry.register("Conv2D", Icon(id="conv2d", label="Conv2D", category="convolution"))
    registry.register("Conv1D", Icon(id="conv1d", label="Conv1D", category="convolution"))
    registry.register("Flatten", Icon(id="flatten", label="Flatten", category="transform"))
    registry.register("Dropout", Icon(id="dropout", label="Dropout", category="regularization"))
    registry.register(
        "BatchNormalization",
        Icon(id="batch_norm", label="Batch Normalization", category="normalization"),
    )
    registry.register("ReLU", Icon(id="relu", label="ReLU", category="activation"))
    registry.register("Sigmoid", Icon(id="sigmoid", label="Sigmoid", category="activation"))
    registry.register("Softmax", Icon(id="softmax", label="Softmax", category="activation"))
    registry.register("Tanh", Icon(id="tanh", label="Tanh", category="activation"))
    registry.register("LeakyReLU", Icon(id="leaky_relu", label="LeakyReLU", category="activation"))
    registry.register(
        "MaxPooling2D",
        Icon(id="max_pool", label="Max Pool", category="pooling"),
    )
    registry.register(
        "AveragePooling2D",
        Icon(id="avg_pool", label="Avg Pool", category="pooling"),
    )
    registry.register("UnknownLayer", Icon(id="unknown", label="?", category="unknown"))

    return registry


def register_builtin_icons(registry: IconRegistry) -> None:
    """Register all built-in icons into the provided registry.

    Args:
        registry: Target icon registry.
    """
    builtin = _create_builtin_registry()
    for name in builtin.available():
        registry.register(name, builtin.get(name))
