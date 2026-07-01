"""Model serialization and persistence utilities.

Provides save/load functionality for complete models and weights using a custom
.krx format (JSON metadata + NPZ weights in a zip archive).
"""
import io
import json
import sys
import zipfile
from datetime import datetime, timezone

import numpy as np

from kronyx.exceptions import NeuralnetError


class SerializationError(NeuralnetError):
    """Raised when serialization operations fail."""
    pass


def save(model, filename):
    """Save model weights to a .npz file (legacy format).

    Args:
        model: Model to save.
        filename: Path to save weights.

    Note:
        For complete model saving including architecture, use model.save() instead.
    """
    params = {}
    idx = 0

    for layer in model.layers:
        if hasattr(layer, "weights"):
            params[f"w{idx}"] = layer.weights
            params[f"b{idx}"] = layer.biases
            idx += 1

    np.savez(filename, **params)


def load(model, filename):
    """Load model weights from a .npz file (legacy format).

    Args:
        model: Model to load weights into.
        filename: Path to load weights from.

    Note:
        For complete model loading, use load_model() instead.
    """
    data = np.load(filename)
    idx = 0

    for layer in model.layers:
        if hasattr(layer, "weights"):
            layer.weights = data[f"w{idx}"]
            layer.biases = data[f"b{idx}"]
            idx += 1


def _get_metadata():
    """Get serialization metadata.

    Returns:
        Dictionary with framework and environment information.
    """
    import kronyx
    return {
        "framework": "kronyx",
        "version": kronyx.__version__,
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}"
            f".{sys.version_info.micro}"
        ),
        "numpy_version": np.__version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "format_version": 1
    }


def _get_layer_config(layer):
    """Extract layer configuration as a dictionary.

    Args:
        layer: Layer to extract config from.

    Returns:
        Dictionary with layer configuration.
    """
    config = {"type": type(layer).__name__}

    if hasattr(layer, 'weights'):
        config["weights_shape"] = list(layer.weights.shape)
        config["biases_shape"] = list(layer.biases.shape)

    if hasattr(layer, 'rate'):
        config["rate"] = layer.rate

    if hasattr(layer, 'momentum'):
        config["momentum"] = layer.momentum

    if hasattr(layer, 'epsilon'):
        config["epsilon"] = layer.epsilon

    if hasattr(layer, 'padding'):
        config["padding"] = layer.padding

    if hasattr(layer, 'stride'):
        config["stride"] = layer.stride

    if hasattr(layer, 'filters'):
        config["filters"] = layer.filters

    if hasattr(layer, 'kernel_size'):
        config["kernel_size"] = layer.kernel_size

    if hasattr(layer, 'kernel_regularizer'):
        reg = layer.kernel_regularizer
        if reg:
            config["regularizer"] = {"type": "L2", "lambda_": reg.lambda_}

    return config


def save_model(model, filename):
    """Save complete model to .krx format.

    Creates a zip archive containing metadata.json, architecture.json,
    weights.npz, and optimizer.npz.

    Args:
        model: Model to save.
        filename: Path to save (should end with .krx).

    Raises:
        SerializationError: If saving fails.
    """

    if not filename.endswith('.krx'):
        filename = filename + '.krx'

    try:
        metadata = _get_metadata()

        architecture = {
            "layers": [_get_layer_config(layer) for layer in model.layers],
            "optimizer": {
                "type": type(model.optimizer).__name__ if model.optimizer else None,
                "learning_rate": getattr(model.optimizer, 'learning_rate', None),
            },
            "loss": type(model.loss_function).__name__ if model.loss_function else None,
            "metric": type(model.metric).__name__ if model.metric else None,
        }

        weights = {}
        trainable_idx = 0
        for layer in model.layers:
            if hasattr(layer, "weights"):
                weights[f"w{trainable_idx}"] = layer.weights
                weights[f"b{trainable_idx}"] = layer.biases
                trainable_idx += 1

        optimizer_state = {}
        if model.optimizer and hasattr(model.optimizer, 'state'):
            optimizer_state = model.optimizer.state

        with zipfile.ZipFile(filename, 'w') as zf:
            zf.writestr('metadata.json', json.dumps(metadata, indent=2))
            zf.writestr('architecture.json', json.dumps(architecture, indent=2))

            weights_buffer = io.BytesIO()
            np.savez(weights_buffer, **weights)
            weights_buffer.seek(0)
            zf.writestr('weights.npz', weights_buffer.read())

            if optimizer_state:
                opt_buffer = io.BytesIO()
                np.savez(opt_buffer, **optimizer_state)
                opt_buffer.seek(0)
                zf.writestr('optimizer.npz', opt_buffer.read())

    except Exception as e:
        raise SerializationError(f"Failed to save model: {e}") from e


def load_model(filename):
    """Load complete model from .krx format.

    Args:
        filename: Path to model file.

    Returns:
        Loaded Sequential model.

    Raises:
        SerializationError: If loading fails.
    """
    try:
        with zipfile.ZipFile(filename, 'r') as zf:
            metadata = json.loads(zf.read('metadata.json').decode('utf-8'))

            if metadata.get("format_version") != _get_metadata()["format_version"]:
                import warnings
                warnings.warn(
                    f"Model format version {metadata.get('format_version')} may not be "
                    f"compatible with current version {_get_metadata()['format_version']}",
                    UserWarning,
                    stacklevel=2
                )

            architecture = json.loads(zf.read('architecture.json').decode('utf-8'))
            weights_data = zf.read('weights.npz')

        weights_buffer = io.BytesIO(weights_data)
        return _reconstruct_model(architecture, weights_buffer, None)

    except FileNotFoundError:
        raise SerializationError(f"Model file not found: {filename}") from None
    except KeyError as e:
        raise SerializationError(f"Invalid model archive: missing {e}") from None
    except json.JSONDecodeError as e:
        raise SerializationError(f"Invalid JSON in model file: {e}") from None
    except zipfile.BadZipFile as e:
        raise SerializationError(f"Invalid or corrupted archive: {e}") from None


def _reconstruct_model(architecture, weights_buffer, optimizer_buffer):
    """Reconstruct a model from architecture and weights.

    Args:
        architecture: Dictionary with model architecture.
        weights_buffer: BytesIO buffer containing weights.
        optimizer_buffer: BytesIO buffer containing optimizer state.

    Returns:
        Reconstructed Sequential model.
    """
    import kronyx
    from kronyx.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten

    layer_map = {
        "Dense": Dense,
        "Conv2D": Conv2D,
        "Flatten": Flatten,
        "Dropout": Dropout,
        "BatchNormalization": BatchNormalization,
        "ReLU": kronyx.ReLU,
        "Sigmoid": kronyx.Sigmoid,
        "Tanh": kronyx.Tanh,
        "Softmax": kronyx.Softmax,
    }

    model = kronyx.Sequential()

    for layer_config in architecture.get("layers", []):
        layer_type = layer_config["type"]

        layer_class = layer_map.get(layer_type)
        if layer_class is None:
            raise SerializationError(f"Unsupported layer type: {layer_type}")

        if layer_type == "Dense":
            w_shape = layer_config["weights_shape"]
            layer = Dense(w_shape[0], w_shape[1])
        elif layer_type == "Conv2D":
            layer = Conv2D(
                filters=layer_config["filters"],
                kernel_size=layer_config["kernel_size"],
                padding=layer_config.get("padding", "valid")
            )
        elif layer_type == "Flatten":
            layer = Flatten()
        elif layer_type == "Dropout":
            layer = Dropout(rate=layer_config["rate"])
        elif layer_type == "BatchNormalization":
            layer = BatchNormalization(
                momentum=layer_config.get("momentum", 0.9)
            )
        else:
            layer = layer_class()

        model.add(layer)

    if weights_buffer:
        weights_buffer.seek(0)
        data = np.load(weights_buffer)
        trainable_idx = 0
        for layer in model.layers:
            if hasattr(layer, "weights"):
                key = f"w{trainable_idx}"
                if key in data:
                    layer.weights = data[key]
                    layer.biases = data[f"b{trainable_idx}"]
                trainable_idx += 1

    return model


def to_json(model):
    """Export model architecture to JSON string.

    Args:
        model: Model to export.

    Returns:
        JSON string of architectural configuration.
    """
    architecture = {
        "layers": [_get_layer_config(layer) for layer in model.layers],
        "optimizer": {
            "type": type(model.optimizer).__name__ if model.optimizer else None,
        },
        "loss": type(model.loss_function).__name__ if model.loss_function else None,
        "metric": type(model.metric).__name__ if model.metric else None,
    }
    return json.dumps(architecture, indent=2)


def from_json(json_string):
    """Create a model from JSON architecture string.

    Args:
        json_string: JSON string containing architecture.

    Returns:
        Sequential model with layers matching the architecture.

    Raises:
        SerializationError: If JSON is invalid or unsupported layer types.
    """
    try:
        architecture = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise SerializationError(f"Invalid JSON: {e}") from e

    return _reconstruct_model(architecture, None, None)
