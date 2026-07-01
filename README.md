# Kronyx

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/Kronyx/kronyx/actions/workflows/tests.yml/badge.svg)](https://github.com/Kronyx/kronyx/actions/workflows/tests.yml)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)
[![mypy](https://img.shields.io/badge/types-mypy-blue.svg)](https://mypy-lang.org/)

A lightweight deep learning framework built from first principles using NumPy.

## Features

- **Layers**: Dense, Conv2D, BatchNormalization, Dropout, Flatten
- **Optimizers**: SGD, Adam
- **Callbacks**: EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
- **Serialization**: Save/load model weights
- **NumPy backend**: Pure NumPy implementation, no external ML dependencies
- **Gradient checking**: Built-in utilities for verifying gradient correctness

## Installation

```bash
pip install kronyx
```

For development:

```bash
git clone https://github.com/Kronyx/kronyx.git
cd kronyx
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from kronyx import Sequential, Dense, ReLU, Sigmoid, BinaryCrossEntropy, Adam, Accuracy

# XOR problem - binary classification
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

model = Sequential()
model.add(Dense(2, 8))  # input_size=2, hidden_size=8
model.add(ReLU())
model.add(Dense(8, 1))
model.add(Sigmoid())

model.compile(
    loss=BinaryCrossEntropy(),
    optimizer=Adam(learning_rate=0.1),
    metric=Accuracy()
)

model.fit(X, y, epochs=1000)
predictions = model.predict(X)
print(f"Accuracy: {(predictions.round() == y).mean():.2%}")
```

## Examples

- `examples/xor.py` - XOR problem with binary classification
- `examples/iris_classification.py` - Iris dataset multi-class classification
- `examples/iris_dropout.py` - Dropout regularization example
- `examples/iris_l2.py` - L2 weight regularization
- `examples/batchnorm_iris.py` - Batch normalization demonstration
- `examples/flatten_demo.py` - Multi-dimensional input handling
- `examples/conv2d_demo.py` - Convolutional neural network example
- `examples/mnist_classifier.py` - MNIST digit classification

## API Overview

```python
from kronyx import (
    Sequential,
    Dense,
    Conv2D,
    Flatten,
    Dropout,
    BatchNormalization,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    BinaryCrossEntropy,
    SoftmaxCategoricalCrossEntropy,
    Adam,
    Accuracy,
)
```

## Documentation

```
kronyx/
├── activations.py    # ReLU, Sigmoid, Tanh, Softmax
├── layers.py         # Dense, Conv2D, Flatten, Dropout, BatchNormalization
├── model.py          # Sequential, History
├── optimizers.py     # SGD, Adam
├── losses.py         # BinaryCrossEntropy, CategoricalCrossEntropy
├── metrics.py        # Accuracy
├── callbacks.py      # Callback base, EarlyStopping, etc.
├── regularizers.py   # L2
├── initializers.py   # he_normal, xavier_uniform, lecun_normal
├── serialization.py  # Save/load model (.krx format)
└── exceptions.py     # Error types
```

## Model Serialization

Kronyx supports saving and loading complete models and weights using the `.krx` format.

### Saving and Loading Models

```python
# Save complete model
model.save('model.krx')

# Load complete model
loaded = load_model('model.krx')
```

### Saving and Loading Weights Only

```python
# Save only trainable weights
model.save_weights('weights.npz')

# Load weights into compatible architecture
model.load_weights('weights.npz')
```

### JSON Export/Import

```python
# Export architecture to JSON
json_str = model.to_json()

# Create model from JSON
model = Sequential.from_json(json_str)
```

### .krx Archive Format

The `.krx` format is a zip archive containing:

```
model.krx
├── metadata.json      # Framework version, Python/numpy versions, timestamp
├── architecture.json  # Layer configuration, loss, optimizer settings
├── weights.npz        # Trainable weights and biases (numpy archive)
└── optimizer.npz      # Optional: optimizer state for resumable training
```

Metadata fields:
- `framework`: Always "kronyx"
- `version`: Framework version
- `format_version`: Archive format version (currently 1)
- `python_version`: Python version used for saving
- `numpy_version`: NumPy version used for saving
- `created_at`: ISO 8601 timestamp

Architecture fields:
- `layers`: List of layer configurations with types and shapes
- `optimizer`: Optimizer type and learning rate
- `loss`: Loss function class name
- `metric`: Metric class name (if any)

Weights are stored as NumPy arrays in the `weights.npz` file:
- `w{i}`: Weights for trainable layer i
- `b{i}`: Biases for trainable layer i

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

MIT License - see [LICENSE](LICENSE)