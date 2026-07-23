# Kronyx

A lightweight deep learning framework built from first principles using NumPy.

[![PyPI](https://img.shields.io/badge/PyPI-kronyx-blue.svg)](https://pypi.org/project/kronyx/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://kronyx.github.io/kronyx)

## Features

| Feature | Description |
|---------|-------------|
| **Pure NumPy** | No external ML dependencies, just NumPy |
| **Clean API** | Keras-like Sequential model interface |
| **Layers** | Dense, Conv2D, Flatten, Dropout, BatchNormalization |
| **Activations** | ReLU, Sigmoid, Tanh, Softmax |
| **Optimizers** | SGD, Adam with full state management |
| **Callbacks** | EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau |
| **Serialization** | Save/load models with .krx format |
| **Datasets** | xor, spiral, circles, moons, blobs, iris |
| **Preprocessing** | StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder |
| **Metrics** | Accuracy, Precision, Recall, F1Score, ConfusionMatrix, TopKAccuracy |
| **Visualization** | plot_training_curves, plot_decision_boundary, plot_confusion_matrix |
| **Architecture** | Native SVG engine, themes, icons, layered pipeline |

## Quick Example

```python
import numpy as np
from kronyx import Sequential, Dense, ReLU, Sigmoid, BinaryCrossEntropy, Adam, Accuracy

# XOR problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

model = Sequential()
model.add(Dense(2, 8))
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

## Installation

```bash
pip install kronyx
```

## Documentation

| Guide | Description |
|--------|-------------|
| [Getting Started](getting_started.md) | Installation and first model |
| [Sequential API](sequential.md) | Building models |
| [Layers](layers.md) | Dense, Conv2D, Flatten, Dropout |
| [Callbacks](callbacks.md) | EarlyStopping, CSVLogger |
| [Serialization](serialization.md) | Saving and loading `.krx` models |
| [Examples](examples.md) | Complete working examples |
| [Visualization Overview](visualization/overview.md) | Architecture pipeline and usage |
| [Visualization Themes](visualization/themes.md) | Theme system and built-in themes |
| [Visualization Icons](visualization/icons.md) | Layer icon system |
| [Visualization Renderers](visualization/renderers.md) | Native SVG renderer |
| [Architecture](architecture/visualization_engine.md) | Engine internals and design |
| [Roadmap](../ROADMAP.md) | Future development |
| [Changelog](../CHANGELOG.md) | Release history |

