<p align="center">
  <a href="https://github.com/BillyMuthiani/Kronyx">
    <img src="assets/logo.png" alt="Kronyx" width="650">
  </a>
</p>

<h1 align="center">Kronyx</h1>

<p align="center">
<b>Built from Scratch • NumPy Powered • Educational Deep Learning Framework</b>
</p>

<p align="center">
A lightweight deep learning framework that makes neural networks easy to learn, inspect, visualize, and build.
</p>

<p align="center">

[![PyPI](https://img.shields.io/pypi/v/kronyx?style=for-the-badge)](https://pypi.org/project/kronyx/)
[![Python](https://img.shields.io/pypi/pyversions/kronyx?style=for-the-badge)](https://pypi.org/project/kronyx/)
[![License](https://img.shields.io/github/license/BillyMuthiani/Kronyx?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/BillyMuthiani/Kronyx/tests.yml?style=for-the-badge&label=Tests)](https://github.com/BillyMuthiani/Kronyx/actions)
[![Documentation](https://img.shields.io/badge/docs-online-blue?style=for-the-badge)](https://kronyx.github.io/kronyx)

</p>

---

## Why Kronyx?

Most deep learning frameworks are designed primarily for production.

**Kronyx is designed to help you understand deep learning.**

Whether you're learning neural networks, teaching machine learning, prototyping new ideas, or building lightweight AI applications, Kronyx provides a familiar Keras-like API with tools that make every layer transparent.

### What makes Kronyx different?

- Built entirely from **NumPy**
- Designed for **education first**
- 📈 Built-in training visualizations (`history.plot()`)
- Rich model inspection (`model.summary()`)
- Architecture visualization (`model.visualize()`)
- Native `.krx` model serialization
- ⚡ Lightweight with zero heavyweight ML dependencies
- Familiar Sequential API inspired by Keras
- 📦 Installable directly from PyPI

---

## 🚀 Installation

```bash
pip install kronyx
```

For development:

```bash
git clone https://github.com/BillyMuthiani/Kronyx.git

cd Kronyx

pip install -e ".[dev]"
```

For plotting support:

```bash
pip install kronyx[plotting]
```

For DataFrame support:

```bash
pip install kronyx[dataframe]
```

---

## ⚡ Quick Example

```python
import numpy as np
from kronyx import *

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

model = Sequential()

model.add(Dense(2,16))
model.add(ReLU())

model.add(Dense(16,1))
model.add(Sigmoid())

model.compile(
    loss=BinaryCrossEntropy(),
    optimizer=Adam(),
    metric=Accuracy()
)

history = model.fit(X,y,epochs=1000)

history.plot()

model.summary()

model.visualize()

model.save("xor.krx")
```

---

# Features

| | |
|:--|:--|
| Backend | Pure NumPy |
| Educational | Built for learning neural networks |
| API | Keras-like Sequential interface |
| 📊 Visualization | `history.plot()` |
| Inspection | `model.summary()` |
| Architecture | `model.visualize()` |
| Serialization | `.krx` format |
| Optimizers | SGD, Adam |
| Losses | BCE, CCE |
|  Metrics | Accuracy |
| Callbacks | EarlyStopping, CSVLogger, ModelCheckpoint |
| Type Checked | mypy |
| Linted | Ruff |
| Distribution | PyPI |

---

# 📚 Documentation

| Guide | Description |
|--------|-------------|
| [Getting Started](docs/getting_started.md) | Installation and first model |
| [Sequential API](docs/sequential.md) | Building models |
| [Layers](docs/layers.md) | Dense, Conv2D, Flatten, Dropout |
| [Callbacks](docs/callbacks.md) | EarlyStopping, CSVLogger |
| [Serialization](docs/serialization.md) | Saving and loading `.krx` models |
| [Examples](docs/examples.md) | Complete working examples |
| [Roadmap](ROADMAP.md) | Future development |
| [Changelog](CHANGELOG.md) | Release history |

# 🌟 Vision

Kronyx exists to make deep learning understandable.

Instead of treating neural networks as black boxes, Kronyx exposes every layer, every parameter, and every training step through intuitive visualization and inspection tools.

Our mission is to become the best framework for learning how deep learning works under the hood.

---

<p align="center">

Made with ❤️ using NumPy.

⭐ If Kronyx helps you learn or build, consider giving the repository a star.

</p>