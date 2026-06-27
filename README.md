# Neuralnet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A lightweight deep learning framework built from scratch in NumPy, implementing neural networks, backpropagation, optimizers, and automatic differentiation without TensorFlow or PyTorch.

## Installation

```bash
pip install Neuralnet
```

For development:

```bash
pip install -e ".[dev]"
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from Neuralnet import Sequential, Dense, ReLU, Softmax
from Neuralnet.losses import SoftmaxCategoricalCrossEntropy
from Neuralnet.optimizers import Adam

# Build model
model = Sequential()
model.add(Dense(4, 16))
model.add(ReLU())
model.add(Dense(16, 3))
model.add(Softmax())

model.compile(
    loss=SoftmaxCategoricalCrossEntropy(),
    optimizer=Adam(learning_rate=0.001)
)

# Train
model.fit(X_train, y_train, epochs=100)

# Predict
predictions = model.predict(X_test)
```

## Sequential API

```python
model = Sequential()
model.add(Dense(input_size, hidden_size))
model.add(ReLU())
model.add(Dense(hidden_size, output_size))
model.add(Softmax())
```

## Training

```python
model.compile(
    loss=SoftmaxCategoricalCrossEntropy(),
    optimizer=Adam(learning_rate=0.001),
    metric=Accuracy()
)

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    shuffle=True,
    validation_data=(X_val, y_val),
    callbacks=[EarlyStopping(patience=10)]
)
```

## Callbacks

- **EarlyStopping**: Stop training when monitored metric stops improving
- **ModelCheckpoint**: Save model when monitored metric improves
- **CSVLogger**: Log training metrics to CSV
- **ReduceLROnPlateau**: Reduce learning rate when plateau detected

```python
from Neuralnet import EarlyStopping, ModelCheckpoint

model.fit(
    X_train, y_train,
    epochs=1000,
    callbacks=[
        EarlyStopping(patience=50),
        ModelCheckpoint("best_model.npz")
    ]
)
```

## Saving & Loading

```python
# Save
model.save("model.npz")

# Load
model.load("model.npz")
```

## Examples

- `examples/xor.py` - XOR problem with binary classification
- `examples/iris_classification.py` - Iris dataset classification
- `examples/iris_dropout.py` - Dropout regularization example
- `examples/iris_l2.py` - L2 weight regularization
- `examples/batchnorm_iris.py` - Batch normalization
- `examples/flatten_demo.py` - Multi-dimensional input handling
- `examples/conv2d_demo.py` - Convolutional neural network
- `examples/mnist_classifier.py` - MNIST digit classification

## Testing

```bash
pytest tests/ -v
pytest --cov=Neuralnet
```

Coverage: <!-- codecov -->

## Benchmarks

```bash
python benchmarks/benchmark_dense.py
python benchmarks/benchmark_conv2d.py
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features.

## License

MIT License - see [LICENSE](LICENSE)