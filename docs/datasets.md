# Datasets

Kronyx provides synthetic datasets for learning and testing neural networks.

## Available Datasets

### XOR

The XOR dataset is a classic non-linearly-separable binary classification problem.

```python
from kronyx import xor

X, y = xor(samples=100, noise=0.1, random_state=42)
```

**Parameters:**
- `samples`: Number of samples (must be even)
- `noise`: Gaussian noise standard deviation
- `random_state`: Random seed for reproducibility

### Spiral

Multi-class classification with spiral-shaped clusters.

```python
from kronyx import spiral

X, y = spiral(samples=300, classes=3, noise=0.2, random_state=42)
```

**Parameters:**
- `samples`: Total number of samples
- `classes`: Number of spiral classes
- `noise`: Gaussian noise standard deviation
- `random_state`: Random seed for reproducibility

### Circles

Concentric circles for binary classification.

```python
from kronyx import circles

X, y = circles(samples=200, noise=0.05, factor=0.5, random_state=42)
```

**Parameters:**
- `samples`: Number of samples
- `noise`: Gaussian noise standard deviation
- `factor`: Scale factor for inner circle radius
- `random_state`: Random seed for reproducibility

### Moons

Two interleaving crescent shapes.

```python
from kronyx import moons

X, y = moons(samples=200, noise=0.1, random_state=42)
```

**Parameters:**
- `samples`: Number of samples
- `noise`: Gaussian noise standard deviation
- `random_state`: Random seed for reproducibility

### Blobs

Gaussian clusters for multi-class classification.

```python
from kronyx import blobs

X, y = blobs(samples=300, centers=3, cluster_std=1.0, random_state=42)
```

**Parameters:**
- `samples`: Number of samples
- `centers`: Number of cluster centers
- `cluster_std`: Standard deviation of clusters
- `random_state`: Random seed for reproducibility

### Iris

The classic Iris flower dataset (synthetic version).

```python
from kronyx import iris

X, y = iris(random_state=42)
```

**Parameters:**
- `random_state`: Random seed for shuffling

## Example: Training on XOR

```python
import numpy as np
from kronyx import *

# Load data
X, y = xor(samples=200, noise=0.1, random_state=42)
y = y.reshape(-1, 1)

# Build model
model = Sequential()
model.add(Dense(2, 16))
model.add(ReLU())
model.add(Dense(16, 1))
model.add(Sigmoid())

model.compile(
    loss=BinaryCrossEntropy(),
    optimizer=Adam(learning_rate=0.1),
    metric=Accuracy()
)

# Train
history = model.fit(X, y, epochs=1000, verbose=0)
print(f"Final accuracy: {history.accuracy[-1]*100:.1f}%")