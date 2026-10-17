# Getting Started

This guide will help you get started with Kronyx for your first neural network project.

## Installation

```bash
pip install kronyx
```

## Your First Model

```python
import numpy as np
from kronyx import Sequential, Dense, ReLU, Accuracy, Adam, BinaryCrossEntropy

# Create a sequential model
model = Sequential()

# Add layers
model.add(Dense(4, 8))
model.add(ReLU())
model.add(Dense(8, 1))

# Compile with optimizer, loss, and metric
model.compile(
    loss=BinaryCrossEntropy(),
    optimizer=Adam(),
    metric=Accuracy()
)

model.summary()
```

## Training a Model

```python
# Prepare data
X_train = np.array([[0, 0, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0], [1, 1, 1, 1]])
y_train = np.array([[0], [1], [0], [1]])

# Train
history = model.fit(X_train, y_train, epochs=100, verbose=1)

# Make predictions
predictions = model.predict(X_train)
```

## Saving Your Model

```python
# Save complete model
model.save('mymodel.krx')

# Load later
from kronyx import load_model
loaded = load_model('mymodel.krx')
```

## Next Steps

- See [sequential.md](sequential.md) for detailed Sequential API
- See [layers.md](layers.md) for layer reference
- See [callbacks.md](callbacks.md) for training callbacks