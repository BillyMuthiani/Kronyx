"""First neural network example.

Learn how to build and train your first neural network with Kronyx.
This example uses the XOR dataset to demonstrate a simple binary classification.
"""
import numpy as np

from kronyx import (
    Accuracy,
    Adam,
    BinaryCrossEntropy,
    Dense,
    ReLU,
    Sequential,
    Sigmoid,
    set_seed,
)
from kronyx.datasets import xor

# Set seed for reproducibility
set_seed(42)

# Generate XOR dataset
X, y = xor(samples=200, noise=0.1, random_state=42)

print(f"Dataset shape: X={X.shape}, y={y.shape}")
print(f"Classes: {np.unique(y)}")

# Build a simple neural network
model = Sequential()
model.add(Dense(2, 16))  # Input layer: 2 features -> 16 neurons
model.add(ReLU())          # Activation function
model.add(Dense(16, 1))    # Output layer: 16 -> 1 output
model.add(Sigmoid())       # Sigmoid for binary classification

# Compile the model
model.compile(
    loss=BinaryCrossEntropy(),
    optimizer=Adam(learning_rate=0.1),
    metric=Accuracy()
)

# Show model architecture
model.summary()

# Reshape y for binary classification
y = y.reshape(-1, 1)

# Train the model
print("\nTraining...")
history = model.fit(X, y, epochs=1000, verbose=0)

# Show final results
print(f"\nFinal loss: {history.loss[-1]:.4f}")
print(f"Final accuracy: {history.accuracy[-1]*100:.1f}%")

# Make predictions
predictions = model.predict(X[:5])
print(f"\nSample predictions:\n{predictions.flatten()}")
print(f"True labels:\n{y[:5]}")
