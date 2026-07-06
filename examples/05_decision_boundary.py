"""Decision boundary visualization example.

Learn how to visualize the decision boundary of a neural network
to understand how it separates classes.
"""

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
from kronyx.datasets import circles, moons
from kronyx.visualization import plot_decision_boundary

# Set seed for reproducibility
set_seed(42)

# Generate non-linear dataset
X, y = circles(samples=200, factor=0.4, noise=0.05)

print(f"Dataset shape: X={X.shape}, y={y.shape}")

# Build a neural network
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

# Visualize decision boundary
print("\nPlotting decision boundary...")
plot_decision_boundary(model, X, y)

# Also try with moons
X2, y2 = moons(samples=200, noise=0.1)
model2 = Sequential()
model2.add(Dense(2, 16))
model2.add(ReLU())
model2.add(Dense(16, 1))
model2.add(Sigmoid())
model2.compile(loss=BinaryCrossEntropy(), optimizer=Adam(learning_rate=0.1))
model2.fit(X2, y2, epochs=1000, verbose=0)

print("\nPlotting moons decision boundary...")
plot_decision_boundary(model2, X2, y2)
