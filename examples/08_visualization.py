"""Visualization example.

Learn how to use visualization tools to understand your data and model.
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
from kronyx.datasets import blobs
from kronyx.visualization import (
    plot_dataset,
    plot_feature_space,
    plot_predictions,
    plot_training_curves,
)

# Set seed for reproducibility
set_seed(42)

# Generate dataset
X, y = blobs(samples=200, centers=3)

print("Available visualization functions:")
print("  - plot_training_curves(history)")
print("  - plot_confusion_matrix(cm)")
print("  - plot_decision_boundary(model, X, y)")
print("  - plot_dataset(X, y)")
print("  - plot_predictions(X, y_true, y_pred)")
print("  - plot_feature_space(X, y)")

# Build and train model
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

# Train and get history
history = model.fit(X, y, epochs=500, verbose=0)

# Plot training curves
print("\nPlotting training curves...")
plot_training_curves(history)

# Plot dataset
print("\nPlotting dataset...")
plot_dataset(X, y)

# Plot predictions
y_pred = model.predict(X)
y_pred_labels = (y_pred > 0.5).astype(int).flatten()
print("\nPlotting predictions...")
plot_predictions(X, y, y_pred_labels)

# Feature space (for multi-feature data)
X_multi = np.random.randn(100, 4)
y_multi = (X_multi[:, 0] + X_multi[:, 1] > 0).astype(int)
print("\nPlotting feature space...")
plot_feature_space(X_multi, y_multi)
