"""Confusion matrix example.

Learn how to compute and visualize confusion matrices
for understanding model performance.
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
    Softmax,
    SoftmaxCategoricalCrossEntropy,
    set_seed,
)
from kronyx.datasets import moons, spiral
from kronyx.metrics import ConfusionMatrix
from kronyx.visualization import plot_confusion_matrix

# Set seed for reproducibility
set_seed(42)

# Generate dataset
X, y = moons(samples=200, noise=0.1)

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

model.fit(X, y, epochs=1000, verbose=0)

# Get predictions
y_pred = model.predict(X)
y_pred_labels = (y_pred > 0.5).astype(int).flatten()

# Compute confusion matrix
cm = ConfusionMatrix().calculate(y, y_pred_labels)
print("Confusion Matrix:")
print(cm)
print("\nInterpretation:")
print("  - Top-left: True Negatives (correctly predicted class 0)")
print("  - Top-right: False Positives (class 0 predicted as class 1)")
print("  - Bottom-left: False Negatives (class 1 predicted as class 0)")
print("  - Bottom-right: True Positives (correctly predicted class 1)")

# Visualize
print("\nPlotting confusion matrix...")
plot_confusion_matrix(cm, classes=["Class 0", "Class 1"])

# Multi-class example
X, y = spiral(samples=300, classes=3)

model2 = Sequential()
model2.add(Dense(2, 16))
model2.add(ReLU())
model2.add(Dense(16, 3))
model2.add(Softmax())

model2.compile(
    loss=SoftmaxCategoricalCrossEntropy(),
    optimizer=Adam(learning_rate=0.01),
    metric=Accuracy()
)

model2.fit(X, y, epochs=1000, verbose=0)

y_pred = model2.predict(X)
y_pred_labels = np.argmax(y_pred, axis=1)

cm2 = ConfusionMatrix().calculate(y, y_pred_labels)
print("\nMulti-class Confusion Matrix:")
print(cm2)

print("\nPlotting multi-class confusion matrix...")
plot_confusion_matrix(cm2, classes=["Class 0", "Class 1", "Class 2"])
