"""Metrics example.

Learn about different evaluation metrics and when to use them.
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
from kronyx.datasets import moons
from kronyx.metrics import F1Score, Precision, Recall

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

# Calculate all metrics
print("Binary Classification Metrics:")
print("=" * 40)

accuracy = Accuracy().calculate(y, y_pred)
precision = Precision().calculate(y, y_pred)
recall = Recall().calculate(y, y_pred)
f1 = F1Score().calculate(y, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\n" + "=" * 40)
print("When to use each metric:")
print("=" * 40)
print("Accuracy: Overall correctness (balanced classes)")
print("Precision: Minimize false positives (positive predictions matter)")
print("Recall: Minimize false negatives (catch all positives)")
print("F1: Balance between precision and recall")

# Demonstrate with imbalanced data
print("\n" + "=" * 40)
print("Imbalanced dataset example:")
print("=" * 40)

# Create imbalanced data
X_imb = np.vstack([
    np.random.randn(20, 2) + [2, 2],  # 20 samples class 1
    np.random.randn(80, 2) - [2, 2],  # 80 samples class 0
])
y_imb = np.hstack([np.ones(20), np.zeros(80)])

model2 = Sequential()
model2.add(Dense(2, 16))
model2.add(ReLU())
model2.add(Dense(16, 1))
model2.add(Sigmoid())
model2.compile(loss=BinaryCrossEntropy(), optimizer=Adam(learning_rate=0.1))
model2.fit(X_imb, y_imb, epochs=500, verbose=0)

y_pred_imb = model2.predict(X_imb)
y_pred_imb_labels = (y_pred_imb > 0.5).astype(int).flatten()

print(f"Class distribution: {np.bincount(y_imb.astype(int))}")
print(f"Accuracy:  {Accuracy().calculate(y_imb, y_pred_imb):.4f}")
print(f"Precision: {Precision().calculate(y_imb, y_pred_imb):.4f}")
print(f"Recall:    {Recall().calculate(y_imb, y_pred_imb):.4f}")
print(f"F1 Score:  {F1Score().calculate(y_imb, y_pred_imb):.4f}")
