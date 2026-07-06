# Visualization

Kronyx provides visualization tools to understand your data and model performance.

## Training Curves

Plot training and validation metrics over epochs.

```python
from kronyx import plot_training_curves

# After training
history = model.fit(X, y, epochs=100)
plot_training_curves(history)
```

## Decision Boundary

Visualize the decision boundary of a classifier.

```python
from kronyx import plot_decision_boundary

# For 2D classification
plot_decision_boundary(model, X, y)
```

## Confusion Matrix

Plot a confusion matrix with labels.

```python
from kronyx import plot_confusion_matrix

cm = ConfusionMatrix().calculate(y_true, y_pred)
plot_confusion_matrix(cm, classes=["Class 0", "Class 1"])
```

## Dataset Plot

Visualize a 2D dataset.

```python
from kronyx import plot_dataset

plot_dataset(X, y)
```

## Predictions Plot

Visualize predictions vs true labels.

```python
from kronyx import plot_predictions

y_pred = model.predict(X)
y_pred_labels = (y_pred > 0.5).astype(int).flatten()
plot_predictions(X, y, y_pred_labels)
```

## Feature Space

Visualize high-dimensional data in feature space.

```python
from kronyx import plot_feature_space

plot_feature_space(X, y)
```

## Example: Complete Visualization

```python
import numpy as np
from kronyx import *

# Load data
X, y = moons(samples=200, random_state=42)

# Build and train
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

history = model.fit(X, y, epochs=500, verbose=0)

# Visualize
plot_training_curves(history)
plot_decision_boundary(model, X, y)