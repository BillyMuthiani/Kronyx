# Metrics

Kronyx provides evaluation metrics for assessing model performance.

## Accuracy

Overall classification accuracy.

```python
from kronyx import Accuracy

metric = Accuracy()
acc = metric.calculate(y_true, y_pred)
```

For binary: `accuracy = (TP + TN) / (TP + TN + FP + FN)`

## BinaryAccuracy

Binary classification accuracy with configurable threshold.

```python
from kronyx import BinaryAccuracy

metric = BinaryAccuracy(threshold=0.5)
acc = metric.calculate(y_true, y_pred)
```

## CategoricalAccuracy

Multi-class classification accuracy.

```python
from kronyx import CategoricalAccuracy

metric = CategoricalAccuracy()
acc = metric.calculate(y_true, y_pred)
```

## Precision

Precision = TP / (TP + FP)

```python
from kronyx import Precision

metric = Precision()
prec = metric.calculate(y_true, y_pred)
```

## Recall

Recall = TP / (TP + FN)

```python
from kronyx import Recall

metric = Recall()
rec = metric.calculate(y_true, y_pred)
```

## F1Score

Harmonic mean of precision and recall.

```python
from kronyx import F1Score

metric = F1Score()
f1 = metric.calculate(y_true, y_pred)
```

**Formula:** `F1 = 2 * (precision * recall) / (precision + recall)`

## ConfusionMatrix

Compute confusion matrix for classification.

```python
from kronyx import ConfusionMatrix

metric = ConfusionMatrix()
cm = metric.calculate(y_true, y_pred)
```

## TopKAccuracy

For multi-class, checks if true label is among top k predictions.

```python
from kronyx import TopKAccuracy

metric = TopKAccuracy(k=3)
acc = metric.calculate(y_true, y_pred)
```

## Example: Using Multiple Metrics

```python
import numpy as np
from kronyx import *

# Get predictions
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# Calculate all metrics
print(f"Accuracy: {Accuracy().calculate(y_test, y_pred):.4f}")
print(f"Precision: {Precision().calculate(y_test, y_pred):.4f}")
print(f"Recall: {Recall().calculate(y_test, y_pred):.4f}")
print(f"F1: {F1Score().calculate(y_test, y_pred):.4f}")

# Confusion matrix
cm = ConfusionMatrix().calculate(y_test, y_pred_labels)
print(f"Confusion Matrix:\n{cm}")