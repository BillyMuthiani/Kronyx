# Data Utilities

Kronyx provides data utilities for preparing and loading data.

## train_test_split

Split arrays into random train and test subsets.

```python
from kronyx import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Parameters:**
- `arrays`: Sequence of arrays to split
- `test_size`: Proportion of data for test set
- `random_state`: Random seed for reproducibility
- `shuffle`: Whether to shuffle before splitting

## BatchLoader

Iterate over data in batches.

```python
from kronyx import BatchLoader

loader = BatchLoader(X, y, batch_size=32, shuffle=True)

for X_batch, y_batch in loader:
    # Train on batch
    model.fit(X_batch, y_batch, epochs=1)
```

**Parameters:**
- `X`: Features array
- `y`: Labels array (optional)
- `batch_size`: Number of samples per batch
- `shuffle`: Whether to shuffle data
- `drop_last`: Whether to drop last incomplete batch

## TensorDataset

Create a dataset from tensors.

```python
from kronyx import TensorDataset

dataset = TensorDataset(X, y)
X0, y0 = dataset[0]  # Get first sample
```

## Dataset

Base dataset class.

```python
from kronyx import Dataset

dataset = Dataset(X, y)
X0, y0 = dataset[0]
```

## Example: Using BatchLoader

```python
import numpy as np
from kronyx import *

# Load data
X, y = blobs(samples=1000, centers=3, random_state=42)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create batch loader
loader = BatchLoader(X_train, y_train, batch_size=32, shuffle=True)

# Train with batches
for epoch in range(10):
    for X_batch, y_batch in loader:
        # Forward pass
        output = model.forward(X_batch)
        # Compute loss, backward, etc.