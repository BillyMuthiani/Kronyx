# Preprocessing

Kronyx provides preprocessing utilities to prepare data for neural network training.

## StandardScaler

Standardize features by removing mean and scaling to unit variance.

```python
from kronyx import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_original = scaler.inverse_transform(X_scaled)
```

**Formula:** `z = (x - mean) / std`

## MinMaxScaler

Transform features to a specified range (default [0, 1]).

```python
from kronyx import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Custom range
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)
```

**Formula:** `X_std = (X - X.min) / (X.max - X.min)`

## RobustScaler

Scale features using statistics robust to outliers (median and IQR).

```python
from kronyx import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Custom quantile range
scaler = RobustScaler(quantile_range=(10.0, 90.0))
X_scaled = scaler.fit_transform(X)
```

**Formula:** `z = (x - median) / IQR`

## OneHotEncoder

Encode integer labels as one-hot vectors.

```python
from kronyx import OneHotEncoder

encoder = OneHotEncoder()
y_onehot = encoder.fit_transform(y)
y_labels = encoder.inverse_transform(y_onehot)
```

## Example: Complete Pipeline

```python
import numpy as np
from kronyx import *

# Load data
X, y = blobs(samples=300, centers=3, random_state=42)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One-hot encode labels
encoder = OneHotEncoder()
y_train_onehot = encoder.fit_transform(y_train)
y_test_onehot = encoder.transform(y_test)

# Build model
model = Sequential()
model.add(Dense(2, 16))
model.add(ReLU())
model.add(Dense(16, 3))
model.add(Softmax())

model.compile(
    loss=SoftmaxCategoricalCrossEntropy(),
    optimizer=Adam(learning_rate=0.01),
    metric=Accuracy()
)

# Train
history = model.fit(X_train_scaled, y_train_onehot, epochs=500, verbose=0)