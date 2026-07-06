"""Preprocessing scalers example.

Learn how to use StandardScaler, MinMaxScaler, and RobustScaler
to prepare data for neural network training.
"""
import numpy as np

from kronyx import set_seed
from kronyx.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# Set seed for reproducibility
set_seed(42)

# Create sample data with different scales
X = np.array([
    [1, 100, 0.001],
    [2, 200, 0.002],
    [3, 300, 0.003],
    [4, 400, 0.004],
    [5, 500, 0.005],
])

print("Original data:")
print(X)
print(f"Mean: {X.mean(axis=0)}")
print(f"Std: {X.std(axis=0)}")

# StandardScaler - zero mean, unit variance
print("\n" + "=" * 50)
print("StandardScaler: z = (x - mean) / std")
print("=" * 50)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaled data:")
print(X_scaled)
print(f"Mean: {X_scaled.mean(axis=0)}")
print(f"Std: {X_scaled.std(axis=0)}")

# MinMaxScaler - scale to [0, 1]
print("\n" + "=" * 50)
print("MinMaxScaler: scale to [0, 1] range")
print("=" * 50)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print("Scaled data:")
print(X_scaled)
print(f"Min: {X_scaled.min(axis=0)}")
print(f"Max: {X_scaled.max(axis=0)}")

# RobustScaler - use median and IQR
print("\n" + "=" * 50)
print("RobustScaler: (x - median) / IQR")
print("=" * 50)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
print("Scaled data:")
print(X_scaled)
print(f"Median: {np.median(X_scaled, axis=0)}")
print(f"IQR: {np.percentile(X_scaled, 75, axis=0) - np.percentile(X_scaled, 25, axis=0)}")
