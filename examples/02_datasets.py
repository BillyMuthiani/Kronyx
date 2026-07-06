"""Datasets example.

Learn about the synthetic datasets available in Kronyx.
Each dataset is designed to teach different concepts.
"""
import numpy as np

from kronyx import set_seed
from kronyx.datasets import blobs, circles, iris, moons, spiral, xor

# Set seed for reproducibility
set_seed(42)

print("=" * 50)
print("Kronyx Synthetic Datasets")
print("=" * 50)

# XOR - Non-linearly separable binary classification
print("\n1. XOR Dataset (Non-linear binary classification)")
X, y = xor(samples=100, noise=0.1)
print(f"   Shape: X={X.shape}, y={y.shape}")
print(f"   Classes: {np.unique(y)}")

# Spiral - Multi-class with complex boundaries
print("\n2. Spiral Dataset (Multi-class, complex boundaries)")
X, y = spiral(samples=300, classes=3, noise=0.2)
print(f"   Shape: X={X.shape}, y={y.shape}")
print(f"   Classes: {np.unique(y)}")

# Circles - Concentric circles
print("\n3. Circles Dataset (Non-linear binary classification)")
X, y = circles(samples=200, factor=0.5)
print(f"   Shape: X={X.shape}, y={y.shape}")
print(f"   Classes: {np.unique(y)}")

# Moons - Interlocking crescents
print("\n4. Moons Dataset (Non-linear binary classification)")
X, y = moons(samples=200)
print(f"   Shape: X={X.shape}, y={y.shape}")
print(f"   Classes: {np.unique(y)}")

# Blobs - Gaussian clusters
print("\n5. Blobs Dataset (Multi-class clustering)")
X, y = blobs(samples=300, centers=3)
print(f"   Shape: X={X.shape}, y={y.shape}")
print(f"   Classes: {np.unique(y)}")

# Iris - Real-world multi-class
print("\n6. Iris Dataset (Real-world multi-class)")
X, y = iris()
print(f"   Shape: X={X.shape}, y={y.shape}")
print(f"   Classes: {np.unique(y)}")

print("\n" + "=" * 50)
print("All datasets return NumPy arrays ready for training!")
print("=" * 50)
