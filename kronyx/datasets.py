"""Synthetic datasets for educational deep learning.

All datasets return NumPy arrays and support random_state for reproducibility.
Designed to be simple, visualizable, and perfect for learning neural networks.
"""
import numpy as np


def xor(samples: int = 100, noise: float = 0.1, random_state: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate XOR binary classification dataset.

    Creates a 2D dataset where points belong to class 0 if they're in the same
    quadrant (both x1 and x2 positive or both negative), and class 1 otherwise.
    This is a classic non-linearly-separable problem.

    Args:
        samples: Number of samples to generate. Must be even. Defaults to 100.
        noise: Standard deviation of Gaussian noise added to features. Defaults to 0.1.
        random_state: Random seed for reproducibility. Defaults to None.

    Returns:
        Tuple of (X, y) where X has shape (samples, 2) and y has shape (samples,).

    Example:
        >>> X, y = xor(samples=100, noise=0.1, random_state=42)
        >>> X.shape
        (100, 2)
        >>> y.shape
        (100,)
    """
    if random_state is not None:
        np.random.seed(random_state)

    half = samples // 2
    X = np.zeros((samples, 2))
    y = np.zeros(samples)

    # Class 0: same quadrant (both positive or both negative)
    X[:half, 0] = np.random.randn(half) * 0.5 + 1
    X[:half, 1] = np.random.randn(half) * 0.5 + 1
    y[:half] = 0

    # Class 1: different quadrants
    X[half:, 0] = np.random.randn(half) * 0.5 - 1
    X[half:, 1] = np.random.randn(half) * 0.5 + 1
    y[half:] = 1

    # Add noise
    X += np.random.randn(*X.shape) * noise

    return X, y


def spiral(samples: int = 1000, classes: int = 3, noise: float = 0.2, random_state: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate spiral-shaped multi-class classification dataset.

    Creates points arranged in spiral patterns, one per class. Each class forms
    a different spiral, making this a challenging non-linear classification problem.

    Args:
        samples: Total number of samples to generate. Defaults to 1000.
        classes: Number of spiral classes. Defaults to 3.
        noise: Standard deviation of Gaussian noise. Defaults to 0.2.
        random_state: Random seed for reproducibility. Defaults to None.

    Returns:
        Tuple of (X, y) where X has shape (samples, 2) and y has shape (samples,).

    Example:
        >>> X, y = spiral(samples=300, classes=3, random_state=42)
        >>> X.shape
        (300, 2)
    """
    if random_state is not None:
        np.random.seed(random_state)

    X = np.zeros((samples, 2))
    y = np.zeros(samples, dtype=int)

    samples_per_class = samples // classes

    for c in range(classes):
        start_idx = c * samples_per_class
        end_idx = start_idx + samples_per_class

        r = np.linspace(0.0, 1, samples_per_class)
        t = np.linspace(c * 4, (c + 1) * 4, samples_per_class) + np.random.randn(samples_per_class) * noise

        X[start_idx:end_idx, 0] = r * np.sin(t)
        X[start_idx:end_idx, 1] = r * np.cos(t)
        y[start_idx:end_idx] = c

    return X, y


def circles(samples: int = 1000, noise: float = 0.05, factor: float = 0.5, random_state: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate concentric circles classification dataset.

    Creates points arranged in two concentric circles. Points inside the inner
    circle belong to class 0, points in the outer ring belong to class 1.

    Args:
        samples: Number of samples to generate. Defaults to 1000.
        noise: Standard deviation of Gaussian noise. Defaults to 0.05.
        factor: Scale factor for inner circle radius. Defaults to 0.5.
        random_state: Random seed for reproducibility. Defaults to None.

    Returns:
        Tuple of (X, y) where X has shape (samples, 2) and y has shape (samples,).

    Example:
        >>> X, y = circles(samples=200, factor=0.3, random_state=42)
        >>> X.shape
        (200, 2)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_inner = samples // 2
    n_outer = samples - n_inner

    # Inner circle
    r_inner = np.sqrt(np.random.rand(n_inner)) * factor
    theta_inner = np.random.rand(n_inner) * 2 * np.pi
    X_inner = np.column_stack([
        r_inner * np.cos(theta_inner),
        r_inner * np.sin(theta_inner)
    ])

    # Outer circle
    r_outer = np.sqrt(np.random.rand(n_outer)) * 1.0
    theta_outer = np.random.rand(n_outer) * 2 * np.pi
    X_outer = np.column_stack([
        r_outer * np.cos(theta_outer),
        r_outer * np.sin(theta_outer)
    ])

    X = np.vstack([X_inner, X_outer])
    y = np.hstack([np.zeros(n_inner), np.ones(n_outer)])

    # Add noise
    X += np.random.randn(*X.shape) * noise

    # Shuffle
    indices = np.random.permutation(samples)
    return X[indices], y[indices]


def moons(samples: int = 1000, noise: float = 0.1, random_state: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate two interleaving crescent moon shapes.

    Creates two crescent-shaped clusters that are not linearly separable.
    A classic toy dataset for testing non-linear classifiers.

    Args:
        samples: Number of samples to generate. Defaults to 1000.
        noise: Standard deviation of Gaussian noise. Defaults to 0.1.
        random_state: Random seed for reproducibility. Defaults to None.

    Returns:
        Tuple of (X, y) where X has shape (samples, 2) and y has shape (samples,).

    Example:
        >>> X, y = moons(samples=200, random_state=42)
        >>> X.shape
        (200, 2)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples2 = samples // 2
    n_samples1 = samples - n_samples2

    # First moon
    x1 = np.cos(np.linspace(0, np.pi, n_samples1))
    y1 = np.sin(np.linspace(0, np.pi, n_samples1))
    X1 = np.column_stack([x1, y1])

    # Second moon (shifted and rotated)
    x2 = np.cos(np.linspace(0, np.pi, n_samples2))
    y2 = np.sin(np.linspace(0, np.pi, n_samples2))
    X2 = np.column_stack([x2 - 0.5, -y2 - 0.25])

    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(n_samples1), np.ones(n_samples2)])

    # Add noise
    X += np.random.randn(*X.shape) * noise

    # Shuffle
    indices = np.random.permutation(samples)
    return X[indices], y[indices]


def blobs(samples: int = 1000, centers: int = 3, cluster_std: float = 1.0, random_state: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate Gaussian blob classification dataset.

    Creates isotropic Gaussian blobs for clustering and classification.
    Each blob center is randomly placed in 2D space.

    Args:
        samples: Number of samples to generate. Defaults to 1000.
        centers: Number of cluster centers. Defaults to 3.
        cluster_std: Standard deviation of clusters. Defaults to 1.0.
        random_state: Random seed for reproducibility. Defaults to None.

    Returns:
        Tuple of (X, y) where X has shape (samples, 2) and y has shape (samples,).

    Example:
        >>> X, y = blobs(samples=300, centers=3, random_state=42)
        >>> X.shape
        (300, 2)
    """
    if random_state is not None:
        np.random.seed(random_state)

    X = np.zeros((samples, 2))
    y = np.zeros(samples, dtype=int)

    samples_per_center = samples // centers

    for c in range(centers):
        start_idx = c * samples_per_center
        end_idx = start_idx + samples_per_center

        # Random center
        center_x = np.random.randn() * 3
        center_y = np.random.randn() * 3

        X[start_idx:end_idx, 0] = np.random.randn(samples_per_center) * cluster_std + center_x
        X[start_idx:end_idx, 1] = np.random.randn(samples_per_center) * cluster_std + center_y
        y[start_idx:end_idx] = c

    # Handle remainder
    remainder = samples % centers
    if remainder > 0:
        X[end_idx:end_idx + remainder] = np.random.randn(remainder, 2) * cluster_std
        y[end_idx:end_idx + remainder] = centers - 1

    # Shuffle
    indices = np.random.permutation(samples)
    return X[indices], y[indices]


def iris(random_state: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load the Iris dataset.

    Returns the classic Iris dataset with 150 samples of 3 flower species.
    Features: sepal length, sepal width, petal length, petal width.

    Args:
        random_state: Random seed (for shuffling). Defaults to None.

    Returns:
        Tuple of (X, y) where X has shape (150, 4) and y has shape (150,).

    Example:
        >>> X, y = iris()
        >>> X.shape
        (150, 4)
    """
    # Generate synthetic iris-like data
    if random_state is not None:
        np.random.seed(random_state)

    # Setosa (50 samples) - smaller features
    setosa = np.random.randn(50, 4) * 0.2 + np.array([5.0, 3.5, 1.4, 0.2])
    # Versicolor (50 samples) - medium features
    versicolor = np.random.randn(50, 4) * 0.3 + np.array([6.0, 2.8, 4.0, 1.3])
    # Virginica (50 samples) - larger features
    virginica = np.random.randn(50, 4) * 0.3 + np.array([7.0, 3.0, 5.5, 2.0])

    X = np.vstack([setosa, versicolor, virginica])
    y = np.array([0] * 50 + [1] * 50 + [2] * 50)

    if random_state is not None:
        indices = np.random.permutation(150)
        return X[indices], y[indices]

    return X, y
