"""Data utilities for batching and dataset management.

Provides tools for splitting data and iterating over batches during training.
All implementations are NumPy-only and educational.
"""
import numpy as np


def train_test_split(
    *arrays,
    test_size: float | int | None = None,
    random_state: int | None = None,
    shuffle: bool = True
) -> list:
    """Split arrays into random train and test subsets.

    Args:
        *arrays: Sequence of indexables with same length / shape[0].
        test_size: If float, should be between 0.0 and 1.0 and represent
            the proportion of the dataset to include in the test split.
            If int, represents the absolute number of test samples.
            If None, test_size is set to 0.25.
        random_state: Random seed for reproducibility.
        shuffle: Whether to shuffle before splitting. Defaults to True.

    Returns:
        List of split arrays: X_train, X_test, y_train, y_test, ...

    Example:
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    """
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")

    n_samples = len(arrays[0])

    if test_size is None:
        test_size = 0.25
    elif isinstance(test_size, float):
        test_size = int(n_samples * test_size)

    if isinstance(test_size, float):
        test_size = int(n_samples * test_size)

    n_test = int(test_size)
    n_train = n_samples - n_test

    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        indices = np.random.permutation(n_samples)
    else:
        indices = np.arange(n_samples)

    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    result = []
    for arr in arrays:
        result.append(arr[train_indices])
        result.append(arr[test_indices])

    return result


class BatchLoader:
    """Iterate over data in batches.

    Provides an iterable interface for batch processing during training.
    Supports shuffling and optional dropping of incomplete batches.

    Example:
        >>> loader = BatchLoader(X, y, batch_size=32, shuffle=True)
        >>> for X_batch, y_batch in loader:
        ...     # Training step
        ...     pass
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int | None = None
    ):
        """Initialize the BatchLoader.

        Args:
            X: Input features of shape (n_samples, ...).
            y: Optional target labels of shape (n_samples, ...).
            batch_size: Number of samples per batch. Defaults to 32.
            shuffle: Whether to shuffle data each epoch. Defaults to True.
            drop_last: Whether to drop the last incomplete batch. Defaults to False.
            seed: Random seed for reproducibility. Defaults to None.
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.n_samples = len(X)
        self._current_idx = 0
        self._indices = np.arange(self.n_samples)

    def __iter__(self) -> "BatchLoader":
        """Reset and return iterator."""
        if self.shuffle:
            if self.seed is not None:
                np.random.seed(self.seed)
            self._indices = np.random.permutation(self.n_samples)
        else:
            self._indices = np.arange(self.n_samples)
        self._current_idx = 0
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Get the next batch.

        Returns:
            Tuple of (X_batch, y_batch) or just X_batch if y is None.

        Raises:
            StopIteration: When all batches have been yielded.
        """
        if self._current_idx >= self.n_samples:
            raise StopIteration

        end_idx = min(self._current_idx + self.batch_size, self.n_samples)

        if self.drop_last and end_idx >= self.n_samples:
            raise StopIteration

        batch_indices = self._indices[self._current_idx:end_idx]
        self._current_idx = end_idx

        if self.y is not None:
            return self.X[batch_indices], self.y[batch_indices]
        return self.X[batch_indices], None

    def __len__(self) -> int:
        """Return the number of batches."""
        n_batches = self.n_samples // self.batch_size
        if not self.drop_last and self.n_samples % self.batch_size != 0:
            n_batches += 1
        return n_batches

    def __repr__(self) -> str:
        return (
            f"BatchLoader(n_samples={self.n_samples}, "
            f"batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}, "
            f"drop_last={self.drop_last})"
        )


class Dataset:
    """Base class for dataset abstraction.

    Provides a simple interface for accessing data samples.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray | None = None):
        """Initialize the Dataset.

        Args:
            X: Input features.
            y: Optional target labels.
        """
        self.X = X
        self.y = y
        self.n_samples = len(X)

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray | None]:
        """Get a single sample or batch.

        Args:
            idx: Index or slice.

        Returns:
            Tuple of (X, y) or just X if y is None.
        """
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx], None

    def __repr__(self) -> str:
        return f"Dataset(n_samples={self.n_samples})"


class TensorDataset(Dataset):
    """Dataset for combining multiple arrays.

    Combines multiple arrays into a single dataset for easy iteration.

    Example:
        >>> dataset = TensorDataset(X, y)
        >>> X0, y0 = dataset[0]
    """

    def __init__(self, *arrays):
        """Initialize the TensorDataset.

        Args:
            *arrays: Variable number of arrays to combine.
                     First array is treated as features, rest as targets.
        """
        if len(arrays) == 0:
            raise ValueError("At least one array required")
        self.arrays = arrays
        self.n_samples = len(arrays[0])

    def __getitem__(self, idx: int) -> tuple:
        """Get a single sample or batch.

        Args:
            idx: Index or slice.

        Returns:
            Tuple of arrays at the given index.
        """
        return tuple(arr[idx] for arr in self.arrays)

    def __repr__(self) -> str:
        return f"TensorDataset(n_samples={self.n_samples}, n_arrays={len(self.arrays)})"
