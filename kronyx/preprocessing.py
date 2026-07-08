"""Preprocessing utilities for neural network inputs.

Lightweight, NumPy-only implementations of common preprocessing techniques.
Designed to be educational and sklearn-compatible where appropriate.
"""
import numpy as np


class StandardScaler:
    """Standardize features by removing mean and scaling to unit variance.

    Transforms features to have zero mean and unit variance, which helps
    neural networks train faster and more reliably.

    Formula: z = (x - mean) / std

    Example:
        >>> scaler = StandardScaler()
        >>> X_scaled = scaler.fit_transform(X)
        >>> X_original = scaler.inverse_transform(X_scaled)
    """

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Compute mean and std to be used for later scaling.

        Args:
            X: Input array of shape (n_samples, n_features).

        Returns:
            self
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero
        if self.std_ is not None:
            self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Perform standardization by centering and scaling.

        Args:
            X: Input array of shape (n_samples, n_features).

        Returns:
            Transformed array with same shape as X.
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted. Call fit() first.")
        return (X - self.mean_) / self.std_  # type: ignore[no-any-return]

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it.

        Args:
            X: Input array of shape (n_samples, n_features).

        Returns:
            Transformed array with same shape as X.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Undo the standardization.

        Args:
            X: Standardized array of shape (n_samples, n_features).

        Returns:
            Original-scale array.
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted. Call fit() first.")
        return X * self.std_ + self.mean_  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return f"StandardScaler(mean={self.mean_}, std={self.std_})"


class MinMaxScaler:
    """Transform features by scaling each to a given range.

    Scales and translates each feature individually to the specified
    range (default [0, 1]) using the formula:
    X_std = (X - X.min) / (X.max - X.min)
    X_scaled = X_std * (max - min) + min

    Example:
        >>> scaler = MinMaxScaler(feature_range=(0, 1))
        >>> X_scaled = scaler.fit_transform(X)
    """

    def __init__(self, feature_range: tuple[float, float] = (0, 1)):
        self.feature_range = feature_range
        self.data_min_: np.ndarray | None = None
        self.data_max_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.min_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        """Compute min and max to be used for later scaling.

        Args:
            X: Input array of shape (n_samples, n_features).

        Returns:
            self
        """
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        if self.data_max_ is not None and self.data_min_ is not None:
            self.scale_ = (self.data_max_ - self.data_min_)
            self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features to the defined range.

        Args:
            X: Input array of shape (n_samples, n_features).

        Returns:
            Scaled array.
        """
        if self.scale_ is None:
            raise ValueError("Scaler has not been fitted. Call fit() first.")
        if self.data_min_ is None:
            raise ValueError("Scaler has not been fitted. Call fit() first.")
        X_std = (X - self.data_min_) / self.scale_
        return X_std * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]  # type: ignore[no-any-return]

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it.

        Args:
            X: Input array of shape (n_samples, n_features).

        Returns:
            Scaled array.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Undo the scaling.

        Args:
            X: Scaled array.

        Returns:
            Original-scale array.
        """
        if self.scale_ is None:
            raise ValueError("Scaler has not been fitted. Call fit() first.")
        if self.data_min_ is None:
            raise ValueError("Scaler has not been fitted. Call fit() first.")
        X_std = (X - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        return X_std * self.scale_ + self.data_min_  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return f"MinMaxScaler(feature_range={self.feature_range})"


class RobustScaler:
    """Scale features using statistics that are robust to outliers.

    This scaler removes the median and scales the data according to
    the interquartile range (IQR). IQR is the range between the 1st
    and 3rd quartiles.

    Formula: z = (x - median) / IQR

    Example:
        >>> scaler = RobustScaler()
        >>> X_scaled = scaler.fit_transform(X)
    """

    def __init__(self, quantile_range: tuple[float, float] = (25.0, 75.0)):
        self.quantile_range = quantile_range
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "RobustScaler":
        """Compute median and IQR to be used for later scaling.

        Args:
            X: Input array of shape (n_samples, n_features).

        Returns:
            self
        """
        q1 = np.percentile(X, self.quantile_range[0], axis=0)
        q3 = np.percentile(X, self.quantile_range[1], axis=0)
        self.center_ = np.median(X, axis=0)
        self.scale_ = q3 - q1
        if self.scale_ is not None:
            self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features using median and IQR.

        Args:
            X: Input array of shape (n_samples, n_features).

        Returns:
            Scaled array.
        """
        if self.center_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted. Call fit() first.")
        return (X - self.center_) / self.scale_  # type: ignore[no-any-return]

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it.

        Args:
            X: Input array of shape (n_samples, n_features).

        Returns:
            Scaled array.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Undo the scaling.

        Args:
            X: Scaled array.

        Returns:
            Original-scale array.
        """
        if self.center_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted. Call fit() first.")
        return X * self.scale_ + self.center_  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return f"RobustScaler(quantile_range={self.quantile_range})"


class OneHotEncoder:
    """Encode labels as one-hot vectors.

    Converts integer labels to binary vectors where only one element
    is 1 (hot) and all others are 0.

    Example:
        >>> encoder = OneHotEncoder()
        >>> y_onehot = encoder.fit_transform([0, 1, 2])
        >>> y_onehot.shape
        (3, 3)
    """

    def __init__(self):
        self.categories_: np.ndarray | None = None
        self.n_classes_: int | None = None

    def fit(self, y: np.ndarray) -> "OneHotEncoder":
        """Learn the unique classes from the labels.

        Args:
            y: 1D array of integer labels.

        Returns:
            self
        """
        y_flat = y.flatten() if y.ndim > 1 else y
        self.categories_ = np.unique(y_flat)
        self.n_classes_ = len(self.categories_)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Transform labels to one-hot encoding.

        Args:
            y: 1D array of integer labels.

        Returns:
            2D array of shape (n_samples, n_classes).
        """
        if self.n_classes_ is None:
            raise ValueError("Encoder has not been fitted. Call fit() first.")
        if self.categories_ is None:
            raise ValueError("Encoder has not been fitted. Call fit() first.")
        y_flat = y.flatten() if y.ndim > 1 else y
        result = np.zeros((len(y_flat), self.n_classes_))
        for i, label in enumerate(y_flat):
            if label in self.categories_:
                idx = np.where(self.categories_ == label)[0][0]
                result[i, idx] = 1
        return result  # type: ignore[no-any-return]

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it.

        Args:
            y: 1D array of integer labels.

        Returns:
            2D one-hot encoded array.
        """
        return self.fit(y).transform(y)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Convert one-hot vectors back to labels.

        Args:
            y: 2D one-hot encoded array.

        Returns:
            1D array of integer labels.
        """
        if self.categories_ is None:
            raise ValueError("Encoder has not been fitted. Call fit() first.")
        return self.categories_[np.argmax(y, axis=1)]  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return f"OneHotEncoder(n_classes={self.n_classes_})"
