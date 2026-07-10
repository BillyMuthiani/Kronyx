"""Utility functions for Kronyx."""
import numpy as np

from kronyx.exceptions import ShapeError


def set_seed(seed):
    """Set random seed for reproducibility.

    Seeds numpy's random number generator for reproducible results.

    Args:
        seed: Integer seed value.

    Example:
        >>> from kronyx import set_seed
        >>> set_seed(42)
        >>> # Now results will be reproducible
    """
    np.random.seed(seed)


def validate_binary_shapes(y_true, y_pred):
    """Validate and normalize shapes for binary classification.

    Accepts y_true as (N,) or (N, 1) and normalizes to (N, 1).
    Requires y_pred to be (N, 1). Raises ShapeError for any shape
    mismatch or broadcast-unsafe combination.

    Args:
        y_true: True labels array.
        y_pred: Prediction probabilities array.

    Returns:
        Tuple of (y_true_normalized, y_pred).

    Raises:
        ShapeError: If y_pred shape is not (N, 1), if y_true shape is
            not (N,) or (N, 1), or if sample counts differ.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_pred.ndim != 2 or y_pred.shape[1] != 1:
        raise ShapeError(
            f"y_pred shape {y_pred.shape} is invalid for binary classification "
            f"— expected (N, 1). Check your final layer's output size."
        )

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    elif y_true.ndim == 2 and y_true.shape[1] == 1:
        y_true = y_true
    else:
        raise ShapeError(
            f"y_true shape {y_true.shape} is invalid for binary classification "
            f"— expected (N,) or (N, 1)."
        )

    if y_true.shape[0] != y_pred.shape[0]:
        raise ShapeError(
            f"y_true has {y_true.shape[0]} samples but y_pred has "
            f"{y_pred.shape[0]} samples. Shapes {y_true.shape} and "
            f"{y_pred.shape} are not aligned."
        )

    return y_true, y_pred
