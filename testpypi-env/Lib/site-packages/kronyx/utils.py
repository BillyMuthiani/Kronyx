"""Utility functions for Kronyx."""
import numpy as np


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
