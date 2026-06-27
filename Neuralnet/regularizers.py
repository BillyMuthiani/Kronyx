import numpy as np


class L2:
    """L2 regularization for weight decay.

    Adds penalty equal to lambda * sum(weights^2) to loss and adds
    lambda * weights to weight gradients during backpropagation.
    """

    def __init__(self, lambda_=1e-4):
        """Initialize L2 regularizer.

        Args:
            lambda_: Regularization strength. Defaults to 1e-4.
        """
        self.lambda_ = lambda_

    def loss(self, weights):
        """Compute L2 regularization loss.

        Args:
            weights: Weight matrix to regularize.

        Returns:
            Scalar regularization loss value.
        """
        return self.lambda_ * np.sum(weights ** 2)

    def gradient(self, weights):
        """Compute gradient of L2 regularization.

        Args:
            weights: Weight matrix to regularize.

        Returns:
            Gradient array of same shape as weights.
        """
        return 2 * self.lambda_ * weights