import numpy as np
from Neuralnet.initializers import he_normal, xavier_uniform, lecun_normal


class Dense:

    def __init__(self, input_size, output_size, initializer="he_normal", kernel_regularizer=None):

        init_func = {
            "he_normal": he_normal,
            "xavier_uniform": xavier_uniform,
            "lecun_normal": lecun_normal,
        }.get(initializer)

        if init_func is None:
            raise ValueError(
                f"Unknown initializer: {initializer}. "
                f"Available: he_normal, xavier_uniform, lecun_normal"
            )

        self.weights = init_func(input_size, output_size)
        self.biases = np.zeros((1, output_size))
        self.kernel_regularizer = kernel_regularizer
        self.regularization_loss = 0.0

    def forward(self, X, training=True):

        if training:
            self.input = X

        return np.dot(X, self.weights) + self.biases

    def backward(self, dvalues):

        self.dweights = np.dot(
            self.input.T,
            dvalues
        )

        self.dbiases = np.sum(
            dvalues,
            axis=0,
            keepdims=True
        )

        self.dinputs = np.dot(
            dvalues,
            self.weights.T
        )

        if self.kernel_regularizer is not None:
            self.dweights += self.kernel_regularizer.gradient(self.weights)
            self.regularization_loss = self.kernel_regularizer.loss(self.weights)

        return self.dinputs


class Dropout:
    """Dropout layer for regularization.

    Randomly zeros a fraction of input units during training to prevent overfitting.
    Uses inverted dropout scaling to maintain expected sums.
    """

    def __init__(self, rate):
        """Initialize the Dropout layer.

        Args:
            rate: Fraction of input units to drop (0 to 1). Must be in range [0, 1).

        Raises:
            ValueError: If rate is not in valid range.
        """
        if rate < 0 or rate >= 1:
            raise ValueError(
                f"rate must be in range [0, 1), got {rate}"
            )
        self.rate = rate
        self.mask = None
        self.input = None

    def forward(self, X, training=True):
        """Apply dropout to input.

        During training, randomly zeros activations with probability rate and scales
        remaining activations by 1/(1-rate). During inference, returns inputs unchanged.

        Args:
            X: Input array to apply dropout to.
            training: If True, apply dropout; if False, return inputs unchanged.

        Returns:
            Input with dropout applied during training, or original input during
            inference.
        """
        if training:
            self.input = X
            self.mask = np.random.rand(*X.shape) > self.rate
            return X * self.mask / (1 - self.rate)
        return X

    def backward(self, dvalues):
        """Backward pass applying cached dropout mask.

        Args:
            dvalues: Gradient values from the next layer.

        Returns:
            Gradient values with dropout mask applied.
        """
        return dvalues * self.mask / (1 - self.rate)


class BatchNormalization:
    """Batch normalization layer for stabilizing training.

    Normalizes inputs across the batch dimension and learns scale (gamma) and
    shift (beta) parameters during training.
    """

    def __init__(self, momentum=0.9, epsilon=1e-5):
        """Initialize the BatchNormalization layer.

        Args:
            momentum: Momentum for running mean/variance updates. Defaults to 0.9.
            epsilon: Small constant for numerical stability. Defaults to 1e-5.
        """
        self.momentum = momentum
        self.epsilon = epsilon
        self.gamma = None
        self.beta = None
        self.running_mean = None
        self.running_variance = None
        self.input = None
        self.normalized = None
        self.mean = None
        self.var = None
        self.std = None

    def forward(self, X, training=True):
        """Normalize inputs using batch or running statistics.

        During training, computes batch mean/variance and updates running statistics.
        During inference, uses running statistics for normalization.

        Args:
            X: Input array to normalize.
            training: If True, use batch statistics; if False, use running statistics.

        Returns:
            Normalized input with learned scale and shift applied.
        """
        if self.gamma is None:
            self.gamma = np.ones((1, X.shape[1]))
            self.beta = np.zeros((1, X.shape[1]))
            self.running_mean = np.zeros((1, X.shape[1]))
            self.running_variance = np.ones((1, X.shape[1]))

        if training:
            self.input = X
            self.mean = np.mean(X, axis=0, keepdims=True)
            self.var = np.var(X, axis=0, keepdims=True)
            self.std = np.sqrt(self.var + self.epsilon)

            self.normalized = (X - self.mean) / self.std

            self.running_mean = (
                self.momentum * self.running_mean +
                (1 - self.momentum) * self.mean
            )
            self.running_variance = (
                self.momentum * self.running_variance +
                (1 - self.momentum) * self.var
            )

            return self.gamma * self.normalized + self.beta
        else:
            normalized = (X - self.running_mean) / np.sqrt(self.running_variance + self.epsilon)
            return self.gamma * normalized + self.beta

    def backward(self, dvalues):
        """Backward pass for batch normalization gradients.

        Args:
            dvalues: Gradient values from the next layer.

        Returns:
            Gradient with respect to inputs.
        """
        N = self.input.shape[0]

        dgamma = np.sum(dvalues * self.normalized, axis=0, keepdims=True)
        dbeta = np.sum(dvalues, axis=0, keepdims=True)

        self.dgamma = dgamma
        self.dbeta = dbeta

        dnormalized = dvalues * self.gamma
        dvar = np.sum(
            dnormalized * (self.input - self.mean) * -0.5 *
            (self.var + self.epsilon) ** (-3/2),
            axis=0,
            keepdims=True
        )

        dmean = np.sum(
            dnormalized * -1 / self.std,
            axis=0,
            keepdims=True
        ) + dvar * np.sum(-2 * (self.input - self.mean), axis=0, keepdims=True) / N

        dinputs = (
            dnormalized / self.std +
            dvar * 2 * (self.input - self.mean) / N +
            dmean / N
        )

        return dinputs


class Flatten:
    """Flatten layer for reshaping multi-dimensional inputs.

    Flattens input tensors from shape (batch, ...) to (batch, features) during
    forward pass. Restores original shape during backward pass.
    """

    def __init__(self):
        """Initialize the Flatten layer."""
        self.input_shape = None

    def forward(self, X, training=True):
        """Flatten input to 2D shape (batch, features).

        Args:
            X: Input array of any shape.
            training: Ignored (included for API consistency).

        Returns:
            Flattened array of shape (batch, features) where features is the
            product of all dimensions after the batch dimension.
        """
        if training:
            self.input_shape = X.shape

        return X.reshape(X.shape[0], -1)

    def backward(self, dvalues):
        """Restore original input shape from flattened gradients.

        Args:
            dvalues: Gradient values of shape (batch, features).

        Returns:
            Gradient values reshaped to original input dimensions.
        """
        return dvalues.reshape(self.input_shape)