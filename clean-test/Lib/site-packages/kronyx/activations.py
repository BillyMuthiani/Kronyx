"""Activation functions for neural network layers."""
import numpy as np


class ReLU:
    """ReLU (Rectified Linear Unit) activation function.

    Computes element-wise: max(0, x). Helps mitigate vanishing gradient problem.

    Examples:
        >>> layer = ReLU()
        >>> output = layer.forward(np.array([[-1, 0, 1]], training=True))
        >>> output
        array([[0, 0, 1]])
    """

    def forward(self, x, training=True):
        """Apply ReLU activation.

        Args:
            x: Input array of any shape.
            training: If True, cache input for backward pass.

        Returns:
            Array with ReLU activation applied.
        """
        if training:
            self.input = x

        return np.maximum(0, x)

    def backward(self, dvalues):
        """Backward pass for ReLU activation.

        Args:
            dvalues: Gradient from the next layer.

        Returns:
            Gradient with ReLU mask applied (zeros where input was negative).
        """
        self.dinputs = dvalues.copy()

        self.dinputs[self.input <= 0] = 0

        return self.dinputs


class Sigmoid:
    """Sigmoid activation function.

    Computes element-wise: 1 / (1 + exp(-x)). Squashes values to (0, 1).
    """

    def forward(self, x, training=True):
        """Apply sigmoid activation.

        Args:
            x: Input array of any shape.
            training: If True, cache output for backward pass.

        Returns:
            Array with sigmoid activation applied.
        """
        output = 1 / (1 + np.exp(-x))

        if training:
            self.output = output

        return output

    def backward(self, dvalues):
        """Backward pass for sigmoid activation.

        Args:
            dvalues: Gradient from the next layer.

        Returns:
            Gradient with sigmoid derivative applied.
        """
        self.dinputs = (
            dvalues
            * self.output
            * (1 - self.output)
        )

        return self.dinputs


class Tanh:
    """Tanh (Hyperbolic Tangent) activation function.

    Computes element-wise: tanh(x). Squashes values to (-1, 1).
    """

    def forward(self, x, training=True):
        """Apply tanh activation.

        Args:
            x: Input array of any shape.
            training: If True, cache output for backward pass.

        Returns:
            Array with tanh activation applied.
        """
        output = np.tanh(x)

        if training:
            self.output = output

        return output

    def backward(self, dvalues):
        """Backward pass for tanh activation.

        Args:
            dvalues: Gradient from the next layer.

        Returns:
            Gradient with tanh derivative applied.
        """
        self.dinputs = (
            dvalues
            * (1 - self.output ** 2)
        )

        return self.dinputs


class Softmax:
    """Softmax activation function for multi-class classification.

    Computes: exp(x_i) / sum(exp(x)) across the class dimension.
    Outputs sum to 1 and can be interpreted as probabilities.
    """

    def forward(self, x, training=True):
        """Apply softmax activation.

        Args:
            x: Input array of shape (batch, classes).
            training: If True, cache output for backward pass.

        Returns:
            Array of shape (batch, classes) with softmax probabilities.
        """
        exp_values = np.exp(
            x - np.max(x, axis=1, keepdims=True)
        )

        output = (
            exp_values
            /
            np.sum(
                exp_values,
                axis=1,
                keepdims=True
            )
        )

        if training:
            self.output = output

        return output

    def backward(self, dvalues):
        """Backward pass for softmax activation.

        Args:
            dvalues: Gradient from the loss function.

        Returns:
            Gradient with softmax Jacobian applied.
        """
        self.dinputs = np.empty_like(dvalues)

        for index, (
            single_output,
            single_dvalues
        ) in enumerate(
            zip(self.output, dvalues, strict=False)
        ):
            single_output = single_output.reshape(
                -1,
                1
            )

            jacobian = (
                np.diagflat(single_output)
                -
                np.dot(
                    single_output,
                    single_output.T
                )
            )

            self.dinputs[index] = np.dot(
                jacobian,
                single_dvalues
            )

        return self.dinputs
