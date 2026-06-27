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
        if self.mask is not None:
            self.dinputs = dvalues * self.mask / (1 - self.rate)
        else:
            self.dinputs = dvalues
        return self.dinputs


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
            self.input = X  # Store for gradient checking
            self.std = np.sqrt(self.running_variance + self.epsilon)
            self.normalized = (X - self.running_mean) / self.std
            return self.gamma * self.normalized + self.beta

    def backward(self, dvalues):
        """Backward pass for batch normalization gradients.

        Args:
            dvalues: Gradient values from the next layer.

        Returns:
            Gradient with respect to inputs.
        """
        # Check if we have training statistics (needed for backward)
        # Inference mode: running stats are constants, so gradient is simpler
        if self.normalized is None or self.mean is None or self.var is None:
            # Inference mode: use running stats as constants
            # gradient of (x - mean) / std * gamma + beta w.r.t. x is just gamma / std
            self.dinputs = dvalues * (self.gamma / self.std)
            self.dgamma = np.sum(dvalues * self.normalized, axis=0, keepdims=True)
            self.dbeta = np.sum(dvalues, axis=0, keepdims=True)
            return self.dinputs
        
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

        self.dinputs = dinputs
        return dinputs


def _compute_conv_output_shape(input_h, input_w, kernel_size, stride, padding):
    """Compute output height and width for convolution."""
    if isinstance(kernel_size, int):
        kernel_h = kernel_w = kernel_size
    else:
        kernel_h, kernel_w = kernel_size

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if padding == "valid":
        out_h = (input_h - kernel_h) // stride_h + 1
        out_w = (input_w - kernel_w) // stride_w + 1
    elif padding == "same":
        out_h = (input_h + stride_h - 1) // stride_h
        out_w = (input_w + stride_w - 1) // stride_w
    else:
        raise ValueError(f"Unknown padding: {padding}")

    return out_h, out_w


def _get_padding(padding, kernel_size, stride):
    """Compute padding values for 'same' padding."""
    if padding != "same":
        return 0, 0

    if isinstance(kernel_size, int):
        kernel_h = kernel_w = kernel_size
    else:
        kernel_h, kernel_w = kernel_size

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    pad_h = max((kernel_h - 1) // 2, 0)
    pad_w = max((kernel_w - 1) // 2, 0)
    return pad_h, pad_w


def _pad_input(X, pad_h, pad_w):
    """Pad input tensor with zeros."""
    if pad_h == 0 and pad_w == 0:
        return X
    return np.pad(X, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant")


def _init_conv_kernel(kernel_h, kernel_w, in_channels, filters, initializer):
    """Initialize convolution kernels with given initializer."""
    fan_in = kernel_h * kernel_w * in_channels
    fan_out = kernel_h * kernel_w * filters

    if initializer == "he_normal":
        std = np.sqrt(2 / fan_in)
        return np.random.randn(kernel_h, kernel_w, in_channels, filters) * std
    elif initializer == "xavier_uniform":
        scale = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-scale, scale, (kernel_h, kernel_w, in_channels, filters))
    elif initializer == "lecun_normal":
        std = np.sqrt(1 / fan_in)
        return np.random.randn(kernel_h, kernel_w, in_channels, filters) * std
    else:
        raise ValueError(
            f"Unknown initializer: {initializer}. "
            f"Available: he_normal, xavier_uniform, lecun_normal"
        )


class Conv2D:
    """2D Convolution layer.

    Applies a 2D convolution over an input tensor composed of several input
    channels. Supports 'valid' and 'same' padding modes.
    """

    def __init__(
        self,
        filters,
        kernel_size,
        stride=1,
        padding="valid",
        initializer="he_normal"
    ):
        """Initialize the Conv2D layer.

        Args:
            filters: Number of output filters (output channels).
            kernel_size: Size of the convolution kernel. Can be an int or
                tuple of two ints (kernel_h, kernel_w).
            stride: Stride of the convolution. Can be an int or tuple of
                two ints (stride_h, stride_w). Defaults to 1.
            padding: Padding mode, either "valid" or "same". Defaults to "valid".
            initializer: Weight initializer. Defaults to "he_normal".
        """
        if padding not in ("valid", "same"):
            raise ValueError(f"padding must be 'valid' or 'same', got '{padding}'")

        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.initializer = initializer
        self.kernels = None
        self.biases = None
        self.input = None
        self.padded_input = None
        self.pad_h = 0
        self.pad_w = 0

    def forward(self, X, training=True):
        """Forward pass for 2D convolution.

        Args:
            X: Input array of shape (batch, height, width, channels).
            training: If True, cache inputs for backward pass.

        Returns:
            Output array of shape (batch, out_h, out_w, filters).
        """
        batch, in_h, in_w, in_channels = X.shape

        if isinstance(self.kernel_size, int):
            kernel_h = kernel_w = self.kernel_size
        else:
            kernel_h, kernel_w = self.kernel_size

        if isinstance(self.stride, int):
            stride_h = stride_w = self.stride
        else:
            stride_h, stride_w = self.stride

        # Initialize kernels if not already done
        if self.kernels is None:
            self.kernels = _init_conv_kernel(
                kernel_h, kernel_w, in_channels, self.filters, self.initializer
            )
            self.biases = np.zeros((self.filters,))

        # Compute output shape
        out_h, out_w = _compute_conv_output_shape(
            in_h, in_w, self.kernel_size, self.stride, self.padding
        )

        # Compute padding
        self.pad_h, self.pad_w = _get_padding(self.padding, self.kernel_size, self.stride)

        # Pad input if needed
        if training:
            self.input = X
            self.padded_input = _pad_input(X, self.pad_h, self.pad_w)
        else:
            self.padded_input = _pad_input(X, self.pad_h, self.pad_w)

        # Perform convolution using im2col approach
        output = np.zeros((batch, out_h, out_w, self.filters))

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride_h
                h_end = h_start + kernel_h
                w_start = j * stride_w
                w_end = w_start + kernel_w

                patch = self.padded_input[:, h_start:h_end, w_start:w_end, :]
                output[:, i, j, :] = np.tensordot(patch, self.kernels, axes=([1, 2, 3], [0, 1, 2])) + self.biases

        return output

    def backward(self, dvalues):
        """Backward pass for 2D convolution.

        Args:
            dvalues: Gradient values of shape (batch, out_h, out_w, filters).

        Returns:
            Gradient with respect to inputs of shape (batch, in_h, in_w, in_channels).
        """
        batch, in_h, in_w, in_channels = self.input.shape
        kernel_h, kernel_w, _, _ = self.kernels.shape

        if isinstance(self.stride, int):
            stride_h = stride_w = self.stride
        else:
            stride_h, stride_w = self.stride

        out_h, out_w = dvalues.shape[1], dvalues.shape[2]

        # Initialize gradients
        dkernels = np.zeros_like(self.kernels)
        dbiases = np.zeros_like(self.biases)
        dinputs_padded = np.zeros_like(self.padded_input)

        # Compute bias gradients
        dbiases = np.sum(dvalues, axis=(0, 1, 2))

        # Compute kernel and input gradients
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride_h
                h_end = h_start + kernel_h
                w_start = j * stride_w
                w_end = w_start + kernel_w

                patch = self.padded_input[:, h_start:h_end, w_start:w_end, :]

                # Kernel gradient
                # patch: (batch, kh, kw, c) -> dvalues: (batch, f)
                # dkernels: (kh, kw, c, f)
                for b in range(batch):
                    dkernels += np.expand_dims(patch[b], -1) * np.expand_dims(dvalues[b, i, j], (0, 1, 2))

                # Input gradient
                for b in range(batch):
                    dinputs_padded[b, h_start:h_end, w_start:w_end, :] += np.tensordot(
                        dvalues[b, i, j], self.kernels, axes=(0, 3)
                    )

        self.dkernels = dkernels
        self.dbiases = dbiases

        # Remove padding from input gradient
        if self.pad_h > 0 or self.pad_w > 0:
            if self.pad_h > 0 and self.pad_w > 0:
                dinputs = dinputs_padded[:, self.pad_h:-self.pad_h, self.pad_w:-self.pad_w, :]
            elif self.pad_h > 0:
                dinputs = dinputs_padded[:, self.pad_h:-self.pad_h, :, :]
            else:
                dinputs = dinputs_padded[:, :, self.pad_w:-self.pad_w, :]
        else:
            dinputs = dinputs_padded

        self.dinputs = dinputs
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