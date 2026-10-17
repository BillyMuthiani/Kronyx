"""Optimizers for the Kronyx deep learning framework.

Each optimizer exposes an ``update(layer)`` method that the training loop
calls once per layer after every backward pass.  The shared helper
``_trainable_params`` discovers which arrays on a layer are trainable so
individual optimizers stay focused on their update mathematics.
"""

import numpy as np


def _trainable_params(layer):
    """Yield (param_attr, grad_attr, state_prefix) per trainable parameter.

    param_attr   - layer attribute holding the parameter array.
    grad_attr    - layer attribute holding the gradient array.
    state_prefix  - short name for optimiser state variables
                    (e.g. "w", "b", "gamma", "beta").

    Supported layer types:
      - Dense:              weights/dweights, biases/dbiases
      - Conv2D:             kernels/dkernels,  biases/dbiases
      - BatchNormalization: gamma/dgamma,      beta/dbeta

    BatchNorm running_mean / running_variance are update-only
    statistics (corrected via momentum, not gradient descent) and are
    intentionally *excluded* from the pairs returned here.
    """
    if hasattr(layer, "weights") and layer.weights is not None:
        yield "weights", "dweights", "w"
        yield "biases", "dbiases", "b"
    elif hasattr(layer, "kernels") and layer.kernels is not None:
        yield "kernels", "dkernels", "w"
        yield "biases", "dbiases", "b"

    if hasattr(layer, "gamma") and layer.gamma is not None:
        yield "gamma", "dgamma", "gamma"
    if hasattr(layer, "beta") and layer.beta is not None:
        yield "beta", "dbeta", "beta"


class SGD:

    def __init__(
        self,
        learning_rate=0.01
    ):

        self.learning_rate = learning_rate

    def update(self, layer):

        for param_attr, grad_attr, _ in _trainable_params(layer):
            param = getattr(layer, param_attr)
            grad = getattr(layer, grad_attr)
            setattr(
                layer, param_attr,
                param - self.learning_rate * grad
            )


class Adam:

    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iterations = 0

    def update(self, layer):

        params = list(_trainable_params(layer))

        if not params:
            return

        self.iterations += 1

        for param_attr, grad_attr, state_prefix in params:

            param = getattr(layer, param_attr)
            grad = getattr(layer, grad_attr)

            m_attr = f"m_{state_prefix}"
            v_attr = f"v_{state_prefix}"

            if not hasattr(layer, m_attr):
                setattr(layer, m_attr, np.zeros_like(param))
                setattr(layer, v_attr, np.zeros_like(param))

            m = getattr(layer, m_attr)
            v = getattr(layer, v_attr)

            m = (
                self.beta1 * m
                + (1 - self.beta1) * grad
            )

            v = (
                self.beta2 * v
                + (1 - self.beta2) * (grad ** 2)
            )

            setattr(layer, m_attr, m)
            setattr(layer, v_attr, v)

            m_hat = m / (1 - self.beta1 ** self.iterations)
            v_hat = v / (1 - self.beta2 ** self.iterations)

            setattr(layer, param_attr, param - (
                self.learning_rate
                * m_hat
                / (np.sqrt(v_hat) + self.epsilon)
            ))
