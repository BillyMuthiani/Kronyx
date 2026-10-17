from collections.abc import Callable

import numpy as np


def numerical_gradient(
    func: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-7
) -> np.ndarray:
    """Compute numerical gradient using central difference.

    Args:
        func: Function that takes input array and returns scalar output.
        x: Input array at which to compute gradient.
        eps: Small perturbation for finite difference.

    Returns:
        Numerical gradient array of same shape as x.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index

        x_plus = x.copy()
        x_plus[idx] += eps
        f_plus = func(x_plus)

        x_minus = x.copy()
        x_minus[idx] -= eps
        f_minus = func(x_minus)

        grad[idx] = (f_plus - f_minus) / (2 * eps)

        it.iternext()

    return grad


def check_gradient(
    analytical_grad: np.ndarray,
    numerical_grad: np.ndarray,
    tolerance: float = 1e-5,
    name: str = "Gradient",
    abs_tolerance: float = 1e-7,
) -> tuple[bool, float, float]:
    """Compare analytical and numerical gradients.

    Uses a dual-tolerance criterion: a coordinate passes if EITHER the
    relative error is below *tolerance* OR the absolute error is below
    *abs_tolerance*.  The absolute criterion is essential for coordinates
    where the gradient is genuinely near zero -- there the relative
    error is mathematically ill-conditioned (numerator and denominator
    both ~0) and can report ~1.0 even when the absolute discrepancy is
    at the level of floating-point noise.

    Args:
        analytical_grad: Analytical gradient from backprop.
        numerical_grad: Numerical gradient from finite difference.
        tolerance: Maximum allowed relative error.
        name: Name for reporting.
        abs_tolerance: Maximum allowed absolute error (for near-zero coords).

    Returns:
        Tuple of (passed, max_abs_error, max_rel_error).
    """
    abs_error = np.abs(analytical_grad - numerical_grad)
    max_abs_error = np.max(abs_error)

    denom = np.abs(analytical_grad) + np.abs(numerical_grad) + 1e-12
    rel_error = abs_error / denom
    max_rel_error = np.max(rel_error)

    passed = max_rel_error < tolerance or max_abs_error < abs_tolerance

    status = 'PASS' if passed else 'FAIL'
    print(
        f"  {name}: max_abs={max_abs_error:.6e}, "
        f"max_rel={max_rel_error:.6e} -> {status}"
    )

    return passed, max_abs_error, max_rel_error


def check_layer_gradient(
    layer,
    x: np.ndarray,
    layer_name: str = "Layer",
    eps: float = 1e-7,
    tolerance: float = 1e-5,
    training: bool = True
) -> tuple[bool, float, float]:
    """Check gradients for a layer with scalar loss function.

    Uses sum of outputs as scalar loss: loss = sum(forward(x))

    Args:
        layer: Layer instance with forward() and backward() methods.
        x: Input array.
        layer_name: Name for reporting.
        eps: Perturbation for finite difference.
        tolerance: Maximum allowed relative error.
        training: Whether to run in training mode.

    Returns:
        Tuple of (passed, max_abs_error, max_rel_error).
    """
    def forward_sum(x_flat):
        x_reshaped = x_flat.reshape(x.shape)
        out = layer.forward(x_reshaped, training=training)
        return np.sum(out)

    def forward_grad(x_flat):
        x_reshaped = x_flat.reshape(x.shape)
        out = layer.forward(x_reshaped, training=training)
        dout = np.ones_like(out)
        layer.backward(dout)
        if hasattr(layer, 'dinputs') and layer.dinputs is not None:
            return layer.dinputs.flatten()
        elif hasattr(layer, 'input') and layer.input is not None:
            return layer.input.flatten()
        else:
            return layer.dinputs.flatten()

    x_flat = x.flatten()

    numerical_grad = numerical_gradient(forward_sum, x_flat, eps)
    analytical_grad = forward_grad(x_flat)

    return check_gradient(analytical_grad, numerical_grad, tolerance, layer_name)


def check_layer_param_gradient(
    layer,
    x: np.ndarray,
    param_attr: str,
    grad_attr: str,
    layer_name: str = "Layer",
    eps: float = 1e-7,
    tolerance: float = 1e-5,
    training: bool = True
) -> tuple[bool, float, float]:
    """Check gradients for an arbitrary layer parameter.

    Generalises ``check_layer_weight_gradient`` to work with any
    (param_attr, grad_attr) pair -- e.g. weights/dweights, kernels/dkernels,
    gamma/dgamma, beta/dbeta.

    Args:
        layer: Layer instance with forward() and backward() methods.
        x: Input array.
        param_attr: Layer attribute holding the parameter array.
        grad_attr: Layer attribute holding the gradient array.
        layer_name: Name for reporting.
        eps: Perturbation for finite difference.
        tolerance: Maximum allowed relative error.
        training: Whether to run in training mode.

    Returns:
        Tuple of (passed, max_abs_error, max_rel_error).
    """
    param = getattr(layer, param_attr)
    if param is None:
        # Parameter not initialised yet -- initialise via a forward pass
        layer.forward(x, training=training)
        param = getattr(layer, param_attr)

    param_shape = param.shape
    param_flat = param.flatten()

    def forward_sum(p_flat):
        setattr(layer, param_attr, p_flat.reshape(param_shape))
        out = layer.forward(x, training=training)
        return np.sum(out)

    def forward_grad(p_flat):
        setattr(layer, param_attr, p_flat.reshape(param_shape))
        out = layer.forward(x, training=training)
        dout = np.ones_like(out)
        layer.backward(dout)
        return getattr(layer, grad_attr).flatten()

    original_param = param.copy()

    try:
        numerical_grad = numerical_gradient(forward_sum, param_flat, eps)
        analytical_grad = forward_grad(param_flat)

        passed, max_abs, max_rel = check_gradient(
            analytical_grad, numerical_grad, tolerance,
            f"{layer_name} {param_attr}"
        )
        return passed, max_abs, max_rel
    finally:
        setattr(layer, param_attr, original_param)


def check_layer_gradient_weighted(
    layer,
    x: np.ndarray,
    upstream_grad: np.ndarray,
    layer_name: str = "Layer",
    eps: float = 1e-7,
    tolerance: float = 1e-5,
    training: bool = True
) -> tuple[bool, float, float]:
    """Check input-gradient correctness using a non-trivial upstream gradient.

    Unlike ``check_layer_gradient`` (which uses a flat all-ones upstream
    gradient and ``sum(forward(x))`` as the scalar loss), this function
    uses a fixed, non-constant upstream-gradient tensor.  The scalar
    objective is ``sum(forward(x) * upstream_grad)``, so every output
    coordinate contributes with a different weight -- exercising the
    backward implementation along multiple directions simultaneously.

    This is particularly important for BatchNormalization in training
    mode, where a flat upstream gradient can produce gradients that
    appear correct but only probe a single linear combination of the
    batch-statistic derivatives.

    Args:
        layer: Layer instance with ``forward()`` and ``backward()`` methods.
        x: Input array at which to evaluate gradients.
        upstream_grad: Fixed non-trivial upstream gradient tensor with the
            same shape as the layer's output.  Used both as the ``dout``
            argument to ``backward()`` and as the per-element weight in the
            scalar objective ``sum(forward(x) * upstream_grad)``.
        layer_name: Name for reporting.
        eps: Perturbation for finite difference.
        tolerance: Maximum allowed relative error.
        training: Whether to run in training mode.

    Returns:
        Tuple of (passed, max_abs_error, max_rel_error).
    """
    def forward_loss(x_flat):
        x_reshaped = x_flat.reshape(x.shape)
        out = layer.forward(x_reshaped, training=training)
        return np.sum(out * upstream_grad)

    def forward_grad(x_flat):
        x_reshaped = x_flat.reshape(x.shape)
        layer.forward(x_reshaped, training=training)
        layer.backward(upstream_grad)
        if hasattr(layer, 'dinputs') and layer.dinputs is not None:
            return layer.dinputs.flatten()
        elif hasattr(layer, 'input') and layer.input is not None:
            return layer.input.flatten()
        else:
            return layer.dinputs.flatten()

    x_flat = x.flatten()

    numerical_grad = numerical_gradient(forward_loss, x_flat, eps)
    analytical_grad = forward_grad(x_flat)

    return check_gradient(analytical_grad, numerical_grad, tolerance, layer_name)


def check_layer_param_gradient_weighted(
    layer,
    x: np.ndarray,
    upstream_grad: np.ndarray,
    param_attr: str,
    grad_attr: str,
    layer_name: str = "Layer",
    eps: float = 1e-7,
    tolerance: float = 1e-5,
    training: bool = True
) -> tuple[bool, float, float]:
    """Check parameter-gradient correctness using a non-trivial upstream gradient.

    Generalises ``check_layer_param_gradient`` by replacing the flat
    all-ones upstream gradient with a fixed, non-constant tensor.  The
    scalar objective is ``sum(forward(x) * upstream_grad)`` and the
    analytical backward is seeded with ``dout = upstream_grad``.

    Args:
        layer: Layer instance with ``forward()`` and ``backward()`` methods.
        x: Input array.
        upstream_grad: Fixed non-trivial upstream gradient tensor with the
            same shape as the layer's output.
        param_attr: Layer attribute holding the parameter array.
        grad_attr: Layer attribute holding the gradient array.
        layer_name: Name for reporting.
        eps: Perturbation for finite difference.
        tolerance: Maximum allowed relative error.
        training: Whether to run in training mode.

    Returns:
        Tuple of (passed, max_abs_error, max_rel_error).
    """
    param = getattr(layer, param_attr)
    if param is None:
        layer.forward(x, training=training)
        param = getattr(layer, param_attr)

    param_shape = param.shape
    param_flat = param.flatten()

    def forward_loss(p_flat):
        setattr(layer, param_attr, p_flat.reshape(param_shape))
        out = layer.forward(x, training=training)
        return np.sum(out * upstream_grad)

    def forward_grad(p_flat):
        setattr(layer, param_attr, p_flat.reshape(param_shape))
        layer.forward(x, training=training)
        layer.backward(upstream_grad)
        return getattr(layer, grad_attr).flatten()

    original_param = param.copy()

    try:
        numerical_grad = numerical_gradient(forward_loss, param_flat, eps)
        analytical_grad = forward_grad(param_flat)

        passed, max_abs, max_rel = check_gradient(
            analytical_grad, numerical_grad, tolerance,
            f"{layer_name} {param_attr}"
        )
        return passed, max_abs, max_rel
    finally:
        setattr(layer, param_attr, original_param)


def check_layer_weight_gradient(
    layer,
    x: np.ndarray,
    layer_name: str = "Layer",
    eps: float = 1e-7,
    tolerance: float = 1e-5,
    training: bool = True
) -> tuple[bool, float, float]:
    """Check gradients for layer weights (if applicable).

    Args:
        layer: Layer instance with weights attribute.
        x: Input array.
        layer_name: Name for reporting.
        eps: Perturbation for finite difference.
        tolerance: Maximum allowed relative error.
        training: Whether to run in training mode.

    Returns:
        Tuple of (passed, max_abs_error, max_rel_error) or (True, 0, 0) if no weights.
    """
    if not hasattr(layer, 'weights') or layer.weights is None:
        return True, 0.0, 0.0

    return check_layer_param_gradient(
        layer, x, "weights", "dweights", layer_name, eps, tolerance, training
    )


def check_regularizer_gradient(
    regularizer,
    weights: np.ndarray,
    eps: float = 1e-7,
    tolerance: float = 1e-5,
    name: str = "Regularizer"
) -> tuple[bool, float, float]:
    """Check gradients for a regularizer.

    Args:
        regularizer: Regularizer instance with loss() and gradient() methods.
        weights: Weight array.
        eps: Perturbation for finite difference.
        tolerance: Maximum allowed relative error.
        name: Name for reporting.

    Returns:
        Tuple of (passed, max_abs_error, max_rel_error).
    """
    weights_flat = weights.flatten()

    def loss_func(w_flat):
        return regularizer.loss(w_flat.reshape(weights.shape))

    def grad_func(w_flat):
        return regularizer.gradient(w_flat.reshape(weights.shape)).flatten()

    numerical_grad = numerical_gradient(loss_func, weights_flat, eps)
    analytical_grad = grad_func(weights_flat)

    return check_gradient(analytical_grad, numerical_grad, tolerance, name)


def run_gradient_checks():
    """Run all gradient checks for the framework."""
    from kronyx import Conv2D, Dense, Dropout
    from kronyx.regularizers import L2

    np.random.seed(42)
    all_passed = True

    print("=" * 60)
    print("GRADIENT CHECKING")
    print("=" * 60)

    print("\n1. Dense Layer")
    dense = Dense(4, 8, initializer="he_normal")
    x = np.random.randn(2, 4)
    passed, _, _ = check_layer_gradient(dense, x, "Dense")
    all_passed &= passed

    passed, _, _ = check_layer_weight_gradient(dense, x, "Dense")
    all_passed &= passed

    print("\n2. Conv2D Layer")
    conv = Conv2D(filters=4, kernel_size=3, padding="same")
    x = np.random.randn(2, 8, 8, 3)
    passed, _, _ = check_layer_gradient(conv, x, "Conv2D")
    all_passed &= passed

    # Conv2D weight gradients (kernels and biases) -- use the generalised
    # param checker because Conv2D has no 'weights' attribute.
    conv = Conv2D(filters=4, kernel_size=3, padding="same")
    passed, _, _ = check_layer_param_gradient(
        conv, x, "kernels", "dkernels", "Conv2D", training=True
    )
    all_passed &= passed

    conv = Conv2D(filters=4, kernel_size=3, padding="same")
    passed, _, _ = check_layer_param_gradient(
        conv, x, "biases", "dbiases", "Conv2D biases", training=True
    )
    all_passed &= passed

    print("\n3. BatchNormalization Layer")
    from kronyx import BatchNormalization
    x = np.random.randn(16, 8)
    bn = BatchNormalization()

    # Training-mode input gradient
    passed, _, _ = check_layer_gradient(
        bn, x, "BatchNorm (training)", training=True
    )
    all_passed &= passed

    # Training-mode gamma gradient
    bn = BatchNormalization()
    passed, _, _ = check_layer_param_gradient(
        bn, x, "gamma", "dgamma", "BatchNorm", training=True
    )
    all_passed &= passed

    # Training-mode beta gradient
    bn = BatchNormalization()
    passed, _, _ = check_layer_param_gradient(
        bn, x, "beta", "dbeta", "BatchNorm beta", training=True
    )
    all_passed &= passed

    # Inference-mode input gradient (uses cached running stats)
    bn = BatchNormalization()
    # Prime running stats with a training forward pass
    bn.forward(x, training=True)
    passed, _, _ = check_layer_gradient(
        bn, x, "BatchNorm (inference)", training=False
    )
    all_passed &= passed

    # --- Weighted upstream-gradient checks (non-trivial direction) ---
    # Using sum(forward(x) * w) instead of sum(forward(x)) exercises
    # multiple gradient directions simultaneously.  Critical for BN
    # training-mode where the batch-statistic derivatives interact.
    w = np.random.randn(*x.shape)  # fixed non-trivial upstream gradient

    print("\n3b. BatchNorm weighted training-mode gradient checks")
    bn = BatchNormalization()
    passed, _, _ = check_layer_gradient_weighted(
        bn, x, w, "BatchNorm (weighted training)", training=True
    )
    all_passed &= passed

    bn = BatchNormalization()
    # Prime gamma/beta with a training forward so they are initialised
    bn.forward(x, training=True)
    # BN output has same shape as input (16, 8), not gamma shape (1, 8)
    w_out = np.random.randn(*x.shape)
    passed, _, _ = check_layer_param_gradient_weighted(
        bn, x, w_out, "gamma", "dgamma", "BatchNorm", training=True
    )
    all_passed &= passed

    bn = BatchNormalization()
    bn.forward(x, training=True)
    w_out = np.random.randn(*x.shape)
    passed, _, _ = check_layer_param_gradient_weighted(
        bn, x, w_out, "beta", "dbeta", "BatchNorm beta", training=True
    )
    all_passed &= passed

    print("\n3c. BatchNorm weighted inference-mode gradient check")
    bn = BatchNormalization()
    bn.forward(x, training=True)  # prime running stats
    w_out = np.random.randn(*x.shape)
    passed, _, _ = check_layer_gradient_weighted(
        bn, x, w_out, "BatchNorm (weighted inference)", training=False
    )
    all_passed &= passed

    print("\n4. Dropout Layer (inference)")
    dropout = Dropout(rate=0.3)
    x = np.random.randn(2, 10)
    passed, _, _ = check_layer_gradient(
        dropout, x, "Dropout (inference)", training=False
    )
    all_passed &= passed

    print("\n5. L2 Regularizer")
    l2 = L2(lambda_=0.1)
    weights = np.random.randn(4, 8)
    passed, _, _ = check_regularizer_gradient(l2, weights, name="L2")
    all_passed &= passed

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = run_gradient_checks()
    exit(0 if success else 1)
