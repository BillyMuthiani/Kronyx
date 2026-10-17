import numpy as np
import pytest

from kronyx import Conv2D


def _reference_backward(conv, dvalues):
    """Reference (original loop-based) Conv2D backward.

    Kept here solely to verify the optimised implementation produces
    numerically identical results.  Duplicated from the pre-optimisation
    implementation.
    """
    batch, in_h, in_w, in_channels = conv.input.shape
    kernel_h, kernel_w, _, _ = conv.kernels.shape

    if isinstance(conv.stride, int):
        stride_h = stride_w = conv.stride
    else:
        stride_h, stride_w = conv.stride

    out_h, out_w = dvalues.shape[1], dvalues.shape[2]

    dkernels = np.zeros_like(conv.kernels)
    dbiases = np.sum(dvalues, axis=(0, 1, 2))
    dinputs_padded = np.zeros_like(conv.padded_input)

    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride_h
            h_end = h_start + kernel_h
            w_start = j * stride_w
            w_end = w_start + kernel_w

            patch = conv.padded_input[:, h_start:h_end, w_start:w_end, :]

            for b in range(batch):
                dkernels += np.expand_dims(patch[b], -1) * np.expand_dims(
                    dvalues[b, i, j], (0, 1, 2)
                )

            for b in range(batch):
                dinputs_padded[b, h_start:h_end, w_start:w_end, :] += np.tensordot(
                    dvalues[b, i, j], conv.kernels, axes=(0, 3)
                )

    if conv.pad_h > 0 or conv.pad_w > 0:
        if conv.pad_h > 0 and conv.pad_w > 0:
            dinputs = dinputs_padded[:, conv.pad_h:-conv.pad_h, conv.pad_w:-conv.pad_w, :]
        elif conv.pad_h > 0:
            dinputs = dinputs_padded[:, conv.pad_h:-conv.pad_h, :, :]
        else:
            dinputs = dinputs_padded[:, :, conv.pad_w:-conv.pad_w, :]
    else:
        dinputs = dinputs_padded

    return dinputs, dkernels, dbiases


class TestConv2D:
    @pytest.fixture
    def conv(self):
        return Conv2D(filters=4, kernel_size=3, padding="same")

    @pytest.fixture
    def input_data(self):
        np.random.seed(42)
        return np.random.randn(2, 8, 8, 3)

    def test_forward_same_padding(self, conv, input_data):
        output = conv.forward(input_data, training=True)
        assert output.shape == (2, 8, 8, 4)

    def test_forward_valid_padding(self):
        conv = Conv2D(filters=4, kernel_size=3, padding="valid")
        input_data = np.random.randn(2, 8, 8, 3)
        output = conv.forward(input_data, training=True)
        assert output.shape == (2, 6, 6, 4)

    def test_forward_stride(self):
        conv = Conv2D(filters=4, kernel_size=3, stride=2, padding="valid")
        input_data = np.random.randn(2, 8, 8, 3)
        output = conv.forward(input_data, training=True)
        assert output.shape == (2, 3, 3, 4)

    def test_forward_tuple_kernel(self):
        conv = Conv2D(filters=4, kernel_size=(3, 5), padding="same")
        input_data = np.random.randn(2, 8, 8, 3)
        output = conv.forward(input_data, training=True)
        assert output.shape == (2, 8, 8, 4)

    def test_backward_shape(self, conv, input_data):
        output = conv.forward(input_data, training=True)
        dout = np.random.randn(*output.shape)
        dinputs = conv.backward(dout)
        assert dinputs.shape == input_data.shape

    def test_kernel_initialization(self):
        conv = Conv2D(filters=8, kernel_size=3, padding="same")
        input_data = np.random.randn(2, 8, 8, 3)
        _ = conv.forward(input_data, training=True)
        assert conv.kernels.shape == (3, 3, 3, 8)
        assert conv.biases.shape == (8,)

    def test_serialization(self, conv, input_data):
        output1 = conv.forward(input_data, training=True)

        kernels_copy = conv.kernels.copy()
        biases_copy = conv.biases.copy()

        conv.kernels[:] = 0
        conv.biases[:] = 0

        conv.kernels = kernels_copy
        conv.biases = biases_copy

        output2 = conv.forward(input_data, training=True)
        np.testing.assert_allclose(output1, output2)

    def test_gradient_check(self):
        from tests.gradient_check import (
            check_layer_gradient,
            check_layer_param_gradient,
        )

        conv = Conv2D(filters=4, kernel_size=3, padding="same")
        input_data = np.random.randn(2, 8, 8, 3)

        passed, _, _ = check_layer_gradient(conv, input_data, "Conv2D")
        assert passed, "Conv2D input gradient check failed"

        # Conv2D kernel gradient (check_layer_weight_gradient skips layers
        # without a 'weights' attribute, so use the generalised param checker)
        conv = Conv2D(filters=4, kernel_size=3, padding="same")
        passed, _, _ = check_layer_param_gradient(
            conv, input_data, "kernels", "dkernels", "Conv2D", training=True
        )
        assert passed, "Conv2D kernel gradient check failed"

        # Conv2D bias gradient
        conv = Conv2D(filters=4, kernel_size=3, padding="same")
        passed, _, _ = check_layer_param_gradient(
            conv, input_data, "biases", "dbiases", "Conv2D biases", training=True
        )
        assert passed, "Conv2D bias gradient check failed"

    def test_gradient_check_stride(self):
        """Conv2D gradient check with stride=2."""
        from tests.gradient_check import (
            check_layer_gradient,
            check_layer_param_gradient,
        )

        conv = Conv2D(filters=4, kernel_size=3, stride=2, padding="same")
        input_data = np.random.randn(2, 8, 8, 3)

        passed, _, _ = check_layer_gradient(conv, input_data, "Conv2D (stride=2)")
        assert passed, "Conv2D stride input gradient check failed"

        conv = Conv2D(filters=4, kernel_size=3, stride=2, padding="same")
        passed, _, _ = check_layer_param_gradient(
            conv, input_data, "kernels", "dkernels", "Conv2D (stride=2)", training=True
        )
        assert passed, "Conv2D stride kernel gradient check failed"

    def test_gradient_check_multi_channel(self):
        """Conv2D gradient check with multiple input channels."""
        from tests.gradient_check import (
            check_layer_gradient,
            check_layer_param_gradient,
        )

        conv = Conv2D(filters=4, kernel_size=3, padding="same")
        input_data = np.random.randn(2, 8, 8, 6)

        passed, _, _ = check_layer_gradient(conv, input_data, "Conv2D (6-channel)")
        assert passed, "Conv2D multi-channel input gradient check failed"

        conv = Conv2D(filters=4, kernel_size=3, padding="same")
        passed, _, _ = check_layer_param_gradient(
            conv, input_data, "kernels", "dkernels", "Conv2D (6-channel)", training=True
        )
        assert passed, "Conv2D multi-channel kernel gradient check failed"


class TestConv2DBackwardOptimized:
    """Verify the optimised Conv2D backward matches the reference implementation."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _setup(self, **kwargs):
        conv = Conv2D(**kwargs)
        in_shape = (2, 8, 8, 3)
        x = np.random.randn(*in_shape)
        out = conv.forward(x, training=True)
        dout = np.random.randn(*out.shape)
        return conv, x, out, dout

    def test_matches_reference_same_padding(self):
        conv, x, out, dout = self._setup(filters=4, kernel_size=3, padding="same")
        dinputs_opt = conv.backward(dout)
        dkernels_opt = conv.dkernels.copy()
        dbiases_opt = conv.dbiases.copy()

        conv2 = Conv2D(filters=4, kernel_size=3, padding="same")
        conv2.kernels = conv.kernels.copy()
        conv2.biases = conv.biases.copy()
        conv2.forward(x, training=True)
        dinputs_ref, dkernels_ref, dbiases_ref = _reference_backward(conv2, dout)

        np.testing.assert_allclose(dinputs_opt, dinputs_ref, atol=1e-12)
        np.testing.assert_allclose(dkernels_opt, dkernels_ref, atol=1e-12)
        np.testing.assert_allclose(dbiases_opt, dbiases_ref, atol=1e-12)

    def test_matches_reference_stride2(self):
        conv, x, out, dout = self._setup(
            filters=4, kernel_size=3, stride=2, padding="same"
        )
        dinputs_opt = conv.backward(dout)
        dkernels_opt = conv.dkernels.copy()

        conv2 = Conv2D(filters=4, kernel_size=3, stride=2, padding="same")
        conv2.kernels = conv.kernels.copy()
        conv2.biases = conv.biases.copy()
        conv2.forward(x, training=True)
        dinputs_ref, dkernels_ref, _ = _reference_backward(conv2, dout)

        np.testing.assert_allclose(dinputs_opt, dinputs_ref, atol=1e-12)
        np.testing.assert_allclose(dkernels_opt, dkernels_ref, atol=1e-12)

    def test_matches_reference_multi_channel(self):
        np.random.seed(7)
        conv = Conv2D(filters=8, kernel_size=3, padding="same")
        x = np.random.randn(2, 8, 8, 6)
        out = conv.forward(x, training=True)
        dout = np.random.randn(*out.shape)

        dinputs_opt = conv.backward(dout)
        dkernels_opt = conv.dkernels.copy()

        conv2 = Conv2D(filters=8, kernel_size=3, padding="same")
        conv2.kernels = conv.kernels.copy()
        conv2.biases = conv.biases.copy()
        conv2.forward(x, training=True)
        dinputs_ref, dkernels_ref, _ = _reference_backward(conv2, dout)

        np.testing.assert_allclose(dinputs_opt, dinputs_ref, atol=1e-12)
        np.testing.assert_allclose(dkernels_opt, dkernels_ref, atol=1e-12)

    def test_matches_reference_valid_padding(self):
        np.random.seed(11)
        conv = Conv2D(filters=4, kernel_size=3, padding="valid")
        x = np.random.randn(2, 8, 8, 3)
        out = conv.forward(x, training=True)
        dout = np.random.randn(*out.shape)

        dinputs_opt = conv.backward(dout)
        dkernels_opt = conv.dkernels.copy()

        conv2 = Conv2D(filters=4, kernel_size=3, padding="valid")
        conv2.kernels = conv.kernels.copy()
        conv2.biases = conv.biases.copy()
        conv2.forward(x, training=True)
        dinputs_ref, dkernels_ref, _ = _reference_backward(conv2, dout)

        np.testing.assert_allclose(dinputs_opt, dinputs_ref, atol=1e-12)
        np.testing.assert_allclose(dkernels_opt, dkernels_ref, atol=1e-12)

    def test_matches_reference_tuple_kernel(self):
        np.random.seed(13)
        conv = Conv2D(filters=4, kernel_size=(3, 5), padding="same")
        x = np.random.randn(2, 8, 8, 3)
        out = conv.forward(x, training=True)
        dout = np.random.randn(*out.shape)

        dinputs_opt = conv.backward(dout)
        dkernels_opt = conv.dkernels.copy()

        conv2 = Conv2D(filters=4, kernel_size=(3, 5), padding="same")
        conv2.kernels = conv.kernels.copy()
        conv2.biases = conv.biases.copy()
        conv2.forward(x, training=True)
        dinputs_ref, dkernels_ref, _ = _reference_backward(conv2, dout)

        np.testing.assert_allclose(dinputs_opt, dinputs_ref, atol=1e-12)
        np.testing.assert_allclose(dkernels_opt, dkernels_ref, atol=1e-12)
