import pytest
import numpy as np
from Neuralnet import Conv2D


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
        from tests.gradient_check import check_layer_gradient, check_layer_weight_gradient
        
        conv = Conv2D(filters=4, kernel_size=3, padding="same")
        input_data = np.random.randn(2, 8, 8, 3)
        
        passed, _, _ = check_layer_gradient(conv, input_data, "Conv2D")
        assert passed, "Conv2D input gradient check failed"
        
        passed, _, _ = check_layer_weight_gradient(conv, input_data, "Conv2D")
        assert passed, "Conv2D weight gradient check failed"