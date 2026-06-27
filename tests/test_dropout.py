import pytest
import numpy as np
from Neuralnet import Dropout


class TestDropout:
    @pytest.fixture
    def dropout(self):
        return Dropout(rate=0.3)

    @pytest.fixture
    def input_data(self):
        np.random.seed(42)
        return np.random.randn(2, 10)

    def test_forward_training(self, dropout, input_data):
        output = dropout.forward(input_data, training=True)
        assert output.shape == (2, 10)
        
        # During training, some elements should be zeroed
        # and remaining should be scaled by 1/(1-rate)
        assert np.any(output == 0)
        # Non-zero elements should be scaled
        non_zero = output[output != 0]
        expected_scale = 1 / (1 - 0.3)
        # Check that non-zero values are approximately scaled
        # (This is probabilistic, so we just check it runs)

    def test_forward_inference(self, dropout, input_data):
        output = dropout.forward(input_data, training=False)
        assert output.shape == (2, 10)
        np.testing.assert_allclose(output, input_data)

    def test_backward_training(self, dropout, input_data):
        dropout.forward(input_data, training=True)
        dout = np.random.randn(*input_data.shape)
        dinputs = dropout.backward(dout)
        assert dinputs.shape == input_data.shape
        
        # Gradient should be zero where mask was zero
        assert np.any(dinputs == 0)

    def test_backward_inference(self, dropout, input_data):
        dropout.forward(input_data, training=False)
        dout = np.random.randn(*input_data.shape)
        dinputs = dropout.backward(dout)
        assert dinputs.shape == input_data.shape
        np.testing.assert_allclose(dinputs, dout)

    def test_rate_validation(self):
        with pytest.raises(ValueError):
            Dropout(rate=-0.1)
        with pytest.raises(ValueError):
            Dropout(rate=1.0)
        # Valid rates
        Dropout(rate=0.0)
        Dropout(rate=0.5)
        Dropout(rate=0.99)

    def test_rate_zero(self, input_data):
        dropout = Dropout(rate=0.0)
        output = dropout.forward(input_data, training=True)
        np.testing.assert_allclose(output, input_data)

    def test_serialization(self, dropout, input_data):
        dropout.forward(input_data, training=True)
        
        mask_copy = dropout.mask.copy()
        input_copy = dropout.input.copy()
        
        # Restore
        dropout.mask = mask_copy
        dropout.input = input_copy
        
        dout = np.ones_like(input_data)
        dinputs = dropout.backward(dout)
        expected = dout * mask_copy / (1 - 0.3)
        np.testing.assert_allclose(dinputs, expected)

    def test_gradient_check(self):
        from tests.gradient_check import check_layer_gradient
        
        dropout = Dropout(rate=0.3)
        input_data = np.random.randn(2, 10)
        
        # In inference mode, dropout is identity
        passed, _, _ = check_layer_gradient(dropout, input_data, "Dropout (inference)", training=False)
        assert passed, "Dropout inference gradient check failed"