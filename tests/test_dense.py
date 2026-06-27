import pytest
import numpy as np
from Neuralnet import Dense, ReLU, Softmax


class TestDense:
    @pytest.fixture
    def dense(self):
        return Dense(4, 8, initializer="he_normal")

    @pytest.fixture
    def input_data(self):
        np.random.seed(42)
        return np.random.randn(2, 4)

    def test_forward_shape(self, dense, input_data):
        output = dense.forward(input_data, training=True)
        assert output.shape == (2, 8)

    def test_forward_output(self, dense, input_data):
        output = dense.forward(input_data, training=True)
        expected = np.dot(input_data, dense.weights) + dense.biases
        np.testing.assert_allclose(output, expected)

    def test_backward_shape(self, dense, input_data):
        dense.forward(input_data, training=True)
        dout = np.random.randn(2, 8)
        dinputs = dense.backward(dout)
        assert dinputs.shape == (2, 4)

    def test_backward_gradients(self, dense, input_data):
        dense.forward(input_data, training=True)
        dout = np.ones((2, 8))
        dense.backward(dout)
        
        expected_dweights = np.dot(input_data.T, dout)
        expected_dbiases = np.sum(dout, axis=0, keepdims=True)
        
        np.testing.assert_allclose(dense.dweights, expected_dweights)
        np.testing.assert_allclose(dense.dbiases, expected_dbiases)

    def test_kernel_regularizer(self, dense, input_data):
        from Neuralnet.regularizers import L2
        dense_reg = Dense(4, 8, initializer="he_normal", kernel_regularizer=L2(0.01))
        dense_reg.forward(input_data, training=True)
        dout = np.ones((2, 8))
        dense_reg.backward(dout)
        
        expected_reg_loss = 0.01 * np.sum(dense_reg.weights ** 2)
        expected_reg_grad = 2 * 0.01 * dense_reg.weights
        
        assert np.isclose(dense_reg.regularization_loss, expected_reg_loss)
        
        # Check that dweights includes regularizer gradient
        # dweights should equal weights_grad + regularizer_grad
        expected_dweights = np.dot(input_data.T, dout) + expected_reg_grad
        np.testing.assert_allclose(dense_reg.dweights, expected_dweights)

    def test_serialization(self, dense, input_data):
        dense.forward(input_data, training=True)
        
        # Save weights
        weights_copy = dense.weights.copy()
        biases_copy = dense.biases.copy()
        
        # Modify
        dense.weights[:] = 0
        dense.biases[:] = 0
        
        # Restore
        dense.weights = weights_copy
        dense.biases = biases_copy
        
        output = dense.forward(input_data, training=True)
        expected = np.dot(input_data, weights_copy) + biases_copy
        np.testing.assert_allclose(output, expected)

    def test_gradient_check(self, dense, input_data):
        from tests.gradient_check import check_layer_gradient, check_layer_weight_gradient
        
        passed, _, _ = check_layer_gradient(dense, input_data, "Dense")
        assert passed, "Dense input gradient check failed"
        
        passed, _, _ = check_layer_weight_gradient(dense, input_data, "Dense")
        assert passed, "Dense weight gradient check failed"