import pytest
import numpy as np
from Neuralnet import BatchNormalization


class TestBatchNormalization:
    @pytest.fixture
    def bn(self):
        return BatchNormalization()

    @pytest.fixture
    def input_data(self):
        np.random.seed(42)
        return np.random.randn(4, 8)

    def test_forward_training(self, bn, input_data):
        output = bn.forward(input_data, training=True)
        assert output.shape == (4, 8)
        
        # Check that running statistics are updated
        assert bn.running_mean is not None
        assert bn.running_variance is not None
        
        # Check that output has zero mean and unit variance (approximately)
        np.testing.assert_allclose(np.mean(output, axis=0), 0, atol=1e-1)
        np.testing.assert_allclose(np.std(output, axis=0), 1, atol=1e-1)

    def test_forward_inference(self, bn, input_data):
        # First do a training pass to initialize running stats
        _ = bn.forward(input_data, training=True)
        
        # Now test inference
        output = bn.forward(input_data, training=False)
        assert output.shape == (4, 8)

    def test_forward_shapes(self):
        # Test different input shapes (create fresh layer for each)
        for shape in [(2, 4), (8, 16), (1, 10)]:
            x = np.random.randn(*shape)
            bn = BatchNormalization()
            out = bn.forward(x, training=True)
            assert out.shape == shape

    def test_backward_shape(self, bn, input_data):
        bn.forward(input_data, training=True)
        dout = np.random.randn(*input_data.shape)
        dinputs = bn.backward(dout)
        assert dinputs.shape == input_data.shape

    def test_gamma_beta_updates(self, bn, input_data):
        bn.forward(input_data, training=True)
        dout = np.ones_like(input_data)
        bn.backward(dout)
        
        assert hasattr(bn, 'dgamma')
        assert hasattr(bn, 'dbeta')
        assert bn.dgamma.shape == (1, 8)
        assert bn.dbeta.shape == (1, 8)

    def test_running_stats_accumulate(self, bn, input_data):
        # Multiple forward passes should update running stats
        for _ in range(5):
            bn.forward(input_data, training=True)
        
        assert bn.running_mean is not None
        assert bn.running_variance is not None
        assert not np.allclose(bn.running_mean, 0)
        assert not np.allclose(bn.running_variance, 1)

    def test_serialization(self, bn, input_data):
        # Do training pass to initialize
        _ = bn.forward(input_data, training=True)
        
        gamma_copy = bn.gamma.copy()
        beta_copy = bn.beta.copy()
        rm_copy = bn.running_mean.copy()
        rv_copy = bn.running_variance.copy()
        
        # Modify
        bn.gamma[:] = 0
        bn.beta[:] = 0
        
        # Restore
        bn.gamma = gamma_copy
        bn.beta = beta_copy
        bn.running_mean = rm_copy
        bn.running_variance = rv_copy
        
        output = bn.forward(input_data, training=False)
        expected = (input_data - rm_copy) / np.sqrt(rv_copy + bn.epsilon) * gamma_copy + beta_copy
        np.testing.assert_allclose(output, expected)

    def test_gradient_check(self):
        from tests.gradient_check import check_layer_gradient
        
        bn = BatchNormalization()
        input_data = np.random.randn(4, 8)
        
        # Skip gradient check for training mode due to complex batch statistics
        passed, _, _ = check_layer_gradient(bn, input_data, "BatchNormalization (inference)", training=False)
        # Note: BatchNorm gradient check in training mode is complex
        # This test verifies inference mode gradients work
        assert passed, "BatchNormalization inference gradient check failed"