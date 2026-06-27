import pytest
import numpy as np
from Neuralnet.optimizers import SGD, Adam
from Neuralnet.layers import Dense
from Neuralnet.initializers import he_normal


class TestSGD:
    @pytest.fixture
    def optimizer(self):
        return SGD(learning_rate=0.01)

    @pytest.fixture
    def layer(self):
        np.random.seed(42)
        dense = Dense(4, 8, initializer="he_normal")
        # Add some gradients
        dense.dweights = np.random.randn(4, 8) * 0.1
        dense.dbiases = np.random.randn(1, 8) * 0.1
        return dense

    def test_update_weights(self, optimizer, layer):
        weights_before = layer.weights.copy()
        biases_before = layer.biases.copy()
        
        optimizer.update(layer)
        
        expected_weights = weights_before - 0.01 * layer.dweights
        expected_biases = biases_before - 0.01 * layer.dbiases
        
        np.testing.assert_allclose(layer.weights, expected_weights)
        np.testing.assert_allclose(layer.biases, expected_biases)

    def test_learning_rate(self):
        optimizer = SGD(learning_rate=0.1)
        assert optimizer.learning_rate == 0.1

    def test_no_weights(self, optimizer):
        # Layer without weights should not cause error
        class DummyLayer:
            pass
        
        dummy = DummyLayer()
        optimizer.update(dummy)  # Should not raise error

    def test_multiple_updates(self, optimizer, layer):
        weights_initial = layer.weights.copy()
        
        for _ in range(10):
            optimizer.update(layer)
        
        assert not np.allclose(layer.weights, weights_initial)
        expected = weights_initial - 10 * 0.01 * layer.dweights
        np.testing.assert_allclose(layer.weights, expected)


class TestAdam:
    @pytest.fixture
    def optimizer(self):
        return Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)

    @pytest.fixture
    def layer(self):
        np.random.seed(42)
        dense = Dense(4, 8, initializer="he_normal")
        dense.dweights = np.random.randn(4, 8) * 0.1
        dense.dbiases = np.random.randn(1, 8) * 0.1
        return dense

    def test_update_weights(self, optimizer, layer):
        weights_before = layer.weights.copy()
        biases_before = layer.biases.copy()
        
        optimizer.update(layer)
        
        # Weights should have changed
        assert not np.allclose(layer.weights, weights_before)
        assert not np.allclose(layer.biases, biases_before)

    def test_momentum_buffers(self, optimizer, layer):
        optimizer.update(layer)
        
        assert hasattr(layer, 'm_w')
        assert hasattr(layer, 'v_w')
        assert hasattr(layer, 'm_b')
        assert hasattr(layer, 'v_b')
        
        assert layer.m_w.shape == layer.weights.shape
        assert layer.v_w.shape == layer.weights.shape

    def test_bias_correction(self, optimizer, layer):
        # After many updates, bias correction should stabilize
        for _ in range(100):
            optimizer.update(layer)
        
        # Check that momentum buffers exist and have correct shape
        assert hasattr(layer, 'm_w')
        assert hasattr(layer, 'v_w')
        assert layer.m_w.shape == layer.weights.shape
        assert layer.v_w.shape == layer.weights.shape

    def test_learning_rate(self):
        optimizer = Adam(learning_rate=0.01)
        assert optimizer.learning_rate == 0.01

    def test_hyperparameters(self):
        optimizer = Adam(learning_rate=0.001, beta1=0.8, beta2=0.99, epsilon=1e-6)
        assert optimizer.learning_rate == 0.001
        assert optimizer.beta1 == 0.8
        assert optimizer.beta2 == 0.99
        assert optimizer.epsilon == 1e-6

    def test_no_weights(self, optimizer):
        class DummyLayer:
            pass
        
        dummy = DummyLayer()
        optimizer.update(dummy)  # Should not raise error