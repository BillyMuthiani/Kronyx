import numpy as np
import pytest

from kronyx import (
    BinaryCrossEntropy,
    Flatten,
    ReLU,
    Sequential,
    Sigmoid,
)
from kronyx.layers import Conv2D, Dense
from kronyx.optimizers import SGD, Adam


class TestSGD:
    @pytest.fixture
    def optimizer(self):
        return SGD(learning_rate=0.01)

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

        expected_weights = weights_before - 0.01 * layer.dweights
        expected_biases = biases_before - 0.01 * layer.dbiases

        np.testing.assert_allclose(layer.weights, expected_weights)
        np.testing.assert_allclose(layer.biases, expected_biases)

    def test_learning_rate(self):
        optimizer = SGD(learning_rate=0.1)
        assert optimizer.learning_rate == 0.1

    def test_no_weights(self, optimizer):
        class DummyLayer:
            pass

        dummy = DummyLayer()
        optimizer.update(dummy)

    def test_multiple_updates(self, optimizer, layer):
        weights_initial = layer.weights.copy()

        for _ in range(10):
            optimizer.update(layer)

        assert not np.allclose(layer.weights, weights_initial)
        expected = weights_initial - 10 * 0.01 * layer.dweights
        np.testing.assert_allclose(layer.weights, expected)

    @pytest.fixture
    def conv_layer(self):
        np.random.seed(42)
        conv = Conv2D(filters=4, kernel_size=3, padding="same")
        conv.forward(np.random.randn(2, 8, 8, 3), training=True)
        conv.dkernels = np.ones_like(conv.kernels) * 0.1
        conv.dbiases = np.ones_like(conv.biases) * 0.1
        return conv

    def test_conv2d_update_kernels(self, optimizer, conv_layer):
        conv_layer.kernels = np.ones_like(conv_layer.kernels)
        conv_layer.biases = np.zeros(conv_layer.biases.shape)
        orig_kernels = conv_layer.kernels.copy()

        optimizer.update(conv_layer)

        expected = orig_kernels - 0.01 * conv_layer.dkernels
        np.testing.assert_allclose(conv_layer.kernels, expected)

    def test_conv2d_update_biases(self, optimizer, conv_layer):
        conv_layer.kernels = np.zeros(conv_layer.kernels.shape)
        conv_layer.biases = np.ones_like(conv_layer.biases)
        orig_biases = conv_layer.biases.copy()

        optimizer.update(conv_layer)

        expected = orig_biases - 0.01 * conv_layer.dbiases
        np.testing.assert_allclose(conv_layer.biases, expected)

    def test_conv2d_zero_gradients(self, optimizer, conv_layer):
        conv_layer.dkernels = np.zeros_like(conv_layer.kernels)
        conv_layer.dbiases = np.zeros_like(conv_layer.biases)
        orig_k = conv_layer.kernels.copy()
        orig_b = conv_layer.biases.copy()

        optimizer.update(conv_layer)

        np.testing.assert_allclose(conv_layer.kernels, orig_k)
        np.testing.assert_allclose(conv_layer.biases, orig_b)

    def test_conv2d_multiple_updates(self, optimizer, conv_layer):
        orig_k = conv_layer.kernels.copy()

        for _ in range(10):
            optimizer.update(conv_layer)

        expected = orig_k - 10 * 0.01 * conv_layer.dkernels
        np.testing.assert_allclose(conv_layer.kernels, expected)


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
        for _ in range(100):
            optimizer.update(layer)

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
        optimizer.update(dummy)

    @pytest.fixture
    def conv_layer(self):
        np.random.seed(42)
        conv = Conv2D(filters=4, kernel_size=3, padding="same")
        conv.forward(np.random.randn(2, 8, 8, 3), training=True)
        conv.dkernels = np.ones_like(conv.kernels) * 0.1
        conv.dbiases = np.ones_like(conv.biases) * 0.1
        return conv

    def test_conv2d_update(self, optimizer, conv_layer):
        orig_kernels = conv_layer.kernels.copy()
        orig_biases = conv_layer.biases.copy()

        optimizer.update(conv_layer)

        assert not np.allclose(conv_layer.kernels, orig_kernels)
        assert not np.allclose(conv_layer.biases, orig_biases)

    def test_conv2d_momentum_buffers(self, optimizer, conv_layer):
        optimizer.update(conv_layer)

        assert hasattr(conv_layer, "m_w")
        assert hasattr(conv_layer, "v_w")
        assert hasattr(conv_layer, "m_b")
        assert hasattr(conv_layer, "v_b")

        assert conv_layer.m_w.shape == conv_layer.kernels.shape
        assert conv_layer.v_w.shape == conv_layer.kernels.shape
        assert conv_layer.m_b.shape == conv_layer.biases.shape
        assert conv_layer.v_b.shape == conv_layer.biases.shape

    def test_conv2d_zero_gradients(self, optimizer, conv_layer):
        conv_layer.dkernels = np.zeros_like(conv_layer.kernels)
        conv_layer.dbiases = np.zeros_like(conv_layer.biases)
        orig_k = conv_layer.kernels.copy()
        orig_b = conv_layer.biases.copy()

        optimizer.update(conv_layer)

        np.testing.assert_allclose(conv_layer.kernels, orig_k)
        np.testing.assert_allclose(conv_layer.biases, orig_b)

    def test_conv2d_multiple_updates(self, optimizer, conv_layer):
        orig_k = conv_layer.kernels.copy()

        for _ in range(100):
            optimizer.update(conv_layer)

        assert not np.allclose(conv_layer.kernels, orig_k)
        assert hasattr(conv_layer, "m_w")
        assert hasattr(conv_layer, "v_w")
        assert conv_layer.m_w.shape == conv_layer.kernels.shape


class TestBatchNormOptimizer:
    """Verify SGD and Adam correctly update BatchNorm gamma/beta and
    do NOT touch running_mean / running_variance."""

    @pytest.fixture
    def bn_layer(self):
        np.random.seed(42)
        from kronyx import BatchNormalization
        bn = BatchNormalization()
        x = np.random.randn(16, 8)
        bn.forward(x, training=True)  # initialise gamma, beta, running stats
        bn.dgamma = np.ones_like(bn.gamma) * 0.1
        bn.dbeta = np.ones_like(bn.beta) * 0.1
        return bn

    def test_sgd_updates_gamma_beta(self, bn_layer):
        from kronyx.optimizers import SGD
        orig_gamma = bn_layer.gamma.copy()
        orig_beta = bn_layer.beta.copy()
        sgd = SGD(learning_rate=0.01)
        sgd.update(bn_layer)
        np.testing.assert_allclose(
            bn_layer.gamma, orig_gamma - 0.01 * bn_layer.dgamma
        )
        np.testing.assert_allclose(
            bn_layer.beta, orig_beta - 0.01 * bn_layer.dbeta
        )

    def test_sgd_zero_updates(self, bn_layer):
        from kronyx.optimizers import SGD
        bn_layer.dgamma = np.zeros_like(bn_layer.gamma)
        bn_layer.dbeta = np.zeros_like(bn_layer.beta)
        orig_g = bn_layer.gamma.copy()
        orig_b = bn_layer.beta.copy()
        SGD().update(bn_layer)
        np.testing.assert_allclose(bn_layer.gamma, orig_g)
        np.testing.assert_allclose(bn_layer.beta, orig_b)

    def test_sgd_no_running_stats_clobber(self, bn_layer):
        """SGD must not touch running statistics."""
        from kronyx.optimizers import SGD
        orig_rm = bn_layer.running_mean.copy()
        orig_rv = bn_layer.running_variance.copy()
        SGD().update(bn_layer)
        np.testing.assert_allclose(bn_layer.running_mean, orig_rm)
        np.testing.assert_allclose(bn_layer.running_variance, orig_rv)

    def test_adam_updates_gamma_beta(self, bn_layer):
        from kronyx.optimizers import Adam
        orig_gamma = bn_layer.gamma.copy()
        orig_beta = bn_layer.beta.copy()
        adam = Adam(learning_rate=0.001)
        adam.update(bn_layer)
        assert not np.allclose(bn_layer.gamma, orig_gamma)
        assert not np.allclose(bn_layer.beta, orig_beta)

    def test_adam_momentum_buffers(self, bn_layer):
        from kronyx.optimizers import Adam
        Adam().update(bn_layer)
        assert hasattr(bn_layer, "m_gamma")
        assert hasattr(bn_layer, "v_gamma")
        assert hasattr(bn_layer, "m_beta")
        assert hasattr(bn_layer, "v_beta")
        assert bn_layer.m_gamma.shape == bn_layer.gamma.shape
        assert bn_layer.v_gamma.shape == bn_layer.gamma.shape
        assert bn_layer.m_beta.shape == bn_layer.beta.shape
        assert bn_layer.v_beta.shape == bn_layer.beta.shape

    def test_adam_zero_gradients(self, bn_layer):
        """With zero gradients Adam should not change params (first step)."""
        from kronyx.optimizers import Adam
        bn_layer.dgamma = np.zeros_like(bn_layer.gamma)
        bn_layer.dbeta = np.zeros_like(bn_layer.beta)
        orig_g = bn_layer.gamma.copy()
        orig_b = bn_layer.beta.copy()
        Adam().update(bn_layer)
        np.testing.assert_allclose(bn_layer.gamma, orig_g)
        np.testing.assert_allclose(bn_layer.beta, orig_b)

    def test_state_isolation(self, bn_layer):
        """Adam state for gamma must not leak into beta and vice-versa."""
        from kronyx.optimizers import Adam
        Adam().update(bn_layer)
        # m_w would be created for Dense/Conv2D kernels, but not for BatchNorm
        assert not hasattr(bn_layer, "m_w")
        # gamma and beta have distinct state buffers
        assert not np.shares_memory(bn_layer.m_gamma, bn_layer.m_beta)


class TestMixedModelTraining:
    """Verify that a model containing both Conv2D and Dense layers
    trains without silently skipping either layer's updates."""

    @pytest.fixture
    def model_data(self):
        np.random.seed(42)
        model = Sequential()
        model.add(Conv2D(filters=4, kernel_size=3, padding="same"))
        model.add(ReLU())
        model.add(Flatten())
        model.add(Dense(4 * 8 * 8, 8))
        model.add(ReLU())
        model.add(Dense(8, 1))
        model.add(Sigmoid())
        model.compile(
            loss=BinaryCrossEntropy(),
            optimizer=SGD(learning_rate=0.1),
            metric=None,
        )
        x = np.random.randn(8, 8, 8, 3)
        y = np.random.randint(0, 2, (8, 1)).astype(float)
        return model, x, y

    def test_conv2d_and_dense_both_update(self, model_data):
        model, x, y = model_data
        # Initialise params via a forward pass, then snapshot them
        _ = model.predict(x, verbose=0)
        conv = model.layers[0]
        dense = model.layers[3]
        conv_kernels_before = conv.kernels.copy()
        dense_weights_before = dense.weights.copy()

        model.fit(x, y, epochs=5, verbose=0)

        assert not np.allclose(conv_kernels_before, conv.kernels), \
            "Conv2D kernels should change during training"
        assert not np.allclose(dense_weights_before, dense.weights), \
            "Dense weights should change during training"

    @pytest.mark.parametrize("OptimizerClass", [SGD, Adam])
    def test_mixed_model_trains(self, OptimizerClass):
        from kronyx.losses import BinaryCrossEntropy as Bce

        np.random.seed(42)
        model = Sequential()
        model.add(Conv2D(filters=4, kernel_size=3, padding="same"))
        model.add(ReLU())
        model.add(Flatten())
        model.add(Dense(4 * 8 * 8, 8))
        model.add(ReLU())
        model.add(Dense(8, 1))
        model.add(Sigmoid())
        model.compile(
            loss=Bce(),
            optimizer=OptimizerClass(learning_rate=0.1),
            metric=None,
        )
        x = np.random.randn(8, 8, 8, 3)
        y = np.random.randint(0, 2, (8, 1)).astype(float)
        _ = model.predict(x, verbose=0)  # initialise params
        conv = model.layers[0]
        dense = model.layers[3]
        conv_before = conv.kernels.copy()
        dense_before = dense.weights.copy()

        model.fit(x, y, epochs=3, verbose=0)

        assert not np.allclose(conv_before, conv.kernels)
        assert not np.allclose(dense_before, dense.weights)
