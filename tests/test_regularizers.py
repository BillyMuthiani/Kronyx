import pytest
import numpy as np
from Neuralnet.regularizers import L2


class TestL2Regularizer:
    @pytest.fixture
    def l2(self):
        return L2(lambda_=0.1)

    @pytest.fixture
    def weights(self):
        np.random.seed(42)
        return np.random.randn(4, 8)

    def test_loss(self, l2, weights):
        loss = l2.loss(weights)
        expected = 0.1 * np.sum(weights ** 2)
        assert np.isclose(loss, expected)

    def test_gradient(self, l2, weights):
        grad = l2.gradient(weights)
        expected = 2 * 0.1 * weights
        np.testing.assert_allclose(grad, expected)

    def test_default_lambda(self):
        l2 = L2()
        assert l2.lambda_ == 1e-4

    def test_custom_lambda(self):
        l2 = L2(lambda_=0.05)
        assert l2.lambda_ == 0.05

    def test_gradient_check(self):
        from tests.gradient_check import check_regularizer_gradient
        
        l2 = L2(lambda_=0.1)
        weights = np.random.randn(4, 8)
        
        passed, _, _ = check_regularizer_gradient(l2, weights, name="L2")
        assert passed, "L2 regularizer gradient check failed"

    def test_zero_weights(self, l2):
        weights = np.zeros((2, 3))
        loss = l2.loss(weights)
        grad = l2.gradient(weights)
        assert loss == 0
        np.testing.assert_allclose(grad, 0)