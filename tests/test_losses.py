import pytest
import numpy as np
from Neuralnet.losses import SoftmaxCategoricalCrossEntropy, BinaryCrossEntropy


class TestSoftmaxCategoricalCrossEntropy:
    @pytest.fixture
    def loss(self):
        return SoftmaxCategoricalCrossEntropy()

    def test_forward(self, loss):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5]
        ])
        loss_val = loss.forward(y_true, y_pred)
        assert isinstance(loss_val, float)
        assert loss_val >= 0

    def test_forward_one_hot(self, loss):
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5]
        ])
        loss_val = loss.forward(y_true, y_pred)
        assert isinstance(loss_val, float)

    def test_backward_shape(self, loss):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5]
        ])
        loss.forward(y_true, y_pred)
        dinputs = loss.backward(y_true, y_pred)
        assert dinputs.shape == (3, 3)

    def test_perfect_prediction(self, loss):
        y_true = np.array([0, 1, 2])
        y_pred = np.eye(3)[y_true]
        loss_val = loss.forward(y_true, y_pred)
        assert np.isclose(loss_val, 0, atol=1e-5)

    def test_clipping(self, loss):
        # Very small predictions should be clipped
        y_true = np.array([0])
        y_pred = np.array([[0.0, 1.0]])
        loss_val = loss.forward(y_true, y_pred)
        assert loss_val < 100  # Should not be infinite


class TestBinaryCrossEntropy:
    @pytest.fixture
    def loss(self):
        return BinaryCrossEntropy()

    def test_forward(self, loss):
        y_true = np.array([[1], [0], [1]])
        y_pred = np.array([[0.9], [0.1], [0.8]])
        loss_val = loss.forward(y_true, y_pred)
        assert isinstance(loss_val, float)
        assert loss_val >= 0

    def test_backward_shape(self, loss):
        y_true = np.array([[1], [0]])
        y_pred = np.array([[0.9], [0.1]])
        loss.forward(y_true, y_pred)
        dinputs = loss.backward(y_true, y_pred)
        assert dinputs.shape == (2, 1)

    def test_perfect_prediction(self, loss):
        y_true = np.array([[1], [0]])
        y_pred = np.array([[1.0], [0.0]])
        loss_val = loss.forward(y_true, y_pred)
        assert np.isclose(loss_val, 0, atol=1e-5)

    def test_clipping(self, loss):
        y_true = np.array([[1]])
        y_pred = np.array([[0.0]])
        loss_val = loss.forward(y_true, y_pred)
        assert loss_val < 100  # Should not be infinite