import numpy as np
import pytest

from kronyx.exceptions import ShapeError
from kronyx.losses import BinaryCrossEntropy, SoftmaxCategoricalCrossEntropy


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


class TestBinaryCrossEntropyShapeValidation:
    def test_forward_n_labels_with_n1_pred(self):
        loss = BinaryCrossEntropy()
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([[0.1], [0.9], [0.8], [0.2]])

        loss_val = loss.forward(y_true, y_pred)
        expected = float(-np.mean(
            y_true.reshape(-1, 1) * np.log(y_pred)
            + (1 - y_true.reshape(-1, 1)) * np.log(1 - y_pred)
        ))
        assert np.isclose(loss_val, expected, rtol=1e-6)

        dloss = loss.backward(y_true, y_pred)
        assert dloss.shape == (4, 1)
        expected_grad = (
            -(y_true.reshape(-1, 1) / y_pred)
            + (1 - y_true.reshape(-1, 1)) / (1 - y_pred)
        ) / 4
        np.testing.assert_allclose(dloss, expected_grad, rtol=1e-6)

    def test_wrong_pred_shape_raises_shape_error(self):
        loss = BinaryCrossEntropy()
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([[0.1, 0.2, 0.3], [0.9, 0.2, 0.3], [0.8, 0.2, 0.3], [0.2, 0.2, 0.3]])

        with pytest.raises(
            ShapeError,
            match=r"y_pred shape .* is invalid for binary classification",
        ):
            loss.forward(y_true, y_pred)

        with pytest.raises(
            ShapeError,
            match=r"y_pred shape .* is invalid for binary classification",
        ):
            loss.backward(y_true, y_pred)

    def test_mismatched_sample_count_raises_shape_error(self):
        loss = BinaryCrossEntropy()
        y_true = np.zeros(70)
        y_pred = np.zeros((75, 1))

        with pytest.raises(ShapeError, match=r"y_true has 70 samples but y_pred has 75 samples"):
            loss.forward(y_true, y_pred)

        with pytest.raises(ShapeError, match=r"y_true has 70 samples but y_pred has 75 samples"):
            loss.backward(y_true, y_pred)

    def test_invalid_y_true_shape_raises_shape_error(self):
        loss = BinaryCrossEntropy()

        y_true_2d = np.zeros((4, 2))
        y_pred = np.array([[0.1], [0.9], [0.8], [0.2]])

        with pytest.raises(ShapeError):
            loss.forward(y_true_2d, y_pred)

        with pytest.raises(ShapeError):
            loss.backward(y_true_2d, y_pred)

        y_true_3d = np.zeros((4, 1, 1))

        with pytest.raises(ShapeError):
            loss.forward(y_true_3d, y_pred)

        with pytest.raises(ShapeError):
            loss.backward(y_true_3d, y_pred)
