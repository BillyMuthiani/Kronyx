"""Integration and regression tests for binary label shape handling."""
import numpy as np

from kronyx import (
    Accuracy,
    Adam,
    BinaryCrossEntropy,
    Dense,
    ReLU,
    Sequential,
    Sigmoid,
)
from kronyx.datasets import circles, xor


class TestBinaryLabelFitLoop:
    """Integration tests for (N,) labels through model.fit()."""

    def test_xor_n_labels_full_batch(self):
        X, y = xor(samples=100, random_state=42)
        assert y.ndim == 1

        model = Sequential()
        model.add(Dense(2, 16))
        model.add(ReLU())
        model.add(Dense(16, 1))
        model.add(Sigmoid())

        model.compile(
            loss=BinaryCrossEntropy(),
            optimizer=Adam(learning_rate=0.1),
            metric=Accuracy()
        )

        history = model.fit(X, y, epochs=10, verbose=0)
        assert len(history.loss) == 10
        assert np.isfinite(history.loss[-1])

    def test_circles_n_labels_mini_batch(self):
        X, y = circles(samples=200, random_state=42)
        assert y.ndim == 1

        model = Sequential()
        model.add(Dense(2, 16))
        model.add(ReLU())
        model.add(Dense(16, 1))
        model.add(Sigmoid())

        model.compile(
            loss=BinaryCrossEntropy(),
            optimizer=Adam(learning_rate=0.1),
            metric=Accuracy()
        )

        history = model.fit(X, y, epochs=20, batch_size=32, verbose=0)
        assert len(history.loss) == 20
        assert np.isfinite(history.loss[-1])
        assert not np.isnan(history.loss).any()


class TestV110Regression:
    """Regression tests for the v1.1.0 broadcast-incompatibility bug."""

    def test_v110_regression_direct_backward(self):
        np.random.seed(42)
        N = 75
        input_dim = 16

        X = np.random.randn(N, input_dim)
        y_true = np.random.randint(0, 2, size=N)

        model = Sequential()
        model.add(Dense(input_dim, 16))
        model.add(ReLU())
        model.add(Dense(16, 1))
        model.add(Sigmoid())

        model.compile(
            loss=BinaryCrossEntropy(),
            optimizer=Adam(learning_rate=0.1),
        )

        y_pred = model.forward(X, training=True)
        assert y_pred.shape == (N, 1)

        dloss = model.loss_function.backward(y_true, y_pred)
        assert dloss.shape == (N, 1), f"Expected (N, 1), got {dloss.shape}"

        model.backward(dloss)

        for layer in model.layers:
            if hasattr(layer, 'weights'):
                assert layer.dweights.shape == layer.weights.shape
            if hasattr(layer, 'dinputs'):
                assert layer.dinputs.shape[0] == N

    def test_v110_regression_full_batch_fit(self):
        np.random.seed(42)
        X = np.random.randn(75, 16)
        y_true = np.random.randint(0, 2, size=75)

        model = Sequential()
        model.add(Dense(16, 16))
        model.add(ReLU())
        model.add(Dense(16, 1))
        model.add(Sigmoid())

        model.compile(
            loss=BinaryCrossEntropy(),
            optimizer=Adam(learning_rate=0.1),
        )

        history = model.fit(X, y_true, epochs=5, batch_size=75, verbose=0)
        assert len(history.loss) == 5
        assert np.isfinite(history.loss[-1])
