"""Integration tests for end-to-end learning in the Kronyx framework.

These tests go beyond unit-level gradient checks and optimizer updates to
verify that *complete models* can actually learn useful functions through the
public ``compile`` / ``fit`` / ``predict`` / ``evaluate`` API.

Coverage areas:
  1. XOR learning — a non-linearly-separable binary task.
  2. Multi-class classification — a synthetic moons dataset.
  3. Convolutional learning — Conv2D kernels that adapt during training.
  4. Persistence & reproducibility — .krx save/load and seed-based determinism.
  5. Optimizer comparison — SGD vs Adam on XOR.
"""
import os
import tempfile

import numpy as np
import pytest

from kronyx import (
    SGD,
    Accuracy,
    Adam,
    BatchNormalization,
    BinaryCrossEntropy,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    ReLU,
    Sequential,
    Sigmoid,
    set_seed,
)
from kronyx.data import train_test_split
from kronyx.datasets import moons
from kronyx.serialization import load_model, save_model

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

XOR_INPUT = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
XOR_TARGET = np.array([[0], [1], [1], [0]], dtype=float)


def _build_xor_model(seed=42, lr=0.05):
    """Create a compact Dense network suited to the XOR problem."""
    set_seed(seed)
    model = Sequential()
    model.add(Dense(2, 8, initializer="he_normal"))
    model.add(ReLU())
    model.add(Dense(8, 1))
    model.add(Sigmoid())
    model.compile(
        loss=BinaryCrossEntropy(),
        optimizer=Adam(learning_rate=lr),
        metric=Accuracy(),
    )
    return model


# ---------------------------------------------------------------------------
# 1. XOR learning
# ---------------------------------------------------------------------------


class TestXORLearning:
    """Verify that a small Dense network can learn the classic XOR problem."""

    def test_xor_loss_decreases_substantially(self):
        """Loss must drop by at least an order of magnitude from initial."""
        model = _build_xor_model(seed=42)
        # Initial loss should be far from zero (random init ≈ 0.69)
        pred0 = model.predict(XOR_INPUT, verbose=0)
        initial_loss = model.loss_function.forward(XOR_TARGET, pred0)
        history = model.fit(XOR_INPUT, XOR_TARGET, epochs=1000, verbose=0)
        final_loss = history.loss[-1]

        assert initial_loss > 0.3, (
            f"Initial XOR loss {initial_loss:.4f} unexpectedly low — "
            "network may have started near-optimal weights."
        )
        assert final_loss < initial_loss * 0.1, (
            f"Final loss {final_loss:.6f} did not decrease substantially "
            f"from initial {initial_loss:.6f}."
        )
        assert final_loss < 0.05, (
            f"Final XOR loss {final_loss:.6f} too high — model has not converged."
        )

    def test_xor_all_inputs_classified_correctly(self):
        """All four XOR input combinations must be classified correctly."""
        model = _build_xor_model(seed=42)
        model.fit(XOR_INPUT, XOR_TARGET, epochs=1000, verbose=0)

        preds = model.predict(XOR_INPUT, verbose=0)
        binary_preds = (preds > 0.5).astype(float)
        accuracy = np.mean(binary_preds == XOR_TARGET)

        assert accuracy == 1.0, (
            f"Not all XOR inputs classified correctly. "
            f"Predictions: {preds.flatten()}, expected: {XOR_TARGET.flatten()}"
        )

    @pytest.mark.parametrize("seed", [42, 123, 999])
    def test_xor_convex_from_multiple_seeds(self, seed):
        """XOR must converge regardless of the random seed (not lucky init)."""
        model = _build_xor_model(seed=seed)
        history = model.fit(XOR_INPUT, XOR_TARGET, epochs=1000, verbose=0)
        preds = model.predict(XOR_INPUT, verbose=0)
        binary_preds = (preds > 0.5).astype(float)
        accuracy = np.mean(binary_preds == XOR_TARGET)

        assert accuracy == 1.0, (
            f"XOR failed to converge with seed={seed}. "
            f"Final loss: {history.loss[-1]:.6f}"
        )
        assert history.loss[-1] < 0.01, (
            f"XOR did not converge with seed={seed}: "
            f"final loss={history.loss[-1]:.6f}"
        )

    def test_xor_reproducibility_same_seed(self):
        """Two runs with the same seed must produce identical predictions."""
        model_a = _build_xor_model(seed=42)
        model_a.fit(XOR_INPUT, XOR_TARGET, epochs=1000, verbose=0)
        preds_a = model_a.predict(XOR_INPUT, verbose=0)

        model_b = _build_xor_model(seed=42)
        model_b.fit(XOR_INPUT, XOR_TARGET, epochs=1000, verbose=0)
        preds_b = model_b.predict(XOR_INPUT, verbose=0)

        np.testing.assert_allclose(
            preds_a, preds_b, rtol=1e-8,
            err_msg="XOR training is not reproducible with the same seed."
        )

    def test_xor_evaluate_matches_predict(self):
        """evaluate() must report the same loss and accuracy as predict()."""
        model = _build_xor_model(seed=42)
        model.fit(XOR_INPUT, XOR_TARGET, epochs=1000, verbose=0)

        eval_loss, eval_acc = model.evaluate(XOR_INPUT, XOR_TARGET, verbose=0)

        preds = model.predict(XOR_INPUT, verbose=0)
        pred_loss = model.loss_function.forward(XOR_TARGET, preds)
        model.metric.calculate(XOR_TARGET, preds)

        assert abs(eval_loss - pred_loss) < 1e-10, (
            f"evaluate() loss {eval_loss} != predict() loss {pred_loss}"
        )

# ---------------------------------------------------------------------------
# 2. Classification learning (moons)
# ---------------------------------------------------------------------------


class TestClassificationLearning:
    """Verify that a Dense network can learn a non-linearly-separable
    2-D classification task using the library's own ``moons`` dataset."""

    @pytest.fixture
    def moons_data(self):
        """Split the moons dataset into train/test (80/20)."""
        X, y = moons(samples=200, noise=0.1, random_state=42)
        y = y.reshape(-1, 1).astype(float)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return X_tr, X_te, y_tr, y_te

    def _build_model(self, seed=42):
        set_seed(seed)
        model = Sequential()
        model.add(Dense(2, 16, initializer="he_normal"))
        model.add(ReLU())
        model.add(Dense(16, 1))
        model.add(Sigmoid())
        model.compile(
            loss=BinaryCrossEntropy(),
            optimizer=Adam(learning_rate=0.01),
            metric=Accuracy(),
        )
        return model

    def test_moons_loss_decreases(self, moons_data):
        """Training loss must decrease substantially."""
        X_tr, _, y_tr, _ = moons_data
        model = self._build_model()

        pred0 = model.predict(X_tr, verbose=0)
        initial_loss = model.loss_function.forward(y_tr, pred0)

        history = model.fit(X_tr, y_tr, epochs=300, verbose=0)
        final_loss = history.loss[-1]

        assert initial_loss > 0.5, (
            f"Initial moons loss {initial_loss:.4f} unexpectedly low."
        )
        assert final_loss < initial_loss * 0.3, (
            f"Final loss {final_loss:.6f} did not decrease sufficiently "
            f"from initial {initial_loss:.6f}."
        )

    def test_moons_reaches_reasonable_accuracy(self, moons_data):
        """The model should achieve high accuracy on both train and test."""
        X_tr, X_te, y_tr, y_te = moons_data
        model = self._build_model()
        model.fit(X_tr, y_tr, epochs=300, verbose=0)

        train_acc = model.metric.calculate(y_tr, model.predict(X_tr, verbose=0))
        test_acc = model.metric.calculate(y_te, model.predict(X_te, verbose=0))

        # 0.85 / 0.80 thresholds are justified: the moons dataset with
        # noise=0.1 is easily separable by a 2-layer MLP after 300 Adam epochs.
        assert train_acc >= 0.85, f"Train accuracy {train_acc:.4f} below 0.85"
        assert test_acc >= 0.80, f"Test accuracy {test_acc:.4f} below 0.80"

    def test_moons_evaluate_does_not_mutate_state(self, moons_data):
        """Calling evaluate() must not change model predictions."""
        X_tr, X_te, y_tr, y_te = moons_data
        model = self._build_model()
        model.fit(X_tr, y_tr, epochs=300, verbose=0)

        preds_before = model.predict(X_te, verbose=0).copy()
        model.evaluate(X_tr, y_tr, verbose=0)
        model.evaluate(X_te, y_te, verbose=0)
        preds_after = model.predict(X_te, verbose=0)

        np.testing.assert_allclose(
            preds_before, preds_after, rtol=1e-8,
            err_msg="evaluate() mutated model state between calls."
        )

    def test_moons_can_resume_training_after_evaluate(self, moons_data):
        """After evaluate(), the model must still be able to train."""
        X_tr, _, y_tr, _ = moons_data
        model = self._build_model()
        model.fit(X_tr, y_tr, epochs=50, verbose=0)

        # Evaluate (which runs predict with training=False)
        _ = model.evaluate(X_tr, y_tr, verbose=0)

        # Resume training — must not crash or produce NaN
        history = model.fit(X_tr, y_tr, epochs=50, verbose=0)
        assert np.isfinite(history.loss[-1]), \
            "Loss is NaN/inf after resuming training"
        assert history.loss[-1] < history.loss[0], \
            "Loss did not decrease during resumed training."


# ---------------------------------------------------------------------------
# 3. Convolutional learning
# ---------------------------------------------------------------------------


class TestConvLearnability:
    """Verify that a Conv2D-based model can learn a synthetic visual task."""

    @staticmethod
    def _make_conv_dataset(n=50, seed=42):
        """Create 4x4 single-channel images.

        Class 0 – bright pixel at top-left corner (0, 0).
        Class 1 – bright pixel at bottom-right corner (3, 3).

        A 3x3 kernel can distinguish these by inspecting where
        activation energy concentrates in the feature map.
        """
        np.random.RandomState(seed)
        X = np.zeros((n, 4, 4, 1))
        y = np.zeros((n, 1))
        for i in range(n):
            if i % 2 == 0:
                X[i, 0, 0, 0] = 1.0
                y[i, 0] = 0
            else:
                X[i, 3, 3, 0] = 1.0
                y[i, 0] = 1
        return X, y

    @pytest.fixture
    def conv_data(self):
        return self._make_conv_dataset()

    def _build_model(self, seed=42):
        set_seed(seed)
        model = Sequential()
        model.add(Conv2D(filters=2, kernel_size=3, padding="same"))
        model.add(ReLU())
        model.add(Flatten())
        # After Conv2D(same) on 4x4x1 -> 4x4x2 -> Flatten -> 32
        model.add(Dense(32, 1))
        model.add(Sigmoid())
        model.compile(
            loss=BinaryCrossEntropy(),
            optimizer=Adam(learning_rate=0.01),
            metric=Accuracy(),
        )
        return model

    def test_conv_kernels_and_biases_change(self, conv_data):
        """Conv2D parameters must actually be updated by the optimizer."""
        X, y = conv_data
        model = self._build_model()

        # Lazy init — trigger first forward to materialise kernels/biases
        model.predict(X, verbose=0)
        conv = model.layers[0]
        assert conv.kernels is not None, "Conv2D kernels never initialised"
        kernels_before = conv.kernels.copy()
        biases_before = conv.biases.copy()

        model.fit(X, y, epochs=200, verbose=0)

        assert not np.array_equal(kernels_before, conv.kernels), (
            "Conv2D kernels did not change during training."
        )
        assert not np.array_equal(biases_before, conv.biases), (
            "Conv2D biases did not change during training."
        )

    def test_conv_loss_decreases(self, conv_data):
        """Training loss must decrease substantially."""
        X, y = conv_data
        model = self._build_model()

        pred0 = model.predict(X, verbose=0)
        initial_loss = model.loss_function.forward(y, pred0)

        history = model.fit(X, y, epochs=200, verbose=0)
        final_loss = history.loss[-1]

        assert initial_loss > 0.4, (
            f"Initial conv loss {initial_loss:.4f} unexpectedly low."
        )
        assert final_loss < initial_loss * 0.1, (
            f"Final loss {final_loss:.6f} did not decrease substantially "
            f"from initial {initial_loss:.6f}."
        )

    def test_conv_reaches_good_accuracy(self, conv_data):
        """The Conv2D model should classify the synthetic task correctly."""
        X, y = conv_data
        model = self._build_model()
        model.fit(X, y, epochs=200, verbose=0)

        preds = model.predict(X, verbose=0)
        binary_preds = (preds > 0.5).astype(float)
        accuracy = np.mean(binary_preds == y)

        assert accuracy >= 0.9, (
            f"Conv2D accuracy {accuracy:.4f} below 0.9 on synthetic task."
        )

    def test_conv_save_load_preserves_predictions(self, conv_data):
        """A trained Conv2D model must produce identical predictions
        after a .krx save/load round-trip."""
        X, y = conv_data
        model = self._build_model()
        model.fit(X, y, epochs=200, verbose=0)

        orig_pred = model.predict(X, verbose=0)

        model_path = tempfile.mktemp(suffix=".krx")
        try:
            save_model(model, model_path)
            loaded = load_model(model_path)
            loaded_pred = loaded.predict(X, verbose=0)

            np.testing.assert_allclose(
                orig_pred, loaded_pred, rtol=1e-5,
                err_msg="Conv2D predictions differ after save/load."
            )
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)


# ---------------------------------------------------------------------------
# 4. Persistence & reproducibility (mixed-layer model)
# ---------------------------------------------------------------------------


class TestPersistenceAndReproducibility:
    """Verify .krx save/load and seed-based reproducibility for a
    model that combines Dense, BatchNorm, Dropout and multiple activations."""

    def _build_mixed_model(self, seed=42):
        set_seed(seed)
        model = Sequential()
        model.add(Dense(2, 8, initializer="he_normal"))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(0.1))
        model.add(Dense(8, 1))
        model.add(Sigmoid())
        model.compile(
            loss=BinaryCrossEntropy(),
            optimizer=Adam(learning_rate=0.05),
            metric=Accuracy(),
        )
        return model

    def test_save_load_preserves_predictions(self):
        """Trained mixed-layer model: predictions must match after .krx
        round-trip."""
        model = self._build_mixed_model(seed=42)
        model.fit(XOR_INPUT, XOR_TARGET, epochs=500, verbose=0)

        orig_pred = model.predict(XOR_INPUT, verbose=0)

        model_path = tempfile.mktemp(suffix=".krx")
        try:
            save_model(model, model_path)
            assert os.path.exists(model_path)

            loaded = load_model(model_path)
            assert isinstance(loaded, Sequential)
            assert len(loaded.layers) == len(model.layers)

            loaded_pred = loaded.predict(XOR_INPUT, verbose=0)
            np.testing.assert_allclose(
                orig_pred, loaded_pred, rtol=1e-5,
                err_msg="Mixed-layer predictions differ after .krx round-trip."
            )
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_save_load_preserves_parameters(self):
        """All trainable parameters and BN running stats must be preserved."""
        model = self._build_mixed_model(seed=42)
        model.fit(XOR_INPUT, XOR_TARGET, epochs=200, verbose=0)

        model_path = tempfile.mktemp(suffix=".krx")
        try:
            save_model(model, model_path)
            loaded = load_model(model_path)

            for i, (orig_layer, loaded_layer) in enumerate(
                zip(model.layers, loaded.layers, strict=False)
            ):
                if hasattr(orig_layer, "weights") and orig_layer.weights is not None:
                    np.testing.assert_allclose(
                        orig_layer.weights, loaded_layer.weights, rtol=1e-6,
                        err_msg=f"Dense weights differ in layer {i}"
                    )
                    np.testing.assert_allclose(
                        orig_layer.biases, loaded_layer.biases, rtol=1e-6,
                        err_msg=f"Dense biases differ in layer {i}"
                    )
                if hasattr(orig_layer, "gamma") and orig_layer.gamma is not None:
                    np.testing.assert_allclose(
                        orig_layer.gamma, loaded_layer.gamma, rtol=1e-6,
                        err_msg=f"BN gamma differs in layer {i}"
                    )
                    np.testing.assert_allclose(
                        orig_layer.beta, loaded_layer.beta, rtol=1e-6,
                        err_msg=f"BN beta differs in layer {i}"
                    )
                    np.testing.assert_allclose(
                        orig_layer.running_mean, loaded_layer.running_mean, rtol=1e-6,
                        err_msg=f"BN running_mean differs in layer {i}"
                    )
                    np.testing.assert_allclose(
                        orig_layer.running_variance, loaded_layer.running_variance,
                        rtol=1e-6,
                        err_msg=f"BN running_variance differs in layer {i}"
                    )
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_reproducibility_same_seed(self):
        """Two independent trainings with the same seed must match."""
        model_a = self._build_mixed_model(seed=42)
        model_a.fit(XOR_INPUT, XOR_TARGET, epochs=500, verbose=0)
        preds_a = model_a.predict(XOR_INPUT, verbose=0)

        model_b = self._build_mixed_model(seed=42)
        model_b.fit(XOR_INPUT, XOR_TARGET, epochs=500, verbose=0)
        preds_b = model_b.predict(XOR_INPUT, verbose=0)

        np.testing.assert_allclose(
            preds_a, preds_b, rtol=1e-6,
            err_msg="Mixed-layer training not reproducible with same seed."
        )

    def test_predict_does_not_alter_running_stats(self):
        """BatchNorm running stats must be unchanged by prediction-only calls."""
        model = self._build_mixed_model(seed=42)
        model.fit(XOR_INPUT, XOR_TARGET, epochs=200, verbose=0)

        bn = model.layers[1]
        rm_before = bn.running_mean.copy()
        rv_before = bn.running_variance.copy()

        # Multiple prediction calls (training=False)
        for _ in range(3):
            model.predict(XOR_INPUT, verbose=0)

        np.testing.assert_allclose(
            bn.running_mean, rm_before, rtol=1e-10,
            err_msg="BatchNorm running_mean changed during prediction."
        )
        np.testing.assert_allclose(
            bn.running_variance, rv_before, rtol=1e-10,
            err_msg="BatchNorm running_variance changed during prediction."
        )


# ---------------------------------------------------------------------------
# 5. Optimizer comparison (SGD vs Adam on XOR)
# ---------------------------------------------------------------------------


class TestOptimizerComparison:
    """Verify that SGD (with a higher learning rate) can also solve XOR."""

    def test_sgd_learns_xor(self):
        """SGD with a suitable learning rate must also solve XOR."""
        set_seed(42)
        model = Sequential()
        model.add(Dense(2, 8, initializer="he_normal"))
        model.add(ReLU())
        model.add(Dense(8, 1))
        model.add(Sigmoid())
        model.compile(
            loss=BinaryCrossEntropy(),
            optimizer=SGD(learning_rate=0.5),
            metric=Accuracy(),
        )

        history = model.fit(XOR_INPUT, XOR_TARGET, epochs=2000, verbose=0)
        preds = model.predict(XOR_INPUT, verbose=0)
        binary_preds = (preds > 0.5).astype(float)
        accuracy = np.mean(binary_preds == XOR_TARGET)

        assert history.loss[-1] < 0.1, (
            f"SGD final XOR loss {history.loss[-1]:.6f} too high."
        )
        assert accuracy == 1.0, (
            f"SGD failed to solve XOR. Accuracy: {accuracy}"
        )
