"""Regression tests for public-API robustness and error handling.

These tests guard against regressions in input validation, error reporting,
and state management discovered during the Phase 3 audit. They complement the
learning-validation tests (test_learning_validation.py) and the existing
unit tests.
"""
import os
import tempfile

import numpy as np
import pytest

from kronyx import (
    Adam,
    BinaryCrossEntropy,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    NotCompiledError,
    ReLU,
    Sequential,
    SerializationError,
    ShapeError,
    Sigmoid,
    load_model,
    save_model,
    set_seed,
)


@pytest.fixture
def simple_model():
    """A minimal compiled model for error-handling tests."""
    set_seed(42)
    model = Sequential()
    model.add(Dense(2, 4))
    model.add(ReLU())
    model.add(Dense(4, 1))
    model.add(Sigmoid())
    model.compile(
        loss=BinaryCrossEntropy(),
        optimizer=Adam(learning_rate=0.01),
    )
    return model


@pytest.fixture
def xor_data():
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)
    return x, y


# ---------------------------------------------------------------------------
# 1. Compilation guards
# ---------------------------------------------------------------------------


class TestCompilationGuards:
    """Verify that fit() and evaluate() require prior compile()."""

    def test_fit_before_compile_raises_not_compiled(self, xor_data):
        """fit() before compile() must raise NotCompiledError, not AttributeError."""
        x, y = xor_data
        model = Sequential()
        model.add(Dense(2, 4))
        model.add(Sigmoid())
        with pytest.raises(NotCompiledError, match="before calling fit"):
            model.fit(x, y, epochs=1, verbose=0)

    def test_evaluate_before_compile_raises_not_compiled(self, xor_data):
        """evaluate() before compile() must raise NotCompiledError, not AttributeError.

        This was a confirmed bug: evaluate() accessed self.loss_function without
        guarding for None, producing a cryptic 'NoneType has no attribute forward'.
        """
        x, y = xor_data
        model = Sequential()
        model.add(Dense(2, 4))
        model.add(Sigmoid())
        with pytest.raises(NotCompiledError, match="before calling evaluate"):
            model.evaluate(x, y, verbose=0)

    def test_evaluate_after_compile_works(self, simple_model, xor_data):
        """Sanity: evaluate() works after compile()."""
        x, y = xor_data
        loss, acc = simple_model.evaluate(x, y, verbose=0)
        assert isinstance(loss, float)
        assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# 2. Epoch and batch-size validation
# ---------------------------------------------------------------------------


class TestEpochBatchValidation:
    """Verify that fit() validates epochs and batch_size."""

    @pytest.mark.parametrize("bad_epochs", [-1, -10, -100])
    def test_fit_negative_epochs_raises(self, simple_model, xor_data, bad_epochs):
        """Negative epochs must raise ValueError instead of silently doing nothing."""
        x, y = xor_data
        with pytest.raises(ValueError, match="non-negative"):
            simple_model.fit(x, y, epochs=bad_epochs, verbose=0)

    def test_fit_zero_epochs_returns_empty_history(self, simple_model, xor_data):
        """epochs=0 is valid (no training) and returns empty history."""
        x, y = xor_data
        history = simple_model.fit(x, y, epochs=0, verbose=0)
        assert len(history.loss) == 0

    @pytest.mark.parametrize("bad_bs", [0, -1, -10])
    def test_fit_invalid_batch_size_raises(self, simple_model, xor_data, bad_bs):
        """batch_size <= 0 must raise a clear ValueError."""
        x, y = xor_data
        with pytest.raises(ValueError, match="batch_size"):
            simple_model.fit(x, y, epochs=1, batch_size=bad_bs, verbose=0)


# ---------------------------------------------------------------------------
# 3. Shape validation
# ---------------------------------------------------------------------------


class TestShapeValidation:
    """Verify that mismatched sample counts and invalid shapes raise clear errors."""

    def test_fit_mismatched_x_y_raises(self, simple_model):
        """fit() with x/y sample-count mismatch must raise ValueError, not IndexError."""
        x = np.zeros((4, 2), dtype=float)
        y = np.zeros((3, 1), dtype=float)
        with pytest.raises(ValueError, match="Number of samples"):
            simple_model.fit(x, y, epochs=1, verbose=0)

    def test_fit_mismatched_x_y_raises_batched(self, simple_model):
        """Same check with batch_size set."""
        x = np.zeros((4, 2), dtype=float)
        y = np.zeros((3, 1), dtype=float)
        with pytest.raises(ValueError, match="Number of samples"):
            simple_model.fit(x, y, epochs=1, batch_size=2, verbose=0)

    def test_evaluate_mismatched_x_y_raises(self, simple_model):
        """evaluate() with mismatched x/y should not silently produce wrong results."""
        x = np.zeros((4, 2), dtype=float)
        y = np.zeros((3, 1), dtype=float)
        # evaluate() delegates to predict() then loss.forward; the loss
        # function's validate_binary_shapes will raise ShapeError.
        with pytest.raises(ShapeError, match="samples"):
            simple_model.evaluate(x, y, verbose=0)


# ---------------------------------------------------------------------------
# 4. Serialization export and error handling
# ---------------------------------------------------------------------------


class TestSerializationExports:
    """Verify that save_model/load_model are consistently exported."""

    def test_save_model_exported_from_top_level(self):
        """save_model must be importable from the top-level kronyx package,
        consistent with load_model and from_json."""
        from kronyx import save_model as top_save
        from kronyx.serialization import save_model as ser_save
        assert top_save is ser_save

    def test_save_model_in_all(self):
        """save_model must appear in __all__."""
        import kronyx
        assert "save_model" in kronyx.__all__

    def test_load_model_directory_raises_serialization_error(self):
        """load_model() on a directory must raise SerializationError, not
        PermissionError or other OS error."""
        d = tempfile.mkdtemp()
        try:
            with pytest.raises(SerializationError):
                load_model(d)
        finally:
            os.rmdir(d)

    def test_save_model_then_load_roundtrip(self, simple_model, xor_data):
        """save_model (top-level) + load_model must produce consistent predictions."""
        x, y = xor_data
        simple_model.fit(x, y, epochs=50, verbose=0)
        orig_pred = simple_model.predict(x, verbose=0)

        path = tempfile.mktemp(suffix=".krx")
        try:
            save_model(simple_model, path)
            loaded = load_model(path)
            loaded_pred = loaded.predict(x, verbose=0)
            np.testing.assert_allclose(
                orig_pred, loaded_pred, rtol=1e-5,
                err_msg="save_model/load_model round-trip mismatch"
            )
        finally:
            if os.path.exists(path):
                os.remove(path)


# ---------------------------------------------------------------------------
# 5. State isolation
# ---------------------------------------------------------------------------


class TestStateIsolation:
    """Verify optimizer state and model behavior remain isolated across
    independent models and operations."""

    def test_independent_models_do_not_share_state(self, xor_data):
        """Two separately constructed models must not share weights or
        optimizer state."""
        x, y = xor_data

        set_seed(42)
        model_a = Sequential()
        model_a.add(Dense(2, 8, initializer="he_normal"))
        model_a.add(ReLU())
        model_a.add(Dense(8, 1))
        model_a.add(Sigmoid())
        model_a.compile(
            loss=BinaryCrossEntropy(),
            optimizer=Adam(learning_rate=0.05),
        )

        set_seed(42)
        model_b = Sequential()
        model_b.add(Dense(2, 8, initializer="he_normal"))
        model_b.add(ReLU())
        model_b.add(Dense(8, 1))
        model_b.add(Sigmoid())
        model_b.compile(
            loss=BinaryCrossEntropy(),
            optimizer=Adam(learning_rate=0.05),
        )

        # Train model_a
        model_a.fit(x, y, epochs=100, verbose=0)

        # model_b weights should still be at their initialized values
        a_w = model_a.layers[0].weights
        b_w = model_b.layers[0].weights
        assert not np.allclose(a_w, b_w), (
            "Models share the same weight array — state leak detected."
        )

    def test_optimizer_state_per_layer(self):
        """Adam must maintain independent state per layer and per parameter."""
        set_seed(42)
        model = Sequential()
        model.add(Dense(2, 8, initializer="he_normal"))
        model.add(ReLU())
        model.add(Dense(8, 1))
        model.add(Sigmoid())
        model.compile(
            loss=BinaryCrossEntropy(),
            optimizer=Adam(learning_rate=0.01),
        )
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([[0], [1], [1], [0]], dtype=float)

        model.fit(x, y, epochs=10, verbose=0)

        d0 = model.layers[0]  # first Dense
        d1 = model.layers[2]  # second Dense
        assert hasattr(d0, "m_w"), "First Dense layer missing Adam m_w state"
        assert hasattr(d0, "v_w"), "First Dense layer missing Adam v_w state"
        assert hasattr(d1, "m_w"), "Second Dense layer missing Adam m_w state"
        assert hasattr(d1, "v_w"), "Second Dense layer missing Adam v_w state"

        # Adam state arrays must be different objects per layer (not shared)
        assert d0.m_w is not d1.m_w, (
            "Adam m_w state is shared between layers — state collision"
        )
        assert d0.v_w is not d1.v_w, (
            "Adam v_w state is shared between layers — state collision"
        )

    def test_dropout_training_vs_inference(self):
        """Dropout in training mode must scale outputs; in inference mode
        must be an identity (no masking)."""
        set_seed(42)
        x = np.ones((100, 4))
        layer = Dropout(rate=0.5)

        # Training mode — output should differ from input (masking applied)
        out_train = layer.forward(x, training=True)
        assert not np.allclose(out_train, x), (
            "Dropout training mode produced identical output — mask not applied"
        )
        # Scaling: kept neurons should be ~2x (1/(1-0.5) = 2)
        nonzero_mask = out_train != 0
        assert np.allclose(out_train[nonzero_mask], 2.0), (
            "Dropout training scaling factor incorrect"
        )

        # Inference mode — output must equal input
        out_infer = layer.forward(x, training=False)
        np.testing.assert_array_equal(out_infer, x,
            err_msg="Dropout inference mode modified input")


# ---------------------------------------------------------------------------
# 6. Layer constructor validation
# ---------------------------------------------------------------------------


class TestLayerConstructorValidation:
    """Verify that layer constructors validate their arguments."""

    @pytest.mark.parametrize("rate", [-0.5, 1.0, 1.5, 2.0])
    def test_dropout_invalid_rate(self, rate):
        with pytest.raises(ValueError, match="rate must be"):
            Dropout(rate=rate)

    @pytest.mark.parametrize("padding", ["invalid", "VALID", "reflect", ""])
    def test_conv2d_invalid_padding(self, padding):
        with pytest.raises(ValueError, match="padding"):
            Conv2D(filters=2, kernel_size=3, padding=padding)

    def test_conv2d_invalid_initializer(self):
        with pytest.raises(ValueError, match="Unknown initializer"):
            Conv2D(filters=2, kernel_size=3, initializer="unknown")


# ---------------------------------------------------------------------------
# 7. predict() shape validation (Phase 4)
# ---------------------------------------------------------------------------


class TestPredictShapeValidation:
    """Verify that predict() raises ShapeError for mismatched input shapes."""

    def test_predict_wrong_feature_count_dense(self, simple_model, xor_data):
        """predict() on a Dense model with wrong feature count raises ShapeError."""
        x, _ = xor_data
        wrong_x = np.zeros((3, 4), dtype=float)  # Dense(2, 4) expects 2 features
        with pytest.raises(ShapeError, match="Dense input feature mismatch"):
            simple_model.predict(wrong_x)

    def test_predict_correct_features_dense(self, simple_model):
        """predict() with correct feature count should work."""
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        preds = simple_model.predict(x, verbose=0)
        assert preds.shape == (4, 1)

    def test_predict_wrong_channel_count_conv2d(self):
        """predict() on a Conv2D model with wrong channel count raises ShapeError."""
        set_seed(42)
        model = Sequential()
        model.add(Conv2D(filters=4, kernel_size=3, padding="same"))
        model.add(Flatten())
        model.add(Dense(256, 1))
        model.add(Sigmoid())

        # First call initializes kernels with 3 input channels
        correct_x = np.random.randn(2, 8, 8, 3)
        model.predict(correct_x)  # should work, initializes kernels

        # Wrong channel count (5 instead of 3)
        wrong_x = np.random.randn(2, 8, 8, 5)
        with pytest.raises(ShapeError, match="Conv2D channel mismatch"):
            model.predict(wrong_x)

    def test_predict_wrong_ndims_conv2d(self):
        """predict() on a Conv2D model with 3-D input (not 4-D) raises ShapeError."""
        set_seed(42)
        model = Sequential()
        model.add(Conv2D(filters=4, kernel_size=3, padding="same"))

        with pytest.raises(ShapeError, match="4-D"):
            model.predict(np.random.randn(2, 8, 8))

    def test_predict_empty_model_returns_input(self):
        """An empty Sequential model acts as identity in predict()."""
        model = Sequential()
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        result = model.predict(x)
        np.testing.assert_array_equal(result, x)


# ---------------------------------------------------------------------------
# 8. batch_size validation in predict/evaluate (Phase 4)
# ---------------------------------------------------------------------------


class TestBatchSizeValidation:
    """Verify consistent validation of invalid batch sizes across the API."""

    @pytest.mark.parametrize("bad_bs", [0, -1, -10])
    def test_predict_invalid_batch_size_raises(self, simple_model, xor_data, bad_bs):
        """predict() with batch_size <= 0 must raise ValueError."""
        x, _ = xor_data
        with pytest.raises(ValueError, match="batch_size"):
            simple_model.predict(x, batch_size=bad_bs)

    @pytest.mark.parametrize("bad_bs", [0, -1, -10])
    def test_evaluate_invalid_batch_size_raises(self, simple_model, xor_data, bad_bs):
        """evaluate() with batch_size <= 0 must raise ValueError."""
        x, y = xor_data
        with pytest.raises(ValueError, match="batch_size"):
            simple_model.evaluate(x, y, batch_size=bad_bs)

    def test_predict_valid_batch_size_works(self, simple_model, xor_data):
        """predict() with a valid batch_size should produce full results."""
        x, _ = xor_data
        preds = simple_model.predict(x, batch_size=2, verbose=0)
        assert preds.shape == (4, 1)

    def test_evaluate_valid_batch_size_works(self, simple_model, xor_data):
        """evaluate() with a valid batch_size should return (loss, accuracy)."""
        x, y = xor_data
        loss, acc = simple_model.evaluate(x, y, batch_size=2, verbose=0)
        assert isinstance(loss, float)
        assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# 9. Conv2D backward lifecycle guard (Phase 4)
# ---------------------------------------------------------------------------


class TestConv2DBackwardGuard:
    """Verify Conv2D.backward raises a clear error when no forward pass ran."""

    def test_backward_without_forward_raises(self):
        """Conv2D.backward() before forward() must raise a clear RuntimeError."""
        conv = Conv2D(filters=4, kernel_size=3, padding="same")
        dout = np.random.randn(2, 8, 8, 4)
        with pytest.raises(RuntimeError, match="before forward"):
            conv.backward(dout)

    def test_backward_after_forward_works(self):
        """Conv2D.backward() after forward() must work normally."""
        conv = Conv2D(filters=4, kernel_size=3, padding="same")
        x = np.random.randn(2, 8, 8, 3)
        out = conv.forward(x, training=True)
        dout = np.random.randn(*out.shape)
        dinputs = conv.backward(dout)
        assert dinputs.shape == x.shape





