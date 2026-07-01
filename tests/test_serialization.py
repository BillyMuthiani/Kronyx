"""Tests for model serialization and persistence."""
import json
import os
import tempfile
import zipfile

import numpy as np
import pytest

from kronyx import (
    Accuracy,
    BinaryCrossEntropy,
    Dense,
    ReLU,
    Sequential,
    Sigmoid,
    load_model,
)
from kronyx.optimizers import SGD
from kronyx.serialization import SerializationError, save_model


def _get_metadata():
    return {
        "framework": "kronyx",
        "version": "0.6.0",
        "python_version": "3.11.0",
        "numpy_version": "2.3.4",
        "created_at": "2026-01-01T00:00:00+00:00",
        "format_version": 1
    }


class TestSerialization:
    """Test suite for model serialization."""

    @pytest.fixture
    def xor_model(self):
        """Create a simple XOR model for testing."""
        np.random.seed(42)
        model = Sequential()
        model.add(Dense(2, 4))
        model.add(ReLU())
        model.add(Dense(4, 1))
        model.add(Sigmoid())
        model.compile(loss=BinaryCrossEntropy(), optimizer=SGD(), metric=Accuracy())
        return model

    @pytest.fixture
    def xor_data(self):
        """XOR dataset for testing."""
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        return x, y

    def test_save_load_roundtrip(self, xor_model, xor_data):
        """Test that save/load produces identical predictions."""
        x, _ = xor_data

        model_path = tempfile.mktemp(suffix='.krx')
        try:
            save_model(xor_model, model_path)
            assert os.path.exists(model_path)

            loaded = load_model(model_path)
            assert isinstance(loaded, Sequential)

            orig_pred = xor_model.predict(x, verbose=0)
            loaded_pred = loaded.predict(x, verbose=0)
            np.testing.assert_allclose(orig_pred, loaded_pred, rtol=1e-5)
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_save_weights_load_weights(self, xor_model, xor_data):
        """Test weights-only save/load."""
        x, _ = xor_data

        weights_path = tempfile.mktemp(suffix='.npz')
        try:
            xor_model.save_weights(weights_path)
            assert os.path.exists(weights_path)

            new_model = Sequential()
            new_model.add(Dense(2, 4))
            new_model.add(ReLU())
            new_model.add(Dense(4, 1))
            new_model.add(Sigmoid())

            new_model.load_weights(weights_path)
            orig_pred = xor_model.predict(x, verbose=0)
            new_pred = new_model.predict(x, verbose=0)
            np.testing.assert_allclose(orig_pred, new_pred, rtol=1e-5)
        finally:
            if os.path.exists(weights_path):
                os.remove(weights_path)

    def test_to_json_from_json(self, xor_model):
        """Test JSON export/import."""
        json_str = xor_model.to_json()
        assert isinstance(json_str, str)

        reconstructed = Sequential.from_json(json_str)
        assert isinstance(reconstructed, Sequential)
        assert len(reconstructed.layers) == len(xor_model.layers)

    def test_from_json_invalid(self):
        """Test from_json raises error on invalid JSON."""
        with pytest.raises(SerializationError):
            Sequential.from_json("not valid json")

        with pytest.raises((SerializationError, TypeError)):
            Sequential.from_json('{"layers": [{"type": "UnknownLayer"}]}')

    def test_load_missing_file(self):
        """Test load_model raises error on missing file."""
        with pytest.raises(SerializationError):
            load_model("/nonexistent/path/model.krx")

    def test_prediction_equality(self, xor_model, xor_data):
        """Test predictions match exactly after save/load."""
        x, _ = xor_data
        xor_model.fit(x, xor_data[1], epochs=10, verbose=0)

        model_path = tempfile.mktemp(suffix='.krx')
        try:
            save_model(xor_model, model_path)
            loaded = load_model(model_path)

            orig_pred = xor_model.predict(x)
            loaded_pred = loaded.predict(x)
            assert np.allclose(orig_pred, loaded_pred)
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_corrupted_archive(self):
        """Test loading corrupted archive raises SerializationError."""
        model_path = tempfile.mktemp(suffix='.krx')
        try:
            with open(model_path, 'wb') as f:
                f.write(b'not a valid zip file')

            with pytest.raises(SerializationError):
                load_model(model_path)
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_missing_metadata(self, xor_model):
        """Test archive without metadata raises error."""
        model_path = tempfile.mktemp(suffix='.krx')
        try:
            with zipfile.ZipFile(model_path, 'w') as zf:
                architecture = {"layers": [{"type": "Dense", "weights_shape": [2, 4],
                                           "biases_shape": [1, 4]}]}
                zf.writestr('architecture.json', json.dumps(architecture))

            with pytest.raises(SerializationError):
                load_model(model_path)
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_missing_weights(self, xor_model):
        """Test archive without weights.npz raises error."""
        model_path = tempfile.mktemp(suffix='.krx')
        try:
            with zipfile.ZipFile(model_path, 'w') as zf:
                zf.writestr('metadata.json', json.dumps(_get_metadata()))
                architecture = {"layers": [{"type": "Dense", "weights_shape": [2, 4],
                                           "biases_shape": [1, 4]}]}
                zf.writestr('architecture.json', json.dumps(architecture))

            with pytest.raises(SerializationError):
                load_model(model_path)
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_wrong_extension(self, xor_model):
        """Test that .krx extension is added if missing."""
        model_path = tempfile.mktemp(suffix='.tmp')
        try:
            save_model(xor_model, model_path)
            assert model_path.endswith('.krx') or os.path.exists(model_path + '.krx')
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(model_path + '.krx'):
                os.remove(model_path + '.krx')
