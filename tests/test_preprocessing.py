"""Tests for kronyx.preprocessing module."""
import numpy as np
import pytest

from kronyx.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler


class TestStandardScaler:
    def test_fit_transform(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10)

    def test_inverse_transform(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_inv = scaler.inverse_transform(X_scaled)
        np.testing.assert_array_almost_equal(X, X_inv)

    def test_not_fitted(self):
        scaler = StandardScaler()
        with pytest.raises(ValueError, match="not been fitted"):
            scaler.transform(np.array([[1, 2]]))


class TestMinMaxScaler:
    def test_fit_transform(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        # Check that values are in [0, 1] range
        assert X_scaled[0, 0] == 0.0  # min value
        assert X_scaled[2, 0] == 1.0  # max value

    def test_custom_range(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X)
        assert X_scaled[0, 0] == -1.0  # min value
        assert X_scaled[2, 0] == 1.0  # max value

    def test_inverse_transform(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_inv = scaler.inverse_transform(X_scaled)
        np.testing.assert_array_almost_equal(X, X_inv)


class TestRobustScaler:
    def test_fit_transform(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        # Should not raise
        assert X_scaled.shape == X.shape

    def test_custom_quantile(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = RobustScaler(quantile_range=(10.0, 90.0))
        X_scaled = scaler.fit_transform(X)
        assert X_scaled.shape == X.shape


class TestOneHotEncoder:
    def test_fit_transform(self):
        y = np.array([0, 1, 2, 1, 0])
        encoder = OneHotEncoder()
        y_onehot = encoder.fit_transform(y)
        assert y_onehot.shape == (5, 3)
        assert np.all(np.sum(y_onehot, axis=1) == 1)

    def test_inverse_transform(self):
        y = np.array([0, 1, 2, 1, 0])
        encoder = OneHotEncoder()
        y_onehot = encoder.fit_transform(y)
        y_inv = encoder.inverse_transform(y_onehot)
        np.testing.assert_array_equal(y, y_inv)

    def test_not_fitted(self):
        encoder = OneHotEncoder()
        with pytest.raises(ValueError, match="not been fitted"):
            encoder.transform(np.array([0, 1]))
