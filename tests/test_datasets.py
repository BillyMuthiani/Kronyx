"""Tests for kronyx.datasets module."""
import numpy as np

from kronyx.datasets import blobs, circles, iris, moons, spiral, xor


class TestXOR:
    def test_shape(self):
        X, y = xor(samples=100)
        assert X.shape == (100, 2)
        assert y.shape == (100,)

    def test_random_state(self):
        X1, y1 = xor(samples=100, random_state=42)
        X2, y2 = xor(samples=100, random_state=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_classes(self):
        X, y = xor(samples=100)
        assert set(np.unique(y)) == {0.0, 1.0}


class TestSpiral:
    def test_shape(self):
        X, y = spiral(samples=300, classes=3)
        assert X.shape == (300, 2)
        assert y.shape == (300,)

    def test_classes(self):
        X, y = spiral(samples=300, classes=3)
        assert set(np.unique(y)) == {0, 1, 2}

    def test_random_state(self):
        X1, y1 = spiral(samples=100, random_state=42)
        X2, y2 = spiral(samples=100, random_state=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


class TestCircles:
    def test_shape(self):
        X, y = circles(samples=200)
        assert X.shape == (200, 2)
        assert y.shape == (200,)

    def test_classes(self):
        X, y = circles(samples=200)
        assert set(np.unique(y)) == {0, 1}

    def test_factor(self):
        X, y = circles(samples=200, factor=0.3)
        assert X.shape == (200, 2)


class TestMoons:
    def test_shape(self):
        X, y = moons(samples=200)
        assert X.shape == (200, 2)
        assert y.shape == (200,)

    def test_classes(self):
        X, y = moons(samples=200)
        assert set(np.unique(y)) == {0, 1}


class TestBlobs:
    def test_shape(self):
        X, y = blobs(samples=300, centers=3)
        assert X.shape == (300, 2)
        assert y.shape == (300,)

    def test_centers(self):
        X, y = blobs(samples=300, centers=5)
        assert set(np.unique(y)) == {0, 1, 2, 3, 4}

    def test_cluster_std(self):
        X, y = blobs(samples=100, centers=2, cluster_std=2.0)
        assert X.shape == (100, 2)


class TestIris:
    def test_shape(self):
        X, y = iris()
        assert X.shape == (150, 4)
        assert y.shape == (150,)

    def test_classes(self):
        X, y = iris()
        assert set(np.unique(y)) == {0, 1, 2}

    def test_random_state(self):
        X1, y1 = iris(random_state=42)
        X2, y2 = iris(random_state=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
