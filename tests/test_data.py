"""Tests for kronyx.data module."""
import numpy as np

from kronyx.data import BatchLoader, Dataset, TensorDataset, train_test_split


class TestTrainTestSplit:
    def test_split_arrays(self):
        X = np.arange(100).reshape(50, 2)
        y = np.arange(50)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        assert len(X_train) == 40
        assert len(X_test) == 10

    def test_split_single_array(self):
        X = np.arange(100)
        X_train, X_test = train_test_split(X, test_size=0.2)
        assert len(X_train) == 80
        assert len(X_test) == 20

    def test_random_state(self):
        X = np.arange(100).reshape(50, 2)
        y = np.arange(50)
        result1 = train_test_split(X, y, test_size=0.2, random_state=42)
        result2 = train_test_split(X, y, test_size=0.2, random_state=42)
        for a, b in zip(result1, result2):
            np.testing.assert_array_equal(a, b)

    def test_no_shuffle(self):
        X = np.arange(100).reshape(50, 2)
        y = np.arange(50)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        assert len(X_train) == 40
        assert len(X_test) == 10


class TestBatchLoader:
    def test_iteration(self):
        X = np.arange(100).reshape(50, 2)
        y = np.arange(50)
        loader = BatchLoader(X, y, batch_size=10)
        batches = list(loader)
        assert len(batches) == 5

    def test_batch_shapes(self):
        X = np.arange(100).reshape(50, 2)
        y = np.arange(50)
        loader = BatchLoader(X, y, batch_size=10)
        for X_batch, y_batch in loader:
            assert X_batch.shape[0] <= 10
            assert y_batch.shape[0] == X_batch.shape[0]

    def test_drop_last(self):
        X = np.arange(100).reshape(50, 2)
        y = np.arange(50)
        loader = BatchLoader(X, y, batch_size=15, drop_last=True)
        assert len(loader) == 3  # 45 samples / 15 = 3

    def test_len(self):
        X = np.arange(100).reshape(50, 2)
        y = np.arange(50)
        loader = BatchLoader(X, y, batch_size=10)
        assert len(loader) == 5

    def test_no_y(self):
        X = np.arange(100).reshape(50, 2)
        loader = BatchLoader(X, batch_size=10)
        for X_batch, y_batch in loader:
            assert y_batch is None


class TestDataset:
    def test_getitem(self):
        X = np.arange(100).reshape(50, 2)
        y = np.arange(50)
        dataset = Dataset(X, y)
        X0, y0 = dataset[0]
        np.testing.assert_array_equal(X0, X[0])
        np.testing.assert_array_equal(y0, y[0])

    def test_len(self):
        X = np.arange(100).reshape(50, 2)
        y = np.arange(50)
        dataset = Dataset(X, y)
        assert len(dataset) == 50


class TestTensorDataset:
    def test_getitem(self):
        X = np.arange(100).reshape(50, 2)
        y = np.arange(50)
        dataset = TensorDataset(X, y)
        X0, y0 = dataset[0]
        np.testing.assert_array_equal(X0, X[0])
        np.testing.assert_array_equal(y0, y[0])

    def test_multiple_arrays(self):
        X = np.arange(100).reshape(50, 2)
        y1 = np.arange(50)
        y2 = np.arange(50) * 2
        dataset = TensorDataset(X, y1, y2)
        X0, y1_0, y2_0 = dataset[0]
        assert X0.shape == (2,)
        assert y1_0 == 0
        assert y2_0 == 0

    def test_len(self):
        X = np.arange(100).reshape(50, 2)
        y = np.arange(50)
        dataset = TensorDataset(X, y)
        assert len(dataset) == 50
