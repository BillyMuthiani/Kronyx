"""Tests for kronyx.metrics module."""
import numpy as np

from kronyx.metrics import (
    Accuracy,
    BinaryAccuracy,
    CategoricalAccuracy,
    ConfusionMatrix,
    F1Score,
    Precision,
    Recall,
    TopKAccuracy,
)


class TestAccuracy:
    def test_binary(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8])
        acc = Accuracy().calculate(y_true, y_pred)
        assert acc == 1.0

    def test_multiclass(self):
        y_true = np.array([0, 1, 2, 2])
        y_pred = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
            [0.1, 0.1, 0.8],
        ])
        acc = Accuracy().calculate(y_true, y_pred)
        assert acc == 1.0

    def test_partial(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.9, 0.1, 0.1, 0.9])
        acc = Accuracy().calculate(y_true, y_pred)
        assert acc == 0.5


class TestBinaryAccuracy:
    def test_threshold(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8])
        acc = BinaryAccuracy().calculate(y_true, y_pred)
        assert acc == 1.0

    def test_custom_threshold(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.4, 0.6, 0.3, 0.7])
        acc = BinaryAccuracy(threshold=0.5).calculate(y_true, y_pred)
        # 0.4 < 0.5 -> 0, 0.6 > 0.5 -> 1, 0.3 < 0.5 -> 0, 0.7 > 0.5 -> 1
        # Predicted: [0, 1, 0, 1], True: [0, 1, 0, 1] -> 100% accuracy
        assert acc == 1.0


class TestCategoricalAccuracy:
    def test_perfect(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
        ])
        acc = CategoricalAccuracy().calculate(y_true, y_pred)
        assert acc == 1.0

    def test_partial(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.8, 0.1, 0.1],
        ])
        acc = CategoricalAccuracy().calculate(y_true, y_pred)
        assert acc == 0.0


class TestPrecision:
    def test_perfect(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        prec = Precision().calculate(y_true, y_pred)
        assert prec == 1.0

    def test_no_positives(self):
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        prec = Precision().calculate(y_true, y_pred)
        assert prec == 0.0


class TestRecall:
    def test_perfect(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        rec = Recall().calculate(y_true, y_pred)
        assert rec == 1.0

    def test_no_positives(self):
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        rec = Recall().calculate(y_true, y_pred)
        assert rec == 0.0


class TestF1Score:
    def test_perfect(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        f1 = F1Score().calculate(y_true, y_pred)
        assert f1 == 1.0

    def test_partial(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 1])
        f1 = F1Score().calculate(y_true, y_pred)
        # TP = 1, FP = 0, FN = 1
        # Precision = 1.0, Recall = 0.5, F1 = 2/3
        assert f1 == 2/3


class TestConfusionMatrix:
    def test_binary(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        cm = ConfusionMatrix().calculate(y_true, y_pred)
        assert cm.shape == (2, 2)
        assert cm[0, 0] == 1  # TN
        assert cm[0, 1] == 1  # FP
        assert cm[1, 0] == 1  # FN
        assert cm[1, 1] == 1  # TP


class TestTopKAccuracy:
    def test_k1(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([
            [0.5, 0.3, 0.2],
            [0.1, 0.6, 0.3],
            [0.1, 0.2, 0.7],
        ])
        acc = TopKAccuracy(k=1).calculate(y_true, y_pred)
        assert acc == 1.0

    def test_k2(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([
            [0.1, 0.5, 0.4],  # top 2: [1, 2], true=0 -> not in top 2
            [0.1, 0.6, 0.3],  # top 2: [1, 2], true=1 -> in top 2
            [0.1, 0.2, 0.7],  # top 2: [2, 1], true=2 -> in top 2
        ])
        acc = TopKAccuracy(k=2).calculate(y_true, y_pred)
        assert acc == 2/3  # 2 out of 3 correct

    def test_k3(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([
            [0.5, 0.3, 0.2],
            [0.1, 0.6, 0.3],
            [0.1, 0.2, 0.7],
        ])
        acc = TopKAccuracy(k=3).calculate(y_true, y_pred)
        assert acc == 1.0
