"""Evaluation metrics for neural network models.

All metrics are NumPy-only and designed to be educational.
Each includes clear equations and explanations.
"""
import numpy as np


class Accuracy:
    """Calculate classification accuracy.

    For binary classification: accuracy = (TP + TN) / (TP + TN + FP + FN)
    For multi-class: uses argmax on predictions.

    Example:
        >>> metric = Accuracy()
        >>> acc = metric.calculate(y_true, y_pred)
    """

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy score.

        Args:
            y_true: True labels, shape (n_samples,) or (n_samples, n_classes).
            y_pred: Predicted probabilities, shape (n_samples,) or (n_samples, n_classes).

        Returns:
            Accuracy as a float between 0 and 1.
        """
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            predictions = np.argmax(y_pred, axis=1)
        else:
            predictions = (y_pred > 0.5).astype(int).flatten()

        y_true_flat = y_true.flatten() if len(y_true.shape) > 1 else y_true
        return float(np.mean(predictions == y_true_flat))


class BinaryAccuracy:
    """Calculate binary classification accuracy.

    Specifically for binary classification tasks.
    Threshold is applied at 0.5 for probability predictions.

    Example:
        >>> metric = BinaryAccuracy()
        >>> acc = metric.calculate(y_true, y_pred)
    """

    def __init__(self, threshold: float = 0.5):
        """Initialize BinaryAccuracy.

        Args:
            threshold: Classification threshold. Defaults to 0.5.
        """
        self.threshold = threshold

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate binary accuracy.

        Args:
            y_true: True binary labels, shape (n_samples,).
            y_pred: Predicted probabilities, shape (n_samples,).

        Returns:
            Binary accuracy as a float between 0 and 1.
        """
        y_true_flat = y_true.flatten() if y_true.ndim > 1 else y_true
        predictions = (y_pred.flatten() > self.threshold).astype(int)
        return float(np.mean(predictions == y_true_flat))


class CategoricalAccuracy:
    """Calculate multi-class classification accuracy.

    Uses argmax to convert probability predictions to class labels.

    Example:
        >>> metric = CategoricalAccuracy()
        >>> acc = metric.calculate(y_true, y_pred)
    """

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate categorical accuracy.

        Args:
            y_true: True class labels, shape (n_samples,).
            y_pred: Predicted probabilities, shape (n_samples, n_classes).

        Returns:
            Categorical accuracy as a float between 0 and 1.
        """
        y_true_flat = y_true.flatten() if y_true.ndim > 1 else y_true
        predictions = np.argmax(y_pred, axis=1)
        return float(np.mean(predictions == y_true_flat))


class Precision:
    """Calculate precision score.

    Precision = TP / (TP + FP)
    The ratio of correctly predicted positive observations to total predicted positives.

    Example:
        >>> metric = Precision()
        >>> prec = metric.calculate(y_true, y_pred)
    """

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate precision.

        Args:
            y_true: True binary labels.
            y_pred: Predicted probabilities or binary labels.

        Returns:
            Precision as a float between 0 and 1.
        """
        y_true_flat = y_true.flatten() if y_true.ndim > 1 else y_true
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            predictions = np.argmax(y_pred, axis=1)
        else:
            predictions = (y_pred.flatten() > 0.5).astype(int)

        true_positives = np.sum((predictions == 1) & (y_true_flat == 1))
        predicted_positives = np.sum(predictions == 1)

        if predicted_positives == 0:
            return 0.0
        return float(true_positives / predicted_positives)


class Recall:
    """Calculate recall score.

    Recall = TP / (TP + FN)
    The ratio of correctly predicted positive observations to all actual positives.

    Example:
        >>> metric = Recall()
        >>> rec = metric.calculate(y_true, y_pred)
    """

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate recall.

        Args:
            y_true: True binary labels.
            y_pred: Predicted probabilities or binary labels.

        Returns:
            Recall as a float between 0 and 1.
        """
        y_true_flat = y_true.flatten() if y_true.ndim > 1 else y_true
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            predictions = np.argmax(y_pred, axis=1)
        else:
            predictions = (y_pred.flatten() > 0.5).astype(int)

        true_positives = np.sum((predictions == 1) & (y_true_flat == 1))
        actual_positives = np.sum(y_true_flat == 1)

        if actual_positives == 0:
            return 0.0
        return float(true_positives / actual_positives)


class F1Score:
    """Calculate F1 score.

    F1 = 2 * (precision * recall) / (precision + recall)
    The harmonic mean of precision and recall.

    Example:
        >>> metric = F1Score()
        >>> f1 = metric.calculate(y_true, y_pred)
    """

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1 score.

        Args:
            y_true: True binary labels.
            y_pred: Predicted probabilities or binary labels.

        Returns:
            F1 score as a float between 0 and 1.
        """
        precision = Precision().calculate(y_true, y_pred)
        recall = Recall().calculate(y_true, y_pred)

        if precision + recall == 0:
            return 0.0
        return float(2 * (precision * recall) / (precision + recall))


class ConfusionMatrix:
    """Compute confusion matrix.

    A table showing correct and incorrect predictions broken down by class.
    [[TN, FP], [FN, TP]] for binary classification.

    Example:
        >>> metric = ConfusionMatrix()
        >>> cm = metric.calculate(y_true, y_pred)
    """

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix.

        Args:
            y_true: True labels.
            y_pred: Predicted labels or probabilities.

        Returns:
            Confusion matrix as 2D array.
        """
        y_true_flat = y_true.flatten() if y_true.ndim > 1 else y_true
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            predictions = np.argmax(y_pred, axis=1)
        else:
            predictions = (y_pred.flatten() > 0.5).astype(int)

        classes = np.unique(np.concatenate([y_true_flat, predictions]))
        n_classes = len(classes)
        matrix = np.zeros((n_classes, n_classes), dtype=int)

        for i, true_class in enumerate(classes):
            for j, pred_class in enumerate(classes):
                matrix[i, j] = np.sum((y_true_flat == true_class) & (predictions == pred_class))

        return matrix


class TopKAccuracy:
    """Calculate top-k accuracy.

    For multi-class classification, checks if the true label is among
    the k most probable predictions.

    Example:
        >>> metric = TopKAccuracy(k=3)
        >>> acc = metric.calculate(y_true, y_pred)
    """

    def __init__(self, k: int = 1):
        """Initialize TopKAccuracy.

        Args:
            k: Number of top predictions to consider. Defaults to 1.
        """
        self.k = k

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate top-k accuracy.

        Args:
            y_true: True class labels.
            y_pred: Predicted probabilities, shape (n_samples, n_classes).

        Returns:
            Top-k accuracy as a float between 0 and 1.
        """
        y_true_flat = y_true.flatten() if y_true.ndim > 1 else y_true

        if y_pred.ndim == 1:
            return Accuracy().calculate(y_true, y_pred)

        top_k_preds = np.argsort(y_pred, axis=1)[:, -self.k:]
        matches = [y_true_flat[i] in top_k_preds[i] for i in range(len(y_true_flat))]
        return float(np.mean(matches))
