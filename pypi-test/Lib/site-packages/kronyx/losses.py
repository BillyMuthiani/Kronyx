import numpy as np


class BinaryCrossEntropy:

    def forward(
        self,
        y_true,
        y_pred
    ):

        y_pred = np.clip(
            y_pred,
            1e-7,
            1 - 1e-7
        )

        return -np.mean(
            y_true * np.log(y_pred)
            +
            (1 - y_true)
            * np.log(1 - y_pred)
        )

    def backward(
        self,
        y_true,
        y_pred
    ):

        y_pred = np.clip(
            y_pred,
            1e-7,
            1 - 1e-7
        )

        samples = len(y_true)

        return (
            -(y_true / y_pred)
            +
            ((1 - y_true)
             / (1 - y_pred))
        ) / samples


class SoftmaxCategoricalCrossEntropy:

    def forward(
        self,
        y_true,
        y_pred
    ):

        samples = len(y_pred)

        y_pred = np.clip(
            y_pred,
            1e-7,
            1 - 1e-7
        )

        correct_confidences = y_pred[
            range(samples),
            y_true
        ]

        return -np.mean(
            np.log(correct_confidences)
        )

    def backward(
        self,
        y_true,
        y_pred
    ):

        samples = len(y_pred)

        self.dinputs = y_pred.copy()

        self.dinputs[
            range(samples),
            y_true
        ] -= 1

        self.dinputs = (
            self.dinputs
            / samples
        )

        return self.dinputs


CategoricalCrossEntropy = SoftmaxCategoricalCrossEntropy
