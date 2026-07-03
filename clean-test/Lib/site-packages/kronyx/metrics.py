import numpy as np


class Accuracy:

    def calculate(
        self,
        y_true,
        y_pred
    ):

        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:

            predictions = np.argmax(
                y_pred,
                axis=1
            )
        else:

            predictions = (
                y_pred > 0.5
            ).astype(int).flatten()

        y_true_flat = y_true.flatten() if len(y_true.shape) > 1 else y_true

        return np.mean(
            predictions == y_true_flat
        )
