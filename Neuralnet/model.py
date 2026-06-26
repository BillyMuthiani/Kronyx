import warnings

import numpy as np


class Sequential:

    def __init__(self):

        self.layers = []
        self.optimizer = None
        self.loss_function = None
        self.metric = None

    def add(self, layer):

        self.layers.append(layer)

    def forward(self, X):

        output = X

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, dvalues):

        for layer in reversed(self.layers):
            dvalues = layer.backward(dvalues)

    def predict(self, X):

        return self.forward(X)

    def compile(
        self,
        loss,
        optimizer,
        metric=None
    ):

        self.loss_function = loss
        self.optimizer = optimizer
        self.metric = metric

    def save(self, filename):
        """Save model weights to a file."""
        from Neuralnet.serialization import save
        save(self, filename)

    def load(self, filename):
        """Load model weights from a file."""
        from Neuralnet.serialization import load
        load(self, filename)

    def fit(
        self,
        X,
        y,
        epochs=1000,
        loss=None,
        optimizer=None,
        metric=None
    ):

        # Handle deprecated signature for backwards compatibility
        if loss is not None or optimizer is not None or metric is not None:
            warnings.warn(
                "Passing loss, optimizer, or metric to fit() is deprecated. "
                "Use compile() to configure the model before calling fit().",
                DeprecationWarning,
                stacklevel=2
            )
            if loss is not None:
                self.loss_function = loss
            if optimizer is not None:
                self.optimizer = optimizer
            if metric is not None:
                self.metric = metric

        history = {
            "loss": []
        }

        if self.metric:
            history["accuracy"] = []

        for epoch in range(epochs):

            predictions = self.forward(X)

            loss_value = self.loss_function.forward(
                y,
                predictions
            )

            history["loss"].append(loss_value)

            dloss = self.loss_function.backward(
                y,
                predictions
            )

            self.backward(dloss)

            for layer in self.layers:
                self.optimizer.update(layer)

            if self.metric:

                score = self.metric.calculate(
                    y,
                    predictions
                )

                history["accuracy"].append(score)

                if epoch % 100 == 0:
                    print(
                        f"Epoch {epoch} "
                        f"Loss: {loss_value:.6f} "
                        f"Accuracy: {score:.4f}"
                    )

            else:

                if epoch % 100 == 0:
                    print(
                        f"Epoch {epoch} "
                        f"Loss: {loss_value:.6f}"
                    )

        return history