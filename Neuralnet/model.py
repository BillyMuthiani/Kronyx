import warnings

import numpy as np

from Neuralnet.exceptions import NotCompiledError


class History:

    def __init__(self):
        self.loss = []
        self.accuracy = []
        self.val_loss = []
        self.val_accuracy = []

    def __getitem__(self, key):
        if key == "loss":
            return self.loss
        elif key == "accuracy":
            return self.accuracy
        elif key == "val_loss":
            return self.val_loss
        elif key == "val_accuracy":
            return self.val_accuracy
        raise KeyError(key)

    def __setitem__(self, key, value):
        if key == "loss":
            self.loss = value
        elif key == "accuracy":
            self.accuracy = value
        elif key == "val_loss":
            self.val_loss = value
        elif key == "val_accuracy":
            self.val_accuracy = value
        else:
            raise KeyError(key)

    def items(self):
        return [
            ("loss", self.loss),
            ("accuracy", self.accuracy),
            ("val_loss", self.val_loss),
            ("val_accuracy", self.val_accuracy),
        ]


class Sequential:

    def __init__(self):

        self.layers = []
        self.optimizer = None
        self.loss_function = None
        self.metric = None

    def add(self, layer):

        self.layers.append(layer)

    def forward(self, X, training=True):

        output = X

        for layer in self.layers:
            output = layer.forward(output, training=training)

        return output

    def backward(self, dvalues):

        for layer in reversed(self.layers):
            dvalues = layer.backward(dvalues)

    def predict(self, X):

        return self.forward(X, training=False)

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
        metric=None,
        batch_size=None,
        shuffle=True,
        validation_data=None,
        callbacks=None
    ):

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

        if self.loss_function is None:
            raise NotCompiledError(
                "Compile the model before calling fit()."
            )

        if callbacks is None:
            callbacks = []

        logs = {}

        for callback in callbacks:
            callback.on_train_begin(logs)

        history = History()

        samples = len(X)

        batch_size = batch_size if batch_size is not None else samples

        for epoch in range(epochs):

            for callback in callbacks:
                callback.on_epoch_begin(epoch, logs)

            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0

            if shuffle:
                indices = np.random.permutation(samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                X_shuffled = X
                y_shuffled = y

            for batch_idx, start_idx in enumerate(range(0, samples, batch_size)):

                for callback in callbacks:
                    callback.on_batch_begin(batch_idx, logs)

                end_idx = min(start_idx + batch_size, samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                predictions = self.forward(X_batch, training=True)

                loss_value = self.loss_function.forward(
                    y_batch,
                    predictions
                )

                epoch_loss += loss_value
                num_batches += 1

                dloss = self.loss_function.backward(
                    y_batch,
                    predictions
                )

                self.backward(dloss)

                for layer in self.layers:
                    self.optimizer.update(layer)

                if self.metric:

                    score = self.metric.calculate(
                        y_batch,
                        predictions
                    )
                    epoch_accuracy += score

                for callback in callbacks:
                    callback.on_batch_end(batch_idx, logs)

            avg_loss = epoch_loss / num_batches

            history.loss.append(avg_loss)

            logs["loss"] = avg_loss

            if self.metric:
                avg_accuracy = epoch_accuracy / num_batches
                history.accuracy.append(avg_accuracy)
                logs["accuracy"] = avg_accuracy

            if validation_data is not None:

                X_val, y_val = validation_data
                val_predictions = self.forward(X_val, training=False)
                val_loss = self.loss_function.forward(y_val, val_predictions)
                history.val_loss.append(val_loss)
                logs["val_loss"] = val_loss

                if self.metric:
                    val_accuracy = self.metric.calculate(y_val, val_predictions)
                    history.val_accuracy.append(val_accuracy)
                    logs["val_accuracy"] = val_accuracy

            logs["epoch"] = epoch

            if self.optimizer is not None:
                logs["learning_rate"] = getattr(
                    self.optimizer,
                    "learning_rate",
                    None
                )

            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)

            if epoch % 100 == 0:
                if self.metric:
                    msg = (
                        f"Epoch {epoch} "
                        f"Loss: {avg_loss:.6f} "
                        f"Accuracy: {avg_accuracy:.4f}"
                    )
                    if validation_data is not None:
                        msg += (
                            f" Val Loss: {val_loss:.6f} "
                            f"Val Accuracy: {val_accuracy:.4f}"
                        )
                    print(msg)
                else:
                    print(f"Epoch {epoch} Loss: {avg_loss:.6f}")

        for callback in callbacks:
            callback.on_train_end(logs)

        return history