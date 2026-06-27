class Callback:

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass


class ModelCheckpoint(Callback):
    """Callback that saves the model during training when a monitored metric improves.

    Saves model weights to a filepath when the monitored metric achieves a new best value.
    Can save either the full model or weights-only, and supports both minimization
    and maximization modes.
    """

    def __init__(
        self,
        filepath,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=True
    ):
        """Initialize the ModelCheckpoint callback.

        Args:
            filepath: Path where the model will be saved. Parent directories will be
                created automatically if they don't exist.
            monitor: Name of the metric to monitor. Defaults to "val_loss".
            mode: One of "min" or "max". Determines whether lower or higher metric
                values are considered better. Defaults to "min".
            save_best_only: If True, only save when the monitored metric improves.
                If False, save at the end of every epoch. Defaults to True.
            save_weights_only: If True, save only the model weights (not the full
                model). Defaults to False.
            verbose: If True, print messages when saving. Defaults to True.
        """
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        if not filepath:
            raise ValueError("filepath must be a non-empty string")

        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose
        self.best = None

    def on_epoch_end(self, epoch, logs=None):
        """Save model at epoch end if conditions are met.

        Args:
            epoch: Current epoch number (0-indexed).
            logs: Dictionary containing training metrics from the current epoch.
        """
        if logs is None:
            return

        current = logs.get(self.monitor)

        if current is None:
            raise ValueError(
                f"Monitored metric '{self.monitor}' not found in logs. "
                f"Available metrics: {list(logs.keys())}"
            )

        is_best = self._is_best(current)

        if self.save_best_only and not is_best:
            return

        self._save_model(epoch, current)

        if is_best:
            self.best = current

    def _is_best(self, current):
        """Check if current value is the best seen so far.

        Args:
            current: Current value of the monitored metric.

        Returns:
            True if current value is an improvement over the previous best.
        """
        if self.best is None:
            return True
        if self.mode == "min":
            return current < self.best
        return current > self.best

    def _save_model(self, epoch, current):
        """Save the model to the specified filepath.

        Args:
            epoch: Current epoch number for verbose output.
            current: Current value of the monitored metric for verbose output.
        """
        import os
        parent_dir = os.path.dirname(self.filepath)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        if self.verbose:
            if self.best is not None:
                print(
                    f"Epoch {epoch}: {self.monitor} improved from "
                    f"{self.best:.4f} to {current:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch}: {self.monitor} improved to {current:.4f}"
                )
            print(f"Saving model to {self.filepath}")

        if self.save_weights_only:
            self.model.save_weights(self.filepath)
        else:
            self.model.save(self.filepath)


class CSVLogger(Callback):
    """Callback that logs training metrics to a CSV file.

    Writes epoch-level metrics to a CSV file at the end of each epoch.
    Automatically handles file creation and appending.
    """

    def __init__(self, filepath, separator=",", append=False):
        """Initialize the CSVLogger callback.

        Args:
            filepath: Path where the CSV log will be saved. Parent directories will
                be created automatically if they don't exist.
            separator: Column separator character. Defaults to ",".
            append: If True, append to existing file without duplicating headers.
                If False, overwrite existing file. Defaults to False.
        """
        if not filepath:
            raise ValueError("filepath must be a non-empty string")

        self.filepath = filepath
        self.separator = separator
        self.append = append
        self.keys = None

    def on_train_begin(self, logs=None):
        """Initialize file and determine which metrics to log.

        Args:
            logs: Dictionary of training logs (may be None).
        """
        import os
        parent_dir = os.path.dirname(self.filepath)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        if logs:
            self.keys = sorted([k for k in logs.keys() if k != "epoch"])

    def on_epoch_end(self, epoch, logs=None):
        """Write metrics to CSV file at the end of each epoch.

        Args:
            epoch: Current epoch number (0-indexed).
            logs: Dictionary containing training metrics from the current epoch.
        """
        if logs is None:
            return

        if self.keys is None:
            self.keys = sorted([k for k in logs.keys() if k != "epoch"])

        if not self.append and epoch == 0:
            self._write_header()

        self._write_row(epoch, logs)

    def _write_header(self):
        """Write the CSV header row."""
        with open(self.filepath, "w") as f:
            header = [str(k) for k in self.keys]
            f.write(self.separator.join(header) + "\n")

    def _write_row(self, epoch, logs):
        """Write a data row to the CSV file.

        Args:
            epoch: Current epoch number.
            logs: Dictionary of metrics to log.
        """
        with open(self.filepath, "a") as f:
            values = [str(logs.get(k, "")) for k in self.keys]
            f.write(self.separator.join(values) + "\n")


class EarlyStopping(Callback):
    """Callback that stops training when a monitored metric stops improving.

    Monitors a specified metric and terminates training if no improvement is seen
    for a specified number of epochs (patience).
    """

    def __init__(
        self,
        monitor="val_loss",
        patience=10,
        min_delta=0.0,
        restore_best_weights=True
    ):
        """Initialize the EarlyStopping callback.

        Args:
            monitor: Name of the metric to monitor. Defaults to "val_loss".
            patience: Number of epochs to wait for improvement before stopping.
                Defaults to 10.
            min_delta: Minimum change to qualify as an improvement. Defaults to 0.0.
            restore_best_weights: If True, restore model weights from the epoch
                with the best monitored metric value. Defaults to True.
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best = None
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        """Reset state at the beginning of training.

        Args:
            logs: Dictionary of training logs (unused).
        """
        self.best = None
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        """Check metric and potentially stop training.

        Args:
            epoch: Current epoch number (0-indexed).
            logs: Dictionary containing training metrics from the current epoch.
        """
        if logs is None:
            return

        current = logs.get(self.monitor)

        if current is None:
            return

        if self.best is None:
            self.best = current
            self.best_weights = self._get_weights()
            self.wait = 0
        elif self._is_improvement(current):
            self.best = current
            self.best_weights = self._get_weights()
            self.wait = 0
        else:
            self.wait += 1

            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        """Restore best weights if configured and training was stopped early.

        Args:
            logs: Dictionary of training logs (unused).
        """
        if self.restore_best_weights and self.best_weights is not None:
            self._set_weights(self.best_weights)

    def _get_weights(self):
        """Extract weights from all trainable layers.

        Returns:
            List of (weights, biases) tuples for each layer with weights.
        """
        weights = []
        for layer in self.model.layers:
            if hasattr(layer, "weights"):
                weights.append((layer.weights.copy(), layer.biases.copy()))
        return weights

    def _set_weights(self, weights):
        """Load weights into all trainable layers.

        Args:
            weights: List of (weights, biases) tuples to restore.
        """
        for layer, weight_data in zip(
            [l for l in self.model.layers if hasattr(l, "weights")],
            weights
        ):
            layer.weights, layer.biases = weight_data[0].copy(), weight_data[1].copy()

    def _is_improvement(self, current):
        """Check if current metric value is an improvement.

        Uses min_delta to determine minimum change for improvement.
        For validation metrics, lower is better; for training metrics, higher is better.

        Args:
            current: Current value of the monitored metric.

        Returns:
            True if the current value represents an improvement.
        """
        if self.monitor.startswith("val_"):
            return current < self.best - self.min_delta
        else:
            return current > self.best + self.min_delta