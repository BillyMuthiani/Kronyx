"""Model implementation with Sequential API and training utilities."""
from __future__ import annotations

import warnings

import numpy as np

from kronyx.exceptions import NotCompiledError


class History:
    """Training history container with metrics and utility methods.

    Stores loss and accuracy values for training and validation across epochs.

    Examples:
        >>> history = History()
        >>> history.loss.append(0.5)
        >>> history.summary()
        'loss: 0.5000'
    """

    def __init__(self):
        self.loss = []
        self.accuracy = []
        self.val_loss = []
        self.val_accuracy = []
        self.learning_rate = []
        self.epochs = []

    def __getitem__(self, key):
        if key == "loss":
            return self.loss
        elif key == "accuracy":
            return self.accuracy
        elif key == "val_loss":
            return self.val_loss
        elif key == "val_accuracy":
            return self.val_accuracy
        elif key == "learning_rate":
            return self.learning_rate
        elif key == "epochs":
            return self.epochs
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
        elif key == "learning_rate":
            self.learning_rate = value
        elif key == "epochs":
            self.epochs = value
        else:
            raise KeyError(key)

    def items(self):
        return [
            ("loss", self.loss),
            ("accuracy", self.accuracy),
            ("val_loss", self.val_loss),
            ("val_accuracy", self.val_accuracy),
            ("learning_rate", self.learning_rate),
            ("epochs", self.epochs),
        ]

    def to_dict(self):
        """Convert history to a dictionary.

        Returns:
            Dictionary with all metric lists.
        """
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
        }

    def to_csv(self, filepath):
        """Save history to CSV file.

        Args:
            filepath: Path to save the CSV file.
        """
        import csv

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "loss", "accuracy",
                "val_loss", "val_accuracy", "learning_rate"
            ])
            for i in range(len(self.epochs)):
                writer.writerow([
                    self.epochs[i] if self.epochs else i,
                    self.loss[i] if i < len(self.loss) else "",
                    self.accuracy[i] if i < len(self.accuracy) else "",
                    self.val_loss[i] if i < len(self.val_loss) else "",
                    self.val_accuracy[i] if i < len(self.val_accuracy) else "",
                    self.learning_rate[i] if i < len(self.learning_rate) else "",
                ])

    def summary(self):
        """Return a summary string of the training history.

        Returns:
            String with final metrics summary.
        """
        lines = []
        if self.loss:
            lines.append(f"loss: {self.loss[-1]:.4f}")
        if self.accuracy:
            lines.append(f"accuracy: {self.accuracy[-1]*100:.1f}%")
        if self.val_loss:
            lines.append(f"val_loss: {self.val_loss[-1]:.4f}")
        if self.val_accuracy:
            lines.append(f"val_accuracy: {self.val_accuracy[-1]*100:.1f}%")
        return "\n".join(lines)

    def plot(self, metric: str | None = None, filename: str | None = None):
        """Plot training and validation metrics from history.

        Creates informative plots to visualize model training progress. Automatically
        detects available metrics and plots them with proper labels, legends, and grids.

        Args:
            metric: Optional specific metric to plot ('loss', 'accuracy', 'val_loss',
                'val_accuracy'). If None, plots all available metrics.
            filename: Optional path to save the figure. If None, displays the plot.

        Raises:
            ImportError: If matplotlib is not installed, with installation instructions.

        Examples:
            >>> history.plot()  # Plot all metrics
            >>> history.plot(metric='loss')  # Plot only loss
            >>> history.plot(filename='training_history.png')  # Save to file
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for plotting. Install it with: pip install matplotlib"
            ) from e

        metrics_to_plot = self._get_plot_metrics(metric)
        if not metrics_to_plot:
            print("No metrics available to plot.")
            return

        fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(8, 4 * len(metrics_to_plot)))
        if len(metrics_to_plot) == 1:
            axes = [axes]

        epochs = list(range(1, len(self.epochs) + 1))

        for ax, (name, data, label) in zip(axes, metrics_to_plot, strict=True):
            ax.plot(epochs[:len(data)], data, label=label, marker='o', markersize=3)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(name.replace('_', ' ').title())
            ax.set_title(f'Training {name.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
            if len(data) > 0:
                ax.legend()

        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {filename}")
        else:
            plt.show()

    def _get_plot_metrics(self, metric: str | None) -> list[tuple]:
        """Get list of metrics to plot based on metric filter.

        Args:
            metric: Optional metric name filter.

        Returns:
            List of (metric_name, data, label) tuples.
        """
        available = [
            ('loss', self.loss, 'Training Loss'),
            ('val_loss', self.val_loss, 'Validation Loss'),
        ]

        if self.accuracy and self.val_accuracy:
            available.insert(1, ('accuracy', self.accuracy, 'Training Accuracy'))
            available.insert(2, ('val_accuracy', self.val_accuracy, 'Validation Accuracy'))

        if metric is not None:
            metric_lower = metric.lower()
            for _i, (m, data, label) in enumerate(available):
                if metric_lower in m or m.startswith(metric_lower):
                    return [(m, data, label)]
        return [(m, d, lbl) for m, d, lbl in available if d]

    def to_dataframe(self):
        """Convert history to a pandas DataFrame.

        Returns:
            pandas DataFrame with columns: epoch, loss, accuracy,
                val_loss, val_accuracy, learning_rate.

        Raises:
            ImportError: If pandas is not installed, with installation instructions.

        Examples:
            >>> df = history.to_dataframe()
            >>> df.head()
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for DataFrame conversion. Install it with: pip install pandas"
            ) from e

        data = {}
        if self.epochs:
            data['epoch'] = self.epochs
        else:
            data['epoch'] = list(range(1, len(self.loss) + 1)) if self.loss else []

        if self.loss:
            data['loss'] = self.loss
        if self.accuracy:
            data['accuracy'] = self.accuracy
        if self.val_loss:
            data['val_loss'] = self.val_loss
        if self.val_accuracy:
            data['val_accuracy'] = self.val_accuracy
        if self.learning_rate:
            data['learning_rate'] = self.learning_rate

        max_len = max((len(v) for v in data.values()), default=0)
        for key in data:
            while len(data[key]) < max_len:
                data[key].append(None)

        return pd.DataFrame(data)


class Sequential:
    """Sequential neural network model with Keras-like API.

    Example:
        >>> model = Sequential()
        >>> model.add(Dense(4, 16))
        >>> model.add(ReLU())
        >>> model.compile(loss=BinaryCrossEntropy(), optimizer=Adam())
        >>> history = model.fit(x, y, epochs=100)
    """

    def __init__(self):
        self.layers = []
        self.optimizer = None
        self.loss_function = None
        self.metric = None

    @classmethod
    def from_json(cls, json_string):
        """Create a model from JSON architecture string.

        Args:
            json_string: JSON string containing architecture.

        Returns:
            Sequential model with layers matching the architecture.

        Raises:
            SerializationError: If JSON is invalid or unsupported layer types.
        """
        from kronyx.serialization import from_json as _from_json
        model = _from_json(json_string)
        return model

    def add(self, layer):
        """Add a layer to the model.

        Args:
            layer: Layer instance to add.
        """
        self.layers.append(layer)

    def forward(self, x, training=True):
        """Forward pass through all layers.

        Args:
            x: Input data.
            training: If True, layers may cache values for backward.

        Returns:
            Output after passing through all layers.
        """
        output = x
        for layer in self.layers:
            output = layer.forward(output, training=training)
        return output

    def backward(self, dvalues):
        """Backward pass through all layers in reverse order.

        Args:
            dvalues: Gradient from the loss function.
        """
        for layer in reversed(self.layers):
            dvalues = layer.backward(dvalues)

    def predict(self, x, batch_size=None, verbose=0):
        """Make predictions without updating weights.

        Args:
            x: Input data.
            batch_size: If provided, predict in batches.
            verbose: 0=silent, 1=progress bar.

        Returns:
            ndarray of predictions.
        """
        if batch_size is None:
            return self.forward(x, training=False)

        predictions = []
        samples = len(x)
        for start_idx in range(0, samples, batch_size):
            end_idx = min(start_idx + batch_size, samples)
            batch_preds = self.forward(x[start_idx:end_idx], training=False)
            predictions.append(batch_preds)
            if verbose:
                print(f"Predicting: {end_idx}/{samples}")
        return np.vstack(predictions)

    def predict_proba(self, x, batch_size=None):
        """Return probabilistic predictions.

        Alias for predict() for API consistency.

        Args:
            x: Input data.
            batch_size: If provided, predict in batches.

        Returns:
            ndarray of predictions (same as predict()).
        """
        return self.predict(x, batch_size=batch_size)

    def compile(
        self,
        loss,
        optimizer,
        metric=None
    ):
        """Configure the model for training.

        Args:
            loss: Loss function instance.
            optimizer: Optimizer instance.
            metric: Optional metric for evaluation.
        """
        self.loss_function = loss
        self.optimizer = optimizer
        self.metric = metric

    def summary(self, input_shape: tuple | None = None):
        """Print a summary of the model architecture.

        Shows layer types, output shapes, and parameter counts in a format
        similar to professional frameworks like Keras.

        Args:
            input_shape: Optional input shape tuple. If provided, calculates
                output shapes for each layer. If None, uses '?' for unknown shapes.

        Examples:
            >>> model.summary()
            >>> model.summary(input_shape=(28, 28, 1))  # For Conv2D first layer
        """
        line_length = 75
        separator = "=" * line_length
        header = f"{'Layer (type)':<25} {'Output Shape':<25} {'Param #':<10}"
        print(separator)
        print(header)
        print(separator)

        total_params = 0
        trainable_params = 0
        current_shape = input_shape

        for i, layer in enumerate(self.layers):
            layer_name = type(layer).__name__
            layer_idx = i

            if hasattr(layer, 'weights') and layer.weights is not None:
                params = int(layer.weights.size + layer.biases.size)
                trainable_params += params
            elif hasattr(layer, 'kernels') and layer.kernels is not None:
                params = int(layer.kernels.size + layer.biases.size)
                trainable_params += params
            else:
                params = 0

            total_params += params

            if current_shape is not None:
                try:
                    if isinstance(current_shape, tuple):
                        dummy_input = np.random.randn(1, *current_shape).astype(np.float64)
                    else:
                        dummy_input = np.random.randn(1, 10).astype(np.float64)
                    output = layer.forward(dummy_input, training=False)
                    current_shape = tuple(output.shape[1:])
                except Exception:
                    shape_str = '?'
                else:
                    shape_str = str(current_shape)
            else:
                shape_str = '?'
            print(f"{layer_name}{layer_idx:<20} {shape_str:<25} {params:<10}")

        print(separator)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Non-trainable params: {total_params - trainable_params:,}")
        print(separator)

    def count_params(self) -> int:
        """Count total trainable parameters in the model.

        Returns:
            Total number of trainable parameters across all layers.

        Examples:
            >>> model.count_params()
            101770
        """
        total = 0
        for layer in self.layers:
            if hasattr(layer, 'weights') and layer.weights is not None:
                total += int(layer.weights.size + layer.biases.size)
            elif hasattr(layer, 'kernels') and layer.kernels is not None:
                total += int(layer.kernels.size + layer.biases.size)
        return total

    def layer_summary(self) -> list[dict[str, object]]:
        """Get structured information about each layer.

        Returns:
            List of dictionaries with layer information including:
                - name: Layer class name
                - type: Layer type string
                - params: Number of trainable parameters
                - trainable: Whether layer has trainable parameters

        Examples:
            >>> model.layer_summary()
            [{'name': 'Dense', 'type': 'dense', 'params': 100480, 'trainable': True}, ...]
        """
        summary = []
        for _i, layer in enumerate(self.layers):
            layer_name = type(layer).__name__

            if hasattr(layer, 'weights') and layer.weights is not None:
                params = int(layer.weights.size + layer.biases.size)
            elif hasattr(layer, 'kernels') and layer.kernels is not None:
                params = int(layer.kernels.size + layer.biases.size)
            else:
                params = 0

            summary.append({
                'name': layer_name,
                'type': layer_name.lower(),
                'params': params,
                'trainable': params > 0,
            })
        return summary

    def visualize(self, output_format: str | None = None, show_params: bool = False):
        """Display an ASCII architecture diagram of the model.

        Creates a visual representation of the model layers connected by arrows.
        If graphviz is installed and format is specified, generates an image.

        Args:
            output_format: Optional output format ('png', 'pdf', 'svg'). If provided
                and graphviz is available, saves to file. Otherwise falls back
                to ASCII output.
            show_params: If True, shows parameter counts beside each layer.

        Raises:
            ImportError: If graphviz is requested but not installed.

        Examples:
            >>> model.visualize()
            >>> model.visualize(show_params=True)
            >>> model.visualize(output_format='png')  # Requires graphviz

        Notes:
            Graphviz is optional. ASCII visualization works without it.
            Install graphviz with: pip install graphviz
        """
        if output_format is not None:
            try:
                import graphviz  # noqa: F401
                return self._visualize_graphviz(output_format)
            except ImportError:
                print("graphviz not installed. Falling back to ASCII visualization.")
                print("To install: pip install graphviz")
                output_format = None

        self._visualize_ascii(show_params)

    def _visualize_ascii(self, show_params: bool):
        """Print ASCII architecture diagram.

        Args:
            show_params: If True, shows parameter counts beside layers.
        """
        if not self.layers:
            print("(Empty model)")
            return

        print()
        for i, layer in enumerate(self.layers):
            layer_name = type(layer).__name__

            if show_params:
                if hasattr(layer, 'weights') and layer.weights is not None:
                    params = int(layer.weights.size + layer.biases.size)
                    print(f"{layer_name}({params})")
                elif hasattr(layer, 'kernels') and layer.kernels is not None:
                    params = int(layer.kernels.size + layer.biases.size)
                    print(f"{layer_name}({params})")
                else:
                    print(layer_name)
            else:
                print(layer_name)

            if i < len(self.layers) - 1:
                print("  |")
                print("  v")

        print()

    def _visualize_graphviz(self, output_format: str):
        """Generate graphviz visualization.

        Args:
            output_format: Output format (png, pdf, svg).

        Returns:
            Graphviz Source object.
        """
        import graphviz

        dot = graphviz.Digraph(comment='Model Architecture')
        dot.attr(rankdir='TB')

        for i, layer in enumerate(self.layers):
            layer_name = type(layer).__name__
            if hasattr(layer, 'weights') and layer.weights is not None:
                params = int(layer.weights.size + layer.biases.size)
                label = f"{layer_name}\\nparams: {params}"
            else:
                label = layer_name

            dot.node(str(i), label)

            if i > 0:
                dot.edge(str(i-1), str(i))

        filename = f"model_architecture.{output_format}"
        dot.render(filename, cleanup=True, format=output_format)
        print(f"Saved to {filename}")

    def save(self, filename):
        """Save complete model to .krx format.

        Args:
            filename: Path to save (should end with .krx).
        """
        from kronyx.serialization import save_model as _save_model
        _save_model(self, filename)

    def save_weights(self, filename):
        """Save model weights to a .npz file.

        Args:
            filename: Path where weights will be saved.
        """
        from kronyx.serialization import save as _save
        _save(self, filename)

    def load_weights(self, filename):
        """Load model weights from a .npz file.

        Args:
            filename: Path to load weights from.

        Raises:
            SerializationError: If loading fails or architecture incompatible.
        """
        from kronyx.serialization import load as _load
        _load(self, filename)

    def to_json(self):
        """Export model architecture to JSON string.

        Returns:
            JSON string containing model architecture.
        """
        from kronyx.serialization import to_json as _to_json
        return _to_json(self)

    def load(self, filename):
        """Load model weights from a file.

        Args:
            filename: Path to load weights from.
        """
        from kronyx.serialization import load
        load(self, filename)

    def evaluate(self, x, y, batch_size=None, verbose=0):
        """Evaluate the model on test data.

        Args:
            x: Input data.
            y: True labels.
            batch_size: If provided, evaluate in batches.
            verbose: 0=silent, 1=progress.

        Returns:
            Tuple of (loss, accuracy).
        """
        predictions = self.predict(x, batch_size=batch_size)
        loss = self.loss_function.forward(y, predictions)

        if self.metric:
            accuracy = self.metric.calculate(y, predictions)
        else:
            accuracy = 0.0

        if verbose:
            print(f"loss: {loss:.4f}")
            if self.metric:
                print(f"accuracy: {accuracy*100:.1f}%")

        return (loss, accuracy)

    def fit(
        self,
        x,
        y,
        epochs=1000,
        loss=None,
        optimizer=None,
        metric=None,
        batch_size=None,
        shuffle=True,
        validation_data=None,
        callbacks=None,
        verbose=1
    ):
        """Train the model on the given data.

        Args:
            x: Training input data.
            y: Training target labels.
            epochs: Number of training epochs.
            loss: Deprecated - use compile().
            optimizer: Deprecated - use compile().
            metric: Deprecated - use compile().
            batch_size: Batch size for training (default: all samples).
            shuffle: Whether to shuffle training data.
            validation_data: Optional (x_val, y_val) tuple.
            callbacks: List of callback instances.
            verbose: 0=silent, 1=progress every epoch, 2=detailed metrics.

        Returns:
            History instance with training metrics.
        """
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

        for callback in callbacks:
            callback.model = self

        logs = {}

        for callback in callbacks:
            callback.on_train_begin(logs)

        history = History()

        samples = len(x)

        batch_size = batch_size if batch_size is not None else samples

        for epoch in range(epochs):

            for callback in callbacks:
                callback.on_epoch_begin(epoch, logs)

            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0

            if shuffle:
                indices = np.random.permutation(samples)
                x_shuffled = x[indices]
                y_shuffled = y[indices]
            else:
                x_shuffled = x
                y_shuffled = y

            for batch_idx, start_idx in enumerate(range(0, samples, batch_size)):

                for callback in callbacks:
                    callback.on_batch_begin(batch_idx, logs)

                end_idx = min(start_idx + batch_size, samples)
                x_batch = x_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                predictions = self.forward(x_batch, training=True)

                loss_value = self.loss_function.forward(
                    y_batch,
                    predictions
                )

                reg_loss = sum(
                    getattr(layer, "regularization_loss", 0.0)
                    for layer in self.layers
                )

                epoch_loss += loss_value + reg_loss
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
            history.epochs.append(epoch)
            logs["loss"] = avg_loss

            avg_accuracy = None
            if self.metric:
                avg_accuracy = epoch_accuracy / num_batches
                history.accuracy.append(avg_accuracy)
                logs["accuracy"] = avg_accuracy

            val_loss_val = None
            val_accuracy_val = None
            if validation_data is not None:

                x_val, y_val = validation_data
                val_pred = self.forward(x_val, training=False)
                val_loss_val = self.loss_function.forward(y_val, val_pred)
                history.val_loss.append(val_loss_val)
                logs["val_loss"] = val_loss_val

                if self.metric:
                    val_accuracy_val = self.metric.calculate(y_val, val_pred)
                    history.val_accuracy.append(val_accuracy_val)
                    logs["val_accuracy"] = val_accuracy_val

            logs["epoch"] = epoch

            lr = None
            if self.optimizer is not None:
                lr = getattr(
                    self.optimizer,
                    "learning_rate",
                    None
                )
                history.learning_rate.append(lr)
                logs["learning_rate"] = lr

            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)

            if getattr(self, "stop_training", False):
                break

            if verbose > 0:
                self._print_epoch(epoch, epochs, avg_loss, avg_accuracy,
                                   val_loss_val, val_accuracy_val, lr)

        for callback in callbacks:
            callback.on_train_end(logs)

        return history

    def _print_epoch(self, epoch, epochs, loss, accuracy,
                       val_loss=None, val_accuracy=None, lr=None):
        """Print epoch progress with formatted metrics.

        Args:
            epoch: Current epoch number.
            epochs: Total epochs.
            loss: Training loss.
            accuracy: Training accuracy.
            val_loss: Optional validation loss.
            val_accuracy: Optional validation accuracy.
            lr: Optional learning rate.
        """
        msg = f"Epoch {epoch + 1}/{epochs}"
        if accuracy is not None:
            msg += f"\n  loss: {loss:.4f}"
            msg += f"\n  accuracy: {accuracy*100:.1f}%"
        else:
            msg += f"\n  loss: {loss:.4f}"

        if val_loss is not None:
            msg += f"\n  val_loss: {val_loss:.4f}"
        if val_accuracy is not None:
            msg += f"\n  val_accuracy: {val_accuracy*100:.1f}%"
        if lr is not None:
            msg += f"\n  learning_rate: {lr}"

        print(msg)
