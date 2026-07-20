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

    def summary(self, input_shape: tuple | None = None, return_string: bool = False) -> str | None:
        """Print a summary of the model architecture.

        Shows layer types, output shapes, and parameter counts in a format
        similar to professional frameworks like Keras.

        Args:
            input_shape: Optional input shape tuple. If provided, attempts to
                infer output shapes for each layer using layer properties.
                If None, uses '?' for unknown shapes.
            return_string: If True, returns the summary as a string instead of
                printing it. Useful for testing.

        Returns:
            Summary string if return_string=True, None otherwise.

        Examples:
            >>> model.summary()
            >>> model.summary(input_shape=(28, 28, 1))  # For Conv2D first layer
        """
        lines: list[str] = []

        line_length = 75
        separator = "=" * line_length
        lines.append(separator)
        lines.append("Model: Sequential")
        lines.append(separator)

        layer_rows: list[tuple[str, str, str]] = []
        class_counts: dict[str, int] = {}
        logical_shape = input_shape

        for layer in self.layers:
            base_name = self._get_layer_type_name(layer)
            count = class_counts.get(base_name, 0)
            class_counts[base_name] = count + 1
            if count == 0:
                display_name = base_name
            else:
                display_name = f"{base_name}_{count}"

            shape_str = self._infer_output_shape(layer, logical_shape)
            params = self._get_layer_param_count(layer)
            breakdown = self._get_layer_param_breakdown(layer, logical_shape)
            params_str = f"{params:,}" if breakdown == "0" else breakdown

            layer_rows.append((display_name, shape_str, params_str))

            if shape_str != "?":
                logical_shape = self._parse_shape_for_next_layer(shape_str)

        col1_width = max(len("Layer (type)"), max((len(r[0]) for r in layer_rows), default=10))
        col2_width = max(len("Output Shape"), max((len(r[1]) for r in layer_rows), default=12))
        col3_width = max(len("Param #"), max((len(r[2]) for r in layer_rows), default=8))

        col1_width = max(col1_width, 10)
        col2_width = max(col2_width, 12)
        col3_width = max(col3_width, 8)

        total_width = col1_width + 2 + col2_width + 2 + col3_width
        thin_separator = "-" * max(total_width, line_length)

        header = (
            f"{'Layer (type)':<{col1_width}}  "
            f"{'Output Shape':<{col2_width}}  "
            f"{'Param #':>{col3_width}}"
        )
        lines.append(header)
        lines.append(thin_separator)

        for row_name, row_shape, row_params in layer_rows:
            row = (
                f"{row_name:<{col1_width}}  "
                f"{row_shape:<{col2_width}}  "
                f"{row_params:>{col3_width}}"
            )
            lines.append(row)

        total_params = sum(self._get_layer_param_count(layer) for layer in self.layers)
        trainable_params = total_params
        non_trainable_params = 0

        labels = ["Total params:", "Trainable params:", "Non-trainable params:"]
        values = [total_params, trainable_params, non_trainable_params]
        label_width = max(len(label) for label in labels)
        num_width = max(len(f"{v:,}") for v in values)

        lines.append(separator)
        for label, value in zip(labels, values, strict=True):
            lines.append(f"{label:<{label_width}}  {value:>{num_width},}")
        lines.append(separator)

        summary_text = "\n".join(lines)

        if return_string:
            return summary_text

        print(summary_text)
        return None

    def _get_layer_type_name(self, layer) -> str:
        """Get the human-readable type name for a layer.

        Args:
            layer: A layer instance.

        Returns:
            Type name string.
        """
        layer_class = type(layer).__name__
        if layer_class == "BatchNormalization":
            return "BatchNorm"
        return layer_class

    def _get_layer_param_count(self, layer) -> int:
        """Count the number of parameters in a layer.

        Matches the counting logic in the legacy summary() and
        count_params() implementations. Weights and biases are counted
        for Dense and Conv2D layers.

        Note: BatchNormalization gamma and beta are lazily initialized
        during the first forward pass, so they are not counted here
        unless training has occurred.

        Args:
            layer: A layer instance.

        Returns:
            Total parameter count for the layer.
        """
        if hasattr(layer, 'weights') and layer.weights is not None:
            return int(layer.weights.size + layer.biases.size)
        if hasattr(layer, 'kernels') and layer.kernels is not None:
            return int(layer.kernels.size + layer.biases.size)
        return 0

    def _get_layer_param_breakdown(self, layer, logical_shape: tuple | None) -> str:
        """Get a human-readable parameter breakdown for a layer.

        Returns strings like '48 (32W + 16B)' for Dense,
        '224 (216W + 8B)' for Conv2D, or '32 (16γ + 16β)' for
        BatchNormalization. Returns '0' for layers without parameters
        and '?' when the breakdown cannot be determined.

        Args:
            layer: A layer instance.
            logical_shape: Input shape without batch dimension, or None.

        Returns:
            Parameter breakdown string.
        """
        layer_type = type(layer).__name__

        if layer_type == "Dense":
            if hasattr(layer, 'weights') and layer.weights is not None:
                w_count = int(layer.weights.size)
                b_count = int(layer.biases.size)
                return f"{w_count + b_count} ({w_count}W + {b_count}B)"
            return "0"

        if layer_type == "Conv2D":
            if layer.kernels is not None:
                kh, kw, in_ch, filters = layer.kernels.shape
                w_count = kh * kw * in_ch * filters
                b_count = int(layer.biases.size)
                return f"{w_count + b_count} ({w_count}W + {b_count}B)"

            if logical_shape is not None and len(logical_shape) >= 3:
                in_ch = logical_shape[2]
                kernel_size = layer.kernel_size
                filters = layer.filters
                if isinstance(kernel_size, int):
                    kernel_h = kernel_w = kernel_size
                else:
                    kernel_h, kernel_w = kernel_size
                w_count = kernel_h * kernel_w * in_ch * filters
                b_count = filters
                return f"{w_count + b_count} ({w_count}W + {b_count}B)"

            return "?"

        if layer_type == "BatchNormalization":
            if hasattr(layer, 'gamma') and layer.gamma is not None:
                num_features = int(layer.gamma.size)
                return f"{num_features * 2} ({num_features}γ + {num_features}β)"

            if logical_shape is not None and len(logical_shape) >= 1:
                num_features = logical_shape[-1]
                return f"{num_features * 2} ({num_features}γ + {num_features}β)"

            return "0"

        return "0"

    def _infer_output_shape(self, layer, logical_shape: tuple | None) -> str:
        """Infer the output shape for a layer without running forward pass.

        Uses layer properties where possible. Returns '?' if shape cannot
        be determined.

        Args:
            layer: A layer instance.
            logical_shape: Input shape without batch dimension, or None.

        Returns:
            Shape string like '(None, 16)' or '?'.
        """
        layer_type = type(layer).__name__

        if layer_type == "Dense":
            output_size = None
            if hasattr(layer, 'biases') and layer.biases is not None:
                output_size = layer.biases.shape[1]
            elif hasattr(layer, 'weights') and layer.weights is not None:
                output_size = layer.weights.shape[1]

            if output_size is not None:
                return f"(None, {output_size})"

            return "?"

        if logical_shape is None:
            return "?"

        if layer_type == "Conv2D":
            if len(logical_shape) >= 2:
                in_h = logical_shape[0]
                in_w = logical_shape[1]

                kernel_size = layer.kernel_size
                stride = layer.stride
                padding = layer.padding
                filters = layer.filters

                if isinstance(kernel_size, int):
                    kernel_h = kernel_w = kernel_size
                else:
                    kernel_h, kernel_w = kernel_size

                if isinstance(stride, int):
                    stride_h = stride_w = stride
                else:
                    stride_h, stride_w = stride

                if padding == "valid":
                    out_h = (in_h - kernel_h) // stride_h + 1
                    out_w = (in_w - kernel_w) // stride_w + 1
                elif padding == "same":
                    out_h = (in_h + stride_h - 1) // stride_h
                    out_w = (in_w + stride_w - 1) // stride_w
                else:
                    return "?"

                return f"(None, {out_h}, {out_w}, {filters})"

            return "?"

        if layer_type == "Flatten":
            if len(logical_shape) >= 1:
                features = 1
                for dim in logical_shape:
                    features *= dim
                return f"(None, {features})"

            return "?"

        if layer_type in (
            "Dropout",
            "BatchNormalization",
            "ReLU",
            "Sigmoid",
            "Tanh",
            "Softmax",
        ):
            parts = ", ".join(str(dim) for dim in logical_shape)
            return f"(None, {parts})"

        return "?"

    def _parse_shape_for_next_layer(self, shape_str: str) -> tuple | None:
        """Parse a display shape string to get the logical shape for next layer.

        Args:
            shape_str: Shape string like '(None, 16)' or '?'.

        Returns:
            Logical shape tuple like (16,), or None if unknown.
        """
        if shape_str == "?" or not shape_str.startswith("(None, "):
            return None
        inner = shape_str[len("(None, "):-1]
        try:
            dims = tuple(int(d.strip()) for d in inner.split(","))
        except ValueError:
            return None
        return dims

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

    def visualize(
        self,
        output_format: str | None = None,
        show_params: bool = False,
        style: str = "ascii",
        compact: bool = False,
    ):
        """Display an ASCII architecture diagram of the model.

        Creates a visual representation of the model layers connected by arrows.
        If graphviz is installed and format is specified, generates an image.
        Otherwise displays an ASCII diagram in the specified style.

        Args:
            output_format: Optional output format ('png', 'pdf', 'svg'). If provided
                and graphviz is available, saves to file. Otherwise falls back
                to ASCII output with specified style.
            show_params: Deprecated. Parameter counts are always shown in the
                default visualization.
            style: Visualization style - "ascii", "box", "compact", or "unicode".
                Required when not using graphviz.
            compact: If True, displays a compact arrow-based diagram instead
                of the detailed box view.

        Raises:
            ImportError: If graphviz is requested but not installed.

        Examples:
            >>> model.visualize()
            >>> model.visualize(style="unicode")
            >>> model.visualize(output_format='png')  # Requires graphviz
            >>> model.visualize(compact=True)

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

        if compact:
            self._visualize_compact()
        else:
            self._visualize_default()

    def _visualize_default(self) -> None:
        """Print the default box-based architecture visualization."""
        if not self.layers:
            print("(Empty model)")
            return

        lines: list[str] = []
        line_length = 66
        separator = "═" * line_length

        lines.append(separator)
        title = "KRONYX MODEL VISUALIZATION"
        lines.append(title.center(line_length))
        lines.append(separator)
        lines.append("")
        lines.append("Input")
        lines.append("  │")
        lines.append("  ▼")

        logical_shape: tuple | None = None
        class_counts: dict[str, int] = {}

        for layer in self.layers:
            base_name = self._get_layer_type_name(layer)
            count = class_counts.get(base_name, 0)
            class_counts[base_name] = count + 1
            if count == 0:
                display_name = base_name
            else:
                display_name = f"{base_name}_{count}"

            title_line = display_name
            type_line = self._get_layer_visual_type_name(layer)
            metadata_lines = self._get_layer_visual_metadata(layer, logical_shape)

            box_width = max(
                len(title_line),
                len(type_line),
                max((len(m) for m in metadata_lines), default=0),
            )
            box_width = max(box_width, 20)

            top = "┌" + "─" * (box_width + 2) + "┐"
            bottom = "└" + "─" * (box_width + 2) + "┘"

            title_pad = (box_width - len(title_line)) // 2
            title_line = (
                "│ "
                + " " * title_pad
                + title_line
                + " " * (box_width - len(title_line) - title_pad)
                + " │"
            )

            type_pad = (box_width - len(type_line)) // 2
            type_line = (
                "│ "
                + " " * type_pad
                + type_line
                + " " * (box_width - len(type_line) - type_pad)
                + " │"
            )

            lines.append(top)
            lines.append(title_line)
            lines.append(type_line)

            if metadata_lines:
                divider = "├" + "─" * (box_width + 2) + "┤"
                lines.append(divider)
                for meta in metadata_lines:
                    padded = "│ " + meta.ljust(box_width) + " │"
                    lines.append(padded)

            lines.append(bottom)

            if layer is not self.layers[-1]:
                lines.append("                     │")
                lines.append("                     ▼")

            shape_str = self._infer_output_shape(layer, logical_shape)
            if shape_str != "?":
                logical_shape = self._parse_shape_for_next_layer(shape_str)

        lines.append("")
        lines.append("                     │")
        lines.append("                     ▼")
        lines.append("")
        lines.append("                 Prediction")

        lines.append("")
        lines.append(separator)
        stats_title = "Network Statistics"
        lines.append(stats_title.center(line_length))
        lines.append(separator)

        total_params = sum(self._get_layer_param_count(layer) for layer in self.layers)
        trainable_params = total_params
        non_trainable_params = 0
        num_layers = len(self.layers)
        num_trainable = sum(
            1 for layer in self.layers if self._get_layer_param_count(layer) > 0
        )
        num_activations = sum(
            1 for layer in self.layers
            if type(layer).__name__
            in ("ReLU", "Sigmoid", "Tanh", "Softmax")
        )
        memory_bytes = total_params * 4

        stats = [
            f"Layers               : {num_layers}",
            f"Trainable Layers     : {num_trainable}",
            f"Activation Layers    : {num_activations}",
            f"Total Parameters     : {total_params:,}",
            f"Trainable Params     : {trainable_params:,}",
            f"Non-trainable Params : {non_trainable_params:,}",
            f"Estimated Memory     : {self._format_memory(memory_bytes)}",
        ]

        for stat in stats:
            lines.append(stat)

        lines.append("")
        lines.append(separator)

        print("\n".join(lines))

    def _visualize_compact(self) -> None:
        """Print a compact arrow-based architecture diagram."""
        if not self.layers:
            print("(Empty model)")
            return

        print("Input")
        print(" │")
        print(" ▼")

        logical_shape: tuple | None = None
        class_counts: dict[str, int] = {}

        for layer in self.layers:
            base_name = self._get_layer_type_name(layer)
            count = class_counts.get(base_name, 0)
            class_counts[base_name] = count + 1
            if count == 0:
                display_name = base_name
            else:
                display_name = f"{base_name}_{count}"

            shape_str = self._infer_output_shape(layer, logical_shape)
            if shape_str != "?":
                logical_shape = self._parse_shape_for_next_layer(shape_str)

            if type(layer).__name__ == "Dense":
                if hasattr(layer, 'biases') and layer.biases is not None:
                    units = layer.biases.shape[1]
                else:
                    units = "?"
                print(f"{display_name}({units})")
            elif type(layer).__name__ == "Conv2D":
                print(f"{display_name}(filters={layer.filters})")
            else:
                print(display_name)

            if layer is not self.layers[-1]:
                print(" │")
                print(" ▼")

        print(" │")
        print(" ▼")
        print("Output")

    def _get_layer_visual_type_name(self, layer) -> str:
        """Get the display type name for visualization.

        Args:
            layer: A layer instance.

        Returns:
            Human-readable type name.
        """
        layer_class = type(layer).__name__
        if layer_class in ("ReLU", "Sigmoid", "Tanh", "Softmax"):
            return f"{layer_class} Activation"
        if layer_class == "BatchNormalization":
            return "BatchNorm"
        if layer_class == "Dropout":
            return "Dropout"
        return layer_class

    def _get_layer_visual_metadata(self, layer, logical_shape: tuple | None) -> list[str]:
        """Get metadata lines for a layer in the default visualization.

        Args:
            layer: A layer instance.
            logical_shape: Input shape without batch dimension, or None.

        Returns:
            List of metadata strings like ['Units        : 16', ...].
        """
        layer_type = type(layer).__name__
        metadata: list[str] = []

        if layer_type == "Dense":
            units = None
            if hasattr(layer, 'biases') and layer.biases is not None:
                units = layer.biases.shape[1]
            elif hasattr(layer, 'weights') and layer.weights is not None:
                units = layer.weights.shape[1]

            if units is not None:
                metadata.append(f"Units        : {units}")

            shape_str = self._infer_output_shape(layer, logical_shape)
            if shape_str != "?":
                metadata.append(f"Output Shape : {shape_str}")

            params = self._get_layer_param_count(layer)
            breakdown = self._get_layer_param_breakdown(layer, logical_shape)
            if breakdown != "?":
                metadata.append(f"Parameters   : {breakdown}")
            else:
                metadata.append(f"Parameters   : {params:,}")

        elif layer_type == "Conv2D":
            if isinstance(layer.kernel_size, int):
                metadata.append(f"Kernel Size  : {layer.kernel_size}")
            else:
                metadata.append(f"Kernel Size  : {layer.kernel_size}")
            metadata.append(f"Stride       : {layer.stride}")
            metadata.append(f"Padding      : {layer.padding}")
            metadata.append(f"Filters      : {layer.filters}")

            shape_str = self._infer_output_shape(layer, logical_shape)
            if shape_str != "?":
                metadata.append(f"Output Shape : {shape_str}")

            params = self._get_layer_param_count(layer)
            breakdown = self._get_layer_param_breakdown(layer, logical_shape)
            if breakdown != "?":
                metadata.append(f"Parameters   : {breakdown}")
            else:
                metadata.append(f"Parameters   : {params:,}")

        elif layer_type == "Flatten":
            shape_str = self._infer_output_shape(layer, logical_shape)
            if shape_str != "?":
                metadata.append(f"Output Shape : {shape_str}")

        elif layer_type == "Dropout":
            metadata.append(f"Rate         : {layer.rate}")
            shape_str = self._infer_output_shape(layer, logical_shape)
            if shape_str != "?":
                metadata.append(f"Output Shape : {shape_str}")

        elif layer_type == "BatchNormalization":
            shape_str = self._infer_output_shape(layer, logical_shape)
            if shape_str != "?":
                metadata.append(f"Output Shape : {shape_str}")

            params = self._get_layer_param_count(layer)
            breakdown = self._get_layer_param_breakdown(layer, logical_shape)
            if breakdown != "?" and breakdown != "0":
                metadata.append(f"Parameters   : {breakdown}")

        elif layer_type in ("ReLU", "Sigmoid", "Tanh", "Softmax"):
            shape_str = self._infer_output_shape(layer, logical_shape)
            if shape_str != "?":
                metadata.append(f"Output Shape : {shape_str}")

        return metadata

    def _format_memory(self, num_bytes: int) -> str:
        """Format a byte count into a human-readable string.

        Args:
            num_bytes: Number of bytes.

        Returns:
            Formatted string like '260 Bytes', '2.5 KB', or '1.2 MB'.
        """
        if num_bytes < 1024:
            return f"{num_bytes} Bytes"
        if num_bytes < 1024 * 1024:
            return f"{num_bytes / 1024:.1f} KB"
        return f"{num_bytes / (1024 * 1024):.1f} MB"

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
