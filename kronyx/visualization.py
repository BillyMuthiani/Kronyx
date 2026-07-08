"""Visualization utilities for neural network analysis.

Publication-quality matplotlib visualizations for understanding models and data.
All functions are educational and designed to make learning easier.
"""
import numpy as np


def plot_training_curves(history, metric: str | None = None, filename: str | None = None) -> None:
    """Plot training and validation metrics from history.

    Creates informative plots to visualize model training progress.
    Automatically detects available metrics and plots them with proper labels.

    Args:
        history: History object with loss, accuracy, val_loss, val_accuracy lists.
        metric: Optional specific metric to plot. If None, plots all available.
        filename: Optional path to save the figure. If None, displays the plot.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from e

    if metric is not None:
        _plot_single_metric(history, metric, filename)
        return

    _plot_all_metrics(history, filename)


def _plot_single_metric(history, metric: str, filename: str | None) -> None:
    """Plot a single metric."""
    import matplotlib.pyplot as plt  # noqa: F401

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    if metric == "loss" and history.loss:
        ax.plot(history.loss, label="Training Loss", marker="o", markersize=3)
        ax.set_ylabel("Loss")
    elif metric == "accuracy" and history.accuracy:
        ax.plot(history.accuracy, label="Training Accuracy", marker="o", markersize=3)
        ax.set_ylabel("Accuracy")
    elif metric == "val_loss" and history.val_loss:
        ax.plot(history.val_loss, label="Validation Loss", marker="o", markersize=3)
        ax.set_ylabel("Loss")
    elif metric == "val_accuracy" and history.val_accuracy:
        ax.plot(history.val_accuracy, label="Validation Accuracy", marker="o", markersize=3)
        ax.set_ylabel("Accuracy")

    ax.set_xlabel("Epoch")
    ax.set_title(f"Training {metric.replace('_', ' ').title()}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {filename}")
    else:
        plt.show()


def _plot_all_metrics(history, filename: str | None) -> None:
    """Plot all available metrics."""
    import matplotlib.pyplot as plt  # noqa: F401

    metrics_to_plot = []
    if history.loss:
        metrics_to_plot.append(("loss", history.loss, "Training Loss"))
    if history.val_loss:
        metrics_to_plot.append(("val_loss", history.val_loss, "Validation Loss"))
    if history.accuracy:
        metrics_to_plot.append(("accuracy", history.accuracy, "Training Accuracy"))
    if history.val_accuracy:
        metrics_to_plot.append(("val_accuracy", history.val_accuracy, "Validation Accuracy"))

    if not metrics_to_plot:
        print("No metrics available to plot.")
        return

    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(8, 4 * len(metrics_to_plot)))
    if len(metrics_to_plot) == 1:
        axes = [axes]

    epochs = list(range(1, len(history.epochs) + 1))

    for ax, (name, data, label) in zip(axes, metrics_to_plot, strict=True):
        ax.plot(epochs[:len(data)], data, label=label, marker="o", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name.replace("_", " ").title())
        ax.set_title(f"Training {name.replace('_', ' ').title()}")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {filename}")
    else:
        plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list | None = None,
    filename: str | None = None,
) -> None:
    """Plot a confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix as 2D array.
        classes: Optional list of class names. If None, uses integers.
        filename: Optional path to save the figure.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from e

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    if classes is None:
        classes = [str(i) for i in range(cm.shape[0])]

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix"
    )

    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {filename}")
    else:
        plt.show()


def plot_decision_boundary(
    model,
    x: np.ndarray,
    y: np.ndarray,
    resolution: float = 0.02,
    filename: str | None = None,
) -> None:
    """Plot decision boundary of a classifier.

    Creates a contour plot showing the decision regions of a model.

    Args:
        model: Trained model with predict() method.
        x: Input features, shape (n_samples, 2).
        y: True labels.
        resolution: Grid resolution for decision boundary. Defaults to 0.02.
        filename: Optional path to save the figure.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from e

    # Create mesh grid
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution)
    )

    # Predict on mesh
    mesh = np.c_[xx.ravel(), yy.ravel()]
    z = model.predict(mesh)
    if z.ndim > 1 and z.shape[1] > 1:
        z = np.argmax(z, axis=1)
    else:
        z = (z > 0.5).astype(int).flatten()
    z = z.reshape(xx.shape)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.contourf(xx, yy, z, alpha=0.3, levels=np.unique(z) + 0.5)
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Set1, edgecolors="k")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Decision Boundary")

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {filename}")
    else:
        plt.show()


def plot_dataset(
    x: np.ndarray, y: np.ndarray, filename: str | None = None
) -> None:
    """Plot a 2D dataset with class colors.

    Args:
        x: Input features, shape (n_samples, 2).
        y: True labels.
        filename: Optional path to save the figure.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from e

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Set1, edgecolors="k")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Dataset Visualization")

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {filename}")
    else:
        plt.show()


def plot_predictions(
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filename: str | None = None,
) -> None:
    """Plot predictions vs true labels.

    Args:
        x: Input features, shape (n_samples, 2).
        y_true: True labels.
        y_pred: Predicted labels.
        filename: Optional path to save the figure.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from e

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # True labels
    axes[0].scatter(x[:, 0], x[:, 1], c=y_true, cmap=plt.cm.Set1, edgecolors="k")
    axes[0].set_title("True Labels")
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")

    # Predictions
    axes[1].scatter(x[:, 0], x[:, 1], c=y_pred, cmap=plt.cm.Set1, edgecolors="k")
    axes[1].set_title("Predictions")
    axes[1].set_xlabel("Feature 1")
    axes[1].set_ylabel("Feature 2")

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {filename}")
    else:
        plt.show()


def plot_feature_space(
    x: np.ndarray,
    y: np.ndarray,
    feature_pairs: list | None = None,
    filename: str | None = None,
) -> None:
    """Plot feature space for multiple feature pairs.

    Creates a grid of scatter plots for all feature combinations.

    Args:
        x: Input features, shape (n_samples, n_features).
        y: True labels.
        feature_pairs: Optional list of (i, j) tuples for feature indices.
        filename: Optional path to save the figure.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from e

    n_features = x.shape[1]

    if feature_pairs is None:
        feature_pairs = [(i, j) for i in range(n_features) for j in range(i + 1, n_features)]

    n_plots = len(feature_pairs)
    if n_plots == 0:
        print("No feature pairs to plot.")
        return

    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, (i, j) in zip(axes, feature_pairs, strict=False):
        ax.scatter(x[:, i], x[:, j], c=y, cmap=plt.cm.Set1, edgecolors="k", s=20)
        ax.set_xlabel(f"Feature {i}")
        ax.set_ylabel(f"Feature {j}")

    # Hide unused subplots
    for ax in axes[n_plots:]:
        ax.set_visible(False)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {filename}")
    else:
        plt.show()
