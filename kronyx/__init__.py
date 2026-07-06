from kronyx.activations import ReLU, Sigmoid, Softmax, Tanh
from kronyx.callbacks import (
    Callback,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from kronyx.data import BatchLoader, Dataset, TensorDataset, train_test_split
from kronyx.datasets import blobs, circles, iris, moons, spiral, xor
from kronyx.exceptions import (
    ConfigurationError,
    NeuralnetError,
    NotCompiledError,
    OptimizerError,
    ShapeError,
)
from kronyx.initializers import he_normal, lecun_normal, xavier_uniform
from kronyx.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten
from kronyx.losses import (
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    SoftmaxCategoricalCrossEntropy,
)
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
from kronyx.model import History, Sequential
from kronyx.optimizers import SGD, Adam
from kronyx.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler
from kronyx.regularizers import L2
from kronyx.serialization import SerializationError, from_json, load_model
from kronyx.utils import set_seed
from kronyx.version import __version__
from kronyx.visualization import (
    plot_confusion_matrix,
    plot_dataset,
    plot_decision_boundary,
    plot_feature_space,
    plot_predictions,
    plot_training_curves,
)

__all__ = [
    # Layers
    "Dense",
    "Dropout",
    "BatchNormalization",
    "Flatten",
    "Conv2D",
    # Activations
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    # Losses
    "BinaryCrossEntropy",
    "SoftmaxCategoricalCrossEntropy",
    "CategoricalCrossEntropy",
    # Optimizers
    "SGD",
    "Adam",
    # Metrics
    "Accuracy",
    "BinaryAccuracy",
    "CategoricalAccuracy",
    "Precision",
    "Recall",
    "F1Score",
    "ConfusionMatrix",
    "TopKAccuracy",
    # Model
    "Sequential",
    "History",
    # Exceptions
    "NeuralnetError",
    "ConfigurationError",
    "NotCompiledError",
    "ShapeError",
    "OptimizerError",
    # Initializers
    "he_normal",
    "xavier_uniform",
    "lecun_normal",
    # Callbacks
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "CSVLogger",
    "ReduceLROnPlateau",
    # Regularizers
    "L2",
    # Utils
    "set_seed",
    # Serialization
    "load_model",
    "from_json",
    "SerializationError",
    # Datasets
    "xor",
    "iris",
    "spiral",
    "circles",
    "moons",
    "blobs",
    # Preprocessing
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "OneHotEncoder",
    # Data utilities
    "train_test_split",
    "BatchLoader",
    "Dataset",
    "TensorDataset",
    # Visualization
    "plot_training_curves",
    "plot_confusion_matrix",
    "plot_decision_boundary",
    "plot_dataset",
    "plot_predictions",
    "plot_feature_space",
    # Version
    "__version__",
]
