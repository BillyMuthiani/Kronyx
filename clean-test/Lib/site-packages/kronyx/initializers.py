import numpy as np


def he_normal(input_size, output_size):
    """He initialization for ReLU activations."""
    return np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)


def xavier_uniform(input_size, output_size):
    """Xavier/Glorot uniform initialization."""
    scale = np.sqrt(6 / (input_size + output_size))
    return np.random.uniform(-scale, scale, (input_size, output_size))


def lecun_normal(input_size, output_size):
    """LeCun initialization for SELU activations."""
    return np.random.randn(input_size, output_size) * np.sqrt(1 / input_size)
