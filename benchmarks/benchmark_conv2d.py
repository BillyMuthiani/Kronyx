"""Benchmark Conv2D layer forward and backward pass performance."""
import time
import numpy as np
from Neuralnet import Conv2D, ReLU


def benchmark_conv2d(
    batch_size: int = 16,
    height: int = 28,
    width: int = 28,
    channels: int = 1,
    filters: int = 32,
    kernel_size: int = 3,
    iterations: int = 500
) -> None:
    """Benchmark Conv2D layer performance.

    Args:
        batch_size: Batch size for operations.
        height: Input height.
        width: Input width.
        channels: Number of input channels.
        filters: Number of output filters.
        kernel_size: Convolution kernel size.
        iterations: Number of benchmark iterations.
    """
    np.random.seed(42)

    conv = Conv2D(filters=filters, kernel_size=kernel_size, padding="same")
    activation = ReLU()

    X = np.random.randn(batch_size, height, width, channels)

    # Forward pass to get output shape
    _ = conv.forward(X, training=True)
    _, out_h, out_w, _ = conv.forward(X, training=True).shape
    dvalues = np.random.randn(batch_size, out_h, out_w, filters)

    # Warmup
    for _ in range(100):
        out = conv.forward(X, training=True)
        out = activation.forward(out, training=True)
        d = activation.backward(dvalues.copy())
        conv.backward(d)

    # Benchmark forward
    start = time.perf_counter()
    for _ in range(iterations):
        out = conv.forward(X, training=True)
        out = activation.forward(out, training=True)
    forward_time = (time.perf_counter() - start) / iterations

    # Benchmark backward
    start = time.perf_counter()
    for _ in range(iterations):
        d = activation.backward(dvalues.copy())
        conv.backward(d)
    backward_time = (time.perf_counter() - start) / iterations

    # Benchmark training throughput
    start = time.perf_counter()
    for _ in range(iterations):
        out = conv.forward(X, training=True)
        out = activation.forward(out, training=True)
        d = activation.backward(dvalues.copy())
        conv.backward(d)
    training_time = (time.perf_counter() - start) / iterations

    print(f"Conv2D({height}x{width}, {channels}->{filters}) Benchmark:")
    print(f"  Batch size: {batch_size}")
    print(f"  Forward time: {forward_time * 1000:.3f} ms")
    print(f"  Backward time: {backward_time * 1000:.3f} ms")
    print(f"  Training throughput: {training_time * 1000:.3f} ms")
    print(f"  Samples/sec: {batch_size / training_time:.1f}")


if __name__ == "__main__":
    benchmark_conv2d()
    benchmark_conv2d(filters=64, kernel_size=5)
    benchmark_conv2d(height=64, width=64, filters=128)