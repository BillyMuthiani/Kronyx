"""Benchmark Dense layer forward and backward pass performance."""
import time
import numpy as np
from Neuralnet import Dense, ReLU


def benchmark_dense(
    input_size: int = 100,
    output_size: int = 50,
    batch_size: int = 32,
    iterations: int = 1000
) -> None:
    """Benchmark Dense layer performance.

    Args:
        input_size: Number of input features.
        output_size: Number of output features.
        batch_size: Batch size for forward/backward passes.
        iterations: Number of benchmark iterations.
    """
    np.random.seed(42)

    layer = Dense(input_size, output_size)
    activation = ReLU()

    X = np.random.randn(batch_size, input_size)
    dvalues = np.random.randn(batch_size, output_size)

    # Warmup
    for _ in range(100):
        out = layer.forward(X, training=True)
        out = activation.forward(out, training=True)
        d = activation.backward(dvalues)
        layer.backward(d)

    # Benchmark forward
    start = time.perf_counter()
    for _ in range(iterations):
        out = layer.forward(X, training=True)
        out = activation.forward(out, training=True)
    forward_time = (time.perf_counter() - start) / iterations

    # Benchmark backward
    start = time.perf_counter()
    for _ in range(iterations):
        d = activation.backward(dvalues)
        layer.backward(d)
    backward_time = (time.perf_counter() - start) / iterations

    # Benchmark training throughput
    start = time.perf_counter()
    for _ in range(iterations):
        out = layer.forward(X, training=True)
        out = activation.forward(out, training=True)
        d = activation.backward(dvalues)
        layer.backward(d)
    training_time = (time.perf_counter() - start) / iterations

    print(f"Dense({input_size}, {output_size}) Benchmark:")
    print(f"  Batch size: {batch_size}")
    print(f"  Forward time: {forward_time * 1000:.3f} ms")
    print(f"  Backward time: {backward_time * 1000:.3f} ms")
    print(f"  Training throughput: {training_time * 1000:.3f} ms (forward + backward)")
    print(f"  Samples/sec: {batch_size / training_time:.1f}")


if __name__ == "__main__":
    benchmark_dense()