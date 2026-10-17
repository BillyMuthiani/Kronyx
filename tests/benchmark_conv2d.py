"""Benchmark for Conv2D forward and backward passes.

This script is **not** part of the pytest suite.  It measures raw wall-clock
performance of the Conv2D layer in isolation and reports median timings
across several representative configurations.

Run directly:

    python -m tests.benchmark_conv2d

Keep performance benchmarks separate from ordinary unit tests so that
timing thresholds do not make CI flaky.
"""

import time

import numpy as np

from kronyx import Conv2D


def _time_call(callable_, warmup=3, repeats=20):
    """Time *callable_* and return the median duration in seconds."""
    for _ in range(warmup):
        callable_()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        callable_()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def _benchmark_forward_backward(
    batch, in_h, in_w, in_channels, filters, kernel_size, stride, padding
):
    """Benchmark a single Conv2D configuration.

    Returns a dict with forward_ms, backward_ms, and shape info.
    """
    np.random.seed(42)
    conv = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    x = np.random.randn(batch, in_h, in_w, in_channels)

    # Warm up + materialise kernels/biases
    out = conv.forward(x, training=True)
    dout = np.random.randn(*out.shape)

    # Measure forward (training=True so caches are populated)
    fwd_ms = _time_call(lambda: conv.forward(x, training=True)) * 1000

    # Measure backward
    bwd_ms = _time_call(lambda: conv.backward(dout)) * 1000

    return {
        "input_shape": (batch, in_h, in_w, in_channels),
        "filters": filters,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "output_shape": tuple(out.shape),
        "forward_ms": fwd_ms,
        "backward_ms": bwd_ms,
    }


CONFIGS = [
    # (batch, h, w, channels, filters, kernel, stride, padding, label)
    (2, 8, 8, 3, 4, 3, 1, "same", "small_s8_c3_f4_k3_s1_same"),
    (2, 8, 8, 3, 4, 3, 2, "same", "small_s8_c3_f4_k3_s2_same"),
    (2, 8, 8, 6, 8, 3, 1, "same", "small_s8_c6_f8_k3_s1_same"),
    (4, 16, 16, 3, 8, 3, 1, "same", "med_s16_c3_f8_k3_s1_same"),
    (4, 16, 16, 3, 8, 5, 1, "same", "med_s16_c3_f8_k5_s1_same"),
    (4, 16, 16, 3, 8, 3, 2, "same", "med_s16_c3_f8_k3_s2_same"),
    (4, 32, 32, 3, 16, 3, 1, "same", "large_s32_c3_f16_k3_s1_same"),
    (4, 32, 32, 3, 16, 3, 2, "same", "large_s32_c3_f16_k3_s2_same"),
    (4, 32, 32, 1, 8, 3, 1, "valid", "large_s32_c1_f8_k3_s1_valid"),
]


def run_benchmark(label="Conv2D benchmark"):
    print("=" * 72)
    print(f"{label}")
    print(f"numpy {np.__version__}")
    print("=" * 72)
    print(
        f"{'Config':<36} {'Fwd (ms)':>10} {'Bwd (ms)':>10} "
        f"{'Input':>20} {'Output':>20}"
    )
    print("-" * 72)

    results = []
    for (
        bs, ih, iw, ch, filt, ks, st, pad, name
    ) in CONFIGS:
        r = _benchmark_forward_backward(
            bs, ih, iw, ch, filt, ks, st, pad
        )
        results.append(r)
        print(
            f"{name:<36} {r['forward_ms']:>10.3f} {r['backward_ms']:>10.3f} "
            f"{str(r['input_shape']):>20} {str(r['output_shape']):>20}"
        )

    return results


if __name__ == "__main__":
    run_benchmark()
