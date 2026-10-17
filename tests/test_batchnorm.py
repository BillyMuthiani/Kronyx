import numpy as np
import pytest

from kronyx import BatchNormalization


class TestBatchNormalization:
    @pytest.fixture
    def bn(self):
        return BatchNormalization()

    @pytest.fixture
    def input_data(self):
        np.random.seed(42)
        return np.random.randn(4, 8)

    def test_forward_training(self, bn, input_data):
        output = bn.forward(input_data, training=True)
        assert output.shape == (4, 8)

        # Check that running statistics are updated
        assert bn.running_mean is not None
        assert bn.running_variance is not None

        # Check that output has zero mean and unit variance (approximately)
        np.testing.assert_allclose(np.mean(output, axis=0), 0, atol=1e-1)
        np.testing.assert_allclose(np.std(output, axis=0), 1, atol=1e-1)

    def test_forward_inference(self, bn, input_data):
        # First do a training pass to initialize running stats
        _ = bn.forward(input_data, training=True)

        # Now test inference
        output = bn.forward(input_data, training=False)
        assert output.shape == (4, 8)

    def test_forward_shapes(self):
        # Test different input shapes (create fresh layer for each)
        for shape in [(2, 4), (8, 16), (1, 10)]:
            x = np.random.randn(*shape)
            bn = BatchNormalization()
            out = bn.forward(x, training=True)
            assert out.shape == shape

    def test_backward_shape(self, bn, input_data):
        bn.forward(input_data, training=True)
        dout = np.random.randn(*input_data.shape)
        dinputs = bn.backward(dout)
        assert dinputs.shape == input_data.shape

    def test_gamma_beta_updates(self, bn, input_data):
        bn.forward(input_data, training=True)
        dout = np.ones_like(input_data)
        bn.backward(dout)

        assert hasattr(bn, 'dgamma')
        assert hasattr(bn, 'dbeta')
        assert bn.dgamma.shape == (1, 8)
        assert bn.dbeta.shape == (1, 8)

    def test_running_stats_accumulate(self, bn, input_data):
        # Multiple forward passes should update running stats
        for _ in range(5):
            bn.forward(input_data, training=True)

        assert bn.running_mean is not None
        assert bn.running_variance is not None
        assert not np.allclose(bn.running_mean, 0)
        assert not np.allclose(bn.running_variance, 1)

    def test_serialization(self, bn, input_data):
        # Do training pass to initialize
        _ = bn.forward(input_data, training=True)

        gamma_copy = bn.gamma.copy()
        beta_copy = bn.beta.copy()
        rm_copy = bn.running_mean.copy()
        rv_copy = bn.running_variance.copy()

        # Modify
        bn.gamma[:] = 0
        bn.beta[:] = 0

        # Restore
        bn.gamma = gamma_copy
        bn.beta = beta_copy
        bn.running_mean = rm_copy
        bn.running_variance = rv_copy

        output = bn.forward(input_data, training=False)
        expected = (input_data - rm_copy) / np.sqrt(rv_copy + bn.epsilon) * gamma_copy + beta_copy
        np.testing.assert_allclose(output, expected)

    def test_gradient_check(self):
        from tests.gradient_check import check_layer_gradient

        bn = BatchNormalization()
        input_data = np.random.randn(4, 8)

        # Training-mode input gradient (previously untested)
        passed, _, _ = check_layer_gradient(
             bn, input_data, "BatchNormalization (training)", training=True
         )
        assert passed, "BatchNormalization training input gradient check failed"

        # Inference-mode input gradient (uses cached running stats)
        bn2 = BatchNormalization()
        # Prime running stats with a training forward pass
        bn2.forward(input_data, training=True)
        passed, _, _ = check_layer_gradient(
             bn2, input_data, "BatchNormalization (inference)", training=False
        )
        assert passed, "BatchNormalization inference gradient check failed"

    def test_gradient_check_gamma_beta_training(self):
        """Numerical gradient check for gamma and beta in training mode."""
        from tests.gradient_check import check_layer_param_gradient

        np.random.seed(42)
        input_data = np.random.randn(16, 8)

        # Gamma gradient
        bn = BatchNormalization()
        passed, _, _ = check_layer_param_gradient(
            bn, input_data, "gamma", "dgamma", "BatchNorm", training=True
        )
        assert passed, "BatchNorm gamma gradient check (training) failed"

        # Beta gradient
        bn = BatchNormalization()
        passed, _, _ = check_layer_param_gradient(
            bn, input_data, "beta", "dbeta", "BatchNorm beta", training=True
        )
        assert passed, "BatchNorm beta gradient check (training) failed"


class TestBatchNormWeightedGradient:
    """Weighted upstream-gradient checks for BatchNormalization.

    These tests use a *non-trivial* upstream gradient tensor (deterministic,
    seed-fixed) so that the scalar objective is ``sum(forward(x) * w)``.
    This exercises every output direction simultaneously rather than relying
    on a flat all-ones upstream gradient, which can mask directional errors
    in the batch-statistic derivatives.
    """

    @staticmethod
    def _make_input(seed=123):
        rng = np.random.RandomState(seed)
        return rng.randn(16, 8)

    @staticmethod
    def _make_upstream(out_shape, seed=321):
        rng = np.random.RandomState(seed)
        return rng.randn(*out_shape)

    # -- Input gradients -------------------------------------------------------

    def test_weighted_input_gradient_training(self):
        """Input-gradient check in training mode with non-trivial upstream."""
        from tests.gradient_check import check_layer_gradient_weighted

        x = self._make_input(123)
        w = self._make_upstream(x.shape, 321)

        bn = BatchNormalization()
        passed, max_abs, max_rel = check_layer_gradient_weighted(
            bn, x, w, "BatchNorm weighted input (training)", training=True
        )
        assert passed, (
            f"Weighted training input-grad mismatch: "
            f"max_abs={max_abs:.6e}, max_rel={max_rel:.6e}"
        )

    def test_weighted_input_gradient_inference(self):
        """Input-gradient check in inference mode with non-trivial upstream."""
        from tests.gradient_check import check_layer_gradient_weighted

        x = self._make_input(123)
        w = self._make_upstream(x.shape, 321)

        bn = BatchNormalization()
        bn.forward(x, training=True)  # prime running stats
        passed, max_abs, max_rel = check_layer_gradient_weighted(
            bn, x, w, "BatchNorm weighted input (inference)", training=False
        )
        assert passed, (
            f"Weighted inference input-grad mismatch: "
            f"max_abs={max_abs:.6e}, max_rel={max_rel:.6e}"
        )

    # -- Parameter gradients --------------------------------------------------

    def test_weighted_gamma_gradient_training(self):
        """Gamma-gradient check in training mode with non-trivial upstream."""
        from tests.gradient_check import check_layer_param_gradient_weighted

        x = self._make_input(123)
        w = self._make_upstream(x.shape, 321)

        bn = BatchNormalization()
        bn.forward(x, training=True)  # initialise gamma
        passed, max_abs, max_rel = check_layer_param_gradient_weighted(
            bn, x, w, "gamma", "dgamma", "BatchNorm gamma (weighted)",
            training=True,
        )
        assert passed, (
            f"Weighted training gamma-grad mismatch: "
            f"max_abs={max_abs:.6e}, max_rel={max_rel:.6e}"
        )

    def test_weighted_beta_gradient_training(self):
        """Beta-gradient check in training mode with non-trivial upstream."""
        from tests.gradient_check import check_layer_param_gradient_weighted

        x = self._make_input(123)
        w = self._make_upstream(x.shape, 321)

        bn = BatchNormalization()
        bn.forward(x, training=True)
        passed, max_abs, max_rel = check_layer_param_gradient_weighted(
            bn, x, w, "beta", "dbeta", "BatchNorm beta (weighted)",
            training=True,
        )
        assert passed, (
            f"Weighted training beta-grad mismatch: "
            f"max_abs={max_abs:.6e}, max_rel={max_rel:.6e}"
        )

    # -- Cache isolation ------------------------------------------------------

    def test_training_and_inference_caches_do_not_interfere(self):
        """A training backward must not leak stale mean/var into an
        subsequent inference backward (and vice-versa)."""
        from tests.gradient_check import check_layer_gradient_weighted

        x = self._make_input(999)
        w = self._make_upstream(x.shape, 888)

        # Training path
        bn = BatchNormalization()
        passed_t, _, _ = check_layer_gradient_weighted(
            bn, x, w, "BN train cache", training=True
        )
        assert passed_t

        # After training backward, mean/var were set.
        # Now do inference forward+backward — must use inference path.
        bn.forward(x, training=False)
        out = bn.forward(x, training=False)
        w2 = self._make_upstream(out.shape, 777)
        passed_i, _, _ = check_layer_gradient_weighted(
            bn, x, w2, "BN infer cache", training=False
        )
        assert passed_i, (
            "Inference backward after training must not use stale caches"
        )

    # -- Running-statistics behaviour -----------------------------------------

    def test_running_stats_update_during_training(self):
        """Running mean/variance must move away from init (0/1) after
        training forward passes."""
        x = self._make_input(42)
        bn = BatchNormalization(momentum=0.9)
        bn.forward(x, training=True)
        assert not np.allclose(bn.running_mean, 0), "running_mean unchanged"
        assert not np.allclose(
            bn.running_variance, 1
        ), "running_variance unchanged"

    def test_inference_does_not_mutate_running_stats(self):
        """Inference forward+backward must leave running stats untouched."""
        x = self._make_input(42)
        bn = BatchNormalization(momentum=0.9)
        bn.forward(x, training=True)
        rm_before = bn.running_mean.copy()
        rv_before = bn.running_variance.copy()

        bn.forward(x, training=False)
        w = self._make_upstream(x.shape, 111)
        bn.backward(w)

        np.testing.assert_allclose(bn.running_mean, rm_before, rtol=0)
        np.testing.assert_allclose(bn.running_variance, rv_before, rtol=0)

    def test_multiple_train_infer_transitions(self):
        """Alternating training/inference modes must remain numerically
        correct across transitions."""
        from tests.gradient_check import check_layer_gradient_weighted

        x = self._make_input(7)
        bn = BatchNormalization()

        for _ in range(3):
            passed, _, _ = check_layer_gradient_weighted(
                bn, x,
                self._make_upstream(x.shape, 555),
                "BN transition", training=True,
            )
            assert passed, "Training-mode check failed during transitions"

            bn.forward(x, training=False)
            out = bn.forward(x, training=False)
            passed, _, _ = check_layer_gradient_weighted(
                bn, x,
                self._make_upstream(out.shape, 666),
                "BN transition infer", training=False,
            )
            assert passed, (
                "Inference-mode check failed during transitions"
            )
