
from kronyx import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    ReLU,
    Sequential,
    Sigmoid,
    Softmax,
    Tanh,
)


class TestModelSummary:
    def test_summary_returns_string_with_input_shape(self):
        model = Sequential()
        model.add(Dense(2, 16))
        model.add(ReLU())
        model.add(Dense(16, 1))
        model.add(Sigmoid())

        result = model.summary(input_shape=(2,), return_string=True)
        assert isinstance(result, str)
        assert "Model: Sequential" in result
        assert "Layer (type)" in result
        assert "Total params:" in result
        assert "Trainable params:" in result

    def test_summary_no_duplicate_separators(self):
        model = Sequential()
        model.add(Dense(2, 4))
        result = model.summary(return_string=True)
        lines = result.strip().splitlines()

        separator = "=" * 75
        sep_count = sum(1 for line in lines if line == separator)
        assert sep_count == 4

        header_idx = next(i for i, line in enumerate(lines) if line.startswith("Layer (type)"))
        next_idx = header_idx + 1
        assert lines[next_idx].startswith("-"), "Expected thin separator after header"

    def test_summary_alignment(self):
        model = Sequential()
        model.add(Dense(4, 8))
        model.add(ReLU())
        model.add(Dense(8, 2))

        result = model.summary(input_shape=(4,), return_string=True)

        data_lines = [
            line for line in result.strip().splitlines()
            if line.startswith(("Dense", "ReLU"))
        ]
        assert len(data_lines) == 3

        for line in data_lines:
            assert len(line) == len(data_lines[0]), f"Alignment mismatch: {line!r}"

    def test_summary_parameter_counts(self):
        model = Sequential()
        model.add(Dense(4, 8))
        model.add(ReLU())
        model.add(Dense(8, 2))

        result = model.summary(input_shape=(4,), return_string=True)
        result_lines = result.strip().splitlines()

        dense1_params = 4 * 8 + 8
        dense2_params = 8 * 2 + 2
        total_params = dense1_params + dense2_params

        data_rows = [line for line in result_lines if line.startswith("Dense")]
        assert any(f"{dense1_params:,}" in line for line in data_rows)
        assert any(f"{dense2_params:,}" in line for line in data_rows)

        totals_lines = [line for line in result_lines if "params:" in line]
        totals_text = " ".join(totals_lines)
        assert f"{total_params:,}" in totals_text

    def test_summary_dense_parameter_breakdown(self):
        model = Sequential()
        model.add(Dense(2, 16))
        model.add(ReLU())
        model.add(Dense(16, 1))
        model.add(Sigmoid())

        result = model.summary(input_shape=(2,), return_string=True)
        assert "48 (32W + 16B)" in result
        assert "17 (16W + 1B)" in result

    def test_summary_conv2d_parameter_breakdown(self):
        model = Sequential()
        model.add(Conv2D(filters=8, kernel_size=3, padding="valid", initializer="he_normal"))

        result = model.summary(input_shape=(28, 28, 1), return_string=True)
        assert "80 (72W + 8B)" in result

    def test_summary_batchnorm_parameter_breakdown(self):
        model = Sequential()
        model.add(Dense(4, 8))
        model.add(BatchNormalization())
        model.add(Dense(8, 2))

        result = model.summary(input_shape=(4,), return_string=True)
        assert "16 (8γ + 8β)" in result

    def test_summary_activation_layers_have_zero_params(self):
        model = Sequential()
        model.add(Dense(4, 8))
        model.add(ReLU())
        model.add(Sigmoid())
        model.add(Tanh())
        model.add(Softmax())
        model.add(Dense(8, 2))

        result = model.summary(input_shape=(4,), return_string=True)
        activation_lines = [
            line for line in result.strip().splitlines()
            if any(line.startswith(a) for a in ("ReLU", "Sigmoid", "Tanh", "Softmax"))
        ]
        for line in activation_lines:
            assert "0" in line

    def test_summary_repeated_layer_numbering(self):
        model = Sequential()
        model.add(Dense(4, 8))
        model.add(Dense(8, 8))
        model.add(Dense(8, 8))
        model.add(Dense(8, 2))

        result = model.summary(input_shape=(4,), return_string=True)
        result_lines = result.strip().splitlines()
        layer_names = [
            line.split()[0]
            for line in result_lines
            if not line.startswith(("=", "-", "Layer", "Model"))
            and "params:" not in line
        ]
        assert layer_names == ["Dense", "Dense_1", "Dense_2", "Dense_3"]

    def test_summary_dense_output_shape(self):
        model = Sequential()
        model.add(Dense(4, 16))
        model.add(ReLU())
        model.add(Dense(16, 1))
        model.add(Sigmoid())

        result = model.summary(input_shape=(4,), return_string=True)
        assert "(None, 16)" in result
        assert "(None, 1)" in result

    def test_summary_conv2d_output_shape(self):
        model = Sequential()
        model.add(Conv2D(filters=10, kernel_size=3, padding="valid", initializer="he_normal"))

        result = model.summary(input_shape=(28, 28, 1), return_string=True)
        assert "(None, 26, 26, 10)" in result

    def test_summary_flatten_output_shape(self):
        model = Sequential()
        model.add(Flatten())

        result = model.summary(input_shape=(28, 28, 1), return_string=True)
        assert "(None, 784)" in result

    def test_summary_dropout_inherits_shape(self):
        model = Sequential()
        model.add(Dense(4, 8))
        model.add(Dropout(0.1))
        model.add(Dense(8, 2))

        result = model.summary(input_shape=(4,), return_string=True)
        assert "(None, 8)" in result

    def test_summary_batchnorm_inherits_shape(self):
        model = Sequential()
        model.add(Dense(4, 8))
        model.add(BatchNormalization())
        model.add(Dense(8, 2))

        result = model.summary(input_shape=(4,), return_string=True)
        assert "(None, 8)" in result

    def test_summary_activation_layers_inherit_shape(self):
        model = Sequential()
        model.add(Dense(4, 8))
        model.add(ReLU())
        model.add(Sigmoid())
        model.add(Tanh())
        model.add(Softmax())
        model.add(Dense(8, 2))

        result = model.summary(input_shape=(4,), return_string=True)
        assert "(None, 8)" in result

    def test_summary_without_input_shape_infers_dense_shapes(self):
        model = Sequential()
        model.add(Dense(4, 8))
        model.add(ReLU())
        model.add(Dense(8, 2))

        result = model.summary(return_string=True)
        assert "(None, 8)" in result
        assert "(None, 2)" in result
        assert "?" not in result

    def test_summary_fallback_to_question_marks_when_inference_impossible(self):
        model = Sequential()
        model.add(Conv2D(filters=8, kernel_size=3, padding="valid"))

        result = model.summary(return_string=True)
        assert "?" in result

    def test_summary_prints_when_return_string_false(self):
        model = Sequential()
        model.add(Dense(2, 4))

        result = model.summary(input_shape=(2,), return_string=False)
        assert result is None
