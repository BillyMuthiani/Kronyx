import io
import sys
from unittest.mock import MagicMock, patch

import pytest

from kronyx import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    ReLU,
    Sequential,
)


class TestVisualize:
    def _capture_visualize(self, model, **kwargs):
        output = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = output
        try:
            model.visualize(**kwargs)
        finally:
            sys.stdout = original_stdout
        return output.getvalue()

    def test_default_visualization_returns_none(self):
        model = Sequential()
        model.add(Dense(2, 4))
        result = model.visualize()
        assert result is None

    def test_compact_visualization_returns_none(self):
        model = Sequential()
        model.add(Dense(2, 4))
        result = model.visualize(compact=True)
        assert result is None

    def test_default_contains_title(self):
        model = Sequential()
        model.add(Dense(2, 4))
        output = self._capture_visualize(model)
        assert "KRONYX MODEL VISUALIZATION" in output

    def test_default_contains_layer_boxes(self):
        model = Sequential()
        model.add(Dense(2, 4))
        output = self._capture_visualize(model)
        assert "┌" in output
        assert "│" in output
        assert "└" in output

    def test_compact_contains_input_output(self):
        model = Sequential()
        model.add(Dense(2, 4))
        output = self._capture_visualize(model, compact=True)
        assert "Input" in output
        assert "Output" in output

    def test_repeated_layer_numbering_default(self):
        model = Sequential()
        model.add(Dense(2, 4))
        model.add(Dense(4, 4))
        model.add(Dense(4, 2))
        output = self._capture_visualize(model)
        assert "Dense" in output
        assert "Dense_1" in output
        assert "Dense_2" in output

    def test_repeated_layer_numbering_compact(self):
        model = Sequential()
        model.add(Dense(2, 4))
        model.add(Dense(4, 4))
        model.add(Dense(4, 2))
        output = self._capture_visualize(model, compact=True)
        assert "Dense" in output
        assert "Dense_1" in output
        assert "Dense_2" in output

    def test_dense_metadata_default(self):
        model = Sequential()
        model.add(Dense(2, 8))
        output = self._capture_visualize(model)
        assert "Units" in output
        assert "Output Shape" in output
        assert "Parameters" in output
        assert "8" in output

    def test_conv2d_metadata_default(self):
        model = Sequential()
        model.add(Conv2D(filters=8, kernel_size=3, padding="valid"))
        output = self._capture_visualize(model)
        assert "Filters" in output
        assert "Kernel Size" in output
        assert "Stride" in output
        assert "Padding" in output
        assert "Parameters" in output

    def test_dropout_metadata_default(self):
        model = Sequential()
        model.add(Dense(2, 8))
        model.add(Dropout(0.1))
        output = self._capture_visualize(model)
        assert "Rate" in output
        assert "Output Shape" in output

    def test_batchnorm_metadata_default(self):
        model = Sequential()
        model.add(Dense(2, 8))
        model.add(BatchNormalization())
        output = self._capture_visualize(model)
        assert "Output Shape" in output
        assert "Parameters" in output

    def test_activation_formatting_default(self):
        model = Sequential()
        model.add(ReLU())
        output = self._capture_visualize(model)
        assert "ReLU Activation" in output

    def test_parameter_display_default(self):
        model = Sequential()
        model.add(Dense(2, 8))
        output = self._capture_visualize(model)
        assert "Parameters" in output

    def test_shape_inference_default(self):
        model = Sequential()
        model.add(Dense(2, 8))
        model.add(ReLU())
        model.add(Dense(8, 2))
        output = self._capture_visualize(model)
        assert "(None, 8)" in output
        assert "(None, 2)" in output

    def test_network_statistics_default(self):
        model = Sequential()
        model.add(Dense(2, 8))
        model.add(ReLU())
        output = self._capture_visualize(model)
        assert "Network Statistics" in output
        assert "Total Parameters" in output
        assert "Trainable Params" in output
        assert "Non-trainable Params" in output
        assert "Estimated Memory" in output

    def test_estimated_memory_calculation(self):
        model = Sequential()
        model.add(Dense(2, 8))
        output = self._capture_visualize(model)
        assert "Bytes" in output or "KB" in output or "MB" in output

    def test_compact_dense_shows_units(self):
        model = Sequential()
        model.add(Dense(2, 8))
        output = self._capture_visualize(model, compact=True)
        assert "Dense(8)" in output

    def test_compact_conv2d_shows_filters(self):
        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=3))
        output = self._capture_visualize(model, compact=True)
        assert "Conv2D(filters=16)" in output

    def test_compact_dropout_shows_rate(self):
        model = Sequential()
        model.add(Dropout(0.25))
        output = self._capture_visualize(model, compact=True)
        assert "Dropout" in output

    def test_output_format_graphviz_defaults_to_svg(self):
        model = Sequential()
        model.add(Dense(2, 4))
        mock_dot = MagicMock()
        mock_graphviz = MagicMock()
        mock_graphviz.Digraph.return_value = mock_dot

        with patch.dict(sys.modules, {"graphviz": mock_graphviz}):
            model.visualize(output_format="graphviz")

        mock_dot.render.assert_called_once()
        args, kwargs = mock_dot.render.call_args
        assert args[0] == "model_architecture"
        assert kwargs["format"] == "svg"
        assert kwargs["cleanup"] is True

    def test_output_format_svg(self):
        model = Sequential()
        model.add(Dense(2, 4))
        mock_dot = MagicMock()
        mock_graphviz = MagicMock()
        mock_graphviz.Digraph.return_value = mock_dot

        with patch.dict(sys.modules, {"graphviz": mock_graphviz}):
            model.visualize(output_format="svg")

        mock_dot.render.assert_called_once()
        args, kwargs = mock_dot.render.call_args
        assert args[0] == "model_architecture"
        assert kwargs["format"] == "svg"
        assert kwargs["cleanup"] is True

    def test_output_format_png(self):
        model = Sequential()
        model.add(Dense(2, 4))
        mock_dot = MagicMock()
        mock_graphviz = MagicMock()
        mock_graphviz.Digraph.return_value = mock_dot

        with patch.dict(sys.modules, {"graphviz": mock_graphviz}):
            model.visualize(output_format="png")

        mock_dot.render.assert_called_once()
        args, kwargs = mock_dot.render.call_args
        assert args[0] == "model_architecture"
        assert kwargs["format"] == "png"
        assert kwargs["cleanup"] is True

    def test_output_format_pdf(self):
        model = Sequential()
        model.add(Dense(2, 4))
        mock_dot = MagicMock()
        mock_graphviz = MagicMock()
        mock_graphviz.Digraph.return_value = mock_dot

        with patch.dict(sys.modules, {"graphviz": mock_graphviz}):
            model.visualize(output_format="pdf")

        mock_dot.render.assert_called_once()
        args, kwargs = mock_dot.render.call_args
        assert args[0] == "model_architecture"
        assert kwargs["format"] == "pdf"
        assert kwargs["cleanup"] is True

    def test_invalid_format_raises_clean_value_error(self):
        model = Sequential()
        model.add(Dense(2, 4))
        with pytest.raises(ValueError, match="Unsupported visualization format"):
            model.visualize(output_format="banana")

    def test_graphviz_missing_handled_gracefully(self):
        model = Sequential()
        model.add(Dense(2, 4))

        with patch.dict(sys.modules, {"graphviz": None}):
            output = self._capture_visualize(model, output_format="svg")

        assert "Graphviz is not installed" in output
        assert "pip install graphviz" in output

    def test_graphviz_executable_missing_handled_gracefully(self):
        model = Sequential()
        model.add(Dense(2, 4))

        class ExecutableNotFound(Exception):  # noqa: N818
            pass

        mock_dot = MagicMock()
        mock_dot.render.side_effect = ExecutableNotFound("ExecutableNotFound")
        mock_graphviz = MagicMock()
        mock_graphviz.Digraph.return_value = mock_dot

        with patch.dict(sys.modules, {"graphviz": mock_graphviz}):
            output = self._capture_visualize(model, output_format="svg")

        assert "Graphviz executable was not found" in output
        assert "graphviz.org" in output
