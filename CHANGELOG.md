# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.1] - 2026-10-17

### Added

- Native visualization engine with layered architecture:
  - `Graph` and `GraphBuilder` for model abstraction
  - `LayoutEngine` and `LayoutRegistry` for geometry computation
  - `Scene` and `SceneStyler` for presentation
  - `RendererRegistry` and native `SvgRenderer` for output
  - `ThemeRegistry` with built-in light, dark, blueprint, terminal, and neon themes
  - `IconRegistry` with semantic layer icons
- Native SVG rendering pipeline for `model.visualize(output_format="svg")`
- Runtime theme switching without Graphviz dependency
- Layer icon system across all renderers
- Comprehensive visualization test suite (396 tests)

### Fixed

- `evaluate()` before `compile()` now raises `NotCompiledError` instead of `AttributeError`
- `fit()` with negative `epochs` now raises `ValueError` instead of silently no-oping
- `fit()` with `batch_size <= 0` now raises `ValueError` with a clear message
- `fit()` with mismatched `x`/`y` lengths now raises `ValueError` early instead of `IndexError`
- `Conv2D` now validates `initializer` eagerly at construction (matching `Dense` behavior)
- `load_model()` on a directory now raises `SerializationError` instead of `PermissionError`
- Exported `save_model` and `KronyxError` from the top-level `kronyx` package for API consistency

### Phase 4

- `predict()` now validates input shape against the first layer's expected input,
  raising `ShapeError` with a descriptive message when feature counts or channels
  mismatch (Dense, Conv2D, BatchNormalization layers)
- `predict()` and `evaluate()` now validate `batch_size <= 0` and raise `ValueError`
  (consistent with the existing `fit()` guard)
- `Conv2D.backward()` now raises a clear `RuntimeError` when called before
  `forward()` instead of a cryptic `AttributeError`
- `Conv2D.backward()` optimised with `numpy.lib.stride_tricks.sliding_window_view`
  for kernel-gradient computation and batch-vectorised `tensordot` for
  input-gradient computation, yielding 3×–7× speedup on backward pass
- Added `check_layer_gradient_weighted` and `check_layer_param_gradient_weighted`
  to the gradient-check utilities, using a non-trivial upstream-gradient tensor
  for more rigorous BatchNorm training-mode gradient verification

### Phase 5

- Added `MANIFEST.in` to ensure `kronyx/assets/` (logo.png, logo.svg) are
  included in wheel and sdist distributions
- Fixed ruff F841 lint errors in `tests/test_optimizers.py` (unused variables
  in `test_sgd_zero_updates`)
- Fixed ruff lint errors in `tests/test_conv2d.py` (unused imports, line length)
- Fixed ruff lint errors in `tests/test_learning_validation.py` (import sorting,
  unused variables, whitespace)
- Updated `ROADMAP.md` test count from 468 to 496
- Updated `RELEASE.md` version examples from 0.7.0 to 1.2.0
- Fixed `docs/getting_started.md` `compile()` example to use real loss/optimizer
  instead of `None` placeholders
- Added `testpypi-env/` to `.gitignore`

### Architecture

- Visualization pipeline: `Sequential` → `GraphBuilder` → `Graph` → `LayoutEngine` → `Scene` → `SceneStyler` → `StyledScene` → `Renderer` → SVG
- Renderer-independent icon and theme abstractions
- Registry-based plugin system for layouts, renderers, themes, and icons

## [1.1.0] - 2026-07-05

### Added

- Educational tools for model understanding:
  - `History.plot()` - Visualize training metrics with matplotlib
  - `History.to_dataframe()` - Convert history to pandas DataFrame
  - `model.count_params()` - Count total trainable parameters
  - `model.layer_summary()` - Get structured layer information
  - `model.visualize()` - ASCII/graphviz architecture diagrams
- Enhanced `model.summary()` with professional Keras-style formatting

## [0.6.0] - 2026-07-01

### Added

- Model serialization system with `.krx` format (zip archive with JSON + NPZ)
- `Sequential.save(filename)` - Save complete model to .krx
- `Sequential.save_weights(filename)` - Save weights to .npz
- `Sequential.load_weights(filename)` - Load weights from .npz
- `Sequential.to_json()` - Export architecture to JSON string
- `Sequential.from_json(json_string)` - Create model from JSON
- `load_model(filename)` - Load model from .krx archive
- `from_json(json_string)` - Module-level JSON deserialization
- `SerializationError` exception class for all serialization failures
- Cross-platform compatibility (Windows/Linux/macOS)

### Changed

- Renamed framework from "NeuralNet" to "Kronyx"
- Improved error messages with actionable suggestions
- Standardized naming conventions

## [0.5.0] - 2026-06-15

### Added

- Dense layer with proper gradient computation
- Conv2D layer with configurable kernel size and padding
- Flatten layer for multi-dimensional inputs
- Dropout layer for regularization
- BatchNormalization layer with running statistics
- ReLU, Sigmoid, Tanh, Softmax activation functions
- SGD optimizer with learning rate scheduling
- Adam optimizer with momentum buffers
- BinaryCrossEntropy and CategoricalCrossEntropy losses
- Accuracy metric
- L2 weight regularization
- HE normal, Xavier uniform, LeCun normal initializers
- Training callbacks: EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
- Gradient checking utilities
- History tracking for training metrics

### Changed

- Improved training loop performance
- Added batch processing support
- Better numerical stability in loss functions

## [0.4.0] - 2026-05-01

### Added

- Sequential model API
- Basic Dense layer implementation
- Forward and backward pass infrastructure
- NumPy-based tensor operations

## [0.3.0] - 2026-04-01

### Added

- Initial project structure
- Basic layer abstractions
- Activation function prototypes

## [0.2.0] - 2026-03-01

### Added

- Project initialization
- Core NumPy utilities

## [0.1.0] - 2026-02-01

### Added

- Initial commit
- Research and planning phase