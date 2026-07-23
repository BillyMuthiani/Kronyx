# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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