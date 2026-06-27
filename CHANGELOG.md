# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-06-27

### Added
- Dropout layer with inverted dropout during training
- BatchNormalization layer with learnable gamma/beta parameters
- Flatten layer for reshaping multi-dimensional inputs
- Conv2D layer with valid/same padding support
- L2 regularizer with corrected gradient
- EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau callbacks
- History class for tracking training metrics
- Support for batch_size and shuffle in fit()
- Tanh activation function
- CategoricalCrossEntropy alias
- Custom exceptions module

### Changed
- Improved serialization API for weight loading
- Added docstrings to public methods

## [0.5.0] - Initial Release

### Added
- Dense layer with he_normal, xavier_uniform, lecun_normal initializers
- ReLU, Sigmoid, Tanh, Softmax activation functions
- BinaryCrossEntropy and SoftmaxCategoricalCrossEntropy loss functions
- SGD and Adam optimizers
- Sequential model API
- Accuracy metric
- Basic serialization support