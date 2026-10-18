# Roadmap

## Current (v1.1.1)

- [x] Layers: Dense, Conv2D, BatchNormalization, Dropout, Flatten
- [x] Activations: ReLU, Sigmoid, Tanh, Softmax
- [x] Optimizers: SGD, Adam
- [x] Callbacks: EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
- [x] Losses: BinaryCrossEntropy, CategoricalCrossEntropy, SoftmaxCategoricalCrossEntropy
- [x] Metrics: Accuracy, BinaryAccuracy, CategoricalAccuracy, Precision, Recall, F1Score, ConfusionMatrix, TopKAccuracy
- [x] Regularizers: L2
- [x] Initializers: he_normal, xavier_uniform, lecun_normal
- [x] Serialization: .krx format with save/load
- [x] Model JSON export/import
- [x] Type annotations (mypy clean - inline annotations on 63% of functions)
- [x] Code style (Ruff clean)
- [x] 496 unit tests
- [x] Educational tools (plot, visualize, count_params, layer_summary)

## Upcoming (v0.7)

- [ ] LSTM/GRU layers for sequence modeling
- [ ] Embedding layer for categorical inputs
- [ ] Learning rate schedulers (StepLR, ExponentialLR)
- [ ] Model checkpointing improvements

## Future (v0.8-v1.0)

- [ ] Model evaluation (train/val/test split utilities)
- [ ] Additional metrics (Precision, Recall, F1)
- [ ] Layer normalization
- [ ] Residual/skip connections
- [ ] Multi-GPU support
- [ ] ONNX export support

## Long-term Vision

- [ ] Distributed training support
- [ ] TensorFlow/PyTorch model import
- [ ] Production deployment utilities
- [ ] Neural architecture search
- [ ] Auto-differentiation improvements
- [ ] JIT compilation for speed