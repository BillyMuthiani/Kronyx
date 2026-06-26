# Neuralnet Refactoring Plan

## Architecture Overview

The Neuralnet framework is a minimal NumPy-based deep learning library with:
- **Sequential model** for stacking layers
- **Dense layers** with He initialization
- **Activations**: ReLU, Sigmoid, Softmax (Tanh referenced but missing)
- **Losses**: BinaryCrossEntropy, SoftmaxCategoricalCrossEntropy
- **Optimizers**: SGD, Adam with state stored on layer objects
- **Metrics**: Accuracy
- **Serialization**: save/load using npz format

## Technical Debt Identified

### Critical Issues (Must Fix)

1. **Missing Tanh activation** - Referenced in `main.py` but not implemented, causing import errors
2. **API Inconsistency in fit()** - `Sequential.fit()` has conflicting signatures:
   - `iris_classification.py`: Uses `compile()` then `fit()` with epochs only
   - `main.py`: Calls `fit()` with `loss`, `optimizer`, `metric` as parameters
   - `mnist_classifier.py`: Uses `compile()` then `fit()` correctly
3. **Missing CategoricalCrossEntropy** - Imported in `iris_classification.py` but doesn't exist (only `SoftmaxCategoricalCrossEntropy` exists)
4. **Serialization API mismatch** - `save()`/`load()` are standalone functions but `mnist_classifier.py` calls them as model methods

### High Priority (Should Fix)

5. **No docstrings** - All classes and methods lack documentation
6. **No type hints** - Code lacks type annotations for better IDE support
7. **Empty utils.py** - File exists but is empty (potential consolidation point)
8. **Empty __init__.py** - Package doesn't export public API
9. **Empty requirements.txt** - Dependencies not declared
10. **Softmax.backward() inefficiency** - Uses O(n²) loop-based Jacobian computation

### Medium Priority (Nice to Have)

11. **No gradient zeroing** - Gradients accumulate if backward() called multiple times before optimizer update
12. **Optimizer state on layers** - Adam pollution of layer namespace with `m_w`, `v_b`, etc.
13. **No input validation** - Parameters not validated (shapes, types, values)
14. **No tests** - No test suite exists
15. **Hardcoded print statements** - In `model.fit()` for logging

## Refactoring Plan

### Phase 1: Critical Bug Fixes (Backwards Compatible)

| # | Task | File | Changes |
|---|------|------|---------|
| 1 | Add Tanh activation | `Neuralnet/activations.py` | Implement Tanh class with forward/backward methods |
| 2 | Add CategoricalCrossEntropy alias | `Neuralnet/losses.py` | Add `CategoricalCrossEntropy = SoftmaxCategoricalCrossEntropy` for compatibility |
| 3 | Fix serialization methods | `Neuralnet/model.py` | Add `save()` and `load()` methods to Sequential class |
| 4 | Consolidate fit() signature | `Neuralnet/model.py` | Support both `compile()+fit()` and direct `fit()` with parameters |

### Phase 2: Code Quality Improvements (Backwards Compatible)

| # | Task | File | Changes |
|---|------|------|---------|
| 5 | Add comprehensive docstrings | All modules | Document all public classes and methods |
| 6 | Add type hints | All modules | Use typing module for function signatures |
| 7 | Export public API | `Neuralnet/__init__.py` | Export main classes: Dense, ReLU, Sigmoid, Softmax, Sequential, SGD, Adam, etc. |
| 8 | Define dependencies | `requirements.txt` | Add numpy, scikit-learn, matplotlib |
| 9 | Fix Softmax backward | `Neuralnet/activations.py` | Vectorize to O(n) using index-based gradient computation |
| 10 | Add gradient zeroing | `Neuralnet/model.py` | Reset gradients before each backward pass |

### Phase 3: Enhancements (Backwards Compatible)

| # | Task | File | Changes |
|---|------|------|---------|
| 11 | Add input validation | All modules | Validate shapes, types, and parameter constraints |
| 12 | Add logging callback | `Neuralnet/model.py` | Replace hardcoded prints with optional callback |
| 13 | Create tests | `tests/` | Add unit tests for layers, activations, losses, optimizers |
| 14 | Add random seed support | `Neuralnet/layers.py` | Optional seed parameter for reproducibility |
| 15 | Add layer naming | `Neuralnet/layers.py` | Optional name parameter for debugging |

## File Structure After Refactoring

```
Neuralnet/
├── __init__.py          # Public API exports
├── layers.py            # Dense layer (with docstrings, type hints)
├── activations.py       # ReLU, Sigmoid, Softmax, Tanh
├── losses.py            # BinaryCrossEntropy, CategoricalCrossEntropy
├── optimizers.py        # SGD, Adam (with docstrings, type hints)
├── model.py             # Sequential model (fixed API, save/load methods, type hints)
├── metrics.py           # Accuracy (with docstrings, type hints)
├── serialization.py     # save/load functions (for external use)
└── utils.py             # Helper functions (consolidate common ops)
```

## Backwards Compatibility Notes

- All existing imports will continue to work via `__init__.py` exports
- `Sequential.fit()` will accept both old and new calling conventions
- `CategoricalCrossEntropy` will be an alias to `SoftmaxCategoricalCrossEntropy`
- `Model.save()`/`Model.load()` will wrap existing `serialization` functions
- All existing example code will run unchanged after Phase 1 fixes

## Estimated Impact

- **Phase 1**: ~150 lines changed, fixes runtime errors
- **Phase 2**: ~300 lines changed, improves code quality
- **Phase 3**: ~500 lines added (tests, validation), no breaking changes 