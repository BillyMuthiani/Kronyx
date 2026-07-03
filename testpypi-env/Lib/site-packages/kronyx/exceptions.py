class KronyxError(Exception):
    """Base exception for all Kronyx errors."""
    pass


NeuralnetError = KronyxError


class ConfigurationError(KronyxError):
    """Raised when model configuration is invalid."""
    pass


class NotCompiledError(ConfigurationError):
    """Raised when fit() is called before compile()."""
    pass


class ShapeError(KronyxError):
    """Raised when tensor shapes are incompatible."""
    pass


class OptimizerError(KronyxError):
    """Raised when optimizer operations fail."""
    pass
