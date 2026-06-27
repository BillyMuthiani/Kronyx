# Contributing to Neuralnet

Thank you for your interest in contributing to Neuralnet! This document provides guidelines and instructions for contributing.

## How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/ -v`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Development Setup

```bash
pip install -e ".[dev]"
```

This installs the package in development mode along with all development dependencies.

## Running Tests

```bash
pytest tests/ -v
pytest --cov=Neuralnet
```

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting:

```bash
ruff check .
```

## Type Checking

This project uses [mypy](https://mypy-lang.org/) for type checking:

```bash
mypy Neuralnet
```

## Code of Conduct

By participating, you are expected to uphold our [Code of Conduct](CODE_OF_CONDUCT.md).