# Contributing to Kronyx

Thank you for your interest in contributing to Kronyx! This document provides guidelines and instructions for contributing.

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
git clone https://github.com/Kronyx/kronyx.git
cd kronyx
pip install -e ".[dev]"
```

This installs the package in development mode along with all development dependencies.

## Running Tests

```bash
pytest tests/ -v
ruff check .
mypy kronyx
```

## Scope Declaration

Every pull request must include a `SCOPE:` line in its description listing
the files or glob patterns the change is expected to touch. Example:

```
SCOPE: kronyx/losses.py, kronyx/utils.py, tests/
```

This is enforced by the **Scope-check** CI job, which compares the actual
changed files in the PR against the declared scope. If a file appears in
the diff that is not covered by the `SCOPE:` line, the check posts a
warning on the PR so it can be reviewed before merge.

Why this exists: the CI gate catches scope creep mechanically — files
modified accidentally, or changes that drift beyond the original intent,
are surfaced immediately rather than relying solely on manual diff review
during the review round.

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting:

```bash
ruff check .
ruff format .
```

## Type Checking

This project uses [mypy](https://mypy-lang.org/) for type checking:

```bash
mypy kronyx
```

## Documentation

- Add docstrings to all public methods (Google style)
- Update relevant documentation files in `docs/`
- Ensure examples in documentation are testable

## Code of Conduct

By participating, you are expected to uphold our [Code of Conduct](CODE_OF_CONDUCT.md).