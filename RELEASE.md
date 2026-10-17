# Release Guide

How to create and publish a new release of Kronyx.

## Prerequisites

- PyPI account with trusted publishing configured
- Write access to the GitHub repository
- All changes committed and tests passing

## Release Steps

### 1. Update Version

Edit `kronyx/version.py`:

```python
__version__ = "1.2.0"  # New version
```

### 2. Update Changelog

Edit `CHANGELOG.md` with the new version notes under the appropriate heading.

### 3. Run Tests

```bash
pytest
ruff check .
mypy kronyx
```

### 4. Build Package

```bash
python -m build
```

This creates:
- `dist/kronyx-X.Y.Z-py3-none-any.whl`
- `dist/kronyx-X.Y.Z.tar.gz`

### 5. Verify Package

```bash
twine check dist/*
```

### 6. Commit and Tag

```bash
git add .
git commit -m "Release v1.2.0"
git tag -a v1.2.0 -m "Release v1.2.0"
git push origin main --tags
```

### 7. Create GitHub Release

Go to GitHub → Releases → Draft new release

- Select the tag you created
- Title: "v0.7.0"
- Description: Copy from CHANGELOG.md
- Attach the wheel and sdist from `dist/`

### 8. Publish to PyPI

The `publish.yml` workflow will automatically publish when a release is created.

To publish manually:

```bash
twine upload dist/*
```

### 9. Verify Installation

```bash
pip install kronyx
python -c "import kronyx; print(kronyx.__version__)"
```

## Trusted Publishing Setup

1. Go to PyPI → Account Settings → Trusted Publishers
2. Add a new trusted publisher
3. Select the GitHub repository
4. Configure the workflow to use `pypa/gh-action-pypi-publish@release/v1`

## Workflow Configuration

The `.github/workflows/publish.yml` workflow handles automatic publishing
when a release is created on GitHub.