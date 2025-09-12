# Migration Guide: Poetry to uv

This document outlines the migration from Poetry to uv and the improvements made to the project structure.

## Summary of Changes

### Package Management
- ✅ **Replaced Poetry with uv**: Faster dependency resolution and installation
- ✅ **Updated pyproject.toml**: Modern Python packaging standards with PEP 621
- ✅ **Added hatchling**: Modern build backend with setuptools-scm for version management
- ✅ **Automatic versioning**: Version derived from git tags using setuptools-scm

### Task Management
- ✅ **Removed poethepoet**: Replaced with simple `uv run` commands and optional justfile
- ✅ **Added justfile**: Optional task runner with common development commands
- ✅ **Direct uv commands**: All tasks can be run with `uv run <command>`

### CI/CD Improvements
- ✅ **Updated GitHub Actions**: All workflows now use uv for faster builds
- ✅ **Modern release workflow**: Trusted publishing to PyPI with security signatures
- ✅ **Added dependency updates**: Automated weekly dependency updates
- ✅ **Added security scanning**: CodeQL and pip-audit security checks

### Development Environment
- ✅ **Enhanced pre-commit hooks**: More comprehensive code quality checks
- ✅ **Better caching**: Improved CI caching for uv dependencies
- ✅ **Faster installs**: uv is significantly faster than Poetry

## Bug Fixes Included

### NumPy 2.0 Compatibility
- ✅ **Fixed deprecated np.product**: Replaced with np.prod for NumPy 2.0+ compatibility
- ✅ **Future-proof dependencies**: Updated version constraints for long-term compatibility

## Migration Instructions

### For Developers

#### Old Poetry workflow:
```bash
# Install dependencies
poetry install

# Run tests
poetry run poe test

# Format code
poetry run poe format

# Run linting  
poetry run poe lint
```

#### New uv workflow:
```bash
# Install dependencies
uv sync --dev

# Run tests
uv run pytest -v

# Format code
uv run black .

# Run linting
uv run flake8 .
```

#### With justfile (optional):
```bash
# Install just: https://just.systems/
# Then use simple commands:
just test
just format 
just lint
just help  # See all available tasks
```

### For CI/CD

All GitHub Actions workflows have been updated to use uv. No changes needed for existing workflows.

### Version Management

- **Before**: Manual version bumps with bump2version
- **After**: Automatic version from git tags using setuptools-scm

To create a release:
```bash
git tag v1.0.0
git push origin v1.0.0
```

The version is automatically computed from git history.

### Build System

- **Before**: Poetry build backend
- **After**: Hatchling with setuptools-scm

Building packages:
```bash
# Old
poetry build

# New  
uv build
```

## Benefits of Migration

### Performance
- **Faster installs**: uv is 10-100x faster than Poetry
- **Better caching**: More efficient dependency caching
- **Parallel installs**: uv installs packages in parallel

### Modern Standards
- **PEP 621**: Modern pyproject.toml format
- **Trusted publishing**: Secure PyPI releases without API tokens
- **Security scanning**: Automated vulnerability detection
- **Dependency updates**: Automated dependency maintenance

### Developer Experience
- **Simpler commands**: Direct `uv run` instead of `poetry run poe`
- **Better error messages**: uv provides clearer error reporting
- **Cross-platform**: Consistent behavior across platforms

## Troubleshooting

### Common Issues

1. **Import errors after migration**
   ```bash
   # Ensure you're in the right environment
   uv sync --dev
   uv run python -c "import zarrnii; print('Success')"
   ```

2. **Missing commands**
   ```bash
   # Make sure dev dependencies are installed
   uv sync --dev --all-extras
   ```

3. **Version issues**
   ```bash
   # Check version is computed correctly
   uv run python -c "import zarrnii; print(zarrnii.__version__)"
   ```

### Getting Help

- **uv documentation**: https://docs.astral.sh/uv/
- **justfile documentation**: https://just.systems/
- **Project issues**: https://github.com/khanlab/zarrnii/issues

## Compatibility

### Python Versions
- Minimum: Python 3.11+ (unchanged)
- Tested: Python 3.11, 3.12
- Build system works with all supported versions

### Dependencies
- All existing dependencies maintained
- Version constraints updated for compatibility
- New dev dependencies for improved tooling

### API
- **No breaking changes** to the zarrnii Python API
- All existing code continues to work unchanged
- Same import paths and function signatures