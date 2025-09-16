# Contributing to ZarrNii

Thank you for your interest in contributing to ZarrNii! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- uv package manager (recommended) or pip

### Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/khanlab/zarrnii.git
   cd zarrnii
   ```

2. **Install dependencies**:
   ```bash
   # Using uv (recommended)
   uv sync --dev
   
   # Or using pip
   pip install -e ".[dev]"
   ```

3. **Set up pre-commit hooks**:
   ```bash
   uv run pre-commit install
   ```

## Development Workflow

### Code Style and Quality

ZarrNii follows Python best practices and uses several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting  
- **flake8**: Linting and style checking
- **pre-commit**: Automated checks before commits

Run quality checks:
```bash
# Format code
uv run black zarrnii tests

# Sort imports
uv run isort zarrnii tests

# Lint code
uv run flake8 zarrnii tests
```

### Testing

ZarrNii uses pytest for testing. Tests are located in the `tests/` directory.

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=zarrnii

# Run specific test file
uv run pytest tests/test_io.py
```

### Documentation

Documentation is built with MkDocs and hosted on GitHub Pages.

```bash
# Serve documentation locally
uv run mkdocs serve

# Build documentation
uv run mkdocs build

# Deploy to GitHub Pages (maintainers only)
uv run mkdocs gh-deploy
```

## Contribution Types

### Bug Reports

When reporting bugs, please include:

- Python version and platform
- ZarrNii version
- Minimal code example that reproduces the issue
- Expected vs. actual behavior
- Full error traceback if applicable

### Feature Requests

For feature requests, please provide:

- Clear description of the proposed feature
- Use case and motivation
- Potential implementation approach
- Any relevant literature or references

### Code Contributions

#### Pull Request Process

1. **Fork the repository** and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Follow existing code style and conventions
   - Add tests for new functionality
   - Update documentation as needed
   - Ensure all tests pass

3. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: brief description of your changes"
   ```

4. **Push and create a Pull Request**:
   ```bash
   git push origin feature/your-feature-name
   ```

#### Code Review Guidelines

- All code must pass CI checks
- New features require tests and documentation
- Breaking changes need detailed justification
- Performance implications should be considered

### Documentation Contributions

Documentation improvements are always welcome! This includes:

- Fixing typos and grammatical errors
- Adding examples and tutorials
- Improving API documentation
- Translating content

## Development Guidelines

### Code Organization

- **Core functionality**: `zarrnii/core.py`
- **Transformations**: `zarrnii/transform.py`  
- **Utilities**: `zarrnii/utils.py`
- **Enumerations**: `zarrnii/enums.py`

### Naming Conventions

- Use descriptive variable and function names
- Follow PEP 8 naming conventions
- Use type hints where appropriate
- Document complex algorithms and edge cases

### Error Handling

- Raise informative exceptions with clear messages
- Use appropriate exception types
- Handle common error scenarios gracefully
- Log warnings for non-fatal issues

### Performance Considerations

- Use Dask for lazy evaluation when possible
- Minimize memory allocation in hot paths
- Profile performance-critical code
- Consider memory usage for large datasets

## API Design Principles

### Consistency

- Methods should have predictable interfaces
- Similar operations should use similar naming patterns
- Return types should be consistent across methods

### Flexibility  

- Support multiple input formats where reasonable
- Provide sensible defaults for optional parameters
- Allow customization through optional arguments

### Usability

- Prioritize common use cases in the main API
- Provide clear error messages
- Include comprehensive docstrings

## Testing Guidelines

### Test Categories

- **Unit tests**: Test individual functions and methods
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows
- **Performance tests**: Benchmark critical operations

### Test Data

- Use synthetic data when possible
- Keep test files small
- Document data requirements clearly
- Provide utilities for generating test data

### Test Structure

```python
def test_feature_description():
    """Test that feature works correctly under normal conditions."""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result.shape == expected_shape
    assert np.allclose(result.data, expected_data)
```

## Release Process

### Version Numbering

ZarrNii uses [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality, backwards compatible
- **PATCH**: Bug fixes, backwards compatible

### Release Checklist

1. Update version number
2. Update changelog
3. Run full test suite
4. Build and test documentation
5. Create release tag
6. Deploy to PyPI
7. Update GitHub release notes

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- Be respectful and considerate
- Use inclusive language
- Focus on constructive feedback
- Help create a positive community

### Communication

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Pull Requests**: Code contributions and reviews

## Getting Help

If you need help with development:

1. Check existing documentation and examples
2. Search through GitHub issues
3. Create a new issue with your question
4. Join community discussions

## Recognition

Contributors will be recognized in:
- Project README
- Release notes
- Documentation acknowledgments

Thank you for helping make ZarrNii better!