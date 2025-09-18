# Development task runner for zarrnii
# Requires: just (https://just.systems/)

# Install pre-commit hooks
setup:
    uv run pre-commit install

# Run linting with flake8
lint:
    uv run flake8 .

# Format code with black
format:
    uv run black .

# Sort imports with isort
sort-imports:
    uv run isort .

# Check code formatting with black (no changes)
check-format:
    uv run black --check .

# Check import sorting with isort (no changes) 
check-imports:
    uv run isort --check .

# Run tests with pytest
test:
    uv run pytest -v

# Run tests with coverage reporting
test-cov:
    uv run pytest --cov=zarrnii --cov-report=term-missing

# Run tests with HTML coverage report
test-cov-html:
    uv run pytest --cov=zarrnii --cov-report=html
    @echo "Coverage report generated in htmlcov/index.html"

# Alias for test
pytest: test

# Serve documentation locally
serve-docs:
    uv run mkdocs serve

# Build documentation
build-docs:
    uv run mkdocs build

# Deploy documentation to GitHub Pages
deploy-docs:
    uv run mkdocs gh-deploy

# Install dependencies
install:
    uv sync --dev

# Run basic import test
test-import:
    uv run python -c "from zarrnii import ZarrNii; print('Import successful')"

# Run all quality checks
quality: check-format check-imports lint

# Build package
build:
    uv build

# Clean build artifacts
clean:
    rm -rf dist/ build/ *.egg-info/

# Show available tasks
help:
    @just --list