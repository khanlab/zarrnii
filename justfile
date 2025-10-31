# Development task runner for zarrnii
# Requires: just (https://just.systems/)

# Install pre-commit hooks
setup:
    uv run pre-commit install

# Run linting with ruff
lint:
    uv run ruff check .

# Format code with ruff
format:
    uv run ruff format .

# Sort imports with ruff (included in format)
sort-imports:
    uv run ruff check --select I --fix .

# Check code formatting with ruff (no changes)
check-format:
    uv run ruff format --check .

# Check linting with ruff (no changes)
check-lint:
    uv run ruff check .

# Run tests with pytest
test:
    uv run pytest -v

# Quality fix (format and auto-fix lint issues)
quality_fix: format
    uv run ruff check --fix .

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

# Run all quality checks (same as CI)
quality:
    ./scripts/quality-check.sh

# Run individual quality checks
quality-individual: check-format check-lint

# Build package
build:
    uv build

# Clean build artifacts
clean:
    rm -rf dist/ build/ *.egg-info/

# Show available tasks
help:
    @just --list
