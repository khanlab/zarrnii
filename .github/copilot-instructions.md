# ZarrNii Development Instructions

**ZarrNii** is a Python library for working with OME-Zarr and NIfTI formats, enabling seamless data transformation, metadata preservation, and efficient processing of biomedical images. It uses uv for fast dependency management, pytest for testing, and mkdocs for documentation.

**ALWAYS** reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap and Setup
- **Install uv first**: `pip install uv` (takes ~10 seconds)
- **Install all dependencies**: `uv sync --dev` (takes ~30 seconds, NEVER CANCEL - set timeout to 120+ seconds)
- **Set up pre-commit hooks**: `uv run pre-commit install` (takes ~2 seconds)

### Building and Testing  
- **Run linting**: `uv run flake8 .` (takes ~10-30 seconds, max-line-length: 88 chars to match Black)
- **Format code**: `uv run black .` (takes ~1 second, line-length: 88)  
- **Sort imports**: `uv run isort .` (takes ~1 second, profile: black, line-length: 88)
- **Run tests**: `uv run pytest -v` (takes ~6-14 seconds, NEVER CANCEL - set timeout to 60+ seconds)
  - **Expected**: 37 tests pass, 1 skipped, 1 expected failure (existing issue with synthetic data generation)
  - **NEVER** try to fix unrelated test failures - they are pre-existing
- **Build documentation**: `uv run mkdocs build` (takes ~2 seconds)
  - **Expected**: Some warnings about missing documentation files and type annotations are normal
- **Run all quality checks**: `./scripts/quality-check.sh` (matches CI exactly)

### Development Environment
- **Python version**: 3.11+ (requires Python >=3.11)
- **Virtual environment**: uv automatically creates and manages .venv
- **Available tools**: pytest, flake8, black, isort, mkdocs, jupyterlab, pre-commit

### Running the Application
- **Import library**: `uv run python -c "from zarrnii import ZarrNii; print('Import successful')"`
- **Launch Jupyter**: `uv run jupyter lab` (for notebook development)
- **Serve docs locally**: `uv run mkdocs serve` (mkdocs dev server)

## Validation

### Pre-commit Validation Steps
ALWAYS run these before committing changes:
- `uv run black .` - Format code with black (line-length: 88)
- `uv run isort .` - Sort imports with isort (profile: black, line-length: 88)  
- `uv run flake8 .` - Check code style (max-line-length: 88, extend-ignore: E203,W503)
- `uv run pytest -v` - Run test suite (expect 37 pass, 1 skip, 1 xfail)
- **OR use quality check script**: `./scripts/quality-check.sh` (matches CI exactly)

### Manual Validation Scenarios
ALWAYS test these core workflows after making changes to the library:

#### Basic Import and Usage Test
```bash
uv run python -c "
from zarrnii import ZarrNii, AffineTransform
print('Available classes:', [ZarrNii, AffineTransform])
print('ZarrNii methods:', [m for m in dir(ZarrNii) if not m.startswith('_')][:5])
"
```

#### Core Functionality Test  
```bash
uv run python -c "
import numpy as np
from zarrnii import ZarrNii
# Test basic class instantiation without file I/O
znii = ZarrNii.__new__(ZarrNii)
print('ZarrNii class created successfully')
"
```

## Common Tasks

### Repository Structure
```
zarrnii/
├── .github/workflows/    # CI/CD pipelines (ci, gh-pages, release, dependencies)  
├── docs/                 # MkDocs documentation
├── notebooks/            # Jupyter example notebooks
├── tests/                # Pytest test suite
├── zarrnii/              # Main Python package
│   ├── core.py          # Main ZarrNii class
│   ├── transform.py     # Transform classes  
│   └── enums.py         # Enumerations
├── pyproject.toml       # Modern Python packaging with dependencies
├── justfile             # Optional task runner with development commands
├── mkdocs.yml          # Documentation config
└── uv.lock             # Dependency lock file
```

### uv and Justfile Commands
Available via `uv run <command>`:
- `uv sync --dev` - Install dependencies  
- `uv run pytest -v` - Run test suite
- `uv run black .` - Format code with black
- `uv run isort .` - Sort imports with isort
- `uv run flake8 .` - Run flake8 linting  
- `uv run mkdocs serve` - Serve documentation locally
- `uv run mkdocs build` - Build documentation static files
- `uv run mkdocs gh-deploy` - Deploy docs to GitHub Pages
- `uv run pre-commit install` - Install pre-commit hooks
- `uv build` - Build package

Available via justfile (if just is installed):
- `just help` - Show all available tasks
- `just setup` - Install pre-commit hooks
- `just test` - Run tests
- `just format` - Format code
- `just lint` - Run linting
- `just quality` - Run all quality checks (matches CI exactly)
- `just quality-individual` - Run individual quality checks (format, lint, etc.)
- `just serve-docs` - Serve documentation locally
- `just build-docs` - Build documentation
- `just install` - Install dependencies
- `just build` - Build package

### Key Files to Know
- **Main API**: `zarrnii/core.py` (ZarrNii class, ~1400 lines)
- **Transforms**: `zarrnii/transform.py` (AffineTransform, DisplacementTransform classes)  
- **Tests**: `tests/test_io.py`, `tests/test_downsample.py`, `tests/test_transform.py`
- **Config**: `pyproject.toml` (dependencies, tool configuration), `justfile` (optional task runner)
- **Docs**: `docs/index.md`, `docs/walkthrough/getting_started.md`

### Known Issues and Workarounds
- **Black Jupyter warning**: "Skipping .ipynb files as Jupyter dependencies are not installed" - normal, not an error
- **Documentation warnings**: Missing files in nav (transformations.md, etc.) and griffe type annotations - normal
- **Pre-commit network issues**: Pre-commit may fail with network timeouts when installing hooks - use `git commit --no-verify` for documentation-only changes

### Timing Expectations (NEVER CANCEL)
- **uv install**: ~30 seconds (set timeout 120+ seconds)
- **Test suite**: ~6-14 seconds (set timeout 60+ seconds)  
- **Linting**: ~10-30 seconds (set timeout 60+ seconds)
- **Formatting**: ~1 second (set timeout 30+ seconds)
- **Documentation build**: ~2 seconds (set timeout 30+ seconds)

### CI/CD Information
- **GitHub Actions**: `.github/workflows/ci.yml` (testing and quality checks), `.github/workflows/gh-pages.yml` (documentation), `.github/workflows/release.yml` (PyPI), `.github/workflows/dependencies.yml` (automated dependency updates)
- **Documentation deployment**: Automatic on push to main/master branch
- **Release process**: Triggered by version tags (v*) using trusted publishing
- **CI testing**: Automated testing on PRs and pushes with Python 3.11 and 3.12

## Library Overview
ZarrNii bridges OME-Zarr and NIfTI formats for biomedical imaging. Core functionality:
- **Format conversion**: OME-Zarr ⟷ NIfTI with metadata preservation  
- **Transformations**: cropping, downsampling, upsampling, affine transforms
- **Multiscale support**: OME-Zarr pyramid handling
- **Lazy loading**: Dask arrays for large datasets
- **Target domain**: Whole brain lightsheet microscopy, ultra-high field MRI, 3T+ channel datasets