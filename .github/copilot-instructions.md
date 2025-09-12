# ZarrNii Development Instructions

**ZarrNii** is a Python library for working with OME-Zarr and NIfTI formats, enabling seamless data transformation, metadata preservation, and efficient processing of biomedical images. It uses Poetry for dependency management, pytest for testing, and mkdocs for documentation.

**ALWAYS** reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap and Setup
- **Install Poetry first**: `pip install poetry` (takes ~30 seconds)
- **Install all dependencies**: `poetry install` (takes ~45 seconds, NEVER CANCEL - set timeout to 120+ seconds)
- **Set up pre-commit hooks**: `poetry run poe setup` (takes ~2 seconds)

### Building and Testing  
- **Run linting**: `poetry run poe lint` (takes ~2 seconds, expect existing style issues)
- **Format code**: `poetry run poe format` (takes ~2 seconds)  
- **Sort imports**: `poetry run poe sort-imports` (takes ~1 second)
- **Run tests**: `poetry run poe test` (takes ~17 seconds, NEVER CANCEL - set timeout to 60+ seconds)
  - **Expected**: 10 tests pass, 1 test may fail (existing issue with synthetic data generation)
  - **NEVER** try to fix unrelated test failures - they are pre-existing
- **Build documentation**: `poetry run poe build-docs` (takes ~2.5 seconds)
  - **Expected**: Some warnings about missing documentation files are normal

### Development Environment
- **Python version**: 3.12+ (requires Python >=3.11,<4.0)
- **Virtual environment**: Poetry automatically creates and manages virtualenv
- **Available tools**: pytest, flake8, black, isort, mkdocs, jupyterlab, pre-commit

### Running the Application
- **Import library**: `poetry run python -c "from zarrnii import ZarrNii; print('Success')"`
- **Launch Jupyter**: `poetry run jupyter lab` (for notebook development)
- **Serve docs locally**: `poetry run poe serve-docs` (mkdocs dev server)

## Validation

### Pre-commit Validation Steps
ALWAYS run these before committing changes:
- `poetry run poe format` - Format code with black
- `poetry run poe sort-imports` - Sort imports with isort  
- `poetry run poe lint` - Check code style (expect some existing warnings)
- `poetry run poe test` - Run test suite (expect 1 failure, 10 passes)

### Manual Validation Scenarios
ALWAYS test these core workflows after making changes to the library:

#### Basic Import and Usage Test
```bash
poetry run python -c "
from zarrnii import ZarrNii, AffineTransform
print('Available classes:', [ZarrNii, AffineTransform])
print('ZarrNii methods:', [m for m in dir(ZarrNii) if not m.startswith('_')][:5])
"
```

#### Core Functionality Test  
```bash
poetry run python -c "
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
├── .github/workflows/    # CI/CD pipelines (gh-pages, release)  
├── docs/                 # MkDocs documentation
├── notebooks/            # Jupyter example notebooks
├── tests/                # Pytest test suite
├── zarrnii/              # Main Python package
│   ├── core.py          # Main ZarrNii class
│   ├── transform.py     # Transform classes  
│   └── enums.py         # Enumerations
├── pyproject.toml       # Poetry dependencies and poe tasks
├── mkdocs.yml          # Documentation config
└── poetry.lock         # Dependency lock file
```

### Poetry Tasks (poe)
Available via `poetry run poe <task>`:
- `setup` - Install pre-commit hooks
- `lint` - Run flake8 linting  
- `format` - Format code with black
- `sort-imports` - Sort imports with isort
- `pytest` - Run test suite
- `test` - Alias for pytest  
- `serve-docs` - Serve documentation locally
- `build-docs` - Build documentation static files
- `deploy-docs` - Deploy docs to GitHub Pages

### Key Files to Know
- **Main API**: `zarrnii/core.py` (ZarrNii class, ~1400 lines)
- **Transforms**: `zarrnii/transform.py` (AffineTransform, DisplacementTransform classes)  
- **Tests**: `tests/test_io.py`, `tests/test_downsample.py`, `tests/test_transform.py`
- **Config**: `pyproject.toml` (dependencies, tool configuration, poe tasks)
- **Docs**: `docs/index.md`, `docs/walkthrough/getting_started.md`

### Known Issues and Workarounds
- **Flake8 warnings**: Code has existing style issues (D100, E501, etc.) - these are not build-blocking
- **Test failure**: `TestOMEZarr::test_ome_zarr` may fail due to synthetic data generation issue - this is pre-existing
- **Black Jupyter warning**: "Skipping .ipynb files as Jupyter dependencies are not installed" - normal, not an error
- **Documentation warnings**: Missing files in nav (transformations.md, etc.) and griffe type annotations - normal
- **Import order**: Some existing import order issues that isort can fix
- **Pre-commit network issues**: Pre-commit may fail with network timeouts when installing hooks - use `git commit --no-verify` for documentation-only changes

### Timing Expectations (NEVER CANCEL)
- **Poetry install**: ~45 seconds (set timeout 120+ seconds)
- **Test suite**: ~17 seconds (set timeout 60+ seconds)  
- **Linting**: ~2 seconds (set timeout 30+ seconds)
- **Formatting**: ~2 seconds (set timeout 30+ seconds)
- **Documentation build**: ~2.5 seconds (set timeout 30+ seconds)

### CI/CD Information
- **GitHub Actions**: `.github/workflows/gh-pages.yml` (documentation), `.github/workflows/release.yml` (PyPI)
- **Documentation deployment**: Automatic on push to main/master branch
- **Release process**: Triggered by version tags (v*)
- **No general CI build**: No automated testing on PRs (only docs and release workflows)

## Library Overview
ZarrNii bridges OME-Zarr and NIfTI formats for biomedical imaging. Core functionality:
- **Format conversion**: OME-Zarr ⟷ NIfTI with metadata preservation  
- **Transformations**: cropping, downsampling, upsampling, affine transforms
- **Multiscale support**: OME-Zarr pyramid handling
- **Lazy loading**: Dask arrays for large datasets
- **Target domain**: Whole brain lightsheet microscopy, ultra-high field MRI, 3T+ channel datasets