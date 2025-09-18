# zarrnii

 **ZarrNii** is a Python library for working with OME-Zarr and NIfTI formats. ZarrNii bridges the gap between these two popular formats, enabling seamless data transformation, metadata preservation, and efficient processing of biomedical images. The motivating application is for whole brain lightsheet microscopy and ultra-high field MRI, but it can generally be used for any 3T+channel datasets.

ZarrNii allows you to:

 - Read and write OME-Zarr and NIfTI datasets
 - Perform transformations like cropping, downsampling, and interpolation.
 - Preserve and manipulate metadata from OME-Zarr (e.g., axes, coordinate transformations, OME annotations).

---

## Installation

### Using pip (recommended)
```bash
pip install zarrnii
```

### Development installation  
For contributing or development, clone the repository and install with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/khanlab/zarrnii.git
cd zarrnii
uv sync --dev
```

---

## Key Features

 - **Seamless Format Conversion**: Easily convert between OME-Zarr and NIfTI while preserving spatial metadata.
 - **ZipStore Support**: Read and write OME-Zarr files in compressed ZIP format (.ome.zarr.zip) for efficient storage and sharing.
 - **Transformations**: Apply common operations like affine transformations, downsampling, and upsampling.
 - **Multiscale Support**: Work with multiscale OME-Zarr pyramids.
 - **Metadata Handling**: Access and modify OME-Zarr metadata like axes and transformations.
 - **Lazy Loading**: Leverage Dask arrays for efficient processing of large datasets.

---

## Quick Start

```python
from zarrnii import ZarrNii

# Load an OME-Zarr dataset
znimg = ZarrNii.from_ome_zarr("path/to/zarr_dataset.ome.zarr")

# Load from compressed ZIP format
znimg_zip = ZarrNii.from_ome_zarr("path/to/dataset.ome.zarr.zip")

# Perform a transformation (e.g., downsample)
downsampled_znimg = znimg.downsample(level=2)

# Save as NIfTI
downsampled_znimg.to_nifti("output_dataset.nii")

# Save as compressed OME-Zarr ZIP file
downsampled_znimg.to_ome_zarr("compressed_output.ome.zarr.zip")
```

---

## Development

For development, this project uses:

- **[uv](https://docs.astral.sh/uv/)** for fast dependency management
- **[pytest](https://pytest.org/)** for testing
- **[black](https://black.readthedocs.io/)** for code formatting  
- **[flake8](https://flake8.pycqa.org/)** for linting
- **[mkdocs](https://www.mkdocs.org/)** for documentation

### Available commands (using `uv run`):
```bash
# Run tests
uv run pytest

# Format code
uv run black .

# Check linting  
uv run flake8 .

# Build documentation
uv run mkdocs build

# Serve docs locally
uv run mkdocs serve
```

### Using the justfile:
If you have [just](https://just.systems/) installed:
```bash
# See all available tasks
just help

# Run tests
just test

# Format and lint
just format
just lint
```

---

## Learn More

Explore the [documentation](https://www.khanlab.ca/zarrnii) to get started.

## Contributing

Contributions are welcome! Please read our contributing guidelines and ensure all tests pass before submitting pull requests.
