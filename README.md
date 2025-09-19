# zarrnii

 **ZarrNii** is a Python library for working with OME-Zarr, NIfTI, and Imaris formats. ZarrNii bridges the gap between these popular formats, enabling seamless data transformation, metadata preservation, and efficient processing of biomedical images. The motivating application is for whole brain lightsheet microscopy and ultra-high field MRI, but it can generally be used for any 3D+[channel,time] datasets.

ZarrNii allows you to:

 - Read and write OME-Zarr, NIfTI, and Imaris datasets
 - Perform transformations like cropping, downsampling, and interpolation.
 - Preserve and manipulate metadata from OME-Zarr (e.g., axes, coordinate transformations, OME annotations).

---

## Installation

### Using pip (recommended)
```bash
pip install zarrnii
```

### Optional Dependencies
For additional format support:
```bash
# For Imaris (.ims) file support
pip install zarrnii[imaris]

# For interactive visualization with vizarr
pip install zarrnii[viz]

# For all optional features
pip install zarrnii[imaris,viz]
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
 - **Interactive Visualization**: Explore OME-Zarr data interactively with vizarr integration in Jupyter notebooks.
 - **Transformations**: Apply common operations like affine transformations, downsampling, and upsampling.
 - **Multiscale Support**: Work with multiscale OME-Zarr pyramids.
 - **Metadata Handling**: Access and modify OME-Zarr metadata like axes and transformations.
 - **Lazy Loading**: Leverage Dask arrays for efficient processing of large datasets.
 - **Segmentation Plugins**: Extensible plugin architecture for image segmentation algorithms.

---

## Interactive Visualization

ZarrNii includes interactive visualization capabilities using [vizarr](https://github.com/hms-dbmi/vizarr) for exploring OME-Zarr data in Jupyter notebooks:

```python
from zarrnii import ZarrNii

# Load your OME-Zarr data
znimg = ZarrNii.from_ome_zarr("data.ome.zarr")

# Create interactive widget for Jupyter notebook
widget = znimg.visualize(mode="widget")
widget  # Display the interactive viewer
```

### Visualization Features

- **Interactive Widget**: Full-featured viewer for Jupyter notebooks
- **Multiple Modes**: Widget mode for notebooks, HTML mode for documentation
- **OME-Zarr Native**: Direct visualization of multiscale OME-Zarr data
- **Easy Integration**: Simple one-line visualization from any ZarrNii object

### Installation for Visualization

```bash
pip install zarrnii[viz]
```

**Note**: Visualization requires the `vizarr` package and is optimized for Jupyter notebook environments.

---

## Segmentation Plugin System

ZarrNii includes a plugin architecture for image segmentation algorithms, starting with Otsu thresholding:

```python
from zarrnii import ZarrNii, OtsuSegmentation

# Load your image
znimg = ZarrNii.from_ome_zarr("image.ome.zarr")

# Apply Otsu thresholding segmentation
segmented = znimg.segment_otsu(nbins=256)

# Or use the generic plugin interface
plugin = OtsuSegmentation(nbins=128)
segmented = znimg.segment(plugin)

# Save segmented results
segmented.to_ome_zarr("segmented_image.ome.zarr")
```

### Custom Plugins

Create your own segmentation algorithms by extending the `SegmentationPlugin` base class:

```python
from zarrnii.plugins.segmentation import SegmentationPlugin

class CustomSegmentation(SegmentationPlugin):
    def segment(self, image, metadata=None):
        # Your segmentation logic here
        return binary_mask.astype(np.uint8)
    
    @property
    def name(self):
        return "Custom Algorithm"
    
    @property 
    def description(self):
        return "Description of your algorithm"
```

---

## Quick Start

```python
from zarrnii import ZarrNii

# Load an OME-Zarr dataset
znimg = ZarrNii.from_ome_zarr("path/to/zarr_dataset.ome.zarr")

<<<<<<< HEAD
# Or load from Imaris (requires zarrnii[imaris])
# znimg = ZarrNii.from_imaris("path/to/microscopy_data.ims")
=======
# Load from compressed ZIP format
znimg_zip = ZarrNii.from_ome_zarr("path/to/dataset.ome.zarr.zip")
>>>>>>> main

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
