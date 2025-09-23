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

# For StarDist deep learning segmentation
pip install zarrnii[stardist]
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

 - **Seamless Format Conversion**: Easily convert between OME-Zarr, NIfTI, and Imaris while preserving spatial metadata.
 - **ZipStore Support**: Read and write OME-Zarr files in compressed ZIP format (.ome.zarr.zip) for efficient storage and sharing.
 - **Transformations**: Apply common operations like affine transformations, downsampling, and upsampling.
 - **Multiscale Support**: Work with multiscale OME-Zarr pyramids.
 - **Metadata Handling**: Access and modify OME-Zarr metadata like axes and transformations.
 - **Lazy Loading**: Leverage Dask arrays for efficient processing of large datasets.
 - **Segmentation Plugins**: Extensible plugin architecture for image segmentation algorithms.

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

### StarDist Deep Learning Segmentation

ZarrNii supports StarDist for accurate cell and nuclei instance segmentation:

```python
from zarrnii import ZarrNii

# Load your image  
znimg = ZarrNii.from_ome_zarr("image.ome.zarr")

# Apply StarDist segmentation with pre-trained model
segmented = znimg.segment_stardist(
    model_name="2D_versatile_fluo",  # or "3D_demo" for 3D
    prob_thresh=0.5,
    use_gpu=True,  # Enable GPU acceleration if available
    use_dask_relabeling=False  # Set to True if dask_relabeling is available
)

# Or use custom model
segmented = znimg.segment_stardist(
    model_path="/path/to/custom/model",
    prob_thresh=0.6,
    overlap=64  # Overlap size for tiled processing
)
```

**Note**: StarDist requires additional dependencies. Install with:
```bash
pip install zarrnii[stardist]
```

For advanced tiled processing of very large images, you can optionally install dask_relabeling:
```bash
pip install git+https://github.com/TheJacksonLaboratory/dask_relabeling.git
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
