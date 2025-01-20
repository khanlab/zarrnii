# zarrnii

 **ZarrNii** is a Python library for working with OME-Zarr and NIfTI formats. ZarrNii bridges the gap between these two popular formats, enabling seamless data transformation, metadata preservation, and efficient. The motivating application is for whole brain lightsheet microscopy and ultra-high field MRI, but it can generally be used for any 3T+channel datasets.

ZarrNii allows you to:

 - Read and write OME-Zarr and NIfTI datasets
 - Perform transformations like cropping, downsampling, and interpolation.
 - Preserve and manipulate metadata from OME-Zarr (e.g., axes, coordinate transformations, OME annotations).


---

## Key Features

 - **Seamless Format Conversion**: Easily convert between OME-Zarr and NIfTI while preserving spatial metadata.
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

# Perform a transformation (e.g., downsample)
downsampled_znimg = znimg.downsample(level=2)

# Save as NIfTI
downsampled_znimg.to_nifti("output_dataset.nii")
```

---

## Learn More

Explore the [documentation](https://www.khanlab.ca/zarrnii) to get started.
