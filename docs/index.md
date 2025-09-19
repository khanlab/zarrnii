Welcome to the documentation for **ZarrNii**, a Python library for working with OME-Zarr, NIfTI, and Imaris formats. ZarrNii bridges the gap between these popular formats, enabling seamless data transformation, metadata preservation, and efficient processing of large biomedical images.

---

## What is ZarrNii?

ZarrNii is designed for researchers and engineers working with:

 - **OME-Zarr**: A format for storing multidimensional image data, commonly used in microscopy.
 - **NIfTI**: A standard format for neuroimaging data.
 - **Imaris**: A microscopy file format (.ims) using HDF5 structure for 3D/4D image analysis.

ZarrNii allows you to:

 - Read and write OME-Zarr, NIfTI, and Imaris datasets.
 - Work with 4D and 5D images, including time-series data (T,C,Z,Y,X).
 - Perform transformations like cropping, downsampling, and interpolation.
 - Select specific channels and timepoints from multidimensional datasets.
 - Preserve and manipulate metadata from OME-Zarr (e.g., axes, coordinate transformations, OME annotations).

---

## Key Features

 - **Seamless Format Conversion**: Easily convert between OME-Zarr, NIfTI, and Imaris while preserving spatial metadata.
 - **5D Image Support**: Work with time-series data in (T,C,Z,Y,X) format with timepoint and channel selection.
 - **Transformations**: Apply common operations like affine transformations, downsampling, and upsampling.
 - **Multiscale Support**: Work with multiscale OME-Zarr pyramids.
 - **Metadata Handling**: Access and modify OME-Zarr metadata like axes and transformations.
 - **Lazy Loading**: Leverage Dask arrays for efficient processing of large datasets.
 - **Segmentation Plugins**: Extensible plugin architecture for image segmentation algorithms.

---

## Quick Start

```python
from zarrnii import ZarrNii

# Load an OME-Zarr dataset
znimg = ZarrNii.from_ome_zarr("path/to/zarr_dataset.ome.zarr")

<<<<<<< HEAD
# Or load from Imaris (requires optional dependency)
# znimg = ZarrNii.from_imaris("path/to/microscopy_data.ims")
=======
# Load with specific timepoints and channels (5D support)
znimg_subset = ZarrNii.from_ome_zarr("timeseries.zarr", timepoints=[0, 2], channels=[1])
>>>>>>> main

# Perform a transformation (e.g., downsample)
downsampled_znimg = znimg.downsample(level=2)

# Apply segmentation using Otsu thresholding
segmented_znimg = znimg.segment_otsu(nbins=256)

# Save as NIfTI
downsampled_znimg.to_nifti("output_dataset.nii")
segmented_znimg.to_nifti("segmented_dataset.nii")
```

---

## Learn More

Explore the documentation to get started:

 - [Walkthrough: Overview](walkthrough/overview.md): Understand the core concepts.
 - [API Reference](reference.md): Dive into the technical details.
 - [Examples](examples/zarr_nifti.md): Learn through practical examples.
 - [Segmentation Plugin Examples](examples/segmentation_example.md): Learn how to use and create segmentation plugins.
 - [FAQ](faq.md): Find answers to common questions.


