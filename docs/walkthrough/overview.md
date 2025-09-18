# Walkthrough: Overview

This page provides an overview of the core concepts behind **ZarrNii**. It’s the starting point for understanding how to work with OME-Zarr, NIfTI, and ZarrNii’s transformation tools.

---

## Core Concepts

### **1. Zarr and OME-Zarr**

- **Zarr** is a format for chunked, compressed N-dimensional arrays.
- **OME-Zarr** extends Zarr with metadata for multidimensional microscopy images, supporting axes definitions and multiscale pyramids.

#### Key Features of OME-Zarr:

- **Axes Metadata**: Defines spatial dimensions (e.g., `x`, `y`, `z`).
- **Multiscale Pyramids**: Stores image resolutions at multiple scales.
- **Annotations**: Includes OME metadata for visualization and analysis.

---

### **2. NIfTI**

- **NIfTI** is a neuroimaging file format, commonly used for MRI and fMRI data.
- It supports spatial metadata, such as voxel sizes and affine transformations, for anatomical alignment.

---

### **3. ZarrNii**

- ZarrNii provides tools to bridge these formats while preserving spatial metadata and enabling transformations.

#### Main Features:

- Read and write OME-Zarr and NIfTI formats.
- Apply transformations like cropping, downsampling, and interpolation.
- Convert between ZYX (OME-Zarr) and XYZ (NIfTI) axes orders.

---

## Data Model

ZarrNii wraps datasets using the `ZarrNii` class, which has the following attributes:

- **`darr`**: The dask array containing image data.
- **`affine`**: An affine transformation matrix for spatial alignment.
- **`axes_order`**: Specifies the data layout (`ZYX` or `XYZ`).
- **OME-Zarr Metadata**:
  - **`axes`**: Defines the dimensions and units.
  - **`coordinate_transformations`**: Lists scaling and translation transformations.
  - **`omero`**: Contains channel and visualization metadata.

---

## Example Workflow

Here’s a high-level example workflow using ZarrNii:

1. **Read Data**:
   ```python
   from zarrnii import ZarrNii
   znimg = ZarrNii.from_ome_zarr("path/to/dataset.ome.zarr")
   ```

2. **Apply Transformations**:
   ```python
   znimg_downsampled = znimg.downsample(level=2)
   znimg_cropped = znimg_downsampled.crop((0, 0, 0), (100, 100, 100))
   ```

3. **Convert Formats**:
   ```python
   znimg_cropped.to_nifti("output.nii")
   ```

---

## What’s Next?

- [Getting Started](getting_started.md): Step-by-step guide to installing and using ZarrNii.
- [Basic Tasks](basic_tasks.md): Learn how to read, write, and transform data.

