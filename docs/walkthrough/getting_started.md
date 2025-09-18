# Getting Started

This guide helps you set up **ZarrNii** and get started with its basic functionality. By the end of this guide, you'll be able to read OME-Zarr and NIfTI datasets, perform basic transformations, and save your results.

---

## Installation

ZarrNii requires Python 3.11 or later. Install it using [uv](https://docs.astral.sh/uv/), a modern, fast Python package installer and project manager.

### **1. Clone the Repository**
If you're using the source code, clone the ZarrNii repository:
```bash
git clone https://github.com/khanlab/zarrnii.git
cd zarrnii
```

### **2. Install with uv**
Run the following command to install the library and its dependencies:
```bash
uv sync --dev
```

If you don't use uv, install ZarrNii and its dependencies using `pip`:
```bash
pip install zarrnii
```

---

## Prerequisites

Before using ZarrNii, ensure you have:
- **OME-Zarr datasets**: Multidimensional images in Zarr format.
- **NIfTI datasets**: Neuroimaging data in `.nii` or `.nii.gz` format.

---

## Basic Usage

### **1. Reading Data**

You can load an OME-Zarr or NIfTI dataset into a `ZarrNii` object.

#### **From OME-Zarr**:
```python
from zarrnii import ZarrNii

# Load OME-Zarr
znimg = ZarrNii.from_ome_zarr("path/to/dataset.ome.zarr")

print("Data shape:", znimg.darr.shape)
print("Affine matrix:\n", znimg.affine.matrix)

# For anisotropic data, automatically create more isotropic voxels
znimg_isotropic = ZarrNii.from_ome_zarr(
    "path/to/dataset.ome.zarr", 
    downsample_near_isotropic=True
)
```

#### **From NIfTI**:
```python
# Load NIfTI
znimg = ZarrNii.from_nifti("path/to/dataset.nii")

print("Data shape:", znimg.darr.shape)
print("Affine matrix:\n", znimg.affine.matrix)
```

---

### **2. Performing Transformations**

ZarrNii supports various transformations, such as cropping, downsampling, and upsampling.

#### **Cropping**:
Crop a region from the dataset using voxel coordinates:
```python
cropped = znimg.crop((10, 10, 10), (50, 50, 50))
print("Cropped shape:", cropped.darr.shape)
```

#### **Downsampling**:
Reduce the resolution of your dataset:
```python
downsampled = znimg.downsample(level=2)
print("Downsampled shape:", downsampled.darr.shape)
```

#### **Upsampling**:
Increase the resolution of your dataset:
```python
upsampled = znimg.upsample(along_x=2, along_y=2, along_z=2)
print("Upsampled shape:", upsampled.darr.shape)
```

---

### **3. Saving Data**

ZarrNii makes it easy to save your datasets in both OME-Zarr and NIfTI formats.

#### **To NIfTI**:
Save the dataset as a `.nii` file:
```python
znimg.to_nifti("output_dataset.nii")
```

#### **To OME-Zarr**:
Save the dataset back to OME-Zarr format:
```python
znimg.to_ome_zarr("output_dataset.ome.zarr")
```

---

## Example Workflow

Here’s a full workflow from loading an OME-Zarr dataset to saving a downsampled version as NIfTI:

```python
from zarrnii import ZarrNii

# Load an OME-Zarr dataset
znimg = ZarrNii.from_ome_zarr("path/to/dataset.ome.zarr")

# Perform transformations
cropped = znimg.crop((10, 10, 10), (100, 100, 100))
downsampled = cropped.downsample(level=2)

# Save the result as a NIfTI file
downsampled.to_nifti("downsampled_output.nii")
```

---

## What’s Next?

- [Walkthrough: Basic Tasks](basic_tasks.md): Learn more about common workflows like cropping, interpolation, and combining transformations.
- [API Reference](../reference.md): Explore the detailed API for ZarrNii.

