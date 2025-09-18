# Walkthrough: Basic Tasks

This guide covers the most common tasks you'll perform with ZarrNii, including reading data, performing transformations, and saving results.

---

## Table of Contents
1. [Reading Data](#reading-data)
    - From OME-Zarr
    - From NIfTI
    - Working with 5D Data
2. [Transforming Data](#transforming-data)
    - Cropping
    - Downsampling
    - Upsampling
    - Applying Affine Transformations
3. [Saving Data](#saving-data)
    - To NIfTI
    - To OME-Zarr

---

## Reading Data

### **From OME-Zarr**
Load a dataset from an OME-Zarr file using `from_ome_zarr`:

```python
from zarrnii import ZarrNii

# Load the dataset
znimg = ZarrNii.from_ome_zarr("path/to/dataset.ome.zarr")

# Inspect the data
print("Data shape:", znimg.darr.shape)
print("Affine matrix:\n", znimg.affine.matrix)
```

#### **Automatic Anisotropy Correction**
For datasets with anisotropic voxels (common in lightsheet microscopy), you can automatically downsample to create more isotropic voxels:

```python
# Load with near-isotropic downsampling
znimg_isotropic = ZarrNii.from_ome_zarr(
    "path/to/anisotropic_data.ome.zarr", 
    downsample_near_isotropic=True
)

print("Isotropic data shape:", znimg_isotropic.darr.shape)
print("Isotropic scales:", znimg_isotropic.scale)
```

---

### **From NIfTI**
Load a dataset from a NIfTI file using `from_nifti`:

```python
# Load the dataset
znimg = ZarrNii.from_nifti("path/to/dataset.nii")

# Inspect the data
print("Data shape:", znimg.darr.shape)
print("Affine matrix:\n", znimg.affine.matrix)
```

---

### **Working with 5D Data**
ZarrNii supports 5D images with time and channel dimensions (T,C,Z,Y,X). You can select specific timepoints and channels during loading or after loading.

#### **Loading with Timepoint Selection**:
```python
# Load specific timepoints
znimg_time = ZarrNii.from_ome_zarr("timeseries.zarr", timepoints=[0, 2, 4])
print("Timepoint subset shape:", znimg_time.darr.shape)

# Load specific channels by index
znimg_channels = ZarrNii.from_ome_zarr("multichannel.zarr", channels=[0, 2])

# Load specific channels by label
znimg_labels = ZarrNii.from_ome_zarr("labeled.zarr", channel_labels=["DAPI", "GFP"])

# Combine timepoint and channel selection
znimg_subset = ZarrNii.from_ome_zarr("data.zarr", timepoints=[1, 3], channels=[0])
```

#### **Post-loading Selection**:
```python
# Load full dataset first
znimg = ZarrNii.from_ome_zarr("timeseries.zarr")

# Select timepoints after loading
selected_time = znimg.select_timepoints([0, 2])

# Select channels after loading
selected_channels = znimg.select_channels([1, 2])

# Chain selections
subset = znimg.select_timepoints([0, 1]).select_channels([0])
```

---

## Transforming Data

### **Cropping**
Crop the dataset to a specific bounding box. You can define the bounding box in either voxel space or RAS (real-world) coordinates. For 5D data, cropping operates only on spatial dimensions, preserving time and channel dimensions.

#### **Voxel Space Cropping**:
```python
# Preferred method using crop()
cropped = znimg.crop((10, 10, 10), (50, 50, 50))
print("Cropped shape:", cropped.darr.shape)
```

#### **RAS Space Cropping**:
```python
# Use legacy method for RAS coordinates
cropped_ras = znimg.crop_with_bounding_box(
    (-20, -20, -20), (20, 20, 20), ras_coords=True
)
print("Cropped shape:", cropped_ras.darr.shape)
```

---

### **Downsampling**
Downsample the dataset to reduce its resolution. You can specify either a downsampling level or individual scaling factors for each axis. For 5D data, downsampling operates only on spatial dimensions, preserving time and channel dimensions.

#### **By Level**:
```python
downsampled = znimg.downsample(level=2)
print("Downsampled shape:", downsampled.darr.shape)
```

#### **By Scaling Factors**:
```python
downsampled_manual = znimg.downsample(along_x=2, along_y=2, along_z=1)
print("Downsampled shape:", downsampled_manual.darr.shape)
```

---

### **Upsampling**
Increase the resolution of the dataset by upsampling.

#### **By Scaling Factors**:
```python
upsampled = znimg.upsample(along_x=2, along_y=2, along_z=2)
print("Upsampled shape:", upsampled.darr.shape)
```

#### **To Target Shape**:
```python
upsampled_target = znimg.upsample(to_shape=(1, 256, 256, 256))
print("Upsampled shape:", upsampled_target.darr.shape)
```

---

### **Applying Affine Transformations**
Apply a custom affine transformation to the dataset.

```python
from zarrnii.transform import AffineTransform
import numpy as np

# Define a scaling transformation using a matrix
scaling_matrix = np.array([
    [2.0, 0.0, 0.0, 0.0],
    [0.0, 2.0, 0.0, 0.0], 
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])
scaling_transform = AffineTransform.from_array(scaling_matrix)

# Apply the transformation  
transformed = znimg.apply_transform(scaling_transform, ref_znimg=znimg)
print("Transformed affine matrix:\n", transformed.affine.matrix)
```

---

## Saving Data

### **To NIfTI**
Save the dataset as a NIfTI file using `to_nifti`:

```python
znimg.to_nifti("output_dataset.nii")
```

---

### **To OME-Zarr**
Save the dataset as an OME-Zarr file using `to_ome_zarr`:

```python
znimg.to_ome_zarr("output_dataset.ome.zarr")
```

You can also save additional metadata during the process:

```python
znimg.to_ome_zarr(
    "output_dataset.ome.zarr",
    max_layer=3,
    scaling_method="local_mean"
)
```

---

## Summary

This guide covered the essential operations you can perform with ZarrNii:
- Reading datasets from OME-Zarr and NIfTI formats.
- Transforming datasets through cropping, downsampling, upsampling, and affine transformations.
- Saving datasets back to either format.

Next, explore [Advanced Use Cases](advanced_use_cases.md) or dive into the [API Reference](../reference.md) for detailed technical documentation.

