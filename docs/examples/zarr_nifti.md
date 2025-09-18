# Examples: Working with Zarr and NIfTI

This section provides practical workflows for using **ZarrNii** with OME-Zarr and NIfTI datasets.

---

## Table of Contents
1. [Loading Datasets](#loading-datasets)
    - From OME-Zarr
    - From NIfTI
2. [Performing Transformations](#performing-transformations)
    - Downsampling
    - Cropping
    - Combining Affine Transformations
3. [Saving Results](#saving-results)
    - To OME-Zarr
    - To NIfTI
4. [Advanced Example: Full Workflow](#advanced-example-full-workflow)

---

## Loading Datasets

### **From OME-Zarr**
Load a dataset from an OME-Zarr file and inspect its metadata:

```python
from zarrnii import ZarrNii

# Load OME-Zarr dataset
znimg = ZarrNii.from_ome_zarr("path/to/dataset.zarr")

# Inspect data
print("Shape:", znimg.darr.shape)
print("Affine matrix:\n", znimg.affine.matrix)
```

---

### **From NIfTI**
Load a NIfTI dataset and inspect its attributes:

```python
# Load NIfTI dataset
znimg = ZarrNii.from_nifti("path/to/dataset.nii")

# Inspect data
print("Shape:", znimg.darr.shape)
print("Affine matrix:\n", znimg.affine.matrix)
```

---

## Performing Transformations

### **Downsampling**
Reduce the resolution of the dataset using the `downsample` method:

```python
# Downsample by level
downsampled = znimg.downsample(level=2)
print("Downsampled shape:", downsampled.darr.shape)
```

---

### **Cropping**
Extract a specific region from the dataset using bounding boxes:

#### **Voxel Space**:
```python
cropped = znimg.crop((10, 10, 10), (50, 50, 50))
print("Cropped shape:", cropped.darr.shape)
```

#### **With RAS Coordinates**:
```python
# Note: crop_with_bounding_box is a legacy method that still supports RAS coords
cropped_ras = znimg.crop_with_bounding_box(
    (-20, -20, -20), (20, 20, 20), ras_coords=True
)
print("Cropped shape:", cropped_ras.darr.shape)
```

---

### **Combining Affine Transformations**
Apply multiple transformations to the dataset in sequence:

```python
from zarrnii.transform import AffineTransform
import numpy as np

# Define transformations using matrices
scale_matrix = np.array([
    [2.0, 0.0, 0.0, 0.0],
    [0.0, 2.0, 0.0, 0.0], 
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])
scale = AffineTransform.from_array(scale_matrix)

translate_matrix = np.array([
    [1.0, 0.0, 0.0, 10.0],
    [0.0, 1.0, 0.0, -5.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])
translate = AffineTransform.from_array(translate_matrix)

# Apply transformations
transformed = znimg.apply_transform(scale, translate, ref_znimg=znimg)
print("Transformed affine matrix:\n", transformed.affine.matrix)
```

---

## Saving Results

### **To OME-Zarr**
Save the dataset to OME-Zarr format:

```python
znimg.to_ome_zarr("output.zarr", max_layer=3, scaling_method="local_mean")
```

---

### **To NIfTI**
Save the dataset to NIfTI format:

```python
znimg.to_nifti("output.nii")
```

---

## Advanced Example: Full Workflow

Combine multiple operations in a single workflow:

```python
from zarrnii import ZarrNii
from zarrnii.transform import AffineTransform
import numpy as np

# Load an OME-Zarr dataset
znimg = ZarrNii.from_ome_zarr("path/to/dataset.zarr")

# Crop the dataset
cropped = znimg.crop((10, 10, 10), (100, 100, 100))

# Downsample the dataset
downsampled = cropped.downsample(level=2)

# Apply an affine transformation
scale_matrix = np.array([
    [1.5, 0.0, 0.0, 0.0],
    [0.0, 1.5, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])
scale = AffineTransform.from_array(scale_matrix)
transformed = downsampled.apply_transform(scale, ref_znimg=downsampled)

# Save the result as a NIfTI file
transformed.to_nifti("final_output.nii")
```

---

## Summary

In this section, you learned how to:
- Load datasets from OME-Zarr and NIfTI formats.
- Perform transformations like downsampling, cropping, and affine transformations.
- Save results back to OME-Zarr or NIfTI.

Next:
- Explore the [API Reference](../reference.md) for in-depth details about ZarrNii's classes and methods.
- Check the [FAQ](../faq.md) for answers to common questions.

