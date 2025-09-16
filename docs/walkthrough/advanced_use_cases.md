# Walkthrough: Advanced Use Cases

This guide explores advanced workflows with **ZarrNii**, including metadata preservation, handling multiscale OME-Zarr pyramids, and combining multiple transformations.

---

## Table of Contents
1. [Preserving Metadata](#preserving-metadata)
2. [Working with Multiscale Pyramids](#working-with-multiscale-pyramids)
3. [Combining Transformations](#combining-transformations)
4. [Handling Large Datasets](#handling-large-datasets)

---

## Preserving Metadata

ZarrNii is designed to handle and preserve metadata when converting between formats or applying transformations.

### **Accessing Metadata**
OME-Zarr metadata is automatically extracted and stored in the `axes`, `coordinate_transformations`, and `omero` attributes of a `ZarrNii` instance.

```python
znimg = ZarrNii.from_ome_zarr("path/to/dataset.zarr")

# Access axes metadata
print("Axes metadata:", znimg.axes)

# Access coordinate transformations
print("Coordinate transformations:", znimg.coordinate_transformations)

# Access Omero metadata
print("Omero metadata:", znimg.omero)
```

---

### **Preserving Metadata During Transformations**
When you perform transformations like cropping or downsampling, ZarrNii ensures metadata remains consistent.

```python
cropped = znimg.crop_with_bounding_box((10, 10, 10), (50, 50, 50))
print("Updated metadata:", cropped.coordinate_transformations)
```

---

## Working with Multiscale Pyramids

OME-Zarr datasets often include multiscale pyramids, where each level represents a progressively downsampled version of the data.

### **Loading a Specific Level**
You can load a specific pyramid level using the `level` argument in `from_ome_zarr`:

```python
znimg = ZarrNii.from_ome_zarr("path/to/dataset.zarr", level=2)
print("Loaded shape:", znimg.darr.shape)
```

### **Handling Custom Downsampling**
If the desired level isn't available in the pyramid, ZarrNii computes additional downsampling lazily:

```python
level, do_downsample, ds_kwargs = ZarrNii.get_level_and_downsampling_kwargs(
    "path/to/dataset.zarr", level=5
)
if do_downsample:
    znimg = znimg.downsample(**ds_kwargs)
```

---

## Combining Transformations

ZarrNii allows you to chain multiple transformations into a single workflow. This is useful when applying affine transformations, interpolations, or warping.

### **Chaining Affine Transformations**
```python
from zarrnii.transform import AffineTransform
import numpy as np

# Create transformations using matrices
scaling_matrix = np.array([
    [2.0, 0.0, 0.0, 0.0],
    [0.0, 2.0, 0.0, 0.0], 
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])
scaling = AffineTransform.from_array(scaling_matrix)

translation_matrix = np.array([
    [1.0, 0.0, 0.0, 10.0],
    [0.0, 1.0, 0.0, -5.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])
translation = AffineTransform.from_array(translation_matrix)

# Apply multiple transformations sequentially
combined = znimg.apply_transform(scaling, translation, ref_znimg=znimg)
print("New affine matrix:\n", combined.affine.matrix)
```

---

## Handling Large Datasets

ZarrNii leverages Dask to handle datasets that don't fit into memory.

### **Optimizing Chunking**
Ensure the dataset is chunked appropriately for operations like downsampling or interpolation:

```python
# Rechunk for efficient processing
rechunked = znimg.darr.rechunk((1, 64, 64, 64))
print("Rechunked shape:", rechunked.shape)
```

### **Lazy Evaluation**
Most transformations in ZarrNii are lazy, meaning computations are only triggered when necessary. Use `.compute()` to materialize results.

```python
# Trigger computation
cropped = znimg.crop_with_bounding_box((10, 10, 10), (50, 50, 50))
cropped.darr.compute()
```

---

## Summary

This guide covered:
- Preserving metadata across transformations and format conversions.
- Working with multiscale pyramids in OME-Zarr.
- Combining transformations for complex workflows.
- Handling large datasets efficiently with Dask.

Next, explore:
- [Examples](../examples/zarr_nifti.md): Detailed workflows and practical use cases.
- [API Reference](../reference.md): Technical details for ZarrNii classes and methods.

