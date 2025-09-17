# New NgffImage-based API

This document describes the new NgffImage-based API that works directly with `ngff_zarr` objects instead of wrapping single-scale dask arrays.

## Overview

The new API provides two main approaches:

1. **NgffZarrNii Class**: A wrapper class that works directly with `NgffImage` objects
2. **Function-based API**: Direct functions that operate on `NgffImage` objects

## Benefits of the New Architecture

- **Direct multiscale support**: Works with the full multiscale pyramid structure
- **Better metadata preservation**: Maintains OME-Zarr metadata throughout operations
- **More efficient**: Avoids unnecessary conversions between formats
- **Cleaner separation**: Functions operate directly on data structures instead of wrapped arrays

## Basic Usage

### Using NgffZarrNii Class

```python
from zarrnii import NgffZarrNii

# Load an OME-Zarr dataset
znimg = NgffZarrNii.from_ome_zarr("path/to/dataset.zarr", level=0)

# Access properties
print("Shape:", znimg.shape)
print("Dimensions:", znimg.dims)
print("Scale:", znimg.scale)
print("Translation:", znimg.translation)

# Get affine transformation
affine_transform = znimg.get_affine_transform()
affine_matrix = znimg.get_affine_matrix()

# Save to OME-Zarr with multiscale pyramid
znimg.to_ome_zarr("output.zarr", max_layer=4)
```

### Using Function-based API

```python
import ngff_zarr as nz
from zarrnii import crop_ngff_image, downsample_ngff_image, resample_ngff_image

# Load using ngff_zarr directly
multiscales = nz.from_ngff_zarr("path/to/dataset.zarr")
ngff_image = multiscales.images[0]  # Get full resolution

# Crop the image
cropped = crop_ngff_image(
    ngff_image, 
    bbox_min=(10, 20, 30),  # Z, Y, X
    bbox_max=(50, 60, 70),
    spatial_dims=["z", "y", "x"]
)

# Downsample the image
downsampled = downsample_ngff_image(
    ngff_image,
    factors=2,  # Isotropic downsampling by factor of 2
    spatial_dims=["z", "y", "x"]
)

# Resample to specific resolution
resampled = resample_ngff_image(
    ngff_image,
    target_scale={"z": 4.0, "y": 2.0, "x": 2.0},
    spatial_dims=["z", "y", "x"]
)
```

## Advanced Transformations

### Applying Spatial Transformations

```python
from zarrnii import AffineTransform, apply_transform_to_ngff_image_full

# Create a transformation (e.g., from registration)
transform_matrix = np.array([
    [1.0, 0.0, 0.0, 5.0],
    [0.0, 1.0, 0.0, 10.0], 
    [0.0, 0.0, 1.0, 15.0],
    [0.0, 0.0, 0.0, 1.0]
])
transform = AffineTransform.from_array(transform_matrix)

# Load source and reference images
source = NgffZarrNii.from_ome_zarr("source.zarr", level=0)
reference = NgffZarrNii.from_ome_zarr("reference.zarr", level=0)

# Apply transformation
transformed = apply_transform_to_ngff_image_full(
    source.ngff_image,
    reference.ngff_image, 
    transform,
    spatial_dims=["z", "y", "x"],
    interpolation_method="linear"
)
```

### Composing Multiple Transformations

```python
from zarrnii import compose_transforms

# Create multiple transformations
transform1 = AffineTransform.from_array(matrix1)
transform2 = AffineTransform.from_array(matrix2)

# Compose them into a single transformation
composed = compose_transforms(transform1, transform2)

# Apply the composed transformation
result = apply_transform_to_ngff_image_full(
    source_image, reference_image, composed
)
```

### Creating Reference Spaces

```python
from zarrnii import create_reference_ngff_image

# Create a custom reference space
reference = create_reference_ngff_image(
    shape=(1, 128, 256, 256),  # C, Z, Y, X
    dims=["c", "z", "y", "x"],
    scale={"z": 2.0, "y": 1.0, "x": 1.0},
    translation={"z": 0.0, "y": 0.0, "x": 0.0},
    name="custom_reference"
)
```

## Channel Selection

Both approaches support channel selection:

```python
# Using NgffZarrNii with channel indices
znimg = NgffZarrNii.from_ome_zarr("multi_channel.zarr", channels=[0, 2])

# Using NgffZarrNii with channel labels
znimg = NgffZarrNii.from_ome_zarr("multi_channel.zarr", channel_labels=["DAPI", "GFP"])
```

## Working with Multiscale Data

The new API maintains access to the full multiscale structure:

```python
# Load with multiscale access
znimg = NgffZarrNii.from_ome_zarr("dataset.zarr", level=0)

# Access the full multiscale object
if znimg.multiscales:
    print("Available levels:", len(znimg.multiscales.images))
    for i, image in enumerate(znimg.multiscales.images):
        print(f"Level {i}: shape={image.data.shape}, scale={image.scale}")
```

## Comparison with Legacy API

### Legacy ZarrNii Approach

```python
from zarrnii import ZarrNii

# Legacy: wraps single-scale dask array
znimg = ZarrNii.from_ome_zarr("dataset.zarr", level=0)
print("Data shape:", znimg.darr.shape)  # Direct dask array access
print("Affine:", znimg.affine.matrix)

# Apply transformation (requires reference image)
transformed = znimg.apply_transform(transform, ref_znimg=reference)
```

### New NgffZarrNii Approach

```python
from zarrnii import NgffZarrNii, apply_transform_to_ngff_image_full

# New: works directly with NgffImage
znimg = NgffZarrNii.from_ome_zarr("dataset.zarr", level=0)
print("Data shape:", znimg.data.shape)  # NgffImage data access
print("Scale:", znimg.scale)  # Direct scale access

# Apply transformation (function-based)
transformed = apply_transform_to_ngff_image_full(
    znimg.ngff_image, reference.ngff_image, transform
)
```

## Migration Guide

For existing code using the legacy `ZarrNii` class:

1. **Replace imports**: Add `NgffZarrNii` alongside `ZarrNii`
2. **Update data access**: Use `.data` instead of `.darr`
3. **Use scale/translation**: Access scale and translation directly instead of through affine matrix
4. **Function-based transforms**: Consider using the new transformation functions for cleaner code

The legacy `ZarrNii` class remains available for backward compatibility.

## Future Enhancements

The new architecture enables several planned enhancements:

- **Better multiscale processing**: Operate on multiple pyramid levels simultaneously
- **Lazy multiscale transformations**: Apply transformations to entire pyramids
- **Enhanced metadata handling**: Preserve and update OME-Zarr metadata throughout workflows
- **Integration with other tools**: Better interoperability with other OME-Zarr tools