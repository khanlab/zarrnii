# Transformations

This section covers spatial transformations in ZarrNii, including affine transforms and displacement fields.

## Overview

ZarrNii provides support for spatial transformations through the `AffineTransform` and `DisplacementTransform` classes. These can be applied to register and align images.

## Affine Transformations

### Creating Affine Transformations

```python
from zarrnii import ZarrNii
from zarrnii.transform import AffineTransform
import numpy as np

# Load a dataset
znimg = ZarrNii.from_nifti("path/to/image.nii")

# Create identity transformation
identity_transform = AffineTransform.identity()

# Create transformation from a matrix
matrix = np.array([
    [2.0, 0.0, 0.0, 10.0],  # Scale X by 2, translate by 10
    [0.0, 2.0, 0.0, -5.0],  # Scale Y by 2, translate by -5
    [0.0, 0.0, 1.0, 0.0],   # No change in Z
    [0.0, 0.0, 0.0, 1.0]    # Homogeneous coordinates
])
transform = AffineTransform.from_array(matrix)

# Load transformation from text file
transform_from_file = AffineTransform.from_txt("transform.txt")
```

### Applying Transformations

```python
# Apply transformation (requires reference image)
ref_znimg = ZarrNii.from_nifti("path/to/reference.nii")
transformed = znimg.apply_transform(transform, ref_znimg=ref_znimg)
```

### Working with Displacement Transformations

```python
from zarrnii.transform import DisplacementTransform

# Load displacement field from NIfTI file
disp_transform = DisplacementTransform.from_nifti("displacement_field.nii")

# Apply displacement transformation
deformed = znimg.apply_transform(disp_transform, ref_znimg=ref_znimg)
```

### Multiple Transformations

```python
# Apply multiple transformations in sequence
# Each transformation is applied sequentially
result = znimg.apply_transform(transform1, transform2, ref_znimg=ref_znimg)
```

### Coordinate Transformations

```python
# Transform coordinates using the matrix multiplication operator
voxel_coords = np.array([50, 60, 30])
ras_coords = transform @ voxel_coords

# Transform multiple points
points = np.array([[50, 60, 30], [100, 120, 60]]).T  # 3xN array
transformed_points = transform @ points
```

### Inverting Transformations

```python
# Get the inverse of a transformation
inverse_transform = transform.invert()

# Apply inverse transformation
restored = transformed_znimg.apply_transform(inverse_transform, ref_znimg=znimg)
```

## Coordinate System Handling

```python
# Update transformation for different orientations
updated_transform = transform.update_for_orientation("RPI", "RAS")
```

## Best Practices

1. **Use ref_znimg parameter**: Always provide a reference image when applying transformations
2. **Consider coordinate systems**: Be aware of voxel vs RAS coordinate conventions  
3. **Memory efficiency**: Use lazy evaluation with Dask arrays for large datasets
4. **Transformation order**: Remember that multiple transforms are applied sequentially

## See Also

- [API Reference](../reference.md) for detailed method documentation
- [Downsampling and Upsampling](downsampling.md) for resolution change operations
- [Working with Zarr and NIfTI](zarr_nifti.md) for basic format operations