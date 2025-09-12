# Transformations

This section covers spatial transformations in ZarrNii, including affine transforms, displacement fields, and composite operations.

## Overview

ZarrNii provides comprehensive support for spatial transformations through the `AffineTransform` and `DisplacementTransform` classes. These can be applied individually or combined for complex image processing workflows.

## Affine Transformations

### Basic Affine Operations

```python
from zarrnii import ZarrNii
from zarrnii.transform import AffineTransform

# Load a dataset
znimg = ZarrNii.from_nifti("path/to/image.nii")

# Create scaling transformation
scale_transform = AffineTransform.from_scaling((2.0, 2.0, 1.0))

# Create translation transformation  
translate_transform = AffineTransform.from_translation((10.0, -5.0, 0.0))

# Create rotation transformation (45 degrees around z-axis)
rotate_transform = AffineTransform.from_rotation_z(45.0)

# Apply transformation
transformed = znimg.apply_transform(scale_transform)
```

### Composite Transformations

```python
# Combine multiple transformations
composite = scale_transform.compose(translate_transform).compose(rotate_transform)

# Apply the composite transformation
result = znimg.apply_transform(composite)
```

### Working with Transform Matrices

```python
import numpy as np

# Create transform from 4x4 matrix
matrix = np.array([
    [2.0, 0.0, 0.0, 10.0],
    [0.0, 2.0, 0.0, -5.0], 
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])
matrix_transform = AffineTransform(matrix)

# Apply to image
transformed = znimg.apply_transform(matrix_transform)
```

## Displacement Transformations

```python
from zarrnii.transform import DisplacementTransform

# Create displacement field (example: simple deformation)
displacement_field = np.random.randn(*znimg.darr.shape[:3], 3) * 2.0
disp_transform = DisplacementTransform(displacement_field)

# Apply displacement transformation
deformed = znimg.apply_transform(disp_transform)
```

## Resampling and Interpolation

```python
# Apply transformation with specific interpolation
transformed = znimg.apply_transform(
    scale_transform, 
    interpolation="linear",  # or "nearest", "cubic"
    ref_znimg=znimg
)
```

## Coordinate System Considerations

```python
# Work with different coordinate systems
# Transform coordinates from voxel to RAS space
voxel_coords = np.array([50, 60, 30])
ras_coords = znimg.affine.apply(voxel_coords)

# Apply transformation in RAS space
ras_transformed = scale_transform.apply(ras_coords)
```

## Advanced Usage

### Chain Multiple Operations

```python
# Create a processing pipeline
result = (znimg
          .apply_transform(scale_transform)
          .crop_with_bounding_box((10, 10, 10), (100, 100, 50))
          .downsample(level=2)
          .apply_transform(rotate_transform))
```

### Transform Composition

```python
# Efficiently compose multiple transforms before applying
pipeline_transform = (scale_transform
                     .compose(translate_transform) 
                     .compose(rotate_transform))

result = znimg.apply_transform(pipeline_transform)
```

## Best Practices

1. **Combine transforms before applying**: Compose multiple transformations into a single operation to minimize interpolation artifacts
2. **Choose appropriate interpolation**: Use "nearest" for labels/masks, "linear" or "cubic" for continuous data
3. **Consider coordinate systems**: Be aware of voxel vs RAS coordinate conventions
4. **Memory efficiency**: Use lazy evaluation with Dask arrays for large datasets

## See Also

- [API Reference](../reference.md) for detailed method documentation
- [Basic Tasks](../walkthrough/basic_tasks.md) for fundamental operations
- [Advanced Use Cases](../walkthrough/advanced_use_cases.md) for complex workflows