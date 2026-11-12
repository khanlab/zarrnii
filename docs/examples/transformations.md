# Transformations

This section covers spatial transformations in ZarrNii, including affine transforms and displacement fields.

## Overview

ZarrNii provides support for spatial transformations through the `AffineTransform` and `DisplacementTransform` classes. These can be used to apply transformations
derived from performing ANTS registration on downsampled (e.g. level > 3)  images, then applied to higher resolution (e.g. level < 3) images. ZarrNii performs
these computations in a block-wise manner using Dask, upsampling the displacement field for a block before applying it. You can also provide a sequence of transforms
to perform composition of transformations to resample in a single step.

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

### Resampling Large Images into Template Space

`apply_transform` maps each output (reference) voxel through the transforms
into the floating image, so the transforms go from reference (template)
physical space to floating (subject) physical space, as with
`antsApplyTransforms`. It is lazy and block-wise, and works on whole-brain
light-sheet data:

```python
from zarrnii import AffineTransform, DisplacementTransform, ZarrNii

spim = ZarrNii.from_ome_zarr("sub-01_SPIM.ome.zarr", level=0, channel_labels=["PI"])
template = ZarrNii.from_nifti("template_25um.nii.gz", axes_order="ZYX")

# template -> subject: warp (in template space) first, then the affine
warp = DisplacementTransform.from_nifti("template_to_subject_warp.nii.gz")
affine = AffineTransform.from_txt("template_to_subject_affine_ras.txt")

# 10x finer than the template (2.5 um) over the template's field of view
out = spim.apply_transform(warp, affine, ref_znimg=template, ref_zoom=10, chunks=128)
out.to_ome_zarr("sub-01_space-template_res-2p5um_SPIM.ome.zarr")
```

Useful options:

- `ref_zoom`: zoom factor for the reference grid (see `ZarrNii.ref_grid`),
  keeping its field of view.
- `interpolation`: `"linear"` (default), `"nearest"` for label images, or
  `"cubic"`.
- `dtype`: output dtype (defaults to the input's; integers are rounded).
- `chunks`: output chunk size; each block reads the floating region it maps to,
  so 128 is a good default.
- `coord_step` / `coord_tol`: the transforms are evaluated on a lattice of
  control points and interpolated in between. The step is chosen
  automatically so the error stays below `coord_tol` floating voxels (default
  0.1); affine-only chains are exact, and `coord_step=1` evaluates every
  voxel.
- `io`: `"graph"` (default) shares input chunks between output blocks, so each
  is read once; `"direct"` makes every output block read its own inputs, which
  keeps blocks independent (good for `dask.distributed` and object stores) at
  the cost of re-reading overlaps.

Output blocks that fall outside the floating image are filled without reading
any data, and the floating image may carry lazy operations (e.g. bias-field
correction or channel selection).

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

- [API Reference](../reference/index.md) for detailed method documentation
- [Downsampling and Upsampling](downsampling.md) for resolution change operations
- [Working with Zarr and NIfTI](zarr_nifti.md) for basic format operations
