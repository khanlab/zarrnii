# Near-Isotropic Downsampling

This example demonstrates the automatic near-isotropic downsampling feature, which is particularly useful for lightsheet microscopy data where Z resolution is often much finer than X/Y resolution.

## Problem: Anisotropic Voxels

Many biomedical imaging modalities produce datasets with anisotropic voxels where one dimension has much finer resolution than others:

```python
from zarrnii import ZarrNii

# Load anisotropic lightsheet data
znimg = ZarrNii.from_ome_zarr("lightsheet_data.ome.zarr")

print("Original scales:", znimg.scale)
# Output might show: {'z': 0.25, 'y': 1.0, 'x': 1.0}
# Z has 4x finer resolution than X/Y
```

## Solution: Automatic Downsampling

The `downsample_near_isotropic` parameter automatically identifies and corrects anisotropic voxels:

```python
# Load with automatic near-isotropic downsampling
znimg_isotropic = ZarrNii.from_ome_zarr(
    "lightsheet_data.ome.zarr", 
    downsample_near_isotropic=True
)

print("Isotropic scales:", znimg_isotropic.scale)
# Output: {'z': 1.0, 'y': 1.0, 'x': 1.0}
# All dimensions now have the same resolution

print("Shape comparison:")
print(f"Original:  {znimg.darr.shape}")
print(f"Isotropic: {znimg_isotropic.darr.shape}")
# Z dimension is reduced by the downsampling factor
```

## How It Works

The algorithm:

1. **Identifies the coarsest resolution** (largest scale value) among spatial dimensions
2. **Calculates downsampling factors** as powers of 2 for finer resolution dimensions  
3. **Applies selective downsampling** using the existing downsample method

```python
# For scales z=0.25, y=1.0, x=1.0:
# - Max scale = 1.0 (coarsest resolution)
# - Z ratio = 1.0 / 0.25 = 4.0
# - Z downsampling factor = 2^2 = 4
# - Result: z=1.0, y=1.0, x=1.0 (isotropic)
```

## Benefits

- **Improved processing efficiency**: Isotropic voxels work better with many algorithms
- **Consistent visualization**: Equal sampling in all dimensions for 3D rendering
- **Memory reduction**: Removes unnecessary oversampling in fine dimensions
- **Algorithm compatibility**: Many image processing algorithms assume isotropic voxels

## When to Use

This feature is most beneficial for:
- **Lightsheet microscopy** data with fine Z-sampling
- **High-resolution imaging** with anisotropic acquisition
- **Preprocessing** before analysis algorithms that assume isotropy
- **Visualization** where consistent sampling is desired

## Manual Control

For fine-grained control, you can still use manual downsampling:

```python
# Manual approach - downsample Z dimension by specific factor
znimg_manual = znimg.downsample(along_z=4, along_y=1, along_x=1)

# This gives the same result as downsample_near_isotropic=True
# but allows custom control over factors
```

## Performance Impact

The automatic downsampling:
- Uses the same efficient downsampling algorithm as manual methods
- Is applied lazily with Dask for memory efficiency  
- Reduces data size and subsequent processing time
- Preserves all metadata and coordinate system information

## See Also

- [Downsampling and Upsampling](downsampling.md) for general resolution operations
- [Basic Tasks](../walkthrough/basic_tasks.md) for getting started with transformations
- [API Reference](../reference.md) for detailed parameter documentation