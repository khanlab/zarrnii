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

The `isotropic` parameter automatically identifies and corrects anisotropic voxels:

```python
# Load with automatic near-isotropic downsampling (unbounded)
znimg_isotropic = ZarrNii.from_ome_zarr(
    "lightsheet_data.ome.zarr", 
    isotropic=True
)

print("Isotropic scales:", znimg_isotropic.scale)
# Output: {'z': 1.0, 'y': 1.0, 'x': 1.0}
# All dimensions now have the same resolution

print("Shape comparison:")
print(f"Original:  {znimg.darr.shape}")
print(f"Isotropic: {znimg_isotropic.darr.shape}")
# Z dimension is reduced by the downsampling factor
```

## Level-Constrained Downsampling

For more control, use an integer level to cap the maximum downsampling factor per axis:

```python
# Load with level=2 constraint (max downsampling factor = 2^2 = 4)
znimg_level2 = ZarrNii.from_ome_zarr(
    "lightsheet_data.ome.zarr",
    isotropic=2
)

print("Level 2 scales:", znimg_level2.scale)
# With original z=2.0, y=1.0, x=1.0:
# - x and y downsampled by 4x -> final: x=4.0, y=4.0
# - z downsampled by 2x -> final: z=4.0
# Result: {'z': 4.0, 'y': 4.0, 'x': 4.0} (isotropic)
```

## How It Works

The algorithm:

1. **Calculates target scale** based on the finest resolution and maximum downsampling factor (2^level)
2. **Distributes downsampling** across all dimensions to reach the target scale
3. **Applies selective downsampling** using the existing downsample method

```python
# For scales z=2.0, y=1.0, x=1.0 with level=2:
# - Max downsampling factor = 2^2 = 4
# - Target scale = min_scale (1.0) * max_factor (4) = 4.0
# - x: 1.0 -> 4.0 (downsample by 4x)
# - y: 1.0 -> 4.0 (downsample by 4x)
# - z: 2.0 -> 4.0 (downsample by 2x)
# - Result: z=4.0, y=4.0, x=4.0 (isotropic)
```

## Benefits

- **Improved processing efficiency**: Isotropic voxels work better with many algorithms
- **Consistent visualization**: Equal sampling in all dimensions for 3D rendering
- **Memory reduction**: Removes unnecessary oversampling in fine dimensions
- **Algorithm compatibility**: Many image processing algorithms assume isotropic voxels
- **Controlled downsampling**: Level constraint prevents excessive downsampling

## When to Use

This feature is most beneficial for:
- **Lightsheet microscopy** data with fine Z-sampling
- **High-resolution imaging** with anisotropic acquisition
- **Preprocessing** before analysis algorithms that assume isotropy
- **Visualization** where consistent sampling is desired
- **Pyramid generation** where specific downsampling levels are needed

## Backward Compatibility

The old `downsample_near_isotropic` parameter is still supported but deprecated:

```python
# Old syntax (deprecated but still works)
znimg_old = ZarrNii.from_ome_zarr(
    "lightsheet_data.ome.zarr",
    downsample_near_isotropic=True  # Shows deprecation warning
)

# New syntax (recommended)
znimg_new = ZarrNii.from_ome_zarr(
    "lightsheet_data.ome.zarr",
    isotropic=True  # or isotropic=2 for level constraint
)
```

## Manual Control

For fine-grained control, you can still use manual downsampling:

```python
# Manual approach - downsample Z dimension by specific factor
znimg_manual = znimg.downsample(along_z=4, along_y=1, along_x=1)

# This gives similar results but with explicit control over factors
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