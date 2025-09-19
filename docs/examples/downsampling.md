# Downsampling and Upsampling

This section covers resolution changes in ZarrNii, including downsampling for efficient processing and upsampling for analysis.

## Overview

ZarrNii provides methods for changing image resolution through downsampling and upsampling operations. These are essential for creating multi-resolution datasets and efficient processing workflows.

## Downsampling

### Basic Downsampling

```python
from zarrnii import ZarrNii

# Load a high-resolution dataset
znimg = ZarrNii.from_nifti("path/to/highres.nii")
print("Original shape:", znimg.darr.shape)

# Downsample by level (2^level reduction)
downsampled = znimg.downsample(level=2)
print("Downsampled shape:", downsampled.darr.shape)
```

### Custom Downsampling Factors

```python
# Downsample with specific factors for each axis
downsampled_custom = znimg.downsample(
    along_x=2, 
    along_y=2, 
    along_z=1  # 2x in X,Y; no change in Z
)
```

## Upsampling

### Basic Upsampling

```python
# Upsample by factors
upsampled = znimg.upsample(
    along_x=2, 
    along_y=2, 
    along_z=1  # 2x in X,Y; no change in Z
)

# Upsample to specific target shape
target_shape = (100, 200, 300)
upsampled_to_shape = znimg.upsample(to_shape=target_shape)
```

## Multi-Resolution Workflows

### Creating Image Pyramids

```python
# Create multiple resolution levels
pyramid_levels = []
current = znimg

for level in range(4):
    pyramid_levels.append(current)
    current = current.downsample(level=1)
    print(f"Level {level}: {current.darr.shape}")
```

### Working with OME-Zarr Multi-Resolution

```python
# Load multi-resolution OME-Zarr at specific level
znimg_level0 = ZarrNii.from_ome_zarr("path/to/multires.zarr", level=0)
znimg_level2 = ZarrNii.from_ome_zarr("path/to/multires.zarr", level=2)

# Create new multi-resolution OME-Zarr
znimg.to_ome_zarr(
    "output_multires.zarr",
    max_layer=4
)
```

### Automatic Near-Isotropic Downsampling

For datasets with anisotropic voxels (e.g., lightsheet microscopy with fine Z resolution), you can automatically downsample to create more isotropic voxels:

```python
# Load with automatic near-isotropic downsampling
znimg_isotropic = ZarrNii.from_ome_zarr(
    "path/to/anisotropic_data.ome.zarr", 
    downsample_near_isotropic=True
)

# Example: if original scales are z=0.25μm, y=1.0μm, x=1.0μm
# Z dimension gets downsampled by 4x to make scales isotropic:
# Result: z=1.0μm, y=1.0μm, x=1.0μm (isotropic voxels)
print("Original anisotropic scales:", znimg_normal.scale)
print("Near-isotropic scales:", znimg_isotropic.scale)
```

**How it works:**
- Identifies dimensions with finer resolution (smaller scale values)
- Downsamples by powers of 2 to match the coarsest resolution
- Works for any spatial dimension (X, Y, or Z)
- Only applies when significant anisotropy is detected

## Memory-Efficient Processing

```python
# Work with large images efficiently using Dask
# Downsampling is lazy and computed only when needed
downsampled = znimg.downsample(level=2)

# Compute result when needed
result = downsampled.darr.compute()
```

## Performance Tips

1. **Use appropriate chunk sizes**: For Dask arrays, ensure chunks are well-sized for your operations
2. **Lazy evaluation**: Downsampling operations are lazy and computed only when `.compute()` is called
3. **Memory management**: Use `.rechunk()` if needed to optimize chunk sizes for your workflow
4. **Level-based downsampling**: Use `level` parameter for consistent 2^level reductions

## See Also

- [Multiscale Processing](multiscale.md) for advanced multi-resolution workflows  
- [Working with Zarr and NIfTI](zarr_nifti.md) for basic format operations
- [API Reference](../reference.md) for detailed method documentation