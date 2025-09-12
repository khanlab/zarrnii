# Downsampling and Upsampling

This section covers resolution changes in ZarrNii, including downsampling for efficient processing and upsampling for analysis.

## Overview

ZarrNii provides flexible methods for changing image resolution through downsampling and upsampling operations. These are essential for creating multi-resolution datasets and efficient processing workflows.

## Downsampling

### Basic Downsampling

```python
from zarrnii import ZarrNii

# Load a high-resolution dataset
znimg = ZarrNii.from_nifti("path/to/highres.nii")
print("Original shape:", znimg.darr.shape)

# Downsample by level (2x reduction per level)
downsampled = znimg.downsample(level=2)
print("Downsampled shape:", downsampled.darr.shape)
```

### Custom Downsampling Factors

```python
# Downsample with specific factors for each axis
downsampled_custom = znimg.downsample(
    level=1,
    downsample_factors=(2, 2, 1)  # 2x in X,Y; no change in Z
)
```

### Downsampling Methods

```python
# Different downsampling methods
local_mean = znimg.downsample(level=2, scaling_method="local_mean")
nearest = znimg.downsample(level=2, scaling_method="nearest")

# For labeled data, use nearest neighbor to preserve labels
labels = ZarrNii.from_nifti("path/to/labels.nii")
downsampled_labels = labels.downsample(level=2, scaling_method="nearest")
```

## Upsampling

### Basic Upsampling

```python
# Upsample to increase resolution
upsampled = znimg.upsample(level=2)
print("Upsampled shape:", upsampled.darr.shape)
```

### Custom Upsampling Factors

```python
# Upsample with specific factors
upsampled_custom = znimg.upsample(
    level=1,
    upsample_factors=(2, 2, 1)  # 2x in X,Y; no change in Z
)
```

### Upsampling Methods

```python
# Different interpolation methods for upsampling
linear_upsample = znimg.upsample(level=2, scaling_method="linear")
cubic_upsample = znimg.upsample(level=2, scaling_method="cubic")

# For labels, use nearest neighbor
labels_upsampled = labels.upsample(level=2, scaling_method="nearest")
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
# Load multi-resolution OME-Zarr
znimg_multires = ZarrNii.from_ome_zarr("path/to/multires.zarr")

# Access specific resolution level
level_2 = znimg_multires.get_level(2)
print("Level 2 shape:", level_2.darr.shape)

# Create new multi-resolution OME-Zarr
znimg.to_ome_zarr(
    "output_multires.zarr",
    max_layer=4,
    scaling_method="local_mean"
)
```

## Efficient Processing Patterns

### Progressive Processing

```python
# Process at low resolution first, then refine
low_res = znimg.downsample(level=3)

# Perform computationally expensive operations on low-res
processed_low = low_res.apply_some_expensive_operation()

# Upsample results for fine-scale processing
upsampled_result = processed_low.upsample(level=3)
```

### Memory-Efficient Downsampling

```python
# Use Dask for memory-efficient processing of large images
import dask.array as da

# Load with Dask
znimg = ZarrNii.from_ome_zarr("path/to/large.zarr", use_dask=True)

# Downsample (computed lazily)
downsampled = znimg.downsample(level=2)

# Compute only when needed
result = downsampled.darr.compute()
```

## Resolution Matching

### Match Resolution Between Images

```python
# Load images with different resolutions
img1 = ZarrNii.from_nifti("path/to/img1.nii")  # High res
img2 = ZarrNii.from_nifti("path/to/img2.nii")  # Low res

print("Image 1 voxel size:", img1.voxel_size)
print("Image 2 voxel size:", img2.voxel_size)

# Downsample img1 to match img2's resolution
target_voxel_size = img2.voxel_size
downsampled_img1 = img1.resample_to_voxel_size(target_voxel_size)
```

### Resample to Target Image

```python
# Resample one image to match another's space
img1_resampled = img1.resample_like(img2)
print("Resampled shape:", img1_resampled.darr.shape)
```

## Quality Considerations

### Anti-Aliasing

```python
# Apply anti-aliasing filter before downsampling
from scipy import ndimage

# Pre-filter to reduce aliasing
sigma = 0.5  # Gaussian kernel standard deviation
filtered = znimg.apply_gaussian_filter(sigma)
downsampled = filtered.downsample(level=2)
```

### Optimal Scaling Methods

```python
# Choose scaling method based on data type
continuous_data = znimg.downsample(level=2, scaling_method="local_mean")
categorical_data = labels.downsample(level=2, scaling_method="nearest")

# For preserving sharp features
edge_preserving = znimg.downsample(level=2, scaling_method="local_mean", 
                                   preserve_range=True)
```

## Performance Tips

1. **Use appropriate chunk sizes**: For Dask arrays, ensure chunks are well-sized for your operations
2. **Lazy evaluation**: Combine multiple operations before computing results
3. **Memory management**: Monitor memory usage with large datasets
4. **Method selection**: Choose scaling methods appropriate for your data type

## Common Use Cases

### Whole Brain Processing

```python
# Typical workflow for whole brain lightsheet data
whole_brain = ZarrNii.from_ome_zarr("path/to/brain.zarr")

# Create analysis resolution
analysis_res = whole_brain.downsample(level=3)  # ~8x reduction

# Perform analysis on smaller data
results = analysis_res.analyze_connectivity()

# Upsample results back to original resolution if needed
full_res_results = results.upsample(level=3)
```

### Multi-Modal Registration

```python
# Register images at multiple resolutions for robustness
moving = ZarrNii.from_nifti("path/to/moving.nii")
fixed = ZarrNii.from_nifti("path/to/fixed.nii")

# Coarse registration at low resolution
moving_low = moving.downsample(level=2)
fixed_low = fixed.downsample(level=2)
coarse_transform = register_images(moving_low, fixed_low)

# Refine at higher resolution
moving_med = moving.downsample(level=1)
fixed_med = fixed.downsample(level=1)
refined_transform = refine_registration(moving_med, fixed_med, coarse_transform)

# Final registration at full resolution
final_transform = refine_registration(moving, fixed, refined_transform)
```

## See Also

- [Transformations](transformations.md) for spatial transformation operations
- [Multiscale OME-Zarr](multiscale.md) for working with multi-resolution datasets
- [API Reference](../reference.md) for detailed method documentation