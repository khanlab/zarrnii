# Multiscale OME-Zarr

This section covers working with multi-resolution OME-Zarr datasets, including creation, manipulation, and optimization strategies.

## Overview

OME-Zarr supports multi-resolution image pyramids that enable efficient visualization and analysis at different scales. ZarrNii provides comprehensive support for creating, reading, and manipulating these multiscale datasets.

## Understanding OME-Zarr Multiscale

### Pyramid Structure

```python
from zarrnii import ZarrNii
import zarr

# Load a multiscale OME-Zarr dataset
znimg = ZarrNii.from_ome_zarr("path/to/multiscale.zarr")

# Inspect pyramid structure
print("Available levels:", znimg.get_available_levels())
print("Shapes per level:", [znimg.get_level(i).darr.shape for i in range(znimg.num_levels)])
```

### Metadata Inspection

```python
# Access OME-Zarr metadata
metadata = znimg.get_ome_zarr_metadata()
print("Multiscale metadata:", metadata.get('multiscales', []))

# Check scaling factors between levels
for i in range(znimg.num_levels - 1):
    current = znimg.get_level(i)
    next_level = znimg.get_level(i + 1)
    factor = [c/n for c, n in zip(current.darr.shape, next_level.darr.shape)]
    print(f"Level {i} to {i+1} scaling factor: {factor}")
```

## Creating Multiscale OME-Zarr

### Basic Multiscale Creation

```python
# Load a single-resolution image
znimg = ZarrNii.from_nifti("path/to/highres.nii")

# Create multiscale OME-Zarr with default parameters
znimg.to_ome_zarr(
    "output_multiscale.zarr",
    max_layer=4,  # Create 5 levels (0-4)
    scaling_method="local_mean"
)
```

### Custom Scaling Parameters

```python
# Create multiscale with custom downsampling factors
znimg.to_ome_zarr(
    "custom_multiscale.zarr",
    max_layer=3,
    scaling_method="local_mean",
    downsample_factors=(2, 2, 1),  # Preserve Z resolution
    chunks=(128, 128, 64)  # Optimize chunk size
)
```

### Advanced Multiscale Options

```python
# Fine-tune multiscale creation
znimg.to_ome_zarr(
    "advanced_multiscale.zarr",
    max_layer=5,
    scaling_method="local_mean",
    storage_options={
        'compressor': zarr.Blosc(cname='zstd', clevel=3),
        'chunks': (256, 256, 128)
    },
    omero_metadata={
        'name': 'Brain Dataset',
        'channels': [{'label': 'DAPI', 'color': 'blue'}]
    }
)
```

## Working with Existing Multiscale Data

### Level Selection and Navigation

```python
# Load multiscale dataset
multiscale_img = ZarrNii.from_ome_zarr("path/to/multiscale.zarr")

# Access specific levels
full_res = multiscale_img.get_level(0)  # Highest resolution
thumbnail = multiscale_img.get_level(-1)  # Lowest resolution
mid_res = multiscale_img.get_level(2)  # Intermediate level

print("Full resolution shape:", full_res.darr.shape)
print("Thumbnail shape:", thumbnail.darr.shape)
```

### Selective Level Loading

```python
# Load only specific levels for memory efficiency
low_res_only = ZarrNii.from_ome_zarr(
    "path/to/multiscale.zarr", 
    level=3  # Load only level 3
)

# Load range of levels
mid_levels = ZarrNii.from_ome_zarr(
    "path/to/multiscale.zarr",
    level_range=(1, 3)  # Load levels 1-3
)
```

## Multi-Resolution Processing

### Progressive Analysis

```python
# Process data at multiple resolutions
multiscale_img = ZarrNii.from_ome_zarr("path/to/large_brain.zarr")

# Quick overview at low resolution
thumbnail = multiscale_img.get_level(-1)
rough_mask = thumbnail.create_tissue_mask(threshold=0.1)

# Refine analysis at medium resolution
med_res = multiscale_img.get_level(2)
refined_mask = med_res.refine_mask(rough_mask.upsample_to_level(2))

# Final processing at full resolution
full_res = multiscale_img.get_level(0)
final_result = full_res.apply_refined_mask(refined_mask.upsample_to_level(0))
```

### Region of Interest (ROI) Processing

```python
# Define ROI at low resolution for efficiency
low_res = multiscale_img.get_level(3)
roi_bounds = low_res.find_tissue_bounds()

# Scale ROI to full resolution
scale_factor = 2 ** 3  # Level 3 to level 0
full_res_bounds = [(b[0] * scale_factor, b[1] * scale_factor) for b in roi_bounds]

# Process only the ROI at full resolution
full_res = multiscale_img.get_level(0)
roi_data = full_res.crop_with_bounding_box(*full_res_bounds)
```

## Optimization Strategies

### Chunk Size Optimization

```python
# Analyze current chunk structure
multiscale_img = ZarrNii.from_ome_zarr("path/to/multiscale.zarr")
for level in range(multiscale_img.num_levels):
    level_data = multiscale_img.get_level(level)
    print(f"Level {level} chunks: {level_data.darr.chunks}")
```

### Memory-Efficient Access Patterns

```python
import dask.array as da

# Load with Dask for lazy evaluation
multiscale_img = ZarrNii.from_ome_zarr("path/to/large.zarr", use_dask=True)

# Process in blocks to manage memory
def process_blocks(img_level):
    """Process image in memory-efficient blocks"""
    chunks = img_level.darr.chunks
    for block in img_level.darr.blocks:
        # Process each block independently
        result = process_single_block(block)
        yield result

# Apply to specific level
results = list(process_blocks(multiscale_img.get_level(1)))
```

## Channel and Time Series Support

### Multi-Channel Multiscale

```python
# Create multiscale from multi-channel data
multi_channel = ZarrNii.from_nifti("path/to/multichannel.nii")  # Shape: (C, Z, Y, X)

# Create multiscale preserving channel dimension
multi_channel.to_ome_zarr(
    "multichannel_multiscale.zarr",
    max_layer=3,
    omero_metadata={
        'channels': [
            {'label': 'DAPI', 'color': 'blue'},
            {'label': 'GFP', 'color': 'green'},
            {'label': 'RFP', 'color': 'red'}
        ]
    }
)

# Access specific channels at different resolutions
dapi_full = multi_channel.get_level(0).get_channel(0)
gfp_thumbnail = multi_channel.get_level(-1).get_channel(1)
```

### Time Series Multiscale

```python
# Work with time-series multiscale data
time_series = ZarrNii.from_ome_zarr("path/to/timeseries_multiscale.zarr")

# Access specific timepoints and levels
t0_full = time_series.get_level(0).get_timepoint(0)
t10_mid = time_series.get_level(2).get_timepoint(10)

# Process across time at consistent resolution
level_2_timeseries = [time_series.get_level(2).get_timepoint(t) 
                      for t in range(time_series.num_timepoints)]
```

## Visualization and Interaction

### Level-Appropriate Visualization

```python
import matplotlib.pyplot as plt

multiscale_img = ZarrNii.from_ome_zarr("path/to/brain.zarr")

# Quick visualization with thumbnail
thumbnail = multiscale_img.get_level(-1)
plt.figure(figsize=(8, 6))
plt.imshow(thumbnail.darr[thumbnail.darr.shape[0]//2], cmap='gray')
plt.title(f'Overview (Level {multiscale_img.num_levels-1})')
plt.show()

# Detailed view with full resolution
full_res = multiscale_img.get_level(0)
roi = full_res.crop_with_bounding_box((100, 100, 50), (200, 200, 60))
plt.figure(figsize=(10, 8))
plt.imshow(roi.darr[5], cmap='gray')
plt.title('Detailed View (Full Resolution)')
plt.show()
```

### Interactive Exploration

```python
# Create interactive viewer data
def create_viewer_pyramid(zarr_path, output_dir):
    """Prepare multiscale data for web viewers"""
    img = ZarrNii.from_ome_zarr(zarr_path)
    
    viewer_metadata = {
        'levels': [],
        'pixel_size': img.voxel_size,
        'shape': img.darr.shape
    }
    
    for level in range(img.num_levels):
        level_img = img.get_level(level)
        viewer_metadata['levels'].append({
            'level': level,
            'shape': level_img.darr.shape,
            'scaling_factor': 2 ** level
        })
    
    return viewer_metadata
```

## Best Practices

### Creation Guidelines

1. **Choose appropriate level count**: Balance between storage and access efficiency
2. **Optimize chunk sizes**: Match your typical access patterns
3. **Select proper compression**: Balance compression ratio with decompression speed
4. **Include metadata**: Add comprehensive OME metadata for interoperability

### Access Patterns

1. **Start with overview**: Use low-resolution levels for initial analysis
2. **Progressive refinement**: Move to higher resolution only when needed
3. **Memory awareness**: Monitor memory usage with large datasets
4. **Parallel processing**: Leverage Dask for distributed processing

### Storage Considerations

```python
# Monitor storage efficiency
import os

def analyze_multiscale_storage(zarr_path):
    """Analyze storage usage per level"""
    total_size = 0
    level_sizes = []
    
    for level in range(len(os.listdir(zarr_path))):
        level_path = os.path.join(zarr_path, str(level))
        if os.path.exists(level_path):
            size = sum(os.path.getsize(os.path.join(level_path, f)) 
                      for f in os.listdir(level_path) if os.path.isfile(os.path.join(level_path, f)))
            level_sizes.append(size)
            total_size += size
            print(f"Level {level}: {size / (1024**3):.2f} GB")
    
    print(f"Total storage: {total_size / (1024**3):.2f} GB")
    return level_sizes
```

## Integration with Other Tools

### Napari Integration

```python
# Load multiscale data in napari
import napari

multiscale_img = ZarrNii.from_ome_zarr("path/to/multiscale.zarr")
pyramid_data = [multiscale_img.get_level(i).darr for i in range(multiscale_img.num_levels)]

viewer = napari.Viewer()
viewer.add_image(pyramid_data, multiscale=True, name="Multiscale Brain")
```

### BigDataViewer/ImageJ Integration

```python
# Export for BigDataViewer
multiscale_img.to_bdv_format("output.xml", "output.h5")
```

## See Also

- [Downsampling and Upsampling](downsampling.md) for resolution change operations
- [Working with Zarr and NIfTI](zarr_nifti.md) for basic format operations
- [API Reference](../reference.md) for detailed method documentation