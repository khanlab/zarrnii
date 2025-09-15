# Multiscale OME-Zarr

This section covers working with multi-resolution OME-Zarr datasets, including creation, manipulation, and optimization strategies.

## Overview

OME-Zarr supports multi-resolution image pyramids that enable efficient visualization and analysis at different scales. ZarrNii provides support for creating and reading multiscale datasets.

## Understanding OME-Zarr Multiscale

### Loading Multiscale Data

```python
from zarrnii import ZarrNii
import zarr

# Load different resolution levels of a multiscale OME-Zarr dataset
znimg_level0 = ZarrNii.from_ome_zarr("path/to/multiscale.zarr", level=0)  # Full resolution
znimg_level1 = ZarrNii.from_ome_zarr("path/to/multiscale.zarr", level=1)  # Half resolution
znimg_level2 = ZarrNii.from_ome_zarr("path/to/multiscale.zarr", level=2)  # Quarter resolution

print("Level 0 shape:", znimg_level0.darr.shape)
print("Level 1 shape:", znimg_level1.darr.shape)
print("Level 2 shape:", znimg_level2.darr.shape)
```

### Inspecting Available Levels

```python
# Use zarr directly to inspect the structure
store = zarr.open_group("path/to/multiscale.zarr", mode='r')
print("Available arrays:", list(store.keys()))

# Check shapes at each level
for key in sorted(store.keys()):
    if key.isdigit():
        array = store[key]
        print(f"Level {key}: shape {array.shape}, chunks {array.chunks}")
```

## Creating Multiscale OME-Zarr

### Basic Multiscale Creation

```python
# Load a single-resolution image
znimg = ZarrNii.from_nifti("path/to/highres.nii")

# Create multiscale OME-Zarr with default parameters
znimg.to_ome_zarr(
    "output_multiscale.zarr",
    max_layer=4  # Creates 4 downsampling levels
)
```

### Custom Multiscale Parameters

```python
# Create multiscale with custom settings
znimg.to_ome_zarr(
    "custom_multiscale.zarr",
    max_layer=6,  # More downsampling levels
    scaling_method=None  # Use default downsampling
)
```

## Working with Different Resolution Levels

### Processing at Different Scales

```python
# Load and process at different resolutions for different tasks

# Use low resolution for quick overview
thumbnail = ZarrNii.from_ome_zarr("data.zarr", level=3)
overview_stats = compute_statistics(thumbnail)

# Use medium resolution for analysis  
analysis_res = ZarrNii.from_ome_zarr("data.zarr", level=1)
feature_map = extract_features(analysis_res)

# Use full resolution for final processing
full_res = ZarrNii.from_ome_zarr("data.zarr", level=0)
final_result = apply_detailed_processing(full_res)
```

### Multi-Resolution Workflow

```python
# Progressive processing workflow
def progressive_analysis(zarr_path):
    # Start with thumbnail for parameter estimation
    low_res = ZarrNii.from_ome_zarr(zarr_path, level=3)
    parameters = estimate_parameters(low_res)
    
    # Refine on medium resolution
    med_res = ZarrNii.from_ome_zarr(zarr_path, level=1) 
    refined_params = refine_parameters(med_res, parameters)
    
    # Apply to full resolution
    full_res = ZarrNii.from_ome_zarr(zarr_path, level=0)
    final_result = process_with_params(full_res, refined_params)
    
    return final_result

result = progressive_analysis("multiscale_data.zarr")
```

## Channel and Time Series Support

### Multi-Channel Multiscale

```python
# Load specific channels from multiscale data
# Channel selection works with any resolution level
dapi_full = ZarrNii.from_ome_zarr("multi_channel.zarr", level=0, channels=[0])
gfp_thumbnail = ZarrNii.from_ome_zarr("multi_channel.zarr", level=3, channels=[1])

print("DAPI full resolution:", dapi_full.darr.shape)
print("GFP thumbnail:", gfp_thumbnail.darr.shape)
```

### Channel labels

```python
# If channel labels are present from OME metadata (e.g. for data from SPIMprep)
# You can select channels based on label
abeta_full = ZarrNii.from_ome_zarr("multi_channel.zarr", level=3, channel_labels=["Abeta"])
```

## Memory Management

### Efficient Loading

```python
# Load only what you need
# Zarr and Dask handle lazy loading automatically

# Load with appropriate chunking
znimg = ZarrNii.from_ome_zarr("large_dataset.zarr", level=1)

# Process in blocks to manage memory
def process_large_dataset(znimg):
    # Process data block by block
    result = znimg.darr.map_blocks(
        process_block,
        dtype=np.float32,
        drop_axis=None
    )
    return result

processed = process_large_dataset(znimg)
```

## Best Practices

### Choosing Resolution Levels

```python
# Guidelines for choosing appropriate resolution levels

def choose_resolution_level(task_type, data_size):
    """Choose optimal resolution level based on task and data size"""
    if task_type == "thumbnail":
        return 4  # Very low resolution for quick preview
    elif task_type == "segmentation":
        return 1  # Medium resolution for segmentation
    elif task_type == "measurement":
        return 0  # Full resolution for accurate measurements
    else:
        # Default to medium resolution
        return 2

# Use the function
level = choose_resolution_level("segmentation", data.shape)
znimg = ZarrNii.from_ome_zarr("data.zarr", level=level)
```


## Performance Tips

1. **Choose appropriate levels**: Use lower resolution levels for exploratory analysis
2. **Minimize data loading**: Only load the resolution level you actually need
3. **Leverage lazy evaluation**: Let Dask handle memory management automatically

## See Also

- [Downsampling and Upsampling](downsampling.md) for resolution change operations
- [Working with Zarr and NIfTI](zarr_nifti.md) for basic format operations  
- [API Reference](../reference.md) for detailed method documentation
