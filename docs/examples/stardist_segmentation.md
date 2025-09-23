# StarDist Segmentation Example

This example demonstrates how to use the StarDist segmentation plugin in ZarrNii for deep learning-based instance segmentation.

## Prerequisites

Install ZarrNii with StarDist dependencies:

```bash
pip install zarrnii[stardist]
```

This will install StarDist, TensorFlow, and the dask-relabel package for efficient processing of large images.

## Basic StarDist Segmentation

```python
import numpy as np
import dask.array as da
from zarrnii import ZarrNii

# Load or create your image data
# For this example, we'll create synthetic cell-like data
np.random.seed(42)
image_data = np.random.normal(0.2, 0.05, (1, 512, 512))  # Background

# Add some cell-like structures
for i in range(10):
    x, y = np.random.randint(50, 462, 2)
    size = np.random.randint(20, 40)
    xx, yy = np.meshgrid(np.arange(512), np.arange(512))
    mask = ((xx - x) ** 2 + (yy - y) ** 2) < (size ** 2)
    image_data[0][mask] = np.random.normal(0.8, 0.1, np.sum(mask))

# Create ZarrNii instance
darr = da.from_array(image_data, chunks=(1, 256, 256))
znimg = ZarrNii.from_darr(darr, axes_order="ZYX", orientation="RAS")

# Method 1: Using the convenience method
segmented = znimg.segment_stardist(
    model_name="2D_versatile_fluo",
    prob_thresh=0.5,
    nms_thresh=0.4,
    use_gpu=True,  # Use GPU if available
)

# Method 2: Using the plugin directly
from zarrnii.plugins.segmentation import StarDistSegmentation

plugin = StarDistSegmentation(
    model_name="2D_versatile_fluo",
    prob_thresh=0.6,
    nms_thresh=0.3,
    use_dask_relabeling=True,
    overlap=64,
)
segmented = znimg.segment(plugin)

print(f"Original shape: {znimg.shape}")
print(f"Segmented shape: {segmented.shape}")
print(f"Number of objects found: {np.max(segmented.data.compute())}")

# Save results
segmented.to_ome_zarr("stardist_segmented.ome.zarr")
```

## Using Pre-trained Models

StarDist provides several pre-trained models:

```python
# 2D models
models_2d = [
    "2D_versatile_fluo",  # Fluorescence images (recommended)
    "2D_versatile_he",    # H&E stained images
    "2D_paper_dsb2018",   # Data Science Bowl 2018
]

# 3D models  
models_3d = [
    "3D_demo",           # Demo 3D model
]

# Use different models
for model_name in models_2d:
    try:
        segmented = znimg.segment_stardist(model_name=model_name)
        print(f"Successfully segmented with {model_name}")
        print(f"Found {np.max(segmented.data.compute())} objects")
    except Exception as e:
        print(f"Failed with {model_name}: {e}")
```

## Custom Models

You can also use your own trained StarDist models:

```python
# Use custom model
segmented = znimg.segment_stardist(
    model_path="/path/to/your/custom/stardist/model",
    prob_thresh=0.4,
    nms_thresh=0.5,
)
```

## 3D Segmentation

For 3D data:

```python
# Create 3D test data
image_3d = np.random.normal(0.2, 0.05, (64, 256, 256))

# Add some 3D structures
for i in range(5):
    x, y, z = np.random.randint(20, 236), np.random.randint(20, 236), np.random.randint(10, 54)
    size = np.random.randint(8, 15)
    xx, yy, zz = np.meshgrid(np.arange(256), np.arange(256), np.arange(64))
    mask = ((xx - x) ** 2 + (yy - y) ** 2 + (zz - z) ** 2) < (size ** 2)
    image_3d[mask] = np.random.normal(0.8, 0.1, np.sum(mask))

# Create ZarrNii instance for 3D
darr_3d = da.from_array(image_3d, chunks=(32, 128, 128))
znimg_3d = ZarrNii.from_darr(darr_3d, axes_order="ZYX", orientation="RAS")

# 3D StarDist segmentation
segmented_3d = znimg_3d.segment_stardist(
    model_name="3D_demo",
    prob_thresh=0.5,
    use_dask_relabeling=True,
    overlap=32,
)

print(f"3D segmentation complete: {segmented_3d.shape}")
print(f"Objects found: {np.max(segmented_3d.data.compute())}")
```

## Large Image Processing with Dask Relabeling

For very large images, ZarrNii automatically uses dask_relabeling for efficient tiled processing:

```python
# Create large image (this would typically be loaded from file)
large_image = da.random.random((1, 4000, 4000), chunks=(1, 1000, 1000))
znimg_large = ZarrNii.from_darr(large_image, axes_order="ZYX", orientation="RAS")

# StarDist with automatic tiled processing
segmented_large = znimg_large.segment_stardist(
    model_name="2D_versatile_fluo",
    use_dask_relabeling=True,  # Enabled by default for large images
    overlap=128,  # Larger overlap for better edge handling
    prob_thresh=0.4,
)

# The processing will automatically:
# 1. Split the large image into overlapping tiles
# 2. Run StarDist on each tile independently
# 3. Merge overlapping labels efficiently
# 4. Return a single coherent label image

print(f"Large image processed: {segmented_large.shape}")
```

## Advanced Configuration

```python
# Advanced StarDist configuration
plugin = StarDistSegmentation(
    model_name="2D_versatile_fluo",
    prob_thresh=0.6,
    nms_thresh=0.3,
    use_gpu=True,
    normalize=True,  # Normalize input (recommended)
    use_dask_relabeling=True,
    overlap=96,  # Overlap size for dask_relabeling
)

# Get model information
model_info = plugin.get_model_info()
print("Model Info:")
for key, value in model_info.items():
    print(f"  {key}: {value}")

# Apply segmentation
segmented = znimg.segment(plugin)
```

## Multi-channel Images

StarDist typically works on single-channel images. For multi-channel data, ZarrNii automatically selects the first channel:

```python
# Multi-channel image (e.g., DAPI + GFP)
multichannel_data = np.random.rand(2, 512, 512)
darr_multi = da.from_array(multichannel_data, chunks=(1, 256, 256))
znimg_multi = ZarrNii.from_darr(darr_multi, axes_order="CYX", orientation="RAS")

# StarDist will use the first channel
segmented_multi = znimg_multi.segment_stardist(
    model_name="2D_versatile_fluo",
)

print(f"Multi-channel input: {znimg_multi.shape}")
print(f"Segmentation output: {segmented_multi.shape}")
```

## Performance Tips

1. **Use GPU**: Enable `use_gpu=True` if you have a compatible GPU
2. **Adjust thresholds**: Lower `prob_thresh` finds more objects, `nms_thresh` affects overlapping object handling
3. **Tile size**: For dask_relabeling, larger tiles are more accurate but use more memory
4. **Overlap**: Larger overlap improves edge handling but increases computation time
5. **Model selection**: Choose the model that best matches your data type

## Error Handling

```python
try:
    segmented = znimg.segment_stardist(model_name="2D_versatile_fluo")
except ImportError:
    print("StarDist not installed. Install with: pip install zarrnii[stardist]")
except Exception as e:
    print(f"Segmentation failed: {e}")
    # Fallback to simpler method
    segmented = znimg.segment_otsu()
```