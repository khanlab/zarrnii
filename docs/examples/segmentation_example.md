# Segmentation Plugin Example

This example demonstrates how to use the segmentation plugin system in ZarrNii.

## Basic Otsu Segmentation

```python
import numpy as np
import dask.array as da
from zarrnii import ZarrNii, OtsuSegmentation

# Load or create your image data
# For this example, we'll create synthetic bimodal data
np.random.seed(42)
image_data = np.random.normal(0.2, 0.05, (1, 50, 100, 100))  # Background
image_data[0, 20:30, 40:60, 40:60] = np.random.normal(0.8, 0.05, (10, 20, 20))  # Foreground

# Create ZarrNii instance
darr = da.from_array(image_data, chunks=(1, 25, 50, 50))
znimg = ZarrNii.from_darr(darr, axes_order="ZYX", orientation="RAS")

# Method 1: Using the convenience method
segmented = znimg.segment_otsu(nbins=256)

# Method 2: Using the plugin directly
plugin = OtsuSegmentation(nbins=256)
segmented = znimg.segment(plugin)

# Method 3: Using plugin class with parameters
segmented = znimg.segment(OtsuSegmentation, nbins=128)

# The result is a new ZarrNii instance with binary segmentation
print(f"Original shape: {znimg.shape}")
print(f"Segmented shape: {segmented.shape}")
print(f"Segmented dtype: {segmented.data.dtype}")
print(f"Unique values: {np.unique(segmented.data.compute())}")

# Save segmented result as OME-Zarr
segmented.to_ome_zarr("segmented_image.ome.zarr")
```

## Custom Chunk Processing

For large datasets, you can control the chunk size for blockwise processing:

```python
# Segment with custom chunk size for memory efficiency
custom_chunks = (1, 10, 25, 25)
segmented = znimg.segment_otsu(chunk_size=custom_chunks)

# The segmentation will be applied block-wise using dask
result_data = segmented.data.compute()
```

## Creating Custom Segmentation Plugins

You can create your own segmentation plugins by inheriting from `SegmentationPlugin`:

```python
from zarrnii.plugins.segmentation import SegmentationPlugin
import numpy as np

class ThresholdSegmentation(SegmentationPlugin):
    """Simple threshold-based segmentation plugin."""
    
    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self.threshold = threshold
    
    def segment(self, image: np.ndarray, metadata=None) -> np.ndarray:
        """Apply threshold segmentation."""
        binary_mask = image > self.threshold
        return binary_mask.astype(np.uint8)
    
    @property
    def name(self) -> str:
        return "Threshold Segmentation"
    
    @property
    def description(self) -> str:
        return f"Simple thresholding at value {self.threshold}"

# Use your custom plugin
custom_plugin = ThresholdSegmentation(threshold=0.3)
segmented = znimg.segment(custom_plugin)
```

## Working with Multi-channel Images

The segmentation plugins automatically handle multi-channel images:

```python
# Create multi-channel test data
multichannel_data = np.random.rand(3, 50, 100, 100)  # 3 channels
multichannel_data[0, 20:30, 40:60, 40:60] += 0.5  # Add signal to first channel

darr = da.from_array(multichannel_data, chunks=(1, 25, 50, 50))
znimg = ZarrNii.from_darr(darr, axes_order="ZYX", orientation="RAS")

# Segment - will use first channel for threshold calculation
# but preserve all channel dimensions in output
segmented = znimg.segment_otsu()
print(f"Input shape: {znimg.shape}")       # (3, 50, 100, 100)
print(f"Output shape: {segmented.shape}")   # (3, 50, 100, 100)
```

## Integration with Existing Workflows

The segmentation plugins integrate seamlessly with other ZarrNii operations:

```python
# Complete workflow: load, downsample, segment, save
znimg = ZarrNii.from_ome_zarr("input_image.ome.zarr", level=1)

# Downsample for faster processing
downsampled = znimg.downsample(factors=2, spatial_dims=["z", "y", "x"])

# Apply segmentation
segmented = downsampled.segment_otsu(nbins=128)

# Crop to region of interest
bbox_min = (10, 20, 20)
bbox_max = (40, 80, 80)
cropped = segmented.crop_with_bounding_box(bbox_min, bbox_max)

# Save final result
cropped.to_ome_zarr("processed_segmentation.ome.zarr")
```