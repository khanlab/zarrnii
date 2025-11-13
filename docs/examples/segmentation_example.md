# Segmentation Plugin Example

This example demonstrates how to use the segmentation plugin system in ZarrNii. The plugin system is built on the [pluggy](https://pluggy.readthedocs.io/) framework, providing a flexible and extensible architecture for implementing custom segmentation algorithms.

## Plugin Architecture Overview

ZarrNii's segmentation plugins use pluggy hook specifications and implementations:
- **Hook specifications** define the interface that all plugins must implement
- **Hook implementations** (using `@hookimpl` decorator) provide the actual functionality
- Plugins can be used directly or registered with the plugin manager for discovery

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

You can create your own segmentation plugins by inheriting from `SegmentationPlugin` and implementing the required hook methods with the `@hookimpl` decorator:

```python
from zarrnii.plugins import SegmentationPlugin
from zarrnii.plugins.segmentation.base import hookimpl
import numpy as np

class ThresholdSegmentation(SegmentationPlugin):
    """Simple threshold-based segmentation plugin."""
    
    def __init__(self, threshold: float = 0.5, **kwargs):
        """Initialize the plugin.
        
        Args:
            threshold: Threshold value for segmentation
        """
        super().__init__(threshold=threshold, **kwargs)
        self.threshold = threshold
    
    @hookimpl
    def segment(self, image: np.ndarray, metadata=None) -> np.ndarray:
        """Apply threshold segmentation.
        
        Args:
            image: Input image as numpy array
            metadata: Optional metadata dictionary
            
        Returns:
            Binary segmentation mask
        """
        binary_mask = image > self.threshold
        return binary_mask.astype(np.uint8)
    
    @hookimpl
    def segmentation_plugin_name(self) -> str:
        """Return the name of the plugin."""
        return "Threshold Segmentation"
    
    @hookimpl
    def segmentation_plugin_description(self) -> str:
        """Return a description of the plugin."""
        return f"Simple thresholding at value {self.threshold}"

# Method 1: Direct usage with ZarrNii (recommended for simple cases)
custom_plugin = ThresholdSegmentation(threshold=0.3)
segmented = znimg.segment(custom_plugin)

# Method 2: Using the plugin manager (recommended for external plugins)
from zarrnii.plugins import get_plugin_manager

pm = get_plugin_manager()
plugin = ThresholdSegmentation(threshold=0.3)
pm.register(plugin)

# Now the plugin is available through the plugin manager
# and can be discovered by other tools
registered_plugins = pm.get_plugins()
print(f"Registered {len(registered_plugins)} plugins")
```

## External Segmentation Plugin Development

External plugins allow you to package and distribute your custom segmentation algorithms as separate Python packages. Here's a complete example:

### Step 1: Create Your Plugin Package Structure

```
my_segmentation_plugin/
├── pyproject.toml
└── my_segmentation_plugin/
    ├── __init__.py
    └── adaptive_threshold.py
```

### Step 2: Implement Your Segmentation Plugin

In `adaptive_threshold.py`:

```python
"""Adaptive threshold segmentation plugin."""
from zarrnii.plugins import SegmentationPlugin
from zarrnii.plugins.segmentation.base import hookimpl
import numpy as np
from skimage.filters import threshold_local

class AdaptiveThresholdSegmentation(SegmentationPlugin):
    """Adaptive thresholding segmentation for images with varying illumination."""
    
    def __init__(self, block_size=35, offset=10, method='gaussian', **kwargs):
        """Initialize adaptive threshold segmentation.
        
        Args:
            block_size: Size of pixel neighborhood for threshold calculation
            offset: Constant subtracted from weighted mean
            method: Method for computing threshold ('gaussian' or 'mean')
        """
        super().__init__(
            block_size=block_size,
            offset=offset,
            method=method,
            **kwargs
        )
        self.block_size = block_size
        self.offset = offset
        self.method = method
    
    @hookimpl
    def segment(self, image: np.ndarray, metadata=None) -> np.ndarray:
        """Segment image using adaptive thresholding.
        
        Args:
            image: Input image as numpy array
            metadata: Optional metadata (unused)
            
        Returns:
            Binary segmentation mask
        """
        if image.size == 0:
            raise ValueError("Input image is empty")
        
        if image.ndim < 2:
            raise ValueError("Input image must be at least 2D")
        
        # Work with 2D slice for threshold computation
        work_image = image
        if work_image.ndim > 2:
            # Use first channel/slice if multi-dimensional
            if work_image.ndim == 3 and work_image.shape[0] <= 4:
                work_image = work_image[0]
            elif work_image.ndim > 3:
                work_image = work_image.reshape(-1, *work_image.shape[-2:])[0]
        
        # Compute adaptive threshold
        threshold = threshold_local(
            work_image,
            block_size=self.block_size,
            offset=self.offset,
            method=self.method
        )
        
        # Apply threshold to original image
        binary_mask = image > threshold
        
        return binary_mask.astype(np.uint8)
    
    @hookimpl
    def segmentation_plugin_name(self) -> str:
        """Return the name of the plugin."""
        return "Adaptive Threshold Segmentation"
    
    @hookimpl
    def segmentation_plugin_description(self) -> str:
        """Return a description of the plugin."""
        return (
            f"Adaptive thresholding using {self.method} method "
            f"(block_size={self.block_size}, offset={self.offset})"
        )
```

### Step 3: Configure Package Discovery

In `__init__.py`:

```python
"""My Segmentation Plugin Package."""
from .adaptive_threshold import AdaptiveThresholdSegmentation

__all__ = ["AdaptiveThresholdSegmentation"]
```

In `pyproject.toml`:

```toml
[project]
name = "my-segmentation-plugin"
version = "0.1.0"
description = "Adaptive threshold segmentation plugin for ZarrNii"
dependencies = [
    "zarrnii>=0.1.0",
    "scikit-image>=0.21.0",
]

[project.entry-points."zarrnii.plugins"]
adaptive_threshold = "my_segmentation_plugin:AdaptiveThresholdSegmentation"
```

### Step 4: Use Your External Plugin

After installing (`pip install my-segmentation-plugin`):

```python
from zarrnii import ZarrNii
from my_segmentation_plugin import AdaptiveThresholdSegmentation

# Load your data
znimg = ZarrNii.from_ome_zarr("input.ome.zarr")

# Use the plugin directly
segmented = znimg.segment(
    AdaptiveThresholdSegmentation(block_size=51, offset=5)
)

# Or register with the plugin manager for discovery
from zarrnii.plugins import get_plugin_manager

pm = get_plugin_manager()
plugin = AdaptiveThresholdSegmentation(block_size=51, offset=5)
pm.register(plugin)

# Call hooks through the plugin manager
test_image = znimg.data.compute()
results = pm.hook.segment(image=test_image)

# Get plugin information
names = pm.hook.segmentation_plugin_name()
descriptions = pm.hook.segmentation_plugin_description()
print(f"Using: {names[0]} - {descriptions[0]}")
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