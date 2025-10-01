# Multi-Resolution Plugin Architecture

This section covers the multi-resolution plugin architecture that enables efficient processing where algorithms are computed at low resolution and applied to full resolution data.

## Overview

The scaled processing plugin architecture in ZarrNii allows for efficient multi-resolution operations. This is particularly useful for algorithms that can be computed efficiently at lower resolution and then applied to the full-resolution data. Common use cases include:

- Bias field correction
- Background estimation
- Denoising operations
- Global intensity normalization

## Plugin Interface

All scaled processing plugins must inherit from `ScaledProcessingPlugin` and implement two key methods:

### `lowres_func(lowres_array: np.ndarray) -> np.ndarray`

This function processes the downsampled data and returns a result that will be upsampled and applied to the full-resolution data.

### `highres_func(fullres_array: dask.array, upsampled_output: dask.array) -> dask.array`

This function receives the full-resolution dask array and the upsampled output (already upsampled to match the full-resolution shape), and applies the operation blockwise. The upsampling is handled internally by the `apply_scaled_processing` method.

## Basic Usage

```python
from zarrnii import ZarrNii, GaussianBiasFieldCorrection

# Load your data
znimg = ZarrNii.from_nifti("path/to/image.nii")

# Apply bias field correction
corrected = znimg.apply_scaled_processing(
    GaussianBiasFieldCorrection(sigma=5.0),
    downsample_factor=4
)

# Save result
corrected.to_nifti("corrected_image.nii")
```

## Built-in Plugins

### GaussianBiasFieldCorrection

A simple bias field correction plugin that estimates smooth bias fields using Gaussian smoothing at low resolution and applies correction by division.

```python
# Basic usage with default parameters
corrected = znimg.apply_scaled_processing(GaussianBiasFieldCorrection())

# Custom parameters
corrected = znimg.apply_scaled_processing(
    GaussianBiasFieldCorrection(sigma=3.0, mode='constant'),
    downsample_factor=8
)
```

Parameters:
- `sigma`: Standard deviation for Gaussian smoothing (default: 5.0)
- `mode`: Boundary condition for smoothing (default: 'reflect')

### N4BiasFieldCorrection

A more sophisticated bias field correction plugin that uses the N4 algorithm from ANTsPy for superior bias field estimation at low resolution, then applies correction by division.

**Installation**: Requires `antspyx` to be installed:
```bash
# Install with N4 support
pip install 'zarrnii[n4]'
# or install antspyx directly
pip install antspyx
```

```python
from zarrnii import N4BiasFieldCorrection

# Basic usage with default parameters
corrected = znimg.apply_scaled_processing(N4BiasFieldCorrection())

# Custom parameters for more control
corrected = znimg.apply_scaled_processing(
    N4BiasFieldCorrection(
        spline_spacing=150.0,
        convergence={'iters': [25, 25], 'tol': 0.001},
        shrink_factor=2
    ),
    downsample_factor=4
)
```

Parameters:
- `spline_spacing`: Spacing between knots for spline fitting (default: 200.0)
- `convergence`: Dictionary with 'iters' (list) and 'tol' (float) for convergence criteria (default: {'iters': [50], 'tol': 0.001})
- `shrink_factor`: Shrink factor for processing (default: 1)

## Creating Custom Plugins

You can create custom plugins by inheriting from `ScaledProcessingPlugin`:

```python
from zarrnii import ScaledProcessingPlugin
import numpy as np
import dask.array as da
from scipy import ndimage

class CustomPlugin(ScaledProcessingPlugin):
    def __init__(self, param1=1.0, **kwargs):
        super().__init__(param1=param1, **kwargs)
        self.param1 = param1

    def lowres_func(self, lowres_array: np.ndarray) -> np.ndarray:
        # Your low-resolution algorithm here
        # Example: compute some correction map
        correction_map = np.ones_like(lowres_array) * self.param1
        return correction_map

    def highres_func(self, fullres_array: da.Array, upsampled_output: da.Array) -> da.Array:
        # The upsampling is handled internally by apply_scaled_processing
        # This example shows a simple multiplication operation

        # Apply correction directly (both arrays are same size)
        result = fullres_array * upsampled_output

        return result

    @property
    def name(self) -> str:
        return "Custom Plugin"

    @property
    def description(self) -> str:
        return "A custom multi-resolution processing plugin"

# Use your custom plugin
result = znimg.apply_scaled_processing(CustomPlugin(param1=2.0))
```

## Advanced Usage

### Custom Downsampling Factors

```python
# Use different downsampling factors
result = znimg.apply_scaled_processing(
    GaussianBiasFieldCorrection(),
    downsample_factor=8  # 8x downsampling
)
```

### Custom Chunk Sizes

```python
# Specify custom chunk sizes for low-resolution processing
# The chunk_size parameter controls the chunking of low-resolution intermediate results
result = znimg.apply_scaled_processing(
    GaussianBiasFieldCorrection(),
    chunk_size=(1, 32, 32, 32)  # Used for low-res processing chunks
)
```

### Temporary File Options

The framework uses temporary OME-Zarr files to break up the dask computation graph for better performance. You can control this behavior:

```python
# Disable temporary file usage (may impact performance on large datasets)
result = znimg.apply_scaled_processing(
    GaussianBiasFieldCorrection(),
    use_temp_zarr=False
)

# Use custom temporary file location
result = znimg.apply_scaled_processing(
    GaussianBiasFieldCorrection(),
    temp_zarr_path="/custom/path/temp_processing.ome.zarr"
)
```

### Plugin Class vs Instance

```python
# Using plugin class (parameters passed as kwargs)
result1 = znimg.apply_scaled_processing(GaussianBiasFieldCorrection, sigma=3.0)

# Using plugin instance (parameters set during initialization)
plugin = GaussianBiasFieldCorrection(sigma=3.0)
result2 = znimg.apply_scaled_processing(plugin)
```

## Performance Considerations

1. **Downsampling Factor**: Higher factors reduce computation time but may reduce accuracy
2. **Chunk Sizes**: Optimize for your memory constraints and processing requirements
3. **Algorithm Complexity**: The `lowres_func` runs on small numpy arrays, while `highres_func` uses dask for scalability
4. **Temporary Files**: The default temporary OME-Zarr approach breaks up dask computation graphs for better performance on large datasets. Disable only if you have specific memory/disk constraints
5. **Dask-based Upsampling**: Uses ZarrNii's `.upsample()` method which leverages dask for efficient parallel upsampling

## Integration with Other Operations

The scaled processing plugins integrate seamlessly with other ZarrNii operations:

```python
# Chain operations
result = (znimg
    .apply_scaled_processing(GaussianBiasFieldCorrection())
    .downsample(level=1)
    .segment_otsu())

# Save to different formats
result.to_nifti("processed.nii")
result.to_ome_zarr("processed.ome.zarr")
```

## See Also

- [Segmentation Plugins](segmentation_example.md) for other plugin architectures
- [Downsampling and Upsampling](downsampling.md) for resolution operations
- [API Reference](../reference.md) for detailed method documentation
