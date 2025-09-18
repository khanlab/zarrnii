# Working with Imaris Files

**ZarrNii** provides seamless support for reading and writing Imaris (.ims) files, enabling integration with microscopy workflows that use Imaris for image analysis and visualization.

!!! note "Optional Dependency"
    Imaris support requires the optional `imaris` dependency. Install it with:
    ```bash
    pip install zarrnii[imaris]
    ```

!!! info "PyImarisWriter Integration"
    ZarrNii automatically uses **PyImarisWriter** when available for better Imaris compatibility. This creates files that are fully compatible with Imaris software. If PyImarisWriter libraries are not available, ZarrNii falls back to a custom HDF5 implementation.
    
    For best results with Imaris software:
    1. Install PyImarisWriter: `pip install PyImarisWriter`
    2. Install Imaris SDK libraries (requires Imaris installation)
    
    Without these, ZarrNii will still work but the created files may have limited Imaris compatibility.

## Loading Imaris Files

### Basic Loading

```python
from zarrnii import ZarrNii

# Load an Imaris file
znimg = ZarrNii.from_imaris("microscopy_data.ims")

print(f"Data shape: {znimg.darr.shape}")
print(f"Spacing: {znimg.get_zooms()}")
```

### Selecting Specific Data

Imaris files can contain multiple resolution levels, timepoints, and channels. You can specify which to load:

```python
# Load specific resolution level, timepoint, and channel
znimg = ZarrNii.from_imaris(
    "microscopy_data.ims",
    level=1,        # Resolution level (0 = full resolution)
    timepoint=5,    # Time point
    channel=0       # Channel index
)

# Specify axes order and orientation
znimg = ZarrNii.from_imaris(
    "microscopy_data.ims",
    axes_order="ZYX",     # Spatial axes order
    orientation="RAS"     # Coordinate system orientation
)
```

## Saving to Imaris Format

### Basic Saving

```python
import numpy as np
import dask.array as da

# Create or load data
data = np.random.rand(64, 128, 96).astype(np.float32)
darr = da.from_array(data[np.newaxis, ...], chunks="auto")
znimg = ZarrNii.from_darr(darr, spacing=[2.0, 1.0, 1.0])

# Save to Imaris format
output_path = znimg.to_imaris("output_data.ims")
print(f"Saved to: {output_path}")
```

### Compression Options

```python
# Save with specific compression settings
znimg.to_imaris(
    "compressed_data.ims",
    compression="gzip",      # Compression method
    compression_opts=6       # Compression level (0-9)
)
```

## Format Conversions

### Imaris to NIfTI

```python
# Load from Imaris and save as NIfTI
znimg = ZarrNii.from_imaris("microscopy_data.ims")
znimg.to_nifti("converted_data.nii.gz")
```

### NIfTI to Imaris

```python
# Load from NIfTI and save as Imaris
znimg = ZarrNii.from_nifti("brain_scan.nii.gz")
znimg.to_imaris("brain_scan.ims")
```

### Round-trip Processing

```python
# Load, process, and save back to Imaris
znimg = ZarrNii.from_imaris("original_data.ims")

# Apply transformations
cropped = znimg.crop((10, 10, 10), (100, 100, 100))
downsampled = cropped.downsample(level=2)

# Save processed result
downsampled.to_imaris("processed_data.ims")
```

## Integration with Other Formats

### Multi-format Workflow

```python
# Load from Imaris
microscopy_data = ZarrNii.from_imaris("confocal_stack.ims")

# Convert to OME-Zarr for analysis
microscopy_data.to_ome_zarr("analysis_data.ome.zarr")

# Load analysis results and convert back
results = ZarrNii.from_ome_zarr("analysis_results.ome.zarr")
results.to_imaris("final_results.ims")
```

## Understanding Imaris File Structure

Imaris files use HDF5 format with a specific hierarchy:

```
MyData.ims
├── DataSet/
│   └── ResolutionLevel 0/
│       └── TimePoint 0/
│           └── Channel 0/
│               └── Data          # The actual image data
├── DataSetInfo/
│   ├── Image/                   # Image metadata
│   └── Channel 0/               # Channel information
└── DataSetTimes/
    └── Time                     # Temporal information
```

**ZarrNii** handles this structure automatically, extracting spatial metadata and presenting a unified interface consistent with other supported formats.

## Imaris Compatibility Options

ZarrNii provides two approaches for writing Imaris files:

### PyImarisWriter (Recommended)
When PyImarisWriter and Imaris SDK libraries are available, ZarrNii automatically uses PyImarisWriter for maximum compatibility:

```python
# Automatically uses PyImarisWriter when available
znimg.to_imaris("output.ims", compression="gzip")
```

**Advantages:**
- Maximum compatibility with Imaris software
- Proper handling of all Imaris metadata
- Support for advanced compression options

**Requirements:**
- PyImarisWriter package: `pip install PyImarisWriter`
- Imaris SDK libraries (requires Imaris installation)

### HDF5 Fallback
When PyImarisWriter is not available, ZarrNii falls back to a custom HDF5 implementation:

```python
# Uses HDF5 fallback when PyImarisWriter unavailable
znimg.to_imaris("output.ims", compression="gzip")
```

**Advantages:**
- Works without additional dependencies
- Basic Imaris file structure
- Good for data exchange and storage

**Limitations:**
- May have limited compatibility with Imaris software
- Simpler metadata handling

## Best Practices

### Memory Management

For large Imaris files, consider using chunked processing:

```python
# Load with specific chunking strategy
znimg = ZarrNii.from_imaris("large_dataset.ims", chunks=(1, 64, 64, 64))

# Process in chunks to avoid memory issues
cropped = znimg.crop((0, 0, 0), (500, 500, 500))  # Crop first
downsampled = cropped.downsample(level=2)          # Then downsample
```

### Metadata Preservation

**ZarrNii** automatically extracts and preserves spatial metadata from Imaris files:

```python
znimg = ZarrNii.from_imaris("calibrated_data.ims")

# Access spatial information
print(f"Voxel spacing: {znimg.get_zooms()}")
print(f"Origin: {znimg.get_origin()}")
print(f"Orientation: {znimg.orientation}")
```

### Error Handling

```python
try:
    znimg = ZarrNii.from_imaris("data.ims", level=5)
except ValueError as e:
    print(f"Invalid parameters: {e}")
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install zarrnii[imaris]")
```

## Example: Complete Microscopy Processing Pipeline

```python
from zarrnii import ZarrNii

def process_microscopy_data(input_path, output_path):
    """Complete processing pipeline for microscopy data."""
    
    # Load original Imaris data
    print("Loading Imaris data...")
    znimg = ZarrNii.from_imaris(input_path)
    
    # Apply processing steps
    print("Processing data...")
    
    # 1. Crop to region of interest
    cropped = znimg.crop((50, 50, 50), (400, 400, 300))
    
    # 2. Downsample for faster processing
    downsampled = cropped.downsample(level=1)
    
    # 3. Save intermediate result as OME-Zarr for analysis
    downsampled.to_ome_zarr("intermediate_analysis.ome.zarr")
    
    # 4. Save final result as Imaris
    print("Saving processed data...")
    downsampled.to_imaris(output_path)
    
    print(f"Processing complete. Output saved to: {output_path}")
    return downsampled

# Run the pipeline
result = process_microscopy_data("raw_confocal.ims", "processed_confocal.ims")
```

This example demonstrates **ZarrNii's** ability to seamlessly integrate Imaris files into broader image processing workflows while maintaining spatial accuracy and metadata consistency.