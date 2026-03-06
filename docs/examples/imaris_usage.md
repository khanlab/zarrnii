# Working with Imaris Files

**ZarrNii** provides seamless support for reading and writing Imaris (.ims) files, enabling integration with microscopy workflows that use Imaris for image analysis and visualization.

!!! note "Optional Dependency"
    Imaris support requires the optional `imaris` dependency. Install it with:
    ```bash
    pip install zarrnii[imaris]
    ```

!!! info "HDF5-Based Imaris Support"
    ZarrNii provides robust Imaris (.ims) file support through a carefully crafted HDF5 implementation. The implementation follows the exact structure of correctly-formed Imaris files to ensure maximum compatibility with Imaris software.
    
    **Key Features:**
    - Reads all standard Imaris files with multiple channels, timepoints, and resolution levels
    - Creates Imaris-compatible files using the correct HDF5 structure and metadata
    - Handles both single and multi-channel data automatically
    - Preserves spatial metadata and supports histogram generation

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

# Save to Imaris format with automatic multi-resolution pyramid
output_path = znimg.to_imaris("output_data.ims")
print(f"Saved to: {output_path}")
```

!!! note "Multi-Resolution Pyramid"
    The `to_imaris()` method automatically generates multiple resolution levels (pyramid) for efficient visualization of large datasets. The number of levels is determined by the Imaris algorithm, which creates downsampled versions until the volume size drops below 1 MB. Each level uses memory-safe chunked processing with 16×256×256 (ZYX) chunks.

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

## Imaris File Format Support

ZarrNii provides comprehensive Imaris (.ims) file support through a robust HDF5-based implementation:

### File Creation and Compatibility
ZarrNii creates Imaris files that are compatible with Imaris software by following the exact structure found in correctly-formed Imaris files:

```python
# Create Imaris-compatible files
znimg.to_imaris("output.ims", compression="gzip")
```

**Key Features:**
- **Correct HDF5 structure**: Follows the exact directory hierarchy used by Imaris
- **Proper metadata**: Includes all necessary attributes for Imaris compatibility
- **Multi-channel support**: Automatically handles single and multi-channel data
- **Histogram generation**: Creates proper histograms for each channel
- **Compression support**: Supports various HDF5 compression options

**File Structure Created:**
- Top-level attributes matching Imaris format (ImarisVersion, DataSetDirectoryName, etc.)
- `DataSet/ResolutionLevel 0/TimePoint 0/Channel X/Data` hierarchy (with multiple ResolutionLevels)
- `DataSetInfo` group with channel metadata
- Proper histogram data for each channel

### Multi-Resolution Pyramids

ZarrNii automatically generates multi-resolution pyramids when saving to Imaris format:

```python
import dask.array as da
from zarrnii import ZarrNii

# Create a large dataset
shape = (256, 512, 384)  # Z, Y, X
data = da.random.random(shape, chunks=(16, 256, 256)).astype('float32')
data = data[None, ...]  # Add channel dimension

znimg = ZarrNii.from_darr(data, spacing=[1.0, 1.0, 1.0])

# Save with automatic pyramid generation
znimg.to_imaris("large_dataset.ims")
# Creates ResolutionLevel 0, 1, 2, 3, etc. automatically
```

**How Pyramid Levels Are Determined:**

The number of resolution levels follows the Imaris specification:
- For each dimension, downsample by 2 if `(10 × dimension)² > volume / dimension`
- Continue creating levels until total volume ≤ 1 MB
- Each level uses consistent 16×256×256 (ZYX) chunking

**Example Pyramid Structures:**

| Original Size | Levels | Level Sizes |
|--------------|--------|-------------|
| 64×128×96 | 2 | L0: 64×128×96, L1: 32×64×48 |
| 128×256×192 | 3 | L0: 128×256×192, L1: 64×128×96, L2: 32×64×48 |
| 256×512×384 | 4 | L0: 256×512×384, L1: 128×256×192, L2: 64×128×96, L3: 32×64×48 |

All pyramid generation is **memory-safe**, processing data in small chunks to handle arbitrarily large datasets.

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