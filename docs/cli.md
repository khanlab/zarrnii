# Command Line Interface (CLI)

ZarrNii provides convenient command-line tools for converting between OME-Zarr and NIfTI formats. These console scripts are simple wrappers around the main ZarrNii API, making it easy to perform conversions without writing Python code.

## Installation

The CLI scripts are automatically installed when you install ZarrNii:

```bash
pip install zarrnii
```

After installation, you'll have access to two console commands:
- `z2n` - Convert OME-Zarr to NIfTI or TIFF stack format
- `n2z` - Convert NIfTI to OME-Zarr

## z2n: OME-Zarr to NIfTI/TIFF Conversion

The `z2n` script converts OME-Zarr datasets to NIfTI format or TIFF stack format. TIFF stack export is particularly useful for compatibility with napari plugins like cellseg3d that don't support OME-Zarr multiscale data.

### Basic Usage

```bash
# NIfTI output
z2n input.ome.zarr output.nii.gz

# TIFF stack output (auto-detected by pattern)
z2n input.ome.zarr output_z{z:04d}.tif
```

### Output Format Selection

The output format is automatically detected based on the output filename:
- `.nii` or `.nii.gz` extensions → NIfTI format
- `.tif` or `.tiff` extensions with `{z}` pattern → TIFF stack format
- Other extensions default to NIfTI format

**TIFF Stack Format**: Each Z-slice is saved as a separate 2D TIFF file. This format is useful for:
- Napari plugins that don't support OME-Zarr (e.g., cellseg3d)
- Tools that expect individual image files
- Visual inspection of individual slices

⚠️ **Performance Note**: TIFF stack export loads all data into memory. Consider cropping or downsampling large datasets first.

### Options

```bash
z2n --help
```

```
usage: z2n [-h] [--level LEVEL] [--channels CHANNELS] 
           [--channel-labels CHANNEL_LABELS [CHANNEL_LABELS ...]]
           [--timepoints TIMEPOINTS] [--axes-order {ZYX,XYZ}] 
           [--orientation ORIENTATION] [--downsample-near-isotropic] 
           [--chunks CHUNKS] [--rechunk]
           input output

Convert OME-Zarr to NIfTI format

positional arguments:
  input                 Input OME-Zarr path or store
  output                Output NIfTI file (.nii or .nii.gz)

options:
  -h, --help            show this help message and exit
  --level LEVEL         Pyramid level to load (0 = highest resolution) (default: 0)
  --channels CHANNELS   Comma-separated channel indices to load (e.g., '0,2,3')
  --channel-labels CHANNEL_LABELS [CHANNEL_LABELS ...]
                        Channel names to load by label
  --timepoints TIMEPOINTS
                        Comma-separated timepoint indices to load (e.g., '0,1,2')
  --axes-order {ZYX,XYZ}
                        Spatial axes order for processing (default: ZYX)
  --orientation ORIENTATION
                        Anatomical orientation string (default: RAS)
  --downsample-near-isotropic
                        Apply near-isotropic downsampling
  --chunks CHUNKS       Chunk specification for dask arrays (default: auto)
  --rechunk             Rechunk data arrays
```

### Examples

```bash
# Basic conversion
z2n input.ome.zarr output.nii.gz

# Convert from ZIP store with specific pyramid level
z2n input.ome.zarr.zip output.nii.gz --level 1

# Select specific channels and reorder axes
z2n input.ome.zarr output.nii.gz --channels 0,2 --axes-order ZYX

# Change orientation and apply near-isotropic downsampling
z2n input.ome.zarr output.nii.gz --orientation LPI --downsample-near-isotropic

# TIFF Stack Export
# Save each Z-slice as a separate TIFF file for compatibility with napari plugins
z2n input.ome.zarr output_z{z:04d}.tif --tiff-channel 0

# Export to TIFF stack with custom pattern and settings
z2n input.ome.zarr slices/brain_{z:03d}.tiff --tiff-timepoint 0 --tiff-no-compress

# Export multi-channel data as multi-channel TIFFs (no channel selection)
z2n input.ome.zarr multichannel_z{z:04d}.tif

# Load specific timepoints
z2n input.ome.zarr output.nii.gz --timepoints 0,5,10

# Use channel labels instead of indices
z2n input.ome.zarr output.nii.gz --channel-labels DAPI GFP
```

## n2z: NIfTI to OME-Zarr Conversion

The `n2z` script converts NIfTI files to OME-Zarr format with multiscale pyramid generation. It's a wrapper around `ZarrNii.from_nifti().to_ome_zarr()`.

### Basic Usage

```bash
n2z input.nii.gz output.ome.zarr
```

### Options

```bash
n2z --help
```

```
usage: n2z [-h] [--chunks CHUNKS] [--axes-order {XYZ,ZYX}] [--name NAME] 
           [--as-ref] [--zooms ZOOMS] [--max-layer MAX_LAYER] 
           [--scale-factors SCALE_FACTORS]
           input output

Convert NIfTI to OME-Zarr format

positional arguments:
  input                 Input NIfTI file (.nii or .nii.gz)
  output                Output OME-Zarr path or store

options:
  -h, --help            show this help message and exit
  --chunks CHUNKS       Chunk specification for dask arrays (default: auto)
  --axes-order {XYZ,ZYX}
                        Spatial axes order for processing (default: XYZ)
  --name NAME           Name for the dataset
  --as-ref              Create as reference without loading data
  --zooms ZOOMS         Custom voxel sizes as comma-separated floats (e.g., '2.0,2.0,2.0')
  --max-layer MAX_LAYER
                        Maximum number of pyramid levels (default: 4)
  --scale-factors SCALE_FACTORS
                        Custom scale factors as comma-separated integers (e.g., '2,4,8')
```

### Examples

```bash
# Basic conversion
n2z input.nii.gz output.ome.zarr

# Create compressed ZIP store with more pyramid levels
n2z input.nii.gz output.ome.zarr.zip --max-layer 6

# Use custom voxel sizes and create as reference
n2z input.nii.gz output.ome.zarr --as-ref --zooms 1.5,1.5,2.5

# Specify custom scale factors for pyramid
n2z input.nii.gz output.ome.zarr --scale-factors 2,4,8,16

# Change axes order and add dataset name
n2z input.nii.gz output.ome.zarr --axes-order ZYX --name "brain_scan"

# Custom chunking specification
n2z input.nii.gz output.ome.zarr --chunks 64,64,32
```

## Common Workflows

### Round-trip Conversion

Convert from NIfTI to OME-Zarr and back to verify data integrity:

```bash
# Original NIfTI to OME-Zarr
n2z original.nii.gz intermediate.ome.zarr

# OME-Zarr back to NIfTI
z2n intermediate.ome.zarr final.nii.gz
```

### Working with ZIP Stores

OME-Zarr supports compressed ZIP format for efficient storage and sharing:

```bash
# Create compressed OME-Zarr
n2z input.nii.gz output.ome.zarr.zip

# Convert from compressed OME-Zarr
z2n input.ome.zarr.zip output.nii.gz
```

### Multiscale Processing

Create OME-Zarr with multiple resolution levels for efficient visualization:

```bash
# Create 6 pyramid levels
n2z input.nii.gz output.ome.zarr --max-layer 6

# Extract a lower resolution level
z2n output.ome.zarr downsampled.nii.gz --level 2
```

### Channel and Timepoint Selection

For multi-dimensional datasets:

```bash
# Select specific channels during conversion
z2n multichannel.ome.zarr selected_channels.nii.gz --channels 0,2,4

# Select specific timepoints
z2n timeseries.ome.zarr timepoint_subset.nii.gz --timepoints 0,10,20
```

## Error Handling

The CLI scripts provide informative error messages for common issues:

- **File not found**: Clear error message if input file doesn't exist
- **Invalid arguments**: Helpful guidance for malformed arguments
- **Conversion errors**: Detailed error information for debugging

Example error handling:

```bash
# Missing input file
$ z2n missing.ome.zarr output.nii.gz
Error: Input path 'missing.ome.zarr' does not exist

# Invalid zoom specification
$ n2z input.nii.gz output.ome.zarr --zooms 1.0,2.0
Error during conversion: Expected exactly 3 comma-separated floats
```

## Integration with Python API

The CLI scripts use the same underlying API as the Python interface, so you can easily switch between command-line and programmatic usage:

```python
# Equivalent Python code for: n2z input.nii.gz output.ome.zarr
from zarrnii import ZarrNii

znimg = ZarrNii.from_nifti("input.nii.gz")
znimg.to_ome_zarr("output.ome.zarr")
```

```python
# Equivalent Python code for: z2n input.ome.zarr output.nii.gz --level 1
from zarrnii import ZarrNii

znimg = ZarrNii.from_ome_zarr("input.ome.zarr", level=1)
znimg.to_nifti("output.nii.gz")
```

This consistency makes it easy to prototype with the CLI and then integrate the same functionality into your Python workflows.