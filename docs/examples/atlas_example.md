# Atlas Module Example

This example demonstrates how to use the atlas functionality in ZarrNii for working with brain atlases and performing region-of-interest (ROI) analysis.

## Atlas-Focused Functionality

ZarrNii functionality is **solely for atlases** (dseg.nii.gz + dseg.tsv files), providing:
- **Loading atlases** from local files or TemplateFlow
- **Region analysis** by index, name, or abbreviation
- **ROI aggregation** with multiple statistical functions
- **Feature mapping** to assign values back to regions
- **Format conversion** utilities (CSV, ITK-SNAP to TSV)

### Loading Atlases from Files

```python
from zarrnii import ZarrNiiAtlas

# Load atlas from BIDS-format files
atlas = ZarrNiiAtlas.from_files(
    dseg_path="atlas_dseg.nii.gz",
    labels_path="atlas_dseg.tsv"
)

# Get basic atlas information
print(f"Atlas shape: {atlas.dseg.shape}")
print(f"Number of regions: {len(atlas.labels_df)}")
print(f"Region labels: {atlas.labels_df[atlas.label_column].values}")
```

### Loading Atlases from TemplateFlow

```python
from zarrnii import get_atlas, get_template

# Load atlas directly from TemplateFlow (if available)
# atlas = get_atlas("MNI152NLin2009cAsym", "DKT", resolution=1)

# Or load template first
# template = get_template("MNI152NLin2009cAsym", "T1w", resolution=1)
# Note: These functions require templateflow to be installed
```

### Saving Atlases to TemplateFlow

```python
from zarrnii import save_atlas_to_templateflow

# Save atlas to TemplateFlow directory as BIDS-compliant files
# Requires templateflow to be installed
# template_dir = save_atlas_to_templateflow(atlas, "MyTemplate", "MyAtlas")
print(f"Atlas saved to: {template_dir}")
# Creates: tpl-MyTemplate_atlas-MyAtlas_dseg.nii.gz and .tsv files
```
- **Unified API**: Access built-in and remote templates through same TemplateFlow interface

```python
# Install templateflow extra for lazy loading
# pip install zarrnii[templateflow]

# Lazy loading - templates copy to TEMPLATEFLOW_HOME on first access
from zarrnii import get_builtin_template
template = get_builtin_template("placeholder")  # Copies to TemplateFlow on first call

# After lazy loading, both APIs work identically:
import templateflow.api as tflow
zarrnii_template = tflow.get("placeholder", suffix="SPIM")        # zarrnii built-in
remote_template = tflow.get("MNI152NLin2009cAsym", suffix="T1w")  # remote template

# Manual installation (if preferred over lazy loading)
from zarrnii import install_zarrnii_templates
results = install_zarrnii_templates()  # {'placeholder': True}
```

### Template Installation Behavior

**With `templateflow` extra installed:**
- ✅ **Lazy loading**: Templates copy to `$TEMPLATEFLOW_HOME` on first `get_builtin_template()` call
- ✅ **Proper setup**: Uses TemplateFlow's `@requires_layout` decorator to ensure directory structure
- ✅ **Unified API**: Use TemplateFlow API for both zarrnii and remote templates
- ✅ **Automatic**: No extra steps needed

**Without `templateflow` extra:**
- ✅ **Direct access**: `get_builtin_template()` works normally from zarrnii package
- ❌ **No lazy loading**: Templates stay in zarrnii package only
- ❌ **No unified API**: TemplateFlow functions unavailable

```python
# Check installation status
try:
    results = install_zarrnii_templates()
    print("TemplateFlow integration:", results)
except ImportError:
    print("TemplateFlow extra not installed - using direct access only")
```

### Backward Compatibility

The old atlas functions still work for convenience:

```python
from zarrnii import get_builtin_atlas, list_builtin_atlases

# These still work (equivalent to template system)
atlases = list_builtin_atlases() 
atlas = get_builtin_atlas("placeholder")  # Gets default atlas from default template
```

### Loading an Atlas from Files

```python
from zarrnii import ZarrNiiAtlas
import numpy as np

# Load atlas from BIDS format files
atlas = ZarrNiiAtlas.from_files(
    dseg_path="path/to/atlas_dseg.nii.gz",
    labels_path="path/to/atlas_dseg.tsv"
)

print(f"Atlas loaded: {atlas}")
print(f"Number of regions: {len(atlas.labels_df)}")
print(f"Region names: {atlas.labels_df[atlas.name_column].tolist()[:5]}...")  # First 5 regions
```

### Creating an Atlas from Existing Data

```python
from zarrnii import ZarrNii, ZarrNiiAtlas, AffineTransform
import pandas as pd
import numpy as np

# Create a simple segmentation image
shape = (64, 64, 64)
dseg_data = np.random.randint(0, 5, shape, dtype=np.int32)
dseg = ZarrNii.from_darr(dseg_data, affine=AffineTransform.identity())

# Create lookup table
labels_df = pd.DataFrame({
    'index': [0, 1, 2, 3, 4],
    'name': ['Background', 'Cortex', 'Hippocampus', 'Thalamus', 'Cerebellum'],
    'abbreviation': ['BG', 'CTX', 'HIP', 'THA', 'CB']
})

# Create Atlas
atlas = ZarrNiiAtlas.create_from_dseg(dseg, labels_df)
```

## Region Analysis

### Getting Region Information

```python
# Get information about a specific region by index (traditional way)
region_info = atlas.get_region_info(2)  # Hippocampus
print(f"Region: {region_info['name']}")
print(f"Abbreviation: {region_info['abbreviation']}")

# NEW: Get information by region name
region_info = atlas.get_region_info("Hippocampus")
print(f"Hippocampus index: {region_info['index']}")

# NEW: Get information by abbreviation
region_info = atlas.get_region_info("HIP")
print(f"HIP full name: {region_info['name']}")

# Get all region names and labels
for label, name in zip(atlas.region_labels, atlas.region_names):
    print(f"Label {label}: {name}")
```

### Creating Region Masks

```python
# Get binary mask for hippocampus by index
hippocampus_mask = atlas.get_region_mask(2)
print(f"Hippocampus mask shape: {hippocampus_mask.shape}")

# NEW: Get mask by region name
hippocampus_mask = atlas.get_region_mask("Hippocampus")
print(f"Hippocampus mask shape: {hippocampus_mask.shape}")

# NEW: Get mask by abbreviation
hippocampus_mask = atlas.get_region_mask("HIP")
print(f"Hippocampus mask shape: {hippocampus_mask.shape}")

# Save mask as NIfTI
hippocampus_mask.to_nifti("hippocampus_mask.nii.gz")
```

### Calculating Region Volumes

```python
# Calculate volume for all regions
volumes = {}
for _, row in atlas.labels_df.iterrows():
    label = row[atlas.label_column]
    if label == 0:  # Skip background
        continue
    volume = atlas.get_region_volume(label)
    name = row[atlas.name_column]
    volumes[name] = volume
    print(f"{name}: {volume:.2f} mm³")

# Calculate volume by name or abbreviation
cortex_volume = atlas.get_region_volume("Cortex")
hippocampus_volume = atlas.get_region_volume("HIP")  # By abbreviation
print(f"Cortex volume: {cortex_volume:.2f} mm³")
print(f"Hippocampus volume: {hippocampus_volume:.2f} mm³")
```

## Region-Based Cropping

### Getting Bounding Boxes for Regions

The `get_region_bounding_box` method allows you to extract the spatial extents of one or more atlas regions in physical coordinates, which can then be used to crop images to focus on specific anatomical structures.

```python
from zarrnii import ZarrNiiAtlas

# Load an atlas
atlas = ZarrNiiAtlas.from_files("atlas_dseg.nii.gz", "atlas_dseg.tsv")

# Get bounding box for a single region by name
bbox_min, bbox_max = atlas.get_region_bounding_box("Hippocampus")
print(f"Hippocampus extends from {bbox_min} to {bbox_max} mm in physical space")

# Get bounding box for multiple regions
bbox_min, bbox_max = atlas.get_region_bounding_box(["Hippocampus", "Amygdala"])
print(f"Combined bounding box: {bbox_min} to {bbox_max}")

# Use regex to select regions
bbox_min, bbox_max = atlas.get_region_bounding_box(regex="Hippocampus.*")
print(f"All hippocampal subfields: {bbox_min} to {bbox_max}")
```

### Cropping Images to Regions

The bounding boxes returned by `get_region_bounding_box` are in physical coordinates and can be used directly with the `crop` method:

```python
from zarrnii import ZarrNii

# Load a brain image
image = ZarrNii.from_nifti("brain_image.nii.gz")

# Get bounding box for hippocampus
bbox_min, bbox_max = atlas.get_region_bounding_box("Hippocampus")

# Crop the image to the hippocampus region
cropped_image = image.crop(bbox_min, bbox_max, physical_coords=True)

# Save the cropped region
cropped_image.to_nifti("hippocampus_cropped.nii.gz")
```

### Example: Cropping and Saving as TIFF Stack

Here's a complete example showing how to crop around the hippocampus and save as a z-stack of TIFF images:

```python
from zarrnii import ZarrNii, ZarrNiiAtlas
import numpy as np
from PIL import Image
from pathlib import Path

# Load atlas and image
atlas = ZarrNiiAtlas.from_files("atlas_dseg.nii.gz", "atlas_dseg.tsv")
brain_image = ZarrNii.from_nifti("brain_mri.nii.gz")

# Get bounding box for hippocampus (or use regex for bilateral hippocampi)
bbox_min, bbox_max = atlas.get_region_bounding_box(regex="[Hh]ippocampus.*")

# Crop both the atlas and the image
cropped_atlas = atlas.crop(bbox_min, bbox_max, physical_coords=True)
cropped_image = brain_image.crop(bbox_min, bbox_max, physical_coords=True)

# Get the data (compute if dask array)
image_data = cropped_image.data
if hasattr(image_data, "compute"):
    image_data = image_data.compute()

# Create output directory
output_dir = Path("hippocampus_slices")
output_dir.mkdir(exist_ok=True)

# Save each z-slice as a TIFF
# Handle different axes orders (CZYX or CXYZ)
if cropped_image.axes_order == "ZYX":
    # Data is (C, Z, Y, X)
    channel_idx = 0 if image_data.ndim == 4 else None
    for z_idx in range(image_data.shape[1] if channel_idx is not None else image_data.shape[0]):
        if channel_idx is not None:
            slice_data = image_data[channel_idx, z_idx, :, :]
        else:
            slice_data = image_data[z_idx, :, :]
        
        # Normalize to 0-255 for TIFF
        slice_normalized = ((slice_data - slice_data.min()) / 
                           (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
        
        # Save as TIFF
        img = Image.fromarray(slice_normalized)
        img.save(output_dir / f"hippocampus_z{z_idx:03d}.tif")

print(f"Saved {image_data.shape[1 if channel_idx is not None else 0]} slices to {output_dir}")
```

### Cropping Multiple Regions

You can easily create crops for multiple regions of interest:

```python
# Define regions of interest
regions_of_interest = {
    "hippocampus": "Hippocampus",
    "amygdala": "Amygdala",
    "cortical": "cortex.*",  # regex pattern
}

# Crop and save each region
for region_name, region_pattern in regions_of_interest.items():
    # Get bounding box (use regex if pattern contains special characters)
    if any(char in region_pattern for char in ["*", ".", "[", "]"]):
        bbox_min, bbox_max = atlas.get_region_bounding_box(regex=region_pattern)
    else:
        bbox_min, bbox_max = atlas.get_region_bounding_box(region_pattern)
    
    # Crop image
    cropped = brain_image.crop(bbox_min, bbox_max, physical_coords=True)
    
    # Save
    cropped.to_nifti(f"{region_name}_cropped.nii.gz")
    print(f"Saved {region_name} crop: shape={cropped.shape}")
```

## Image Analysis with Atlas

### Aggregating Image Values by Regions

```python
# Load an image for analysis (e.g., a functional or structural image)
image = ZarrNii.from_nifti("functional_image.nii.gz")

# Make sure image and atlas have the same shape and alignment
# (In practice, you might need to resample/register the image to atlas space)

# Aggregate image values by atlas regions
results = atlas.aggregate_image_by_regions(
    image, 
    aggregation_func="mean"  # Can be "mean", "sum", "std", "median", "min", "max"
)

print("Mean signal by region:")
for _, row in results.iterrows():
    print(f"{row['name']}: {row['mean_value']:.3f}")

# Save results to CSV
results.to_csv("roi_analysis_results.csv", index=False)
```

### Different Aggregation Functions

```python
# Calculate multiple statistics
stats = {}
for func in ["mean", "std", "min", "max"]:
    stats[func] = atlas.aggregate_image_by_regions(image, aggregation_func=func)

# Combine results (note: results have 'label' and 'name' columns by default)
combined_results = stats["mean"][["label", "name"]].copy()
for func in ["mean", "std", "min", "max"]:
    combined_results[f"{func}_value"] = stats[func][f"{func}_value"]

print(combined_results.head())
```

### Analyzing Specific Regions Only

```python
# Get results for all regions, then filter
all_results = atlas.aggregate_image_by_regions(image, aggregation_func="mean")

# Filter to specific regions (e.g., cortical regions with labels 1-3)
cortical_results = all_results[all_results["label"].isin([1, 2, 3])]

print("Cortical region analysis:")
print(cortical_results)
```

## Creating Feature Maps

### Mapping Statistical Values Back to Image Space

```python
# Create a feature DataFrame (e.g., from previous analysis)
feature_data = pd.DataFrame({
    'index': [1, 2, 3, 4],
    'activation_score': [2.5, 3.1, 1.8, 2.9],
    'p_value': [0.001, 0.0001, 0.05, 0.002]
})

# Create feature maps
activation_map = atlas.create_feature_map(
    feature_data, 
    feature_column='activation_score',
    background_value=0.0
)

p_value_map = atlas.create_feature_map(
    feature_data, 
    feature_column='p_value',
    background_value=1.0  # Non-significant p-value for background
)

# Save feature maps
activation_map.to_nifti("activation_map.nii.gz")
p_value_map.to_nifti("p_value_map.nii.gz")
```

## Atlas Summary Information

### Getting Atlas Overview

```python
# Get information about all regions
print("Atlas Summary:")
print(f"Total regions: {len(atlas.labels_df)}")
print(f"\nRegion list:")
for _, row in atlas.labels_df.iterrows():
    label = row[atlas.label_column]
    name = row[atlas.name_column]
    if label != 0:  # Skip background
        volume = atlas.get_region_volume(label)
        print(f"  {label}: {name} - {volume:.2f} mm³")

# Calculate total brain volume (excluding background)
total_volume = sum(
    atlas.get_region_volume(row[atlas.label_column])
    for _, row in atlas.labels_df.iterrows()
    if row[atlas.label_column] != 0
)
print(f"\nTotal brain volume: {total_volume:.2f} mm³")
```

## Lookup Table Conversion Utilities

### Convert CSV to BIDS TSV Format

```python
from zarrnii import import_lut_csv_as_tsv

# Convert a CSV lookup table to BIDS-compatible TSV
import_lut_csv_as_tsv(
    csv_path="atlas_lookup.csv",
    tsv_path="atlas_dseg.tsv",
    csv_columns=["abbreviation", "name", "index"]  # Specify column order
)
```

### Convert ITK-SNAP Label File to TSV

```python
from zarrnii import import_lut_itksnap_as_tsv

# Convert ITK-SNAP format to BIDS TSV
import_lut_itksnap_as_tsv(
    itksnap_path="atlas.txt",
    tsv_path="atlas_dseg.tsv"
)
```

## Advanced Usage

### Working with Multi-Resolution Data

```python
# Load atlas at different resolution levels (if using OME-Zarr format)
atlas = ZarrNiiAtlas.from_files(
    dseg_path="atlas.ome.zarr",
    labels_path="atlas_dseg.tsv",
    level=2  # Use downsampled level
)

# This allows processing at different scales for efficiency
```

### Custom Column Names

```python
# Create atlas with custom column naming conventions
# First load the data
dseg = ZarrNii.from_nifti("custom_atlas.nii.gz")
labels_df = pd.read_csv("custom_labels.tsv", sep="\t")

# Create atlas with custom column names
atlas = ZarrNiiAtlas.create_from_dseg(
    dseg, 
    labels_df,
    label_column="region_id",
    name_column="region_name",
    abbrev_column="short_name"
)
```

### Error Handling

```python
try:
    atlas = ZarrNiiAtlas.from_files("nonexistent.nii.gz", "nonexistent.tsv")
except FileNotFoundError as e:
    print(f"Atlas files not found: {e}")

try:
    results = atlas.aggregate_image_by_regions(wrong_sized_image)
except ValueError as e:
    print(f"Shape mismatch: {e}")
```

## Integration with Existing ZarrNii Workflows

```python
# Complete workflow: load, transform, analyze
from zarrnii import ZarrNii, ZarrNiiAtlas, AffineTransform

# Load atlas and image
atlas = ZarrNiiAtlas.from_files("atlas_dseg.nii.gz", "atlas_dseg.tsv")
image = ZarrNii.from_ome_zarr("microscopy_data.ome.zarr", level=1)

# Apply spatial transformation to align image to atlas
transform = AffineTransform.from_txt("subject_to_atlas.txt")
aligned_image = image.apply_transform(transform, ref_znimg=atlas.dseg)

# Perform ROI analysis
results = atlas.aggregate_image_by_regions(aligned_image, aggregation_func="mean")

# Save results
results.to_csv("roi_quantification.csv", index=False)

# Create and save feature map
feature_map = atlas.create_feature_map(results, "mean_value")
feature_map.to_ome_zarr("quantification_map.ome.zarr")
```

This atlas functionality provides a complete toolkit for working with brain atlases in the ZarrNii ecosystem, enabling sophisticated region-based analysis workflows that can scale from microscopy data to MRI volumes.