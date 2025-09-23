# Atlas Module Example

This example demonstrates how to use the atlas functionality in ZarrNii for working with brain atlases and performing region-of-interest (ROI) analysis.

## Basic Atlas Usage

### Loading an Atlas from Files

```python
from zarrnii import Atlas
import numpy as np

# Load atlas from BIDS format files
atlas = Atlas.from_files(
    dseg_path="path/to/atlas_dseg.nii.gz",
    labels_path="path/to/atlas_dseg.tsv"
)

print(f"Atlas loaded: {atlas}")
print(f"Number of regions: {len(atlas.region_labels)}")
print(f"Region names: {atlas.region_names[:5]}...")  # First 5 regions
```

### Creating an Atlas from Existing Data

```python
from zarrnii import ZarrNii, Atlas, AffineTransform
import pandas as pd

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
atlas = Atlas(dseg=dseg, labels_df=labels_df)
```

## Region Analysis

### Getting Region Information

```python
# Get information about a specific region
region_info = atlas.get_region_info(2)  # Hippocampus
print(f"Region: {region_info['name']}")
print(f"Abbreviation: {region_info['abbreviation']}")

# Get all region names and labels
for label, name in zip(atlas.region_labels, atlas.region_names):
    print(f"Label {label}: {name}")
```

### Creating Region Masks

```python
# Get binary mask for hippocampus
hippocampus_mask = atlas.get_region_mask(2)
print(f"Hippocampus mask shape: {hippocampus_mask.shape}")

# Save mask as NIfTI
hippocampus_mask.to_nifti("hippocampus_mask.nii.gz")
```

### Calculating Region Volumes

```python
# Calculate volume for all regions
volumes = {}
for label in atlas.region_labels:
    volume = atlas.get_region_volume(label)
    name = atlas.get_region_info(label)['name']
    volumes[name] = volume
    print(f"{name}: {volume:.2f} mm³")
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

# Combine results
combined_results = stats["mean"][["index", "name", "abbreviation"]].copy()
for func in ["mean", "std", "min", "max"]:
    combined_results[f"{func}_value"] = stats[func][f"{func}_value"]

print(combined_results.head())
```

### Analyzing Specific Regions Only

```python
# Analyze only cortical regions (assuming labels 1-3 are cortical)
cortical_regions = [1, 2, 3]
cortical_results = atlas.aggregate_image_by_regions(
    image, 
    aggregation_func="mean",
    regions=cortical_regions
)

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

## Summary Statistics

### Getting Atlas Overview

```python
# Get summary statistics for all regions
summary = atlas.get_summary_statistics()
print("Atlas Summary:")
print(summary[['name', 'volume_mm3', 'voxel_count']].head())

# Total brain volume (excluding background)
total_volume = summary[summary['index'] != 0]['volume_mm3'].sum()
print(f"Total brain volume: {total_volume:.2f} mm³")
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
# Load atlas at different resolution levels
atlas = Atlas.from_files(
    dseg_path="atlas.ome.zarr",
    labels_path="atlas_dseg.tsv",
    level=2  # Use downsampled level
)

# This allows processing at different scales for efficiency
```

### Custom Column Names

```python
# Work with custom column naming conventions
atlas = Atlas.from_files(
    dseg_path="custom_atlas.nii.gz",
    labels_path="custom_labels.tsv",
    label_column="region_id",
    name_column="region_name",
    abbrev_column="short_name"
)
```

### Error Handling

```python
try:
    atlas = Atlas.from_files("nonexistent.nii.gz", "nonexistent.tsv")
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
from zarrnii import ZarrNii, Transform

# Load atlas and image
atlas = Atlas.from_files("atlas_dseg.nii.gz", "atlas_dseg.tsv")
image = ZarrNii.from_ome_zarr("microscopy_data.ome.zarr", level=1)

# Apply spatial transformation to align image to atlas
transform = Transform.affine_ras_from_txt("subject_to_atlas.txt")
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