# ML Training: Patch Sampling and Cropping

This example demonstrates how to use `sample_region_patches()` and `crop_centered()` for machine learning workflows where you need to extract fixed-size training patches from anatomical regions defined in an atlas.

## Overview

The typical ML training workflow involves:
1. **Define regions** using a lower-resolution atlas
2. **Sample center coordinates** from those regions
3. **Extract fixed-size patches** at those centers from high-resolution data

This approach is particularly useful when:
- Your atlas is at a different resolution than your training data
- You need consistent patch dimensions for ML models (e.g., 256×256×256 voxels)
- You want to sample from specific anatomical regions

## Basic Workflow

### Step 1: Load Atlas and Sample Centers

```python
from zarrnii import ZarrNiiAtlas

# Load atlas (e.g., at 1mm resolution)
atlas = ZarrNiiAtlas.from_files(
    dseg_path="atlas_dseg.nii.gz",
    labels_path="atlas_dseg.tsv"
)

# Sample 100 centers from cortical regions
centers = atlas.sample_region_patches(
    n_patches=100,
    region_ids="Cortex",
    seed=42  # For reproducibility
)

print(f"Sampled {len(centers)} centers")
print(f"First center: {centers[0]} mm (physical coordinates)")
```

**Output:**
```
Sampled 100 centers
First center: (45.2, 67.8, 123.4) mm (physical coordinates)
```

### Step 2: Extract Fixed-Size Patches

```python
from zarrnii import ZarrNii

# Load high-resolution image (e.g., at 0.1mm resolution)
highres_image = ZarrNii.from_ome_zarr("microscopy_data.ome.zarr", level=0)

# Extract 256x256x256 voxel patches at each center
patches = highres_image.crop_centered(
    centers,
    patch_size=(256, 256, 256)  # Fixed size in voxels
)

print(f"Created {len(patches)} patches")
print(f"First patch shape: {patches[0].shape}")
print(f"All patches have consistent spatial dims: {all(p.shape[1:] == patches[0].shape[1:] for p in patches)}")
```

**Output:**
```
Created 100 patches
First patch shape: (1, 256, 256, 256)
All patches have consistent spatial dims: True
```

## Advanced Examples

### Multi-Region Sampling

Sample from multiple anatomical regions:

```python
# Sample from multiple regions using a list
centers = atlas.sample_region_patches(
    n_patches=50,
    region_ids=["Hippocampus", "Amygdala", "Cortex"],
    seed=42
)

# Or use regex patterns
cortical_centers = atlas.sample_region_patches(
    n_patches=100,
    regex=".*[Cc]ortex.*",  # Match any cortical regions
    seed=42
)
```

### Different Patch Sizes for Different Tasks

```python
# Large patches for context (512³ voxels)
large_patches = highres_image.crop_centered(centers[:10], patch_size=(512, 512, 512))

# Medium patches for training (256³ voxels)
training_patches = highres_image.crop_centered(centers, patch_size=(256, 256, 256))

# Small patches for fine details (128³ voxels)
detail_patches = highres_image.crop_centered(centers, patch_size=(128, 128, 128))
```

### Single Center Cropping

When you need just one patch, `crop_centered()` returns a single `ZarrNii` object (not a list):

```python
# Get one specific center
center = centers[0]

# Extract single patch
patch = highres_image.crop_centered(
    center,  # Single tuple, not a list
    patch_size=(256, 256, 256)
)

print(f"Single patch type: {type(patch)}")  # <class 'zarrnii.core.ZarrNii'>
print(f"Shape: {patch.shape}")
```

## Complete ML Training Pipeline

Here's a complete example showing data preparation for ML training:

```python
import numpy as np
from zarrnii import ZarrNii, ZarrNiiAtlas
from pathlib import Path

# Configuration
PATCH_SIZE = (256, 256, 256)
N_TRAIN_PATCHES = 1000
N_VAL_PATCHES = 200
OUTPUT_DIR = Path("ml_training_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Step 1: Load atlas and define regions of interest
atlas = ZarrNiiAtlas.from_files(
    dseg_path="atlas_1mm_dseg.nii.gz",
    labels_path="atlas_1mm_dseg.tsv"
)

# Step 2: Sample centers for training set
train_centers = atlas.sample_region_patches(
    n_patches=N_TRAIN_PATCHES,
    region_ids=["Gray-Matter", "Cortex"],
    seed=42
)

# Step 3: Sample centers for validation set
val_centers = atlas.sample_region_patches(
    n_patches=N_VAL_PATCHES,
    region_ids=["Gray-Matter", "Cortex"],
    seed=123  # Different seed for validation
)

# Step 4: Load high-resolution data
highres_image = ZarrNii.from_ome_zarr(
    "specimen_highres.ome.zarr",
    level=0  # Highest resolution
)

# Step 5: Extract training patches
print("Extracting training patches...")
train_patches = highres_image.crop_centered(train_centers, patch_size=PATCH_SIZE)

# Step 6: Extract validation patches
print("Extracting validation patches...")
val_patches = highres_image.crop_centered(val_centers, patch_size=PATCH_SIZE)

# Step 7: Save patches (example with one format)
print("Saving patches...")
for i, patch in enumerate(train_patches):
    patch.to_nifti(OUTPUT_DIR / f"train_patch_{i:04d}.nii.gz")

for i, patch in enumerate(val_patches):
    patch.to_nifti(OUTPUT_DIR / f"val_patch_{i:04d}.nii.gz")

print(f"Training patches: {len(train_patches)}")
print(f"Validation patches: {len(val_patches)}")
print(f"Patch size: {PATCH_SIZE}")
print(f"Output directory: {OUTPUT_DIR}")
```

## Integration with PyTorch DataLoader

Convert patches to numpy arrays for direct use with PyTorch:

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PatchDataset(Dataset):
    def __init__(self, patches, transform=None):
        """
        Args:
            patches: List of ZarrNii objects
            transform: Optional transforms to apply
        """
        self.patches = patches
        self.transform = transform
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        # Get patch data
        patch = self.patches[idx]
        
        # Convert to numpy array
        data = patch.data
        if hasattr(data, 'compute'):
            data = data.compute()
        
        # Convert to float32 and normalize
        data = data.astype(np.float32)
        
        # Remove channel dimension if present
        if data.shape[0] == 1:
            data = data[0]
        
        # Apply transforms if any
        if self.transform:
            data = self.transform(data)
        
        return torch.from_numpy(data)

# Create datasets
train_dataset = PatchDataset(train_patches)
val_dataset = PatchDataset(val_patches)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4
)

# Use in training loop
for batch in train_loader:
    # batch shape: (batch_size, z, y, x)
    # Your training code here
    pass
```

## Multi-Resolution Training Strategy

Sample at different atlas resolutions for multi-scale analysis:

```python
# Load atlases at different resolutions
atlas_1mm = ZarrNiiAtlas.from_files("atlas_1mm_dseg.nii.gz", "atlas_dseg.tsv")
atlas_2mm = ZarrNiiAtlas.from_files("atlas_2mm_dseg.nii.gz", "atlas_dseg.tsv")

# Sample centers from each resolution
centers_1mm = atlas_1mm.sample_region_patches(
    n_patches=500,
    region_ids="Cortex",
    seed=42
)

centers_2mm = atlas_2mm.sample_region_patches(
    n_patches=500,
    region_ids="Cortex",
    seed=42
)

# Load image at multiple resolutions
highres_image = ZarrNii.from_ome_zarr("data.ome.zarr", level=0)  # Highest resolution
lowres_image = ZarrNii.from_ome_zarr("data.ome.zarr", level=2)   # Lower resolution

# Extract patches at different scales
# All patches will be 256³ voxels, but represent different physical sizes
highres_patches = highres_image.crop_centered(centers_1mm, patch_size=(256, 256, 256))
lowres_patches = lowres_image.crop_centered(centers_2mm, patch_size=(256, 256, 256))
```

## Balanced Sampling from Multiple Regions

Ensure equal representation from different anatomical regions:

```python
# Define regions and number of patches per region
regions_config = {
    "Cortex": 300,
    "Hippocampus": 200,
    "Thalamus": 200,
    "Cerebellum": 300
}

# Sample from each region
all_centers = []
for region_name, n_patches in regions_config.items():
    centers = atlas.sample_region_patches(
        n_patches=n_patches,
        region_ids=region_name,
        seed=42
    )
    all_centers.extend(centers)

print(f"Total centers sampled: {len(all_centers)}")

# Extract all patches
patches = highres_image.crop_centered(all_centers, patch_size=(256, 256, 256))
```

## Quality Control: Verify Patch Coverage

Check that patches cover the intended regions:

```python
# Sample centers
centers = atlas.sample_region_patches(
    n_patches=50,
    region_ids="Hippocampus",
    seed=42
)

# Extract patches from the atlas itself to verify regions
atlas_patches = atlas.crop_centered(centers, patch_size=(64, 64, 64))

# Check that patches contain the target region label
hippocampus_label = atlas.get_region_info("Hippocampus")["index"]

for i, patch in enumerate(atlas_patches):
    data = patch.data
    if hasattr(data, 'compute'):
        data = data.compute()
    
    # Check if hippocampus label is present
    has_hippocampus = np.any(data == hippocampus_label)
    print(f"Patch {i}: Contains Hippocampus = {has_hippocampus}")
```

## Memory-Efficient Processing

For large datasets, process patches in batches:

```python
def process_in_batches(centers, image, patch_size, batch_size=10):
    """Process patches in batches to save memory."""
    n_patches = len(centers)
    
    for i in range(0, n_patches, batch_size):
        # Get batch of centers
        batch_centers = centers[i:i+batch_size]
        
        # Extract batch of patches
        batch_patches = image.crop_centered(batch_centers, patch_size=patch_size)
        
        # Process batch (e.g., save, train, etc.)
        for j, patch in enumerate(batch_patches):
            patch_idx = i + j
            # Your processing code here
            patch.to_nifti(f"patch_{patch_idx:04d}.nii.gz")
        
        print(f"Processed patches {i} to {i+len(batch_patches)-1}")

# Use the function
process_in_batches(
    centers=train_centers,
    image=highres_image,
    patch_size=(256, 256, 256),
    batch_size=10
)
```

## Tips and Best Practices

### 1. Coordinate Systems
- Centers are always in **physical space (mm)** in (x, y, z) order
- Patch sizes are always in **voxels** in (x, y, z) order
- This allows sampling at one resolution and cropping at another

### 2. Reproducibility
- Always set a `seed` for reproducible sampling
- Save the seed value with your training configuration

```python
# Save configuration
config = {
    'seed': 42,
    'n_patches': 1000,
    'patch_size': (256, 256, 256),
    'regions': ["Cortex", "Hippocampus"],
}
```

### 3. Handling Edge Cases
- Patches near image boundaries may be smaller than requested
- Check patch dimensions before training:

```python
# Filter patches by size
min_size = (200, 200, 200)  # Minimum acceptable size
valid_patches = [
    p for p in patches 
    if all(s >= m for s, m in zip(p.shape[1:], min_size))
]
```

### 4. Data Augmentation
- Combine with standard augmentation techniques:
  - Rotation (use `apply_transform()`)
  - Flipping
  - Intensity scaling
  - Adding noise

### 5. Distributed Training
- Sample centers once and share across workers:

```python
# In main process
centers = atlas.sample_region_patches(n_patches=10000, region_ids="Cortex", seed=42)

# Save centers for workers
import pickle
with open('centers.pkl', 'wb') as f:
    pickle.dump(centers, f)

# In worker processes, load and subset
with open('centers.pkl', 'rb') as f:
    all_centers = pickle.load(f)

# Each worker gets a subset
worker_centers = all_centers[worker_id::n_workers]
patches = image.crop_centered(worker_centers, patch_size=(256, 256, 256))
```

## See Also

- [Atlas Example](atlas_example.md) - General atlas functionality
- [Transformations](transformations.md) - Spatial transformations and alignment
- [Multiscale OME-Zarr](multiscale.md) - Working with multi-resolution data
