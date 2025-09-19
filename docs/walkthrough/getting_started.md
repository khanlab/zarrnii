# Getting Started

This guide helps you set up **ZarrNii** and get started with its basic functionality. By the end of this guide, you'll be able to read OME-Zarr and NIfTI datasets, perform basic transformations, and save your results.

---

## Installation

ZarrNii requires Python 3.11 or later. Install it using [uv](https://docs.astral.sh/uv/), a modern, fast Python package installer and project manager.

### **1. Clone the Repository**
If you're using the source code, clone the ZarrNii repository:
```bash
git clone https://github.com/khanlab/zarrnii.git
cd zarrnii
```

### **2. Install with uv**
Run the following command to install the library and its dependencies:
```bash
uv sync --dev
```

If you don't use uv, install ZarrNii and its dependencies using `pip`:
```bash
pip install zarrnii
```

---

## Prerequisites

Before using ZarrNii, ensure you have:
- **OME-Zarr datasets**: Multidimensional images in Zarr format.
- **NIfTI datasets**: Neuroimaging data in `.nii` or `.nii.gz` format.

---

## Basic Usage

### **1. Reading Data**

You can load an OME-Zarr or NIfTI dataset into a `ZarrNii` object.

#### **From OME-Zarr**:
```python
from zarrnii import ZarrNii

# Load OME-Zarr
znimg = ZarrNii.from_ome_zarr("path/to/dataset.ome.zarr")

print("Data shape:", znimg.darr.shape)
print("Affine matrix:\n", znimg.affine.matrix)

# For anisotropic data, automatically create more isotropic voxels
znimg_isotropic = ZarrNii.from_ome_zarr(
    "path/to/dataset.ome.zarr", 
    downsample_near_isotropic=True
)
# Load specific timepoints and channels from 5D data
znimg_5d = ZarrNii.from_ome_zarr("timeseries.zarr", timepoints=[0, 2], channels=[1])
print("5D subset shape:", znimg_5d.darr.shape)
```

#### **From NIfTI**:
```python
# Load NIfTI (supports 3D, 4D, and 5D data)
znimg = ZarrNii.from_nifti("path/to/dataset.nii")

print("Data shape:", znimg.darr.shape)
print("Affine matrix:\n", znimg.affine.matrix)
```

---

### **2. Working with 5D Data**

ZarrNii supports 5D images with time and channel dimensions (T,C,Z,Y,X format). You can select specific timepoints and channels either during loading or after loading.

#### **Loading with Selection**:
```python
# Load specific timepoints
znimg_time = ZarrNii.from_ome_zarr("timeseries.zarr", timepoints=[0, 2, 4])

# Load specific channels  
znimg_channels = ZarrNii.from_ome_zarr("multichannel.zarr", channels=[0, 2])

# Load specific channels by label
znimg_labels = ZarrNii.from_ome_zarr("labeled.zarr", channel_labels=["DAPI", "GFP"])

# Combine timepoint and channel selection
znimg_subset = ZarrNii.from_ome_zarr("data.zarr", timepoints=[1, 3], channels=[0])
```

#### **Post-loading Selection**:
```python
# Load full dataset first
znimg = ZarrNii.from_ome_zarr("timeseries.zarr")

# Select timepoints after loading
selected_time = znimg.select_timepoints([0, 2])

# Select channels after loading
selected_channels = znimg.select_channels([1, 2])

# Chain selections
subset = znimg.select_timepoints([0, 1]).select_channels([0])
```

---

### **3. Performing Transformations**

ZarrNii supports various transformations, such as cropping, downsampling, and upsampling. When working with 5D data, spatial transformations preserve the time and channel dimensions.

#### **Cropping**:
Crop a region from the dataset using voxel coordinates:
```python
cropped = znimg.crop((10, 10, 10), (50, 50, 50))
print("Cropped shape:", cropped.darr.shape)
```

#### **Downsampling**:
Reduce the resolution of your dataset:
```python
downsampled = znimg.downsample(level=2)
print("Downsampled shape:", downsampled.darr.shape)

# For 5D data: (3, 2, 16, 32, 32) -> (3, 2, 8, 16, 16)
# Time and channel dimensions are preserved
```

#### **Upsampling**:
Increase the resolution of your dataset:
```python
upsampled = znimg.upsample(along_x=2, along_y=2, along_z=2)
print("Upsampled shape:", upsampled.darr.shape)
```

---

### **3. Image Segmentation**

ZarrNii includes a plugin architecture for image segmentation algorithms. You can apply segmentation to identify regions of interest in your data.

#### **Otsu Thresholding**:
Apply automatic Otsu thresholding for binary segmentation:
```python
segmented = znimg.segment_otsu(nbins=256)
print("Segmented shape:", segmented.darr.shape)
print("Unique values:", segmented.darr.compute().unique())  # Should show [0, 1]
```

#### **Using Plugin Interface**:
You can also use the generic plugin interface:
```python
from zarrnii import OtsuSegmentation

plugin = OtsuSegmentation(nbins=128)
segmented = znimg.segment(plugin)
```

#### **Custom Chunk Processing**:
For large datasets, you can control memory usage with custom chunk sizes:
```python
segmented = znimg.segment_otsu(chunk_size=(1, 10, 50, 50))
```

---

### **4. Saving Data**

ZarrNii makes it easy to save your datasets in both OME-Zarr and NIfTI formats.

#### **To NIfTI**:
Save the dataset as a `.nii` file:
```python
znimg.to_nifti("output_dataset.nii")
```

#### **To OME-Zarr**:
Save the dataset back to OME-Zarr format:
```python
znimg.to_ome_zarr("output_dataset.ome.zarr")
```

---

## Example Workflow

Here’s a full workflow from loading an OME-Zarr dataset to saving a downsampled version as NIfTI:

```python
from zarrnii import ZarrNii

# Load a 5D OME-Zarr dataset with specific timepoints and channels
znimg = ZarrNii.from_ome_zarr("path/to/timeseries.zarr", 
                              timepoints=[0, 2, 4], 
                              channels=[0, 1])

# Perform transformations
cropped = znimg.crop((10, 10, 10), (100, 100, 100))
downsampled = cropped.downsample(level=2)

# Save the result as a NIfTI file

downsampled.to_nifti("processed_timeseries.nii")

# Or save as OME-Zarr with metadata preservation
downsampled.to_ome_zarr("processed_timeseries.ome.zarr")


# Apply segmentation to the original image
segmented = znimg.segment_otsu(nbins=256)
segmented.to_nifti("segmented_output.nii")

```

---

## What’s Next?

- [Walkthrough: Basic Tasks](basic_tasks.md): Learn more about common workflows like cropping, interpolation, and combining transformations.
- [Segmentation Plugin Examples](../examples/segmentation_example.md): Learn how to use and create segmentation plugins.
- [API Reference](../reference.md): Explore the detailed API for ZarrNii.

