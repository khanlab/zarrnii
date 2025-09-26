# API Reference

This page documents the core classes, methods, and functions in ZarrNii. 

---

## Core Classes

### ZarrNii
The `ZarrNii` class provides tools for reading, writing, and transforming datasets in OME-Zarr and NIfTI formats.

#### Key Methods
- [`from_ome_zarr`](#from_ome_zarr): Load data from OME-Zarr.
- [`from_nifti`](#from_nifti): Load data from a NIfTI file.
- [`from_imaris`](#from_imaris): Load data from an Imaris (.ims) file.
- [`to_ome_zarr`](#to_ome_zarr): Save data as OME-Zarr.
- [`to_nifti`](#to_nifti): Save data as a NIfTI file.
- [`to_tiff_stack`](#to_tiff_stack): Save data as a stack of 2D TIFF files.
- [`to_imaris`](#to_imaris): Save data as an Imaris (.ims) file.
- [`crop`](#crop): Extract a region from the dataset.
- [`downsample`](#downsample): Reduce resolution of datasets.
- [`upsample`](#upsample): Increase resolution of datasets.
- [`apply_transform`](#apply_transform): Apply spatial transformations.

::: zarrnii.ZarrNii
    options:
        members: []  # Exclude individual methods here; include them manually below.

---

## Methods

### `from_ome_zarr`
::: zarrnii.ZarrNii.from_ome_zarr

### `from_nifti`
::: zarrnii.ZarrNii.from_nifti

### `from_imaris`
::: zarrnii.ZarrNii.from_imaris

### `to_ome_zarr`
::: zarrnii.ZarrNii.to_ome_zarr

### `to_nifti`
::: zarrnii.ZarrNii.to_nifti

### `to_tiff_stack`
::: zarrnii.ZarrNii.to_tiff_stack

### `to_imaris`
::: zarrnii.ZarrNii.to_imaris

### `crop`
::: zarrnii.ZarrNii.crop

### `downsample`
::: zarrnii.ZarrNii.downsample

### `upsample`
::: zarrnii.ZarrNii.upsample

### `apply_transform`
::: zarrnii.ZarrNii.apply_transform

---



## Remaining Methods and Functions

::: zarrnii.ZarrNii
    options:
        members: true  # Automatically include all remaining members not listed above.
        show_source: true

::: zarrnii.transform.AffineTransform
    options:
        members: true
        show_source: true

::: zarrnii.transform.DisplacementTransform
    options:
        members: true
        show_source: true

