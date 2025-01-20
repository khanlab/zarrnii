# API Reference

This page documents the core classes, methods, and functions in ZarrNii. 

---

## Core Classes

### ZarrNii
The `ZarrNii` class provides tools for reading, writing, and transforming datasets in OME-Zarr and NIfTI formats.

#### Key Methods
- [`from_ome_zarr`](#from_ome_zarr): Load data from OME-Zarr.
- [`from_nifti`](#from_nifti): Load data from a NIfTI file.
- [`to_ome_zarr`](#to_ome_zarr): Save data as OME-Zarr.
- [`to_nifti`](#to_nifti): Save data as a NIfTI file.
- [`downsample`](#downsample): Reduce resolution of datasets.
- [`upsample`](#upsample): Increase resolution of datasets.

::: zarrnii.ZarrNii
    options:
        members: []  # Exclude individual methods here; include them manually below.

---

## Methods

### `from_ome_zarr`
::: zarrnii.ZarrNii.from_ome_zarr

### `from_nifti`
::: zarrnii.ZarrNii.from_nifti

### `to_ome_zarr`
::: zarrnii.ZarrNii.to_ome_zarr

### `to_nifti`
::: zarrnii.ZarrNii.to_nifti

### `downsample`
::: zarrnii.ZarrNii.downsample

### `upsample`
::: zarrnii.ZarrNii.upsample

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

