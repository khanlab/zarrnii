"""Unified ZarrNii implementation using NgffImage internally.

This module provides the core ZarrNii class that maintains chainable functionality
while using NgffImage objects under the hood for better multiscale support and
metadata preservation. It bridges OME-Zarr and NIfTI formats with a unified API.

The module includes:
- Core ZarrNii class with transformation, cropping, and resampling capabilities
- Helper functions for loading and saving OME-Zarr data
- Utility functions for metadata extraction and conversion
- Compatibility functions for backward compatibility

Key Classes:
    ZarrNii: Main class for working with OME-Zarr and NIfTI data

Key Functions:
    load_ngff_image: Load NgffImage from OME-Zarr store
    save_ngff_image: Save NgffImage to OME-Zarr store with pyramid
    get_multiscales: Load full multiscales object from store
"""

from __future__ import annotations

import copy
from typing import Any

import dask.array as da
import fsspec
import ngff_zarr as nz
import nibabel as nib
import numpy as np
from attrs import define
from scipy.interpolate import interpn
from scipy.ndimage import zoom

from .transform import AffineTransform, Transform


class MetadataInvalidError(Exception):
    """Raised when an operation would invalidate ZarrNii metadata."""

    pass


# NgffImage-based function library
# These functions operate directly on ngff_zarr.NgffImage objects


def load_ngff_image(
    store_or_path: str | Any,
    level: int = 0,
    channels: list[int] | None = None,
    channel_labels: list[str] | None = None,
    timepoints: list[int] | None = None,
    storage_options: dict[str, Any] | None = None,
) -> nz.NgffImage:
    """Load an NgffImage from an OME-Zarr store.

    This function provides flexible loading of OME-Zarr data with support for
    ZIP stores, channel selection, and timepoint selection. It handles various
    storage backends through fsspec.

    Args:
        store_or_path: Store or path to the OME-Zarr file. Supports local paths,
            remote URLs, and .zip extensions for ZipStore access
        level: Pyramid level to load (0 = highest resolution, higher = lower resolution)
        channels: List of channel indices to load (0-based). If None, loads all channels
        channel_labels: List of channel names to load by label. Requires OMERO metadata
        timepoints: List of timepoint indices to load (0-based). If None, loads all timepoints
        storage_options: Additional options passed to zarr storage backend

    Returns:
        NgffImage object containing the loaded image data and metadata at the specified level

    Raises:
        FileNotFoundError: If the store or path does not exist
        ValueError: If level is out of range or invalid channel/timepoint indices
        KeyError: If channel_labels are specified but not found in metadata

    Examples:
        >>> # Load highest resolution level
        >>> img = load_ngff_image("/path/to/data.zarr")

        >>> # Load specific channels by index
        >>> img = load_ngff_image("/path/to/data.zarr", channels=[0, 2])

        >>> # Load from ZIP store
        >>> img = load_ngff_image("/path/to/data.zarr.zip", level=1)
    """
    import zarr

    # Handle ZIP files by creating a ZipStore
    if isinstance(store_or_path, str) and store_or_path.endswith(".zip"):
        store = zarr.storage.ZipStore(store_or_path, mode="r")
        multiscales = nz.from_ngff_zarr(store, storage_options=storage_options)
        store.close()
    else:
        # Load the multiscales object normally
        multiscales = nz.from_ngff_zarr(store_or_path, storage_options=storage_options)

    # Get the specified level
    ngff_image = multiscales.images[level]

    # Handle channel and timepoint selection if specified
    if channels is not None or channel_labels is not None or timepoints is not None:
        ngff_image = _select_dimensions_from_image(
            ngff_image, multiscales, channels, channel_labels, timepoints
        )

    return ngff_image


def save_ngff_image(
    ngff_image: nz.NgffImage,
    store_or_path: str | Any,
    max_layer: int = 4,
    scale_factors: list[int] | None = None,
    xyz_orientation: str | None = None,
    **kwargs: Any,
) -> None:
    """Save an NgffImage to an OME-Zarr store with multiscale pyramid.

    Creates a multiscale OME-Zarr dataset from the input NgffImage, with automatic
    generation of pyramid levels for efficient viewing and processing at different
    scales.

    Args:
        ngff_image: NgffImage object to save containing data and metadata
        store_or_path: Target store or path. Supports local paths, remote URLs,
            and .zip extensions for ZipStore creation
        max_layer: Maximum number of pyramid levels to create (including level 0)
        scale_factors: Custom scale factors for each pyramid level. If None,
            uses powers of 2: [2, 4, 8, ...]
        orientation: Anatomical orientation string (e.g., 'RAS', 'LPI') to store
            as metadata
        **kwargs: Additional arguments passed to to_ngff_zarr function

    Raises:
        ValueError: If scale_factors length doesn't match max_layer-1
        OSError: If unable to write to the specified location
        TypeError: If ngff_image is not a valid NgffImage object

    Examples:
        >>> # Save with default pyramid levels
        >>> save_ngff_image(img, "/path/to/output.zarr")

        >>> # Save to ZIP with custom pyramid
        >>> save_ngff_image(img, "/path/to/output.zarr.zip",
        ...                 scale_factors=[2, 4], xyz_orientation="RAS")
    """
    import zarr

    if scale_factors is None:
        scale_factors = [2**i for i in range(1, max_layer)]

    # Create multiscales from the image
    multiscales = nz.to_multiscales(ngff_image, scale_factors=scale_factors)

    # Check if the target is a ZIP file (based on extension)
    if isinstance(store_or_path, str) and store_or_path.endswith(".zip"):
        # For ZIP files, use temp directory approach due to zarr v3.x ZipStore compatibility issues
        import os
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save to temporary directory first
            temp_zarr_path = os.path.join(tmpdir, "temp.zarr")
            nz.to_ngff_zarr(temp_zarr_path, multiscales, **kwargs)

            # Add xyz_orientation metadata to the temporary zarr store if provided
            if xyz_orientation:
                try:
                    group = zarr.open_group(temp_zarr_path, mode="r+")
                    group.attrs["xyz_orientation"] = xyz_orientation
                except Exception:
                    # If we can't write orientation metadata, that's not critical
                    pass

            # Create ZIP file from the directory
            zip_base_path = store_or_path.replace(".zip", "")
            shutil.make_archive(zip_base_path, "zip", temp_zarr_path)
    else:
        # Write to zarr store directly
        nz.to_ngff_zarr(store_or_path, multiscales, **kwargs)

        # Add xyz_orientation metadata if provided
        if xyz_orientation:
            try:
                if isinstance(store_or_path, str):
                    group = zarr.open_group(store_or_path, mode="r+")
                else:
                    group = zarr.open_group(store_or_path, mode="r+")
                group.attrs["xyz_orientation"] = xyz_orientation
            except Exception:
                # If we can't write orientation metadata, that's not critical
                pass


def save_ngff_image_with_ome_zarr(
    ngff_image: nz.NgffImage,
    store_or_path: str | Any,
    max_layer: int = 4,
    scale_factors: list[int] | None = None,
    scaling_method: str = "local_mean",
    xyz_orientation: str | None = None,
    compute: bool = True,
    **kwargs: Any,
) -> None:
    """Save an NgffImage to an OME-Zarr store using ome-zarr-py library.

    This function uses the ome-zarr-py library for writing, which can provide
    performance enhancements when using dask and dask distributed. It was the
    default writer before v2.0 and is now offered as an alternative.

    Args:
        ngff_image: NgffImage object to save containing data and metadata
        store_or_path: Target store or path. Supports local paths, remote URLs,
            and .zip extensions for ZipStore creation
        max_layer: Maximum number of pyramid levels to create (including level 0)
        scale_factors: Custom scale factors for each pyramid level. If None,
            uses powers of 2: [2, 4, 8, ...]
        scaling_method: Method for downsampling ('nearest', 'gaussian', etc.)
        xyz_orientation: Anatomical orientation string (e.g., 'RAS', 'LPI') to store
            as metadata
        compute: Whether to compute the write operations immediately (True) or
            return delayed operations (False)
        **kwargs: Additional arguments passed to ome_zarr.writer.write_image

    Raises:
        ValueError: If scale_factors length doesn't match max_layer-1
        OSError: If unable to write to the specified location
        TypeError: If ngff_image is not a valid NgffImage object

    Examples:
        >>> # Save with default pyramid levels
        >>> save_ngff_image_with_ome_zarr(img, "/path/to/output.zarr")

        >>> # Save to ZIP with custom pyramid
        >>> save_ngff_image_with_ome_zarr(img, "/path/to/output.zarr.zip",
        ...                                scale_factors=[2, 4], xyz_orientation="RAS")

        >>> # Use with dask distributed for better performance
        >>> from dask.distributed import Client
        >>> client = Client()
        >>> save_ngff_image_with_ome_zarr(img, "/path/to/output.zarr", compute=True)
    """
    import zarr
    from ome_zarr.format import FormatV04
    from ome_zarr.scale import Scaler
    from ome_zarr.writer import write_image

    # Force use of OME-NGFF v0.4 format (Zarr v2) instead of v0.5 (Zarr v3)
    # This ensures compatibility with existing tools and avoids .zarr.json files
    fmt = FormatV04()

    # Note: ome-zarr's Scaler interprets max_layer as the highest pyramid level index
    # (not count), so max_layer=N creates N+1 levels: 0, 1, ..., N
    # To match save_ngff_image behavior where max_layer=N creates N total levels,
    # we need to adjust: ome_zarr_max_layer = max_layer - 1
    if scale_factors is None:
        # Generate scale factors for each additional level beyond level 0
        # For max_layer total levels, we need max_layer-1 scale factors
        scale_factors = [2**i for i in range(1, max_layer)]

    # Convert NgffImage metadata to ome-zarr format
    axes = _ngff_image_to_ome_zarr_axes(ngff_image)
    coordinate_transformations = _ngff_image_to_ome_zarr_transforms(
        ngff_image, scale_factors
    )

    # Set up scaler for multi-resolution pyramid
    # Adjust max_layer to match ome-zarr's interpretation (highest index, not count)
    if max_layer <= 1:
        # No pyramid, just one level
        scaler = None
    else:
        # ome-zarr max_layer is the highest index, so max_layer-1 for N total levels
        scaler = Scaler(max_layer=max_layer - 1, method=scaling_method)

    # Check if the target is a ZIP file (based on extension)
    if isinstance(store_or_path, str) and store_or_path.endswith(".zip"):
        # For ZIP files, use temp directory approach
        import os
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save to temporary directory first
            temp_zarr_path = os.path.join(tmpdir, "temp.zarr")
            # Use Zarr v2 format for compatibility with OME-NGFF v0.4
            store = zarr.open_group(temp_zarr_path, mode="w", zarr_format=2)

            # Write the data to OME-Zarr
            write_image(
                image=ngff_image.data,
                group=store,
                scaler=scaler,
                coordinate_transformations=coordinate_transformations,
                axes=axes,
                fmt=fmt,
                compute=compute,
                **kwargs,
            )

            # Add xyz_orientation metadata if provided
            if xyz_orientation:
                try:
                    store.attrs["xyz_orientation"] = xyz_orientation
                except Exception:
                    # If we can't write orientation metadata, that's not critical
                    pass

            # Create ZIP file from the directory
            zip_base_path = store_or_path.replace(".zip", "")
            shutil.make_archive(zip_base_path, "zip", temp_zarr_path)
    else:
        # Write to zarr store directly
        if isinstance(store_or_path, str):
            # Use Zarr v2 format for compatibility with OME-NGFF v0.4
            store = zarr.open_group(store_or_path, mode="w", zarr_format=2)
        else:
            store = store_or_path

        # Write the data to OME-Zarr
        write_image(
            image=ngff_image.data,
            group=store,
            scaler=scaler,
            coordinate_transformations=coordinate_transformations,
            axes=axes,
            fmt=fmt,
            compute=compute,
            **kwargs,
        )

        # Add xyz_orientation metadata if provided
        if xyz_orientation:
            try:
                if isinstance(store_or_path, str):
                    group = zarr.open_group(store_or_path, mode="r+")
                else:
                    group = store_or_path
                group.attrs["xyz_orientation"] = xyz_orientation
            except Exception:
                # If we can't write orientation metadata, that's not critical
                pass


def _ngff_image_to_ome_zarr_axes(ngff_image: nz.NgffImage) -> list:
    """Convert NgffImage dims to ome-zarr axes format.

    Args:
        ngff_image: NgffImage object

    Returns:
        List of axis dictionaries for ome-zarr
    """
    axes = []
    for dim in ngff_image.dims:
        axis = {"name": dim}

        # Determine axis type
        if dim == "t":
            axis["type"] = "time"
            # Add unit if available
            if ngff_image.axes_units and dim in ngff_image.axes_units:
                axis["unit"] = ngff_image.axes_units[dim]
        elif dim == "c":
            axis["type"] = "channel"
        elif dim in ["x", "y", "z"]:
            axis["type"] = "space"
            # Add unit if available
            if ngff_image.axes_units and dim in ngff_image.axes_units:
                axis["unit"] = ngff_image.axes_units[dim]
            else:
                # Default to micrometer for space axes if not specified
                axis["unit"] = "micrometer"

        axes.append(axis)

    return axes


def _ngff_image_to_ome_zarr_transforms(
    ngff_image: nz.NgffImage, scale_factors: list[int]
) -> list:
    """Convert NgffImage scale/translation to ome-zarr coordinate_transformations.

    Args:
        ngff_image: NgffImage object
        scale_factors: List of scale factors for each pyramid level

    Returns:
        List of coordinate transformations for each pyramid level
    """
    # Build base scale and translation arrays in the order of dims
    base_scale = []
    base_translation = []

    for dim in ngff_image.dims:
        # Get scale for this dimension (default to 1.0)
        if ngff_image.scale and dim in ngff_image.scale:
            base_scale.append(ngff_image.scale[dim])
        else:
            base_scale.append(1.0)

        # Get translation for this dimension (default to 0.0)
        if ngff_image.translation and dim in ngff_image.translation:
            base_translation.append(ngff_image.translation[dim])
        else:
            base_translation.append(0.0)

    # Create coordinate transformations for each pyramid level
    coordinate_transformations = []

    # Level 0 (original resolution)
    coordinate_transformations.append(
        [
            {"type": "scale", "scale": base_scale.copy()},
            {"type": "translation", "translation": base_translation.copy()},
        ]
    )

    # Additional levels with downsampling
    # Note: ome-zarr-py only downsamples in xy plane, not in z
    for factor in scale_factors:
        level_scale = []
        for i, dim in enumerate(ngff_image.dims):
            # Only apply scaling to x and y dimensions (not z)
            # ome-zarr-py Scaler only downsamples in the xy plane
            if dim in ["x", "y"]:
                level_scale.append(base_scale[i] * factor)
            else:
                level_scale.append(base_scale[i])

        coordinate_transformations.append(
            [
                {"type": "scale", "scale": level_scale},
                {"type": "translation", "translation": base_translation.copy()},
            ]
        )

    return coordinate_transformations


def get_multiscales(
    store_or_path,
    storage_options: dict | None = None,
) -> nz.Multiscales:
    """
    Load the full multiscales object from an OME-Zarr store.

    This provides access to all pyramid levels and metadata.

    Args:
        store_or_path: Store or path to the OME-Zarr file
        storage_options: Storage options for Zarr

    Returns:
        Multiscales: The full multiscales object with all pyramid levels
    """
    return nz.from_ngff_zarr(store_or_path, storage_options=storage_options)


def _select_dimensions_from_image(
    image: nz.NgffImage,
    multiscales: nz.Multiscales,
    channels: list[int] | None = None,
    channel_labels: list[str] | None = None,
    timepoints: list[int] | None = None,
) -> nz.NgffImage:
    """
    Create a new NgffImage with selected channels and timepoints.

    This is a unified function to handle both channel and timepoint selection.
    """
    # Get axis names
    axis_names = [axis.name for axis in multiscales.metadata.axes]

    # Handle channel label resolution
    if channel_labels is not None:
        if multiscales.metadata.omero is None or not hasattr(
            multiscales.metadata.omero, "channels"
        ):
            raise ValueError("Channel labels specified but no omero metadata found")

        # Extract available labels
        omero_channels = multiscales.metadata.omero.channels
        available_labels = []
        for ch in omero_channels:
            if hasattr(ch, "label"):
                available_labels.append(ch.label)
            elif isinstance(ch, dict):
                available_labels.append(ch.get("label", ""))
            else:
                available_labels.append(str(getattr(ch, "label", "")))

        # Resolve labels to indices
        resolved_channels = []
        for label in channel_labels:
            try:
                idx = available_labels.index(label)
                resolved_channels.append(idx)
            except ValueError:
                raise ValueError(
                    f"Channel label '{label}' not found. Available: {available_labels}"
                )

        channels = resolved_channels

    # Set defaults if not specified
    if channels is None:
        c_index = axis_names.index("c") if "c" in axis_names else None
        if c_index is not None:
            num_channels = image.data.shape[c_index]
            channels = list(range(num_channels))

    if timepoints is None:
        t_index = axis_names.index("t") if "t" in axis_names else None
        if t_index is not None:
            num_timepoints = image.data.shape[t_index]
            timepoints = list(range(num_timepoints))

    # Build slices for dimension selection
    slices = []
    new_dims = []

    for i, name in enumerate(axis_names):
        if name == "t":
            if timepoints is not None:
                slices.append(timepoints)
                new_dims.append(name)
            else:
                # No time axis selection, keep full range
                slices.append(slice(None))
                new_dims.append(name)
        elif name == "c":
            if channels is not None:
                slices.append(channels)
                new_dims.append(name)
            else:
                # No channel axis selection, keep full range
                slices.append(slice(None))
                new_dims.append(name)
        else:
            # Keep other dimensions unchanged
            slices.append(slice(None))
            new_dims.append(name)

    # Apply slices to get new data
    new_data = image.data[tuple(slices)]

    # Create new NgffImage with selected data
    new_image = nz.NgffImage(
        data=new_data,
        dims=new_dims,
        scale=image.scale,
        translation=image.translation,
        name=image.name,
    )

    return new_image


def _select_channels_from_image(
    image: nz.NgffImage,
    multiscales: nz.Multiscales,
    channels: list[int] | None = None,
    channel_labels: list[str] | None = None,
) -> nz.NgffImage:
    """
    Create a new NgffImage with selected channels.

    This is a helper function to handle channel selection.
    """
    # Get axis names
    axis_names = [axis.name for axis in multiscales.metadata.axes]

    # Handle channel label resolution
    if channel_labels is not None:
        if multiscales.metadata.omero is None or not hasattr(
            multiscales.metadata.omero, "channels"
        ):
            raise ValueError("Channel labels specified but no omero metadata found")

        # Extract available labels
        omero_channels = multiscales.metadata.omero.channels
        available_labels = []
        for ch in omero_channels:
            if hasattr(ch, "label"):
                available_labels.append(ch.label)
            elif isinstance(ch, dict):
                available_labels.append(ch.get("label", ""))
            else:
                available_labels.append(str(getattr(ch, "label", "")))

        # Resolve labels to indices
        resolved_channels = []
        for label in channel_labels:
            try:
                idx = available_labels.index(label)
                resolved_channels.append(idx)
            except ValueError:
                raise ValueError(
                    f"Channel label '{label}' not found. Available: {available_labels}"
                )

        channels = resolved_channels

    # If no channels specified, load all
    if channels is None:
        c_index = axis_names.index("c") if "c" in axis_names else None
        if c_index is not None:
            num_channels = image.data.shape[c_index]
            channels = list(range(num_channels))
        else:
            # No channel axis, return original image
            return image

    # Build slices for channel selection
    slices = []
    for i, name in enumerate(axis_names):
        if name == "t":
            slices.append(0)  # Drop singleton time axis
        elif name == "c":
            slices.append(channels)  # Select specific channels
        else:
            slices.append(slice(None))  # Keep full range

    # Apply slices to get new data
    new_data = image.data[tuple(slices)]

    # Create new NgffImage with selected data
    new_image = nz.NgffImage(
        data=new_data,
        dims=image.dims,
        scale=image.scale,
        translation=image.translation,
        name=image.name,
    )

    return new_image


# Function-based API for operating on NgffImage objects
def crop_ngff_image(
    ngff_image: nz.NgffImage,
    bbox_min: dict[float],
    bbox_max: dict[float],
    dim_flips: dict[float],
) -> nz.NgffImage:
    """
    Crop an NgffImage using a bounding box.

    Args:
        ngff_image: Input NgffImage to crop
        bbox_min: Minimum corner of bounding box, dict with spatial dim keys
        bbox_max: Maximum corner of bounding box, dict with spatial dim keys
        orientation_flips: orientation flips by dimensions, dict with spatial dim keys, vals as -1 or +1

    Returns:
        New cropped NgffImage
    """
    # Build slices for cropping
    slices = []

    for dim in ngff_image.dims:
        if dim in bbox_min:
            # This is a spatial dimension
            slices.append(slice(bbox_min[dim], bbox_max[dim]))
        else:
            # Non-spatial dimension, keep all
            slices.append(slice(None))

    # Apply crop
    cropped_data = ngff_image.data[tuple(slices)]

    # Update translation to account for cropping
    new_translation = ngff_image.translation.copy()

    for dim in bbox_min.keys():
        new_translation[dim] = (
            new_translation[dim]
            + dim_flips[dim] * bbox_min[dim] * ngff_image.scale[dim]
        )

    # Create new NgffImage
    return nz.NgffImage(
        data=cropped_data,
        dims=ngff_image.dims,
        scale=ngff_image.scale,
        translation=new_translation,
        name=ngff_image.name,
    )


def downsample_ngff_image(
    ngff_image: nz.NgffImage,
    factors: int | list[int],
    spatial_dims: list[str] = ["z", "y", "x"],
) -> nz.NgffImage:
    """
    Downsample an NgffImage by the specified factors.

    Args:
        ngff_image: Input NgffImage to downsample
        factors: Downsampling factors (int for isotropic, list for per-dimension)
        spatial_dims: Names of spatial dimensions

    Returns:
        New downsampled NgffImage
    """
    if isinstance(factors, int):
        factors = [factors] * len(spatial_dims)

    # Build downsampling slices
    slices = []
    spatial_idx = 0

    for dim in ngff_image.dims:
        if dim.lower() in [d.lower() for d in spatial_dims]:
            if spatial_idx < len(factors):
                factor = factors[spatial_idx]
                slices.append(slice(None, None, factor))
                spatial_idx += 1
            else:
                slices.append(slice(None))
        else:
            # Non-spatial dimension, keep all
            slices.append(slice(None))

    # Apply downsampling
    downsampled_data = ngff_image.data[tuple(slices)]

    # Update scale to account for downsampling
    new_scale = ngff_image.scale.copy()
    spatial_idx = 0

    for dim in ngff_image.dims:
        if dim.lower() in [d.lower() for d in spatial_dims]:
            if spatial_idx < len(factors) and dim in new_scale:
                new_scale[dim] *= factors[spatial_idx]
                spatial_idx += 1

    # Create new NgffImage
    return nz.NgffImage(
        data=downsampled_data,
        dims=ngff_image.dims,
        scale=new_scale,
        translation=ngff_image.translation,
        name=ngff_image.name,
    )


def _apply_near_isotropic_downsampling(znimg: ZarrNii, axes_order: str) -> ZarrNii:
    """
    Apply near-isotropic downsampling to a ZarrNii instance.

    This function calculates downsampling factors for dimensions where the pixel sizes
    are smaller than others by at least an integer factor, making the image more isotropic.

    Args:
        znimg: Input ZarrNii instance
        axes_order: Spatial axes order ("ZYX" or "XYZ")

    Returns:
        New ZarrNii instance with downsampling applied if needed
    """
    # Get scale information
    scale = znimg.scale

    # Define spatial dimensions based on axes order
    if axes_order == "ZYX":
        spatial_dims = ["z", "y", "x"]
    else:  # XYZ
        spatial_dims = ["x", "y", "z"]

    # Extract scales for spatial dimensions only
    scales = []
    available_dims = []
    for dim in spatial_dims:
        if dim in scale:
            scales.append(scale[dim])
            available_dims.append(dim)

    if len(scales) < 2:
        # Need at least 2 spatial dimensions to compare
        return znimg

    # Find the largest scale (coarsest resolution) to use as reference
    max_scale = max(scales)

    # Calculate downsampling factors for each dimension
    downsample_factors = []
    for i, current_scale in enumerate(scales):
        if current_scale < max_scale:
            # Calculate ratio and find the nearest power of 2
            ratio = max_scale / current_scale
            level = int(np.log2(round(ratio)))
            if level > 0:
                downsample_factors.append(2**level)
            else:
                downsample_factors.append(1)
        else:
            downsample_factors.append(1)

    # Only apply downsampling if at least one factor is > 1
    if any(factor > 1 for factor in downsample_factors):
        znimg = znimg.downsample(
            factors=downsample_factors, spatial_dims=available_dims
        )

    return znimg


def apply_transform_to_ngff_image(
    ngff_image: nz.NgffImage,
    transform: Transform,
    reference_image: nz.NgffImage,
    spatial_dims: list[str] = ["z", "y", "x"],
) -> nz.NgffImage:
    """
    Apply a spatial transformation to an NgffImage.

    Args:
        ngff_image: Input NgffImage to transform
        transform: Transformation to apply
        reference_image: Reference image defining output space
        spatial_dims: Names of spatial dimensions

    Returns:
        New transformed NgffImage
    """
    # For now, return a placeholder implementation
    # This would need full implementation of interpolation logic
    print("Warning: apply_transform_to_ngff_image is not fully implemented yet")

    return reference_image


# Utility functions for compatibility


def _extract_channel_labels_from_omero(channel_info):
    """
    Extract channel labels from omero metadata, handling both legacy and modern formats.

    Args:
        channel_info: List of channel information, either as dictionaries (legacy)
                     or as structured objects (modern OME-Zarr format)

    Returns:
        List[str]: List of channel labels
    """
    labels = []
    for ch in channel_info:
        if hasattr(ch, "label"):
            # Modern format: OmeroChannel object with .label attribute
            labels.append(ch.label)
        elif isinstance(ch, dict):
            # Legacy format: dictionary with 'label' key
            labels.append(ch.get("label", ""))
        else:
            # Fallback: try to get label as string or use empty string
            labels.append(str(getattr(ch, "label", "")))
    return labels


def _select_dimensions_from_image_with_omero(
    ngff_image, multiscales, channels, channel_labels, timepoints, omero_metadata
):
    """
    Select specific channels and timepoints from an NgffImage and filter omero metadata accordingly.

    Returns:
        Tuple of (selected_ngff_image, filtered_omero_metadata)
    """
    # Handle channel selection by labels
    if channel_labels is not None:
        if omero_metadata is None:
            raise ValueError(
                "Channel labels were specified but no omero metadata found"
            )

        available_labels = _extract_channel_labels_from_omero(omero_metadata.channels)
        channel_indices = []
        for label in channel_labels:
            if label not in available_labels:
                raise ValueError(f"Channel label '{label}' not found")
            channel_indices.append(available_labels.index(label))
        channels = channel_indices

    # If no selection is specified, return original
    if channels is None and timepoints is None:
        return ngff_image, omero_metadata

    # Select dimensions from data - do this sequentially to avoid fancy indexing conflicts
    data = ngff_image.data
    dims = ngff_image.dims

    # First, select timepoints if specified
    if timepoints is not None and "t" in dims:
        t_idx = dims.index("t")
        slices = [slice(None)] * len(data.shape)
        slices[t_idx] = timepoints
        data = data[tuple(slices)]

    # Then, select channels if specified
    if channels is not None and "c" in dims:
        c_idx = dims.index("c")
        slices = [slice(None)] * len(data.shape)
        slices[c_idx] = channels
        data = data[tuple(slices)]

    # Create new NgffImage with selected data
    selected_ngff_image = nz.NgffImage(
        data=data,
        dims=dims,
        scale=ngff_image.scale,
        translation=ngff_image.translation,
        name=ngff_image.name,
    )

    # Filter omero metadata to match selected channels (timepoints don't affect omero metadata)
    filtered_omero = omero_metadata
    if (
        channels is not None
        and omero_metadata is not None
        and hasattr(omero_metadata, "channels")
    ):

        class FilteredOmero:
            def __init__(self, channels):
                self.channels = channels

        filtered_channels = [omero_metadata.channels[i] for i in channels]
        filtered_omero = FilteredOmero(filtered_channels)

    return selected_ngff_image, filtered_omero


def get_bounded_subregion_from_zarr(
    points: np.ndarray,
    store_path: str,
    array_shape: Tuple[int, ...],
    dataset_path: str = "0",
    storage_options: Optional[Dict[str, Any]] = None,
):
    """
    Extract a bounded subregion from a zarr array using direct zarr access.

    This function reads data directly from a zarr store without using dask's compute(),
    avoiding nested compute() calls when used within dask.map_blocks.

    Parameters:
        points (np.ndarray): Nx3 or Nx4 array of coordinates in the array's space.
                            If Nx4, the last column is assumed to be the homogeneous
                            coordinate and is ignored.
        store_path (str): Path or URI to the zarr store
        array_shape (tuple): Shape of the full array (C, Z, Y, X)
        dataset_path (str): Path to the dataset within the zarr group (default: "0")
        storage_options (dict, optional): Additional options for the storage backend

    Returns:
        tuple:
            grid_points (tuple): A tuple of three 1D arrays representing the grid
                                points along each axis (Z, Y, X) in the subregion.
            subvol (np.ndarray or None): The extracted subregion as a NumPy array.
                                        Returns `None` if all points are outside
                                        the array domain.

    Notes:
        - Uses zarr library directly to load the subregion
        - A padding of 1 voxel is applied around the extent of the points
        - Handles ZIP stores automatically if store_path ends with .zip
    """
    import zarr

    # Ensure store_path is a string (could be Path object)
    store_path = str(store_path)

    pad = 1  # Padding around the extent of the points

    # Compute the extent of the points in the array's coordinate space
    min_extent = np.floor(points.min(axis=1)[:3] - pad).astype("int")
    max_extent = np.ceil(points.max(axis=1)[:3] + pad).astype("int")

    # Clip the extents to ensure they stay within the bounds of the array
    clip_min = np.zeros_like(min_extent)
    clip_max = np.array(array_shape[-3:])  # Z, Y, X dimensions

    min_extent = np.clip(min_extent, clip_min, clip_max)
    max_extent = np.clip(max_extent, clip_min, clip_max)

    # Check if all points are outside the domain
    if np.any(max_extent <= min_extent):
        return None, None

    # Open the zarr store and read the subregion directly
    try:
        if store_path.endswith(".zip"):
            store = zarr.storage.ZipStore(store_path, mode="r")
            root = zarr.open_group(store, mode="r")
        else:
            if storage_options:
                # Use fsspec for remote stores with storage options
                mapper = fsspec.get_mapper(store_path, **storage_options)
                root = zarr.open_group(mapper, mode="r")
            else:
                root = zarr.open_group(store_path, mode="r")

        # Access the dataset
        arr = root[dataset_path]

        # Extract the subvolume using direct zarr array access
        subvol = arr[
            :,
            min_extent[0] : max_extent[0],
            min_extent[1] : max_extent[1],
            min_extent[2] : max_extent[2],
        ]

        # Ensure we have a numpy array (zarr v3 may return zarr Array)
        if not isinstance(subvol, np.ndarray):
            subvol = np.asarray(subvol)

    finally:
        # Close ZIP store if used
        if store_path.endswith(".zip"):
            store.close()

    # Generate grid points for interpolation
    grid_points = (
        np.arange(min_extent[0], max_extent[0]),  # Z
        np.arange(min_extent[1], max_extent[1]),  # Y
        np.arange(min_extent[2], max_extent[2]),  # X
    )

    return grid_points, subvol


def interp_by_block(
    x,
    transforms: list[Transform],
    flo_store_path: Optional[str] = None,
    flo_array_shape: Optional[Tuple[int, ...]] = None,
    flo_dataset_path: str = "0",
    flo_storage_options: Optional[Dict[str, Any]] = None,
    flo_znimg: Optional["ZarrNii"] = None,
    block_info=None,
    interp_method="linear",
):
    """
    Interpolates the floating image onto the reference image block (`x`)
    using the provided transformations.

    This function extracts the necessary subset of the floating image for each block
    of the reference image, applies the transformations, and interpolates the floating
    image intensities onto the reference image grid.

    Parameters:
        x (np.ndarray): The reference image block to interpolate onto.
        transforms (list[Transform]): A list of `Transform` objects to apply to the
                                       reference image coordinates.
        flo_store_path (str, optional): Path/URI to the zarr store containing the
                                       floating image. If provided, uses direct zarr
                                       access instead of dask compute().
        flo_array_shape (tuple, optional): Shape of the floating array (C, Z, Y, X).
                                          Required if flo_store_path is provided.
        flo_dataset_path (str, optional): Path to dataset within zarr group.
                                         Defaults to "0".
        flo_storage_options (dict, optional): Storage options for accessing the store.
        flo_znimg (ZarrNii, optional): The floating ZarrNii instance. Used as fallback
                                      if store path not provided (legacy behavior).
        block_info (dict, optional): Metadata about the current block being processed.
        interp_method (str, optional): Interpolation method. Defaults to "linear".

    Returns:
        np.ndarray: The interpolated block of the reference image.

    Notes:
        - When flo_store_path is provided, uses direct zarr access to avoid nested
          compute() calls.
        - Falls back to using flo_znimg.get_bounded_subregion() for backwards
          compatibility.
        - If the transformed coordinates are completely outside the bounds of the
          floating image, a zero-filled array is returned.

    Example:
        # New approach with store path
        interpolated_block = interp_by_block(
            x=ref_block,
            transforms=[transform1, transform2],
            flo_store_path="/path/to/data.zarr",
            flo_array_shape=(3, 100, 100, 100),
            block_info=block_metadata,
        )

        # Legacy approach with ZarrNii instance
        interpolated_block = interp_by_block(
            x=ref_block,
            transforms=[transform1, transform2],
            flo_znimg=floating_image,
            block_info=block_metadata,
        )
    """
    # Extract the array location (block bounds) from block_info
    arr_location = block_info[0]["array-location"]

    # Generate coordinate grids for the reference image block
    xv, yv, zv = np.meshgrid(
        np.arange(arr_location[-3][0], arr_location[-3][1]),
        np.arange(arr_location[-2][0], arr_location[-2][1]),
        np.arange(arr_location[-1][0], arr_location[-1][1]),
        indexing="ij",
    )

    # Reshape grids into vectors for matrix multiplication
    xvf = xv.reshape((1, np.prod(xv.shape)))
    yvf = yv.reshape((1, np.prod(yv.shape)))
    zvf = zv.reshape((1, np.prod(zv.shape)))
    homog = np.ones(xvf.shape)

    xfm_vecs = np.vstack((xvf, yvf, zvf, homog))

    # Apply transformations sequentially
    for tfm in transforms:
        xfm_vecs = tfm.apply_transform(xfm_vecs)

    # Initialize the output array for interpolated values
    interpolated = np.zeros(x.shape)

    # Determine the required subregion of the floating image
    # Use direct zarr access if store path is provided, otherwise use legacy method
    if flo_store_path is not None and flo_array_shape is not None:
        grid_points, flo_vol = get_bounded_subregion_from_zarr(
            xfm_vecs,
            flo_store_path,
            flo_array_shape,
            flo_dataset_path,
            flo_storage_options,
        )
    elif flo_znimg is not None:
        # Legacy fallback
        grid_points, flo_vol = flo_znimg.get_bounded_subregion(xfm_vecs)
    else:
        raise ValueError(
            "Either (flo_store_path and flo_array_shape) or flo_znimg must be provided"
        )

    if grid_points is None and flo_vol is None:
        # Points are fully outside the floating image; return zeros
        return interpolated

    # Interpolate each channel of the floating image
    for c in range(flo_vol.shape[0]):
        interpolated[c, :, :, :] = (
            interpn(
                grid_points,
                flo_vol[c, :, :, :],
                xfm_vecs[:3, :].T,  # Transformed coordinates
                method=interp_method,
                bounds_error=False,
                fill_value=0,
            )
            .reshape((x.shape[-3], x.shape[-2], x.shape[-1]))
            .astype(block_info[None]["dtype"])
        )

    return interpolated


@define
class ZarrNii:
    """
    Zarr-based image with NIfTI compatibility using NgffImage internally.

    This class provides chainable operations on OME-Zarr data while maintaining
    compatibility with NIfTI workflows. It uses NgffImage objects internally for
    better multiscale support and metadata preservation.

    Attributes:
        ngff_image (nz.NgffImage): The internal NgffImage object containing data and metadata.
        axes_order (str): The order of the axes for NIfTI compatibility ('ZYX' or 'XYZ').
        xyz_orientation (str): The anatomical orientation string in XYZ axes order (e.g., 'RAS', 'LPI').
    """

    ngff_image: nz.NgffImage
    axes_order: str = "ZYX"
    xyz_orientation: str = "RAS"
    _omero: object | None = None

    # Properties that delegate to the internal NgffImage
    @property
    def data(self) -> da.Array:
        """Access the image data (dask array)."""
        return self.ngff_image.data

    @property
    def darr(self) -> da.Array:
        """Legacy property name for image data."""
        return self.ngff_image.data

    @darr.setter
    def darr(self, value: da.Array) -> None:
        """Set the image data via the legacy `darr` property."""
        self.ngff_image.data = value

    @data.setter
    def data(self, value: da.Array) -> None:
        """Set the image data via the data property"""
        self.ngff_image.data = value

    @property
    def shape(self) -> tuple:
        """Shape of the image data."""
        return self.ngff_image.data.shape

    @property
    def dims(self) -> list[str]:
        """Dimension names."""
        return self.ngff_image.dims

    @property
    def scale(self) -> dict[str, float]:
        """Scale information from NgffImage."""
        return self.ngff_image.scale

    @property
    def translation(self) -> dict[str, float]:
        """Translation information from NgffImage."""
        return self.ngff_image.translation

    @property
    def name(self) -> str:
        """Image name from NgffImage."""
        return self.ngff_image.name

    @property
    def orientation(self) -> str:
        """
        Legacy property for backward compatibility.

        Returns the xyz_orientation attribute to maintain backward compatibility
        with code that expects the 'orientation' property.

        Returns:
            str: The anatomical orientation string in XYZ axes order
        """
        return self.xyz_orientation

    @orientation.setter
    def orientation(self, value: str) -> None:
        """
        Legacy setter for backward compatibility.

        Sets the xyz_orientation attribute to maintain backward compatibility
        with code that sets the 'orientation' property.

        Args:
            value: The anatomical orientation string in XYZ axes order
        """
        object.__setattr__(self, "xyz_orientation", value)

    @property
    def affine(self) -> AffineTransform:
        """
        Affine transformation matrix derived from NgffImage scale and translation.

        Returns:
            AffineTransform: 4x4 affine transformation matrix in axes order of self.
        """
        return self.get_affine_transform()

    def get_affine_transform(self, axes_order: str = None) -> AffineTransform:
        """
        Get AffineTransform object from NgffImage metadata.

        Args:
            axes_order: Spatial axes order, defaults to self.axes_order

        Returns:
            AffineTransform object
        """
        matrix = self.get_affine_matrix(axes_order)
        return AffineTransform.from_array(matrix)

    def get_zarr_store_info(
        self,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract zarr store information from the dask array if available.

        Attempts to extract the underlying zarr store path and metadata from
        the dask array graph. This information can be used for direct zarr
        access without triggering dask compute() operations.

        Returns:
            Dictionary containing store information if available:
                - 'store_path': Path or URI to the zarr store
                - 'dataset_path': Path to the dataset within the zarr group
                - 'array_shape': Shape of the full array
            Returns None if the data is not backed by a zarr store.

        Raises:
            ValueError: If the dask array shape doesn't match the zarr array shape,
                indicating lazy operations that change shape (e.g., downsampling).

        Notes:
            - Only works if the dask array was created from zarr using da.from_zarr()
            - Returns None for in-memory arrays or arrays from other sources
            - Validates that zarr array shape matches dask array shape to ensure
              compatibility with direct zarr access
        """
        try:
            # Check if the dask array has a graph
            graph = self.data.__dask_graph__()

            # Look for zarr array in the graph
            # The first key in a from_zarr graph typically contains the zarr array
            for key in graph.keys():
                task = graph[key]
                # Check if this is a zarr array
                if hasattr(task, "store") and hasattr(task, "name"):
                    # Extract store information
                    import zarr
                    import zarr.storage

                    store = task.store
                    dataset_path = task.name.strip("/")

                    # Determine store path based on store type
                    # Handle both zarr v2 and v3 store types
                    store_path = None

                    # Try zarr v3 LocalStore first (has 'root' attribute)
                    if hasattr(store, "root"):
                        store_path = store.root
                    # Try zarr v2 DirectoryStore (has 'path' attribute)
                    elif hasattr(store, "path"):
                        store_path = store.path
                    # Try string representation as fallback
                    elif isinstance(store, str):
                        store_path = store
                    # Try str(store) which works for LocalStore
                    else:
                        store_str = str(store)
                        # LocalStore repr is like "file:///path/to/store"
                        if store_str.startswith("file://"):
                            store_path = store_str.replace("file://", "")
                        else:
                            store_path = store_str

                    if store_path:
                        # Validate that the zarr array shape matches the dask array shape
                        # This ensures no lazy operations have changed the shape
                        try:
                            # Convert store_path to string in case it's a Path object
                            store_path_str = str(store_path)

                            # Open the zarr store to get the actual array shape
                            if store_path_str.endswith(".zip"):
                                zarr_store = zarr.storage.ZipStore(
                                    store_path_str, mode="r"
                                )
                                root = zarr.open_group(zarr_store, mode="r")
                                zarr_array = root[dataset_path]
                                zarr_store.close()
                            else:
                                root = zarr.open_group(store_path_str, mode="r")
                                zarr_array = root[dataset_path]

                            zarr_shape = zarr_array.shape
                            dask_shape = self.shape

                            # Check if shapes match
                            if zarr_shape != dask_shape:
                                raise ValueError(
                                    f"Cannot use direct zarr access for apply_transform: "
                                    f"the floating image has lazy operations that change its shape. "
                                    f"Zarr array shape: {zarr_shape}, but dask array shape: {dask_shape}. "
                                    f"This typically happens when using downsample levels beyond what exists "
                                    f"in the zarr store, or when using downsample_near_isotropic option. "
                                    f"To fix this, save the floating image to an intermediate zarr file first:\n"
                                    f"  flo_znimg.to_ome_zarr('intermediate.zarr')\n"
                                    f"  flo_znimg = ZarrNii.from_ome_zarr('intermediate.zarr')\n"
                                    f"  transformed = flo_znimg.apply_transform(...)"
                                )

                        except (KeyError, FileNotFoundError) as e:
                            # Dataset doesn't exist at the specified path
                            raise ValueError(
                                f"Cannot use direct zarr access for apply_transform: "
                                f"the specified dataset '{dataset_path}' does not exist in the zarr store "
                                f"at '{store_path}'. This may happen when using a downsample level that "
                                f"doesn't exist in the zarr store. "
                                f"To fix this, save the floating image to an intermediate zarr file first:\n"
                                f"  flo_znimg.to_ome_zarr('intermediate.zarr')\n"
                                f"  flo_znimg = ZarrNii.from_ome_zarr('intermediate.zarr')\n"
                                f"  transformed = flo_znimg.apply_transform(...)"
                            ) from e

                        return {
                            "store_path": store_path,
                            "dataset_path": dataset_path,
                            "array_shape": self.shape,
                        }
        except ValueError:
            # Re-raise ValueError (our validation errors)
            raise
        except Exception:
            # If we can't extract store info for other reasons, return None
            pass

        return None

    # Legacy compatibility properties
    @property
    def axes(self) -> list[dict] | None:
        """Axes metadata - derived from NgffImage for compatibility."""
        axes = []
        for dim in self.ngff_image.dims:
            if dim == "c":
                axes.append({"name": "c", "type": "channel", "unit": None})
            else:
                axes.append({"name": dim, "type": "space", "unit": "micrometer"})
        return axes

    @property
    def coordinate_transformations(self) -> list[dict] | None:
        """Coordinate transformations - derived from NgffImage scale/translation."""
        transforms = []

        # Scale transform
        scale_list = [
            self.ngff_image.scale.get(dim, 1.0) for dim in self.ngff_image.dims
        ]
        transforms.append({"type": "scale", "scale": scale_list})

        # Translation transform
        translation_list = [
            self.ngff_image.translation.get(dim, 0.0) for dim in self.ngff_image.dims
        ]
        if any(v != 0.0 for v in translation_list):
            transforms.append({"type": "translation", "translation": translation_list})

        return transforms if transforms else None

    @property
    def omero(self) -> object | None:
        """Omero metadata object."""
        return self._omero

    # Constructor methods
    @classmethod
    def from_file(cls, path, **kwargs):
        if path.endswith((".nii", ".nii.gz")):
            return cls.from_nifti(path, **kwargs)
        elif path.endswith(".zarr") or path.endswith(".zip"):
            return cls.from_ome_zarr(path, **kwargs)
        else:
            raise ValueError(f"Unknown file extension: {path}")

    @classmethod
    def from_ngff_image(
        cls,
        ngff_image: nz.NgffImage,
        axes_order: str = "ZYX",
        xyz_orientation: str = "RAS",
        omero: object | None = None,
    ) -> ZarrNii:
        """
        Create ZarrNii from an existing NgffImage.

        Args:
            ngff_image: NgffImage to wrap
            axes_order: Spatial axes order for NIfTI compatibility
            xyz_orientation: Anatomical orientation string in XYZ axes order
            omero: Optional omero metadata object

        Returns:
            ZarrNii instance
        """
        return cls(
            ngff_image=ngff_image,
            axes_order=axes_order,
            xyz_orientation=xyz_orientation,
            _omero=omero,
        )

    @classmethod
    def from_darr(
        cls,
        darr: da.Array,
        affine: AffineTransform | None = None,
        axes_order: str = "ZYX",
        orientation: str = "RAS",
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        name: str = "image",
        omero: object | None = None,
        **kwargs,
    ) -> ZarrNii:
        """
        Create ZarrNii from dask array (legacy compatibility constructor).

        Args:
            darr: Dask array containing image data
            affine: Optional affine transformation
            axes_order: Spatial axes order
            orientation: Anatomical orientation string
            spacing: Voxel spacing (used if no affine provided)
            origin: Origin offset (used if no affine provided)
            name: Image name
            omero: Optional omero metadata

        Returns:
            ZarrNii instance
        """
        # Create scale and translation from affine if provided
        if affine is not None:
            # Extract scale and translation from affine matrix
            affine_matrix = affine.matrix
            if axes_order == "ZYX":
                scale = {
                    "z": affine_matrix[0, 0],
                    "y": affine_matrix[1, 1],
                    "x": affine_matrix[2, 2],
                }
                translation = {
                    "z": affine_matrix[0, 3],
                    "y": affine_matrix[1, 3],
                    "x": affine_matrix[2, 3],
                }
            else:  # XYZ
                scale = {
                    "x": affine_matrix[0, 0],
                    "y": affine_matrix[1, 1],
                    "z": affine_matrix[2, 2],
                }
                translation = {
                    "x": affine_matrix[0, 3],
                    "y": affine_matrix[1, 3],
                    "z": affine_matrix[2, 3],
                }
        else:
            # Use spacing and origin
            if axes_order == "ZYX":
                scale = {"z": spacing[0], "y": spacing[1], "x": spacing[2]}
                translation = {"z": origin[0], "y": origin[1], "x": origin[2]}
            else:  # XYZ
                scale = {"x": spacing[0], "y": spacing[1], "z": spacing[2]}
                translation = {"x": origin[0], "y": origin[1], "z": origin[2]}

        # Create dimensions based on data shape after dimension adjustments
        final_ndim = len(darr.shape)
        if final_ndim == 4:
            # 4D: (c, z, y, x) or (c, x, y, z) - standard case
            dims = ["c"] + list(axes_order.lower())
        elif final_ndim == 5:
            # 5D: (t, c, z, y, x) or (t, c, x, y, z) - time dimension included
            dims = ["t", "c"] + list(axes_order.lower())
        else:
            # Fallback for other cases
            dims = ["c"] + list(axes_order.lower())

        # Create NgffImage
        ngff_image = nz.NgffImage(
            data=darr, dims=dims, scale=scale, translation=translation, name=name
        )

        return cls(
            ngff_image=ngff_image,
            axes_order=axes_order,
            xyz_orientation=orientation,
            _omero=omero,
        )

    # Legacy compatibility method names
    def __init__(
        self,
        darr=None,
        affine=None,
        axes_order="ZYX",
        orientation="RAS",
        xyz_orientation=None,
        ngff_image=None,
        _omero=None,
        **kwargs,
    ):
        """
        Constructor with backward compatibility for old signature.
        """
        # Handle backwards compatibility: if xyz_orientation is provided, use it
        # Otherwise, use orientation for backwards compatibility
        final_orientation = (
            xyz_orientation if xyz_orientation is not None else orientation
        )

        if ngff_image is not None:
            # New signature
            object.__setattr__(self, "ngff_image", ngff_image)
            object.__setattr__(self, "axes_order", axes_order)
            object.__setattr__(self, "xyz_orientation", final_orientation)
            object.__setattr__(self, "_omero", _omero)
        elif darr is not None:
            # Legacy signature - delegate to from_darr
            instance = self.from_darr(
                darr=darr,
                affine=affine,
                axes_order=axes_order,
                orientation=final_orientation,
                **kwargs,
            )
            object.__setattr__(self, "ngff_image", instance.ngff_image)
            object.__setattr__(self, "axes_order", instance.axes_order)
            object.__setattr__(self, "xyz_orientation", instance.xyz_orientation)
            object.__setattr__(self, "_omero", instance._omero)
        else:
            raise ValueError("Must provide either ngff_image or darr")

    @classmethod
    def from_ome_zarr(
        cls,
        store_or_path: str | Any,
        level: int = 0,
        channels: list[int] | None = None,
        channel_labels: list[str] | None = None,
        timepoints: list[int] | None = None,
        storage_options: dict[str, Any] | None = None,
        axes_order: str = "ZYX",
        orientation: str | None = None,
        downsample_near_isotropic: bool = False,
        chunks: tuple[int, Ellipsis] | Literal[auto] = "auto",
        rechunk: bool = False,
    ) -> ZarrNii:
        """Load ZarrNii from OME-Zarr store with flexible options.

        Creates a ZarrNii instance from an OME-Zarr store, supporting multiscale
        pyramids, channel/timepoint selection, and various storage backends.
        Automatically handles metadata extraction and format conversion.

        Args:
            store_or_path: Store or path to OME-Zarr file. Supports:
                - Local file paths
                - Remote URLs (s3://, http://, etc.)
                - ZIP files (.zip extension)
                - Zarr store objects
            level: Pyramid level to load (0 = highest resolution). If level
                exceeds available levels, applies lazy downsampling
            channels: List of channel indices to load (0-based). Mutually
                exclusive with channel_labels
            channel_labels: List of channel names to load by label. Requires
                OMERO metadata. Mutually exclusive with channels
            timepoints: List of timepoint indices to load (0-based). If None,
                loads all available timepoints
            storage_options: Additional options for zarr storage backend
                (e.g., credentials for cloud storage)
            axes_order: Spatial axis order for NIfTI compatibility.
                Either "ZYX" or "XYZ"
            orientation: Default anatomical orientation if not in metadata.
                Standard orientations like "RAS", "LPI", etc. This is always
                interpreted in XYZ axes order for consistency. This setting will override
                any orientation defined in the OME zarr metadata
            downsample_near_isotropic: If True, automatically downsample
                dimensions with smaller voxel sizes to achieve near-isotropic
                resolution
            chunks: chunking strategy, or explicit chunk sizes to use if not automatic
            rechunk: If True, rechunks the dataset after lazy loading, based
                on the chunks parameter

        Returns:
            ZarrNii instance with loaded data and metadata

        Raises:
            ValueError: If both channels and channel_labels are specified,
                or if invalid level/indices are provided
            FileNotFoundError: If store_or_path does not exist
            KeyError: If specified channel labels are not found
            IOError: If unable to read from the storage backend

        Examples:
            >>> # Load full resolution data
            >>> znii = ZarrNii.from_ome_zarr("/path/to/data.zarr")

            >>> # Load specific channels and pyramid level
            >>> znii = ZarrNii.from_ome_zarr(
            ...     "/path/to/data.zarr",
            ...     level=1,
            ...     channels=[0, 2],
            ...     orientation="LPI"
            ... )

            >>> # Load from cloud storage
            >>> znii = ZarrNii.from_ome_zarr(
            ...     "s3://bucket/data.zarr",
            ...     storage_options={"key": "access_key", "secret": "secret"}
            ... )

        Notes:
            **Orientation Metadata Backwards Compatibility:**

            This method implements backwards compatibility for orientation metadata:

            1. **Override**: Setting the orientation here will override
               any orientation defined in the OME Zarr metadata.

            2. **Zarr Metadata**: Checks for 'xyz_orientation' first (new format),
               then falls back to 'orientation' (legacy format)

            3. **Legacy Fallback**: When only legacy 'orientation' is found, the
               orientation string is automatically reversed to convert from ZYX-based
               encoding (legacy) to XYZ-based encoding (current standard)

            4. **Default Fallback**: If no orientation metadata is found, uses RAS
               orientation as the default.

            Examples of the conversion:
            - Legacy 'orientation'='SAR' (ZYX) → 'xyz_orientation'='RAS' (XYZ)
            - Legacy 'orientation'='IPL' (ZYX) → 'xyz_orientation'='LPI' (XYZ)

            This ensures consistent orientation handling while maintaining backwards
            compatibility with existing OME-Zarr files that use the legacy format.
        """
        # Validate channel and timepoint selection arguments
        if channels is not None and channel_labels is not None:
            raise ValueError("Cannot specify both 'channels' and 'channel_labels'")

        # Load the multiscales object
        try:
            if isinstance(store_or_path, str):
                # Handle ZIP files by creating a ZipStore
                if store_or_path.endswith(".zip"):
                    import zarr

                    store = zarr.storage.ZipStore(store_or_path, mode="r")
                    multiscales = nz.from_ngff_zarr(
                        store, storage_options=storage_options or {}
                    )
                    # Note: We'll close the store after extracting metadata
                else:
                    multiscales = nz.from_ngff_zarr(
                        store_or_path, storage_options=storage_options or {}
                    )
            else:
                multiscales = nz.from_ngff_zarr(store_or_path)
        except Exception:
            # Fallback for older zarr/ngff_zarr versions
            if isinstance(store_or_path, str):
                if store_or_path.endswith(".zip"):
                    import zarr

                    store = zarr.storage.ZipStore(store_or_path, mode="r")
                    multiscales = nz.from_ngff_zarr(store)
                else:
                    store = fsspec.get_mapper(store_or_path, **storage_options or {})
                    multiscales = nz.from_ngff_zarr(store)
            else:
                store = store_or_path
                multiscales = nz.from_ngff_zarr(store)

        # Extract omero metadata if available
        omero_metadata = None
        try:
            import zarr

            if isinstance(store_or_path, str):
                if store_or_path.endswith(".zip"):
                    zip_store = zarr.storage.ZipStore(store_or_path, mode="r")
                    group = zarr.open_group(zip_store, mode="r")
                    # Close zip store after getting group
                    zip_store.close()
                else:
                    group = zarr.open_group(store_or_path, mode="r")

            else:
                group = zarr.open_group(store_or_path, mode="r")

            if "omero" in group.attrs:
                omero_dict = group.attrs["omero"]

                # Create a simple object to hold omero metadata
                class OmeroMetadata:
                    def __init__(self, omero_dict):
                        self.channels = []
                        if "channels" in omero_dict:
                            for ch_dict in omero_dict["channels"]:
                                # Create channel objects
                                class ChannelMetadata:
                                    def __init__(self, ch_dict):
                                        self.label = ch_dict.get("label", "")
                                        self.color = ch_dict.get("color", "")
                                        if "window" in ch_dict:

                                            class WindowMetadata:
                                                def __init__(self, win_dict):
                                                    self.min = win_dict.get("min", 0.0)
                                                    self.max = win_dict.get(
                                                        "max", 65535.0
                                                    )
                                                    self.start = win_dict.get(
                                                        "start", 0.0
                                                    )
                                                    self.end = win_dict.get(
                                                        "end", 65535.0
                                                    )

                                            self.window = WindowMetadata(
                                                ch_dict["window"]
                                            )
                                        else:
                                            self.window = None

                                self.channels.append(ChannelMetadata(ch_dict))

                omero_metadata = OmeroMetadata(omero_dict)
        except Exception:
            # If we can't load omero metadata, that's okay
            pass

        # Read orientation metadata with backwards compatibility support
        # Priority: xyz_orientation (new) > orientation (legacy, with reversal)
        try:
            import zarr

            if orientation is None:
                if isinstance(store_or_path, str):
                    if store_or_path.endswith(".zip"):
                        zip_store = zarr.storage.ZipStore(store_or_path, mode="r")
                        group = zarr.open_group(zip_store, mode="r")
                        # Check for new xyz_orientation first, then fallback to legacy orientation
                        if "xyz_orientation" in group.attrs:
                            orientation = group.attrs["xyz_orientation"]
                        elif "orientation" in group.attrs:
                            # Legacy orientation is ZYX-based, reverse it to get XYZ-based orientation
                            legacy_orientation = group.attrs["orientation"]
                            orientation = reverse_orientation_string(legacy_orientation)
                        # If neither found, use the provided default orientation
                        zip_store.close()
                    else:
                        group = zarr.open_group(store_or_path, mode="r")
                        # Check for new xyz_orientation first, then fallback to legacy orientation
                        if "xyz_orientation" in group.attrs:
                            orientation = group.attrs["xyz_orientation"]
                        elif "orientation" in group.attrs:
                            # Legacy orientation is ZYX-based, reverse it to get XYZ-based orientation
                            legacy_orientation = group.attrs["orientation"]
                            orientation = reverse_orientation_string(legacy_orientation)
                        # If neither found, use the provided default orientation
                else:
                    group = zarr.open_group(store_or_path, mode="r")
                    # Check for new xyz_orientation first, then fallback to legacy orientation
                    if "xyz_orientation" in group.attrs:
                        orientation = group.attrs["xyz_orientation"]
                    elif "orientation" in group.attrs:
                        # Legacy orientation is ZYX-based, reverse it to get XYZ-based orientation
                        legacy_orientation = group.attrs["orientation"]
                        orientation = reverse_orientation_string(legacy_orientation)
                    # If neither found, use the provided default orientation

        except Exception:
            # If we can't read orientation metadata, use the provided default
            pass

        # If orientation is still None, use the fallback default
        if orientation is None:
            orientation = "RAS"
        # Determine the available pyramid levels and handle lazy downsampling
        max_level = len(multiscales.images) - 1
        actual_level = min(level, max_level)
        do_downsample = level > max_level

        # Get the highest available level
        ngff_image = multiscales.images[actual_level]

        # Handle channel and timepoint selection and filter omero metadata accordingly
        filtered_omero = omero_metadata
        if channels is not None or channel_labels is not None or timepoints is not None:
            ngff_image, filtered_omero = _select_dimensions_from_image_with_omero(
                ngff_image,
                multiscales,
                channels,
                channel_labels,
                timepoints,
                omero_metadata,
            )

        # Create ZarrNii instance with xyz_orientation
        znimg = cls(
            ngff_image=ngff_image,
            axes_order=axes_order,
            xyz_orientation=orientation,
            _omero=filtered_omero,
        )

        # Apply lazy downsampling if needed
        if do_downsample:
            level_ds = level - max_level
            downsample_factor = 2**level_ds

            # Get spatial dims based on axes order
            spatial_dims = ["z", "y", "x"] if axes_order == "ZYX" else ["x", "y", "z"]

            # Apply downsampling using the existing method
            znimg = znimg.downsample(
                factors=downsample_factor, spatial_dims=spatial_dims
            )

        # Apply near-isotropic downsampling if requested
        if downsample_near_isotropic:
            znimg = _apply_near_isotropic_downsampling(znimg, axes_order)

        if rechunk:
            znimg.data = znimg.data.rechunk(chunks)

        return znimg

    @classmethod
    def from_nifti(
        cls,
        path: str | bytes,
        chunks: str | tuple[int, ...] = "auto",
        axes_order: str = "XYZ",
        name: str | None = None,
        as_ref: bool = False,
        zooms: tuple[float, float, float] | None = None,
    ) -> ZarrNii:
        """Load ZarrNii from NIfTI file with flexible loading options.

        Creates a ZarrNii instance from a NIfTI file, automatically converting
        the data to dask arrays and extracting spatial transformation information.
        Supports both full data loading and reference-only loading for memory
        efficiency.

        Args:
            path: File path to NIfTI file (.nii, .nii.gz, .img/.hdr)
            chunks: Dask array chunking strategy. Can be:
                - "auto": Automatic chunking based on file size
                - Tuple of ints: Manual chunk sizes for each dimension
                - Dict mapping axis to chunk size
            axes_order: Spatial axis ordering convention. Either:
                - "XYZ": X=left-right, Y=anterior-posterior, Z=inferior-superior
                - "ZYX": Z=inferior-superior, Y=anterior-posterior, X=left-right
            name: Optional name for the resulting NgffImage. If None,
                uses filename without extension
            as_ref: If True, creates empty dask array with correct shape/metadata
                without loading actual image data (memory efficient for templates)
            zooms: Target voxel spacing as (x, y, z) in mm. Only valid when
                as_ref=True. Adjusts shape and affine accordingly

        Returns:
            ZarrNii instance containing NIfTI data and spatial metadata

        Raises:
            ValueError: If zooms specified with as_ref=False, or invalid axes_order
            FileNotFoundError: If NIfTI file does not exist
            OSError: If unable to read NIfTI file
            nibabel.filebasedimages.ImageFileError: If file is not valid NIfTI

        Examples:
            >>> # Load full NIfTI data
            >>> znii = ZarrNii.from_nifti("/path/to/brain.nii.gz")

            >>> # Load with custom chunking and axis order
            >>> znii = ZarrNii.from_nifti(
            ...     "/path/to/data.nii",
            ...     chunks=(64, 64, 64),
            ...     axes_order="ZYX"
            ... )

            >>> # Create reference with target resolution
            >>> znii_ref = ZarrNii.from_nifti(
            ...     "/path/to/template.nii.gz",
            ...     as_ref=True,
            ...     zooms=(2.0, 2.0, 2.0)
            ... )

        Notes:
            The method automatically handles NIfTI orientation codes and converts
            them to the specified axes_order for consistency with OME-Zarr workflows.
        """
        if not as_ref and zooms is not None:
            raise ValueError("`zooms` can only be used when `as_ref=True`.")

        # Load NIfTI file
        nifti_img = nib.load(path)
        shape = nifti_img.header.get_data_shape()
        affine_matrix = nifti_img.affine.copy()

        # infer orientation from the affine
        orientation = _affine_to_orientation(affine_matrix)

        # Adjust shape and affine if zooms are provided
        if zooms is not None:
            in_zooms = np.sqrt(
                (affine_matrix[:3, :3] ** 2).sum(axis=0)
            )  # Current voxel spacing
            scaling_factor = in_zooms / zooms
            new_shape = [
                int(np.floor(shape[0] * scaling_factor[2])),  # Z
                int(np.floor(shape[1] * scaling_factor[1])),  # Y
                int(np.floor(shape[2] * scaling_factor[0])),  # X
            ]
            np.fill_diagonal(affine_matrix[:3, :3], zooms)
        else:
            new_shape = shape

        if as_ref:
            # Create an empty dask array with the adjusted shape
            darr = da.empty((1, *new_shape), chunks=chunks, dtype="float32")
        else:
            # Load the NIfTI data and convert to a dask array
            array = nifti_img.get_fdata()
            darr = da.from_array(array, chunks=chunks)

        # Add channel and time dimensions if not present
        original_ndim = len(darr.shape)

        if original_ndim == 3:
            # 3D data: add channel dimension -> (c, z, y, x) or (c, x, y, z)
            darr = darr[np.newaxis, ...]
        elif original_ndim == 4:
            # 4D data: could be (c, z, y, x) or (t, z, y, x) - assume channel by default
            # User can specify if it's time by using appropriate axes_order
            pass  # Keep as is - 4D is already handled
        elif original_ndim == 5:
            # 5D data: assume (t, z, y, x, c) and handle appropriately
            pass  # Keep as is - 5D is already the target format
        else:
            # For 1D, 2D, or >5D data, add channel dimension and let user handle
            darr = darr[np.newaxis, ...]

        # Create dimensions based on data shape after dimension adjustments
        final_ndim = len(darr.shape)
        if final_ndim == 4:
            # 4D: (c, z, y, x) or (c, x, y, z) - standard case
            dims = ["c"] + list(axes_order.lower())
        elif final_ndim == 5:
            # 5D: (t, c, z, y, x) or (t, c, x, y, z) - time dimension included
            dims = ["t", "c"] + list(axes_order.lower())
        else:
            # Fallback for other cases
            dims = ["c"] + list(axes_order.lower())

        # Extract scale and translation from affine
        scale = {}
        translation = {}
        spatial_dims = ["z", "y", "x"] if axes_order == "ZYX" else ["x", "y", "z"]

        for i, dim in enumerate(spatial_dims):
            scale[dim] = np.sqrt((affine_matrix[i, :3] ** 2).sum())
            translation[dim] = affine_matrix[i, 3]

        # Create NgffImage
        if name is None:
            name = f"nifti_image_{path}"

        ngff_image = nz.NgffImage(
            data=darr, dims=dims, scale=scale, translation=translation, name=name
        )

        return cls(
            ngff_image=ngff_image, axes_order=axes_order, xyz_orientation=orientation
        )

    # Chainable operations - each returns a new ZarrNii instance
    def crop(
        self,
        bbox_min: tuple[float, ...] | list[tuple[tuple[float, ...], tuple[float, ...]]],
        bbox_max: tuple[float, ...] | None = None,
        spatial_dims: list[str] | None = None,
        physical_coords: bool = False,
    ) -> ZarrNii | list[ZarrNii]:
        """Extract a spatial region or multiple regions from the image.

        Crops the image to the specified bounding box coordinates, preserving
        all metadata and non-spatial dimensions (channels, time). The cropping
        is performed in voxel coordinates by default, or physical coordinates
        if specified. Can crop a single region or multiple regions at once.

        Args:
            bbox_min: Either:
                - Minimum corner coordinates of bounding box as tuple
                  (when bbox_max is provided). Length should match number of
                  spatial dimensions (x, y, z order)
                - List of (bbox_min, bbox_max) tuples for batch cropping
                  (when bbox_max is None)
            bbox_max: Maximum corner coordinates of bounding box as tuple.
                Length should match number of spatial dimensions (x, y, z order).
                Should be None when bbox_min is a list of bounding boxes.
            spatial_dims: Names of spatial dimensions to crop. If None,
                automatically derived from axes_order ("z","y","x" for ZYX
                or "x","y","z" for XYZ)
            physical_coords: If True, bbox_min and bbox_max are in physical/world
                coordinates (mm). If False, they are in voxel coordinates.
                Default is False.

        Returns:
            New ZarrNii instance with cropped data (single crop) or list of
            ZarrNii instances (batch crop) with updated spatial metadata

        Raises:
            ValueError: If bbox coordinates are invalid or out of bounds, or
                if both list and bbox_max are provided
            IndexError: If bbox dimensions don't match spatial dimensions

        Examples:
            >>> # Crop 3D region (voxel coordinates)
            >>> cropped = znii.crop((10, 20, 30), (110, 120, 130))

            >>> # Crop with physical coordinates
            >>> cropped = znii.crop((10.5, 20.5, 30.5), (110.5, 120.5, 130.5),
            ...                      physical_coords=True)

            >>> # Crop with explicit spatial dimensions
            >>> cropped = znii.crop(
            ...     (50, 60, 70), (150, 160, 170),
            ...     spatial_dims=["x", "y", "z"]
            ... )

            >>> # Batch crop multiple regions
            >>> bboxes = [
            ...     ((10, 20, 30), (60, 70, 80)),
            ...     ((100, 110, 120), (150, 160, 170))
            ... ]
            >>> cropped_list = znii.crop(bboxes, physical_coords=True)

        Notes:
            - Coordinates are in voxel space (0-based indexing) by default
            - Physical coordinates are in RAS orientation (Right-Anterior-Superior)
            - The cropped region includes bbox_min but excludes bbox_max
            - All non-spatial dimensions (channels, time) are preserved
            - Spatial transformations are automatically updated
            - When batch cropping, all patches share the same spatial_dims and
              physical_coords settings
        """
        # Check if this is batch cropping (list of bounding boxes)
        # A batch crop is a list of (bbox_min, bbox_max) tuples
        # Each element should be a tuple/list of two elements
        is_batch_crop = (
            isinstance(bbox_min, list)
            and len(bbox_min) > 0
            and isinstance(bbox_min[0], (tuple, list))
            and len(bbox_min[0]) == 2
        )

        if is_batch_crop:
            if bbox_max is not None:
                raise ValueError(
                    "bbox_max should be None when bbox_min is a list of bounding boxes"
                )
            # Batch crop: recursively call crop for each bounding box
            return [
                self.crop(bmin, bmax, spatial_dims, physical_coords)
                for bmin, bmax in bbox_min
            ]

        # Single crop: original implementation
        if bbox_max is None:
            raise ValueError("bbox_max is required when bbox_min is not a list")

        if spatial_dims is None:
            spatial_dims = (
                ["z", "y", "x"] if self.axes_order == "ZYX" else ["x", "y", "z"]
            )

        # Convert physical coordinates to voxel coordinates if needed
        if physical_coords:
            # Physical coords are always in (x, y, z) order
            # Convert to homogeneous coordinates
            phys_min = np.array(list(bbox_min) + [1.0])
            phys_max = np.array(list(bbox_max) + [1.0])

            # Get inverse affine to convert from physical to voxel
            affine_inv = np.linalg.inv(
                self.get_affine_matrix(axes_order="XYZ")
            )  # TODO: should this always be xyz affine??

            # Transform to voxel coordinates
            voxel_min = affine_inv @ phys_min
            voxel_max = affine_inv @ phys_max

            # Extract voxel coordinates (x, y, z)
            voxel_min_xyz = voxel_min[:3]
            voxel_max_xyz = voxel_max[:3]

            # Round to nearest integer voxel indices
            voxel_min_xyz = np.round(voxel_min_xyz).astype(int)
            voxel_max_xyz = np.round(voxel_max_xyz).astype(int)

            # Ensure max >= min
            voxel_min_xyz = np.minimum(voxel_min_xyz, voxel_max_xyz)
            voxel_max_xyz = np.maximum(
                np.round(affine_inv @ phys_min).astype(int)[:3],
                np.round(affine_inv @ phys_max).astype(int)[:3],
            )

            # Create mapping from x,y,z to voxel coordinates
            bbox_min = voxel_min_xyz
            bbox_max = voxel_max_xyz

        # Create mapping from x,y,z to voxel coordinates
        bbox_vox_min = {
            "x": bbox_min[0],
            "y": bbox_min[1],
            "z": bbox_min[2],
        }
        bbox_vox_max = {
            "x": bbox_max[0],
            "y": bbox_max[1],
            "z": bbox_max[2],
        }

        dim_flips = _axcodes2flips(self.orientation)
        cropped_image = crop_ngff_image(
            self.ngff_image, bbox_vox_min, bbox_vox_max, dim_flips
        )
        return ZarrNii(
            ngff_image=cropped_image,
            axes_order=self.axes_order,
            xyz_orientation=self.xyz_orientation,
            _omero=self._omero,
        )

    def crop_with_bounding_box(self, bbox_min, bbox_max, ras_coords=False):
        """Legacy method name for crop.

        Args:
            bbox_min: Minimum corner coordinates
            bbox_max: Maximum corner coordinates
            ras_coords: If True, coordinates are in RAS physical space (deprecated,
                use physical_coords parameter of crop() instead)
        """
        return self.crop(bbox_min, bbox_max, physical_coords=ras_coords)

    def crop_centered(
        self,
        centers: tuple[float, float, float] | list[tuple[float, float, float]],
        patch_size: tuple[int, int, int],
        spatial_dims: list[str] | None = None,
        fill_value: float = 0.0,
    ) -> ZarrNii | list[ZarrNii]:
        """Extract fixed-size patches centered at specified coordinates.

        Crops the image to extract patches of a fixed size (in voxels) centered
        at the given physical coordinates. This is particularly useful for machine
        learning workflows where training patches must have consistent dimensions.
        The method can process a single center or multiple centers at once.

        Patches that extend beyond image boundaries are padded with the fill_value
        to ensure all patches have exactly the requested size.

        Args:
            centers: Either:
                - Single center coordinate as (x, y, z) tuple in physical space (mm)
                - List of center coordinates for batch processing
            patch_size: Size of the patch in voxels as (x, y, z) tuple.
                This defines the dimensions of each cropped region in voxel space.
                All returned patches will have exactly this size.
            spatial_dims: Names of spatial dimensions to crop. If None,
                automatically derived from axes_order ("z","y","x" for ZYX
                or "x","y","z" for XYZ). Default is None.
            fill_value: Value to use for padding when patches extend beyond
                image boundaries. Default is 0.0.

        Returns:
            Single ZarrNii instance (when centers is a single tuple) or list of
            ZarrNii instances (when centers is a list) with cropped data and
            updated spatial metadata. All patches will have exactly the shape
            specified by patch_size (plus any non-spatial dimensions).

        Raises:
            ValueError: If coordinates/dimensions are invalid
            IndexError: If patch_size dimensions don't match spatial dimensions

        Examples:
            >>> # Extract single 256x256x256 voxel patch at a coordinate
            >>> center = (50.0, 60.0, 70.0)  # physical coordinates in mm
            >>> patch = znii.crop_centered(center, patch_size=(256, 256, 256))
            >>>
            >>> # Extract multiple patches for ML training
            >>> centers = [
            ...     (50.0, 60.0, 70.0),
            ...     (100.0, 110.0, 120.0),
            ...     (150.0, 160.0, 170.0)
            ... ]
            >>> patches = znii.crop_centered(centers, patch_size=(128, 128, 128))
            >>> # Returns list of 3 ZarrNii instances, all with shape (1, 128, 128, 128)
            >>>
            >>> # Use with atlas sampling for ML training workflow
            >>> centers = atlas.sample_region_patches(
            ...     n_patches=100,
            ...     region_ids="cortex",
            ...     seed=42
            ... )
            >>> patches = image.crop_centered(centers, patch_size=(256, 256, 256))
            >>>
            >>> # Use custom fill value for padding
            >>> patch = znii.crop_centered(center, patch_size=(256, 256, 256), fill_value=-1.0)

        Notes:
            - Centers are in physical/world coordinates (mm), always in (x, y, z) order
            - patch_size is in voxels, in (x, y, z) order
            - The patch is centered at the given coordinate, extending ±patch_size/2
            - If patch_size is odd, the center voxel is included
            - Patches near boundaries are padded with fill_value to maintain size
            - All patches are guaranteed to have exactly the requested size
            - Useful for ML training where fixed patch sizes are required
            - Coordinates from atlas.sample_region_patches() can be used directly
        """
        # Check if this is batch processing (list of centers)
        is_batch = isinstance(centers, list)

        if is_batch:
            # Batch processing: recursively call crop_centered for each center
            return [
                self.crop_centered(center, patch_size, spatial_dims, fill_value)
                for center in centers
            ]

        # Single center processing
        if spatial_dims is None:
            spatial_dims = (
                ["z", "y", "x"] if self.axes_order == "ZYX" else ["x", "y", "z"]
            )

        # Convert center from physical to voxel coordinates
        # Centers are always in (x, y, z) order
        center_phys = np.array(list(centers) + [1.0])

        # Get inverse affine to convert from physical to voxel
        affine_inv = np.linalg.inv(self.get_affine_matrix(axes_order="XYZ"))

        # Transform to voxel coordinates
        center_voxel = affine_inv @ center_phys
        center_voxel_xyz = center_voxel[:3]

        # patch_size is in voxels, in (x, y, z) order
        patch_size_np = np.array(patch_size)
        half_patch = patch_size_np / 2.0

        # Calculate desired bounding box in voxel coordinates (may extend beyond image)
        voxel_min_xyz = center_voxel_xyz - half_patch
        voxel_max_xyz = center_voxel_xyz + half_patch

        # Round to nearest integer voxel indices
        voxel_min_xyz = np.round(voxel_min_xyz).astype(int)
        voxel_max_xyz = np.round(voxel_max_xyz).astype(int)

        # Ensure we get exactly the requested patch size
        # Adjust max to ensure patch_size is respected
        voxel_max_xyz = voxel_min_xyz + patch_size_np

        # Get image dimensions in voxel space
        # Map spatial dims to their indices
        spatial_dim_indices = {}
        for i, dim in enumerate(self.ngff_image.dims):
            if dim.lower() in [d.lower() for d in spatial_dims]:
                spatial_dim_indices[dim.lower()] = i

        image_shape_xyz = np.array(
            [
                self.ngff_image.data.shape[spatial_dim_indices["x"]],
                self.ngff_image.data.shape[spatial_dim_indices["y"]],
                self.ngff_image.data.shape[spatial_dim_indices["z"]],
            ]
        )

        # Calculate the actual crop region (clipped to image bounds)
        crop_min_xyz = np.maximum(voxel_min_xyz, 0)
        crop_max_xyz = np.minimum(voxel_max_xyz, image_shape_xyz)

        # Ensure crop_max >= crop_min to avoid empty arrays
        crop_max_xyz = np.maximum(crop_min_xyz, crop_max_xyz)

        # Calculate padding needed on each side
        pad_before_xyz = crop_min_xyz - voxel_min_xyz  # How much we're clipped at start
        pad_after_xyz = voxel_max_xyz - crop_max_xyz  # How much we're clipped at end

        # Check if the entire patch is outside the image bounds
        # This happens when crop_min >= crop_max in any dimension after clipping
        is_completely_outside = np.any(crop_min_xyz >= crop_max_xyz)

        if is_completely_outside:
            # The entire patch is outside the image bounds
            # Create a completely padded array with the fill value
            import dask.array as da

            # Build the full patch shape
            full_shape = []
            spatial_idx = 0
            for dim in self.ngff_image.dims:
                if dim.lower() in [d.lower() for d in spatial_dims]:
                    full_shape.append(patch_size_np[spatial_idx])
                    spatial_idx += 1
                else:
                    # Non-spatial dimension - keep original size
                    dim_idx = self.ngff_image.dims.index(dim)
                    full_shape.append(self.ngff_image.data.shape[dim_idx])

            # Create array filled with fill_value
            padded_data = da.full(
                tuple(full_shape),
                fill_value,
                dtype=self.ngff_image.data.dtype,
                chunks=self.ngff_image.data.chunksize,
            )

            # Calculate translation for the patch center
            # The translation should be at voxel_min_xyz (the desired start of patch)
            new_translation = {}
            for dim in self.ngff_image.dims:
                if dim.lower() in [d.lower() for d in spatial_dims]:
                    dim_lower = dim.lower()
                    if dim_lower == "x":
                        voxel_start = voxel_min_xyz[0]
                    elif dim_lower == "y":
                        voxel_start = voxel_min_xyz[1]
                    elif dim_lower == "z":
                        voxel_start = voxel_min_xyz[2]
                    else:
                        voxel_start = 0

                    # Translation is voxel_start * scale + original translation
                    new_translation[dim] = voxel_start * self.ngff_image.scale.get(
                        dim, 1.0
                    ) + self.ngff_image.translation.get(dim, 0.0)
                elif dim in self.ngff_image.translation:
                    new_translation[dim] = self.ngff_image.translation[dim]

            # Create NgffImage with the padded data
            padded_image = nz.NgffImage(
                data=padded_data,
                dims=self.ngff_image.dims,
                scale=self.ngff_image.scale.copy(),
                translation=new_translation,
                name=self.ngff_image.name,
            )

            return ZarrNii(
                ngff_image=padded_image,
                axes_order=self.axes_order,
                xyz_orientation=self.xyz_orientation,
                _omero=self._omero,
            )

        # Create mapping from x,y,z to voxel coordinates for cropping
        bbox_vox_min = {
            "x": crop_min_xyz[0],
            "y": crop_min_xyz[1],
            "z": crop_min_xyz[2],
        }
        bbox_vox_max = {
            "x": crop_max_xyz[0],
            "y": crop_max_xyz[1],
            "z": crop_max_xyz[2],
        }

        dim_flips = _axcodes2flips(self.orientation)
        # Crop the actual image data that exists
        cropped_image = crop_ngff_image(
            self.ngff_image, bbox_vox_min, bbox_vox_max, dim_flips
        )

        # Check if padding is needed
        needs_padding = np.any(pad_before_xyz > 0) or np.any(pad_after_xyz > 0)

        if needs_padding:
            # Build padding specification for all dimensions
            pad_width = []
            spatial_idx = 0

            for dim in cropped_image.dims:
                if dim.lower() in [d.lower() for d in spatial_dims]:
                    # Spatial dimension - may need padding
                    dim_lower = dim.lower()
                    if dim_lower == "x":
                        pad_width.append((pad_before_xyz[0], pad_after_xyz[0]))
                    elif dim_lower == "y":
                        pad_width.append((pad_before_xyz[1], pad_after_xyz[1]))
                    elif dim_lower == "z":
                        pad_width.append((pad_before_xyz[2], pad_after_xyz[2]))
                    spatial_idx += 1
                else:
                    # Non-spatial dimension - no padding
                    pad_width.append((0, 0))

            # Apply padding
            import dask.array as da

            padded_data = da.pad(
                cropped_image.data,
                pad_width=pad_width,
                mode="constant",
                constant_values=fill_value,
            )

            # Adjust translation for the padding
            new_translation = cropped_image.translation.copy()

            for i, dim in enumerate(bbox_vox_min.keys()):
                new_translation[dim] = new_translation[dim] + dim_flips[
                    dim
                ] * pad_before_xyz[i] * cropped_image.scale.get(dim, 1.0)

            # Create padded NgffImage
            cropped_image = nz.NgffImage(
                data=padded_data,
                dims=cropped_image.dims,
                scale=cropped_image.scale,
                translation=new_translation,
                name=cropped_image.name,
            )

        return ZarrNii(
            ngff_image=cropped_image,
            axes_order=self.axes_order,
            xyz_orientation=self.xyz_orientation,
            _omero=self._omero,
        )

    def downsample(
        self,
        factors: int | list[int] | None = None,
        along_x: int = 1,
        along_y: int = 1,
        along_z: int = 1,
        level: int | None = None,
        spatial_dims: list[str] | None = None,
    ) -> ZarrNii:
        """Reduce image resolution by downsampling.

        Performs spatial downsampling by averaging blocks of voxels, effectively
        reducing image resolution and size. Multiple parameter options provide
        flexibility for different downsampling strategies.

        Args:
            factors: Downsampling factors for spatial dimensions. Can be:
                - int: Same factor applied to all spatial dimensions
                - List[int]: Per-dimension factors matching spatial_dims order
                - None: Use other parameters to determine factors
            along_x: Downsampling factor for X dimension (legacy parameter)
            along_y: Downsampling factor for Y dimension (legacy parameter)
            along_z: Downsampling factor for Z dimension (legacy parameter)
            level: Power-of-2 downsampling level (factors = 2^level).
                Takes precedence over along_* parameters
            spatial_dims: Names of spatial dimensions. If None, derived
                from axes_order

        Returns:
            New ZarrNii instance with downsampled data and updated metadata

        Raises:
            ValueError: If conflicting parameters provided or invalid factors

        Examples:
            >>> # Isotropic downsampling by factor of 2
            >>> downsampled = znii.downsample(factors=2)

            >>> # Anisotropic downsampling
            >>> downsampled = znii.downsample(factors=[1, 2, 2])

            >>> # Using legacy parameters
            >>> downsampled = znii.downsample(along_x=2, along_y=2, along_z=1)

            >>> # Power-of-2 downsampling
            >>> downsampled = znii.downsample(level=2)  # factors = 4

        Notes:
            - Downsampling uses block averaging for anti-aliasing
            - Spatial transformations are automatically scaled
            - Non-spatial dimensions (channels, time) are preserved
            - Original data remains unchanged (creates new instance)
        """
        # Handle legacy parameters
        if factors is None:
            if level is not None:
                factors = 2**level
            else:
                factors = (
                    [along_z, along_y, along_x]
                    if self.axes_order == "ZYX"
                    else [along_x, along_y, along_z]
                )

        if spatial_dims is None:
            spatial_dims = (
                ["z", "y", "x"] if self.axes_order == "ZYX" else ["x", "y", "z"]
            )

        downsampled_image = downsample_ngff_image(
            self.ngff_image, factors, spatial_dims
        )
        return ZarrNii(
            ngff_image=downsampled_image,
            axes_order=self.axes_order,
            xyz_orientation=self.xyz_orientation,
            _omero=self._omero,
        )

    def upsample(self, along_x=1, along_y=1, along_z=1, to_shape=None):
        """
        Upsamples the ZarrNii instance using `scipy.ndimage.zoom`.

        Parameters:
            along_x (int, optional): Upsampling factor along the X-axis (default: 1).
            along_y (int, optional): Upsampling factor along the Y-axis (default: 1).
            along_z (int, optional): Upsampling factor along the Z-axis (default: 1).
            to_shape (tuple, optional): Target shape for upsampling. Should include all dimensions
                                         (e.g., `(c, z, y, x)` for ZYX or `(c, x, y, z)` for XYZ).
                                         If provided, `along_x`, `along_y`, and `along_z` are ignored.

        Returns:
            ZarrNii: A new ZarrNii instance with the upsampled data and updated affine.

        Notes:
            - This method supports both direct scaling via `along_*` factors or target shape via `to_shape`.
            - If `to_shape` is provided, chunk sizes and scaling factors are dynamically calculated.
            - The affine matrix is updated to reflect the new voxel size after upsampling.

        Example:
            # Upsample with scaling factors
            upsampled_znimg = znimg.upsample(along_x=2, along_y=2, along_z=2)

            # Upsample to a specific shape
            upsampled_znimg = znimg.upsample(to_shape=(1, 256, 256, 256))
        """
        # Determine scaling and chunks based on input parameters
        if to_shape is None:
            if self.axes_order == "XYZ":
                scaling = (1, along_x, along_y, along_z)
            else:
                scaling = (1, along_z, along_y, along_x)

            chunks_out = tuple(
                tuple(c * scale for c in chunks_i)
                for chunks_i, scale in zip(self.data.chunks, scaling)
            )
        else:
            chunks_out, scaling = self.__get_upsampled_chunks(to_shape)

        # Define block-wise upsampling function
        def zoom_blocks(x, block_info=None):
            """
            Scales blocks to the desired size using `scipy.ndimage.zoom`.

            Parameters:
                x (np.ndarray): Input block data.
                block_info (dict, optional): Metadata about the current block.

            Returns:
                np.ndarray: The upscaled block.
            """
            # Calculate scaling factors based on input and output chunk shapes
            scaling = tuple(
                out_n / in_n
                for out_n, in_n in zip(block_info[None]["chunk-shape"], x.shape)
            )
            return zoom(x, scaling, order=1, prefilter=False)

        # Perform block-wise upsampling
        darr_scaled = da.map_blocks(
            zoom_blocks, self.data, dtype=self.data.dtype, chunks=chunks_out
        )

        # Update the affine matrix to reflect the new voxel size
        if self.axes_order == "XYZ":
            scaling_matrix = np.diag(
                (1 / scaling[1], 1 / scaling[2], 1 / scaling[3], 1)
            )
        else:
            scaling_matrix = np.diag(
                (1 / scaling[-1], 1 / scaling[-2], 1 / scaling[-3], 1)
            )

        # Create new NgffImage with upsampled data
        dims = self.dims
        if self.axes_order == "XYZ":
            new_scale = {
                dims[1]: self.scale[dims[1]] / scaling[1],
                dims[2]: self.scale[dims[2]] / scaling[2],
                dims[3]: self.scale[dims[3]] / scaling[3],
            }
        else:
            new_scale = {
                dims[1]: self.scale[dims[1]] / scaling[1],
                dims[2]: self.scale[dims[2]] / scaling[2],
                dims[3]: self.scale[dims[3]] / scaling[3],
            }

        upsampled_ngff = nz.to_ngff_image(
            darr_scaled,
            dims=dims,
            scale=new_scale,
            translation=self.translation.copy(),
            name=self.name,
        )

        # Return a new ZarrNii instance with the upsampled data
        return ZarrNii.from_ngff_image(
            upsampled_ngff,
            axes_order=self.axes_order,
            xyz_orientation=self.xyz_orientation,
            omero=self.omero,
        )

    def __get_upsampled_chunks(self, target_shape, return_scaling=True):
        """
        Calculates new chunk sizes for a dask array to match a target shape,
        while ensuring the chunks sum precisely to the target shape. Optionally,
        returns the scaling factors for each dimension.

        This method is useful for upsampling data or ensuring 1:1 correspondence
        between downsampled and upsampled arrays.

        Parameters:
            target_shape (tuple): The desired shape of the array after upsampling.
            return_scaling (bool, optional): Whether to return the scaling factors
                                             for each dimension (default: True).

        Returns:
            tuple:
                new_chunks (tuple): A tuple of tuples specifying the new chunk sizes
                                    for each dimension.
                scaling (list): A list of scaling factors for each dimension
                                (only if `return_scaling=True`).

            OR

            tuple:
                new_chunks (tuple): A tuple of tuples specifying the new chunk sizes
                                    for each dimension (if `return_scaling=False`).

        Notes:
            - The scaling factor for each dimension is calculated as:
              `scaling_factor = target_shape[dim] / original_shape[dim]`
            - The last chunk in each dimension is adjusted to account for rounding
              errors, ensuring the sum of chunks matches the target shape.

        Example:
            # Calculate upsampled chunks and scaling factors
            new_chunks, scaling = znimg.__get_upsampled_chunks((256, 256, 256))
            print("New chunks:", new_chunks)
            print("Scaling factors:", scaling)

            # Calculate only the new chunks
            new_chunks = znimg.__get_upsampled_chunks((256, 256, 256), return_scaling=False)
        """
        new_chunks = []
        scaling = []

        for dim, (orig_shape, orig_chunks, new_shape) in enumerate(
            zip(self.data.shape, self.data.chunks, target_shape)
        ):
            # Calculate the scaling factor for this dimension
            scaling_factor = new_shape / orig_shape

            # Scale each chunk size and round to get an initial estimate
            scaled_chunks = [
                int(round(chunk * scaling_factor)) for chunk in orig_chunks
            ]
            total = sum(scaled_chunks)

            # Adjust the chunks to ensure they sum up to the target shape exactly
            diff = new_shape - total
            if diff != 0:
                # Correct rounding errors by adjusting the last chunk size in the dimension
                scaled_chunks[-1] += diff

            new_chunks.append(tuple(scaled_chunks))
            scaling.append(scaling_factor)

        if return_scaling:
            return tuple(new_chunks), scaling
        else:
            return tuple(new_chunks)

    def get_bounded_subregion(self, points: np.ndarray):
        """
        Extracts a bounded subregion of the dask array containing the specified points,
        along with the grid points for interpolation.

        If the points extend beyond the domain of the dask array, the extent is capped
        at the boundaries. If all points are outside the domain, the function returns
        `(None, None)`.

        Parameters:
            points (np.ndarray): Nx3 or Nx4 array of coordinates in the array's space.
                                 If Nx4, the last column is assumed to be the homogeneous
                                 coordinate and is ignored.

        Returns:
            tuple:
                grid_points (tuple): A tuple of three 1D arrays representing the grid
                                     points along each axis (X, Y, Z) in the subregion.
                subvol (np.ndarray or None): The extracted subregion as a NumPy array.
                                             Returns `None` if all points are outside
                                             the array domain.

        Notes:
            - The function uses `compute()` on the dask array to immediately load the
              subregion, as Dask doesn't support the type of indexing required for
              interpolation.
            - A padding of 1 voxel is applied around the extent of the points.

        Example:
            grid_points, subvol = znimg.get_bounded_subregion(points)
            if subvol is not None:
                print("Subvolume shape:", subvol.shape)
        """
        pad = 1  # Padding around the extent of the points

        # Compute the extent of the points in the array's coordinate space
        min_extent = np.floor(points.min(axis=1)[:3] - pad).astype("int")
        max_extent = np.ceil(points.max(axis=1)[:3] + pad).astype("int")

        # Clip the extents to ensure they stay within the bounds of the array
        clip_min = np.zeros_like(min_extent)
        clip_max = np.array(self.darr.shape[-3:])  # Z, Y, X dimensions

        min_extent = np.clip(min_extent, clip_min, clip_max)
        max_extent = np.clip(max_extent, clip_min, clip_max)

        # Check if all points are outside the domain
        if np.any(max_extent <= min_extent):
            return None, None

        # Extract the subvolume using the computed extents
        subvol = self.darr[
            :,
            min_extent[0] : max_extent[0],
            min_extent[1] : max_extent[1],
            min_extent[2] : max_extent[2],
        ].compute()

        # Generate grid points for interpolation
        grid_points = (
            np.arange(min_extent[0], max_extent[0]),  # Z
            np.arange(min_extent[1], max_extent[1]),  # Y
            np.arange(min_extent[2], max_extent[2]),  # X
        )

        return grid_points, subvol

    def apply_transform(
        self,
        *transforms: Transform,
        ref_znimg: ZarrNii,
        spatial_dims: list[str] | None = None,
    ) -> ZarrNii:
        """Apply spatial transformations to image data.

        Transforms the image data to align with a reference image space using
        the provided transformation(s). This enables registration, resampling,
        and coordinate system conversions.

        Args:
            *transforms: Variable number of Transform objects to apply sequentially.
                Supported transform types:
                - AffineTransform: Linear transformations (rotation, scaling, translation)
                - DisplacementTransform: Non-linear deformation fields
            ref_znimg: Reference ZarrNii image defining the target coordinate system,
                grid spacing, and field of view for the output
            spatial_dims: Names of spatial dimensions for transformation. If None,
                automatically derived from axes_order

        Returns:
            New ZarrNii instance with transformed data in reference space

        Raises:
            ValueError: If no transforms provided or reference image incompatible
            TypeError: If transforms are not valid Transform objects

        Examples:
            >>> # Apply affine transformation
            >>> affine = AffineTransform.from_txt("transform.txt")
            >>> transformed = moving.apply_transform(affine, ref_znimg=reference)

            >>> # Apply multiple transforms sequentially
            >>> affine = AffineTransform.identity()
            >>> warp = DisplacementTransform.from_nifti("warp.nii.gz")
            >>> result = moving.apply_transform(affine, warp, ref_znimg=reference)

        Notes:
            - Transformations are applied in the order specified
            - Output data inherits spatial properties from ref_znimg
            - Uses interpolation for non-integer coordinate mappings
            - Non-spatial dimensions (channels, time) are preserved
        """
        if spatial_dims is None:
            spatial_dims = (
                ["z", "y", "x"] if self.axes_order == "ZYX" else ["x", "y", "z"]
            )

        # Initialize the list of transformations to apply
        tfms_to_apply = [ref_znimg.affine]  # Start with the reference image affine

        # Append all transformations passed as arguments
        tfms_to_apply.extend(transforms)

        # Append the inverse of the current image's affine
        tfms_to_apply.append(self.affine.invert())

        # Create new NgffImage from ref image
        interp_ngff_image = nz.NgffImage(
            data=ref_znimg.data,
            dims=ref_znimg.ngff_image.dims.copy(),
            scale=ref_znimg.ngff_image.scale.copy(),
            translation=ref_znimg.ngff_image.translation.copy(),
            name=f"{self.name}_transformed_to_{ref_znimg.name}",
        )

        # Try to get zarr store information for direct access (avoids nested compute)
        store_info = self.get_zarr_store_info()

        # Lazily apply the transformations using dask
        if store_info is not None:
            # Use direct zarr access to avoid nested compute() calls
            interp_ngff_image.data = da.map_blocks(
                interp_by_block,  # Function to interpolate each block
                ref_znimg.data,  # Reference image data
                dtype=np.float32,  # Output data type
                transforms=tfms_to_apply,  # Transformations to apply
                flo_store_path=store_info["store_path"],
                flo_array_shape=store_info["array_shape"],
                flo_dataset_path=store_info["dataset_path"],
                flo_storage_options=None,  # TODO: Extract from dask array if available
            )
        else:
            # Fall back to passing ZarrNii instance (legacy behavior with nested compute)
            interp_ngff_image.data = da.map_blocks(
                interp_by_block,  # Function to interpolate each block
                ref_znimg.data,  # Reference image data
                dtype=np.float32,  # Output data type
                transforms=tfms_to_apply,  # Transformations to apply
                flo_znimg=self,  # Floating image to align (legacy)
            )

        return ZarrNii.from_ngff_image(
            interp_ngff_image,
            axes_order=ref_znimg.axes_order,
            xyz_orientation=ref_znimg.xyz_orientation,
            omero=self.omero,
        )

    # I/O operations
    def to_ome_zarr(
        self,
        store_or_path: str | Any,
        max_layer: int = 4,
        scale_factors: list[int] | None = None,
        backend: str = "ome-zarr-py",
        **kwargs: Any,
    ) -> ZarrNii:
        """Save to OME-Zarr store with multiscale pyramid.

        Creates an OME-Zarr dataset with automatic multiscale pyramid generation
        for efficient visualization and processing at multiple resolutions.
        Preserves spatial metadata and supports various storage backends.

        Args:
            store_or_path: Target location for OME-Zarr store. Supports:
                - Local directory path
                - Remote URLs (s3://, gs://, etc.)
                - ZIP files (.zip extension for compressed storage)
                - Zarr store objects
            max_layer: Maximum number of pyramid levels to create (including level 0).
                Higher values create more downsampled levels
            scale_factors: Custom downsampling factors for each pyramid level.
                If None, uses powers of 2: [2, 4, 8, 16, ...]
            backend: Backend library to use for writing. Options:
                - 'ngff-zarr': Use ngff-zarr library (default)
                - 'ome-zarr-py': Use ome-zarr-py library for better dask integration
            **kwargs: Additional arguments passed to the save function.
                For 'ngff-zarr': passed to to_ngff_zarr function
                For 'ome-zarr-py': passed to write_image (e.g., scaling_method, compute)

        Returns:
            Self for method chaining

        Raises:
            OSError: If unable to write to target location
            ValueError: If invalid scale_factors or backend provided

        Examples:
            >>> # Save with default pyramid levels
            >>> znii.to_ome_zarr("/path/to/output.zarr")

            >>> # Save to compressed ZIP with custom pyramid
            >>> znii.to_ome_zarr(
            ...     "/path/to/output.zarr.zip",
            ...     max_layer=3,
            ...     scale_factors=[2, 4]
            ... )

            >>> # Use ome-zarr-py backend for better dask performance
            >>> znii.to_ome_zarr(
            ...     "/path/to/output.zarr",
            ...     backend="ome-zarr-py",
            ...     scaling_method="gaussian"
            ... )

            >>> # Chain with other operations
            >>> result = (znii.downsample(2)
            ...               .crop((0,0,0), (100,100,100))
            ...               .to_ome_zarr("processed.zarr"))

        Notes:
            - OME-Zarr files are always saved in ZYX axis order
            - Automatic axis reordering if current order is XYZ
            - Spatial transformations and metadata are preserved
            - Orientation information is stored using the new 'xyz_orientation'
              metadata key for consistency and future compatibility
            - The 'ome-zarr-py' backend provides better performance with dask
              and dask distributed workflows
        """
        # Validate backend parameter
        if backend not in ["ngff-zarr", "ome-zarr-py"]:
            raise ValueError(
                f"Invalid backend '{backend}'. Must be 'ngff-zarr' or 'ome-zarr-py'"
            )

        # Determine the image to save
        if self.axes_order == "XYZ":
            # Need to reorder data from XYZ to ZYX for OME-Zarr
            ngff_image_to_save = self._create_zyx_ngff_image()
        else:
            # Already in ZYX order
            ngff_image_to_save = self.ngff_image

        # Use the appropriate save function based on backend
        if backend == "ngff-zarr":
            save_ngff_image(
                ngff_image_to_save,
                store_or_path,
                max_layer,
                scale_factors,
                xyz_orientation=(
                    self.xyz_orientation if hasattr(self, "xyz_orientation") else None
                ),
                **kwargs,
            )
        elif backend == "ome-zarr-py":
            save_ngff_image_with_ome_zarr(
                ngff_image_to_save,
                store_or_path,
                max_layer,
                scale_factors,
                xyz_orientation=(
                    self.xyz_orientation if hasattr(self, "xyz_orientation") else None
                ),
                **kwargs,
            )

        # Add orientation metadata to the zarr store (only for non-ZIP files)
        # For ZIP files, orientation is handled inside save_ngff_image
        if not (isinstance(store_or_path, str) and store_or_path.endswith(".zip")):
            try:
                import zarr

                if isinstance(store_or_path, str):
                    group = zarr.open_group(store_or_path, mode="r+")
                else:
                    group = zarr.open_group(store_or_path, mode="r+")

                # Add metadata for xyz_orientation (new format)
                if hasattr(self, "xyz_orientation") and self.xyz_orientation:
                    group.attrs["xyz_orientation"] = self.xyz_orientation
            except Exception:
                # If we can't write orientation metadata, that's not critical
                pass

        return self

    def to_nifti(self, filename: str | bytes | None = None) -> nib.Nifti1Image | str:
        """Convert to NIfTI format with automatic dimension handling.

        Converts the ZarrNii image to NIfTI-1 format, handling dimension
        reordering, singleton dimension removal, and spatial transformation
        conversion. NIfTI files are always written in XYZ axis order.

        Args:
            filename: Output file path for saving. Supported extensions:
                - .nii: Uncompressed NIfTI
                - .nii.gz: Compressed NIfTI (recommended)
                If None, returns nibabel image object without saving

        Returns:
            If filename is None: nibabel.Nifti1Image object
            If filename provided: path to saved file

        Raises:
            ValueError: If data has non-singleton time or channel dimensions
                (NIfTI doesn't support >4D data)
            OSError: If unable to write to specified filename

        Notes:
            - Automatically reorders data from ZYX to XYZ if necessary
            - Removes singleton time/channel dimensions automatically
            - Spatial transformations are converted to NIfTI affine format
            - For 5D data (T,C,Z,Y,X), only singleton T/C dimensions are supported

        Examples:
            >>> # Save to compressed NIfTI file
            >>> znii.to_nifti("output.nii.gz")

            >>> # Get nibabel object without saving
            >>> nifti_img = znii.to_nifti()
            >>> print(nifti_img.shape)

            >>> # Handle multi-channel data by selecting single channel first
            >>> znii.select_channels([0]).to_nifti("channel0.nii.gz")

        Warnings:
            Large images will be computed in memory during conversion.
            Consider downsampling or cropping first for very large datasets.
        """
        # Get data and dimensions
        data = self.data.compute()

        dims = self.dims

        # Handle dimensional reduction for NIfTI compatibility
        # NIfTI supports up to 4D, so we need to remove singleton dimensions
        squeeze_axes = []
        remaining_dims = []

        for i, dim in enumerate(dims):
            if dim in ["t", "c"] and data.shape[i] == 1:
                # Remove singleton time or channel dimensions
                squeeze_axes.append(i)
            elif dim in ["t", "c"] and data.shape[i] > 1:
                # Non-singleton time or channel dimensions - NIfTI can't handle this
                raise ValueError(
                    f"NIfTI format doesn't support non-singleton {dim} dimension. "
                    f"Dimension '{dim}' has size {data.shape[i]}. "
                    f"Consider selecting specific timepoints/channels first."
                )
            else:
                remaining_dims.append(dim)

        # Squeeze out singleton dimensions
        if squeeze_axes:
            data = np.squeeze(data, axis=tuple(squeeze_axes))

        # Check final dimensionality
        if data.ndim > 4:
            raise ValueError(
                f"Resulting data has {data.ndim} dimensions, but NIfTI supports maximum 4D"
            )

        # Now handle spatial reordering based on axes_order
        if self.axes_order == "ZYX":
            # Data spatial dimensions are in ZYX order, need to transpose to XYZ
            if data.ndim == 3:
                # Pure spatial data: ZYX -> XYZ
                data = data.transpose(2, 1, 0)
            elif data.ndim == 4:
                # 4D data with one non-spatial dimension remaining
                # Could be (T,Z,Y,X) or (C,Z,Y,X) - spatial part needs ZYX->XYZ
                # The non-spatial dimension stays first
                data = data.transpose(0, 3, 2, 1)

            # Get affine matrix in XYZ order
            affine_matrix = self.get_affine_matrix(axes_order="XYZ")
        else:
            # Data is already in XYZ order
            affine_matrix = self.get_affine_matrix(axes_order="XYZ")

        # Create NIfTI image
        nifti_img = nib.Nifti1Image(data, affine_matrix)

        if filename is not None:
            nib.save(nifti_img, filename)
            return filename
        else:
            return nifti_img

    def to_tiff_stack(
        self,
        filename_pattern: str,
        channel: int | None = None,
        timepoint: int | None = None,
        compress: bool = True,
        dtype: str | None = "uint16",
        rescale: bool = True,
    ) -> str:
        """Save data as a stack of 2D TIFF images.

        Saves the image data as a series of 2D TIFF files, with each Z-slice
        saved as a separate file. This format is useful for compatibility with
        tools that don't support OME-Zarr or napari plugins that require
        individual TIFF files.

        Args:
            filename_pattern: Output filename pattern. Should contain '{z:04d}' or similar
                format specifier for the Z-slice number. Examples:
                - "output_z{z:04d}.tif"
                - "data/slice_{z:03d}.tiff"
                If pattern doesn't contain format specifier, '_{z:04d}' is appended
                before the extension.
            channel: Channel index to save (0-based). If None and data has multiple
                channels, all channels will be saved as separate channel dimensions
                in each TIFF file (multi-channel TIFFs).
            timepoint: Timepoint index to save (0-based). If None and data has multiple
                timepoints, raises ValueError (must select single timepoint).
            compress: Whether to use LZW compression (default: True)
            dtype: Output data type for TIFF files. Options:
                - 'uint8': 8-bit unsigned integer (0-255)
                - 'uint16': 16-bit unsigned integer (0-65535) [default]
                - 'int16': 16-bit signed integer (-32768 to 32767)
                - 'float32': 32-bit float (preserves original data)
                Default 'uint16' provides good range and compatibility.
            rescale: Whether to rescale data to fit the output dtype range.
                If True, data is linearly scaled from [min, max] to the full
                range of the output dtype. If False, data is clipped to the
                output dtype range. Default: True

        Returns:
            Base directory path where files were saved

        Raises:
            ValueError: If data has multiple timepoints but none selected,
                or if selected channel/timepoint is out of range,
                or if dtype is not supported
            OSError: If unable to write to specified directory

        Examples:
            >>> # Save as 16-bit with auto-rescaling (default, recommended)
            >>> znii.to_tiff_stack("output_z{z:04d}.tif")

            >>> # Save as 8-bit for smaller file sizes
            >>> znii.to_tiff_stack("output_z{z:04d}.tif", dtype='uint8')

            >>> # Save specific channel without rescaling
            >>> znii.to_tiff_stack("channel0_z{z:04d}.tif", channel=0, rescale=False)

            >>> # Save as float32 to preserve original precision
            >>> znii.to_tiff_stack("precise_z{z:04d}.tif", dtype='float32')

        Warnings:
            This method loads all data into memory. For large datasets,
            consider cropping or downsampling first to reduce memory usage.
            The cellseg3d napari plugin and similar tools work best with
            cropped regions rather than full-resolution whole-brain images.

        Notes:
            - Z-dimension becomes the stack (file) dimension
            - Time and channel dimensions are handled as specified
            - Spatial transformations are not preserved in TIFF format
            - For 5D data (T,C,Z,Y,X), you must select a single timepoint
            - Multi-channel data can be saved as multi-channel TIFFs or selected
            - Data type conversion helps ensure compatibility with analysis tools
            - uint16 is recommended for most scientific applications (good range + compatibility)
        """
        try:
            import tifffile
        except ImportError:
            raise ImportError(
                "tifffile is required for TIFF stack support. "
                "Install with: pip install tifffile"
            )

        # Get data and dimensions
        data = self.data.compute()
        dims = self.dims

        # Create output directory if needed
        import os

        output_dir = os.path.dirname(filename_pattern)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Handle dimensional selection and validation
        # Remove singleton dimensions first, similar to to_nifti
        squeeze_axes = []
        remaining_dims = []
        time_dim_size = 1
        channel_dim_size = 1

        for i, dim in enumerate(dims):
            if dim == "t":
                time_dim_size = data.shape[i]
                if data.shape[i] == 1:
                    squeeze_axes.append(i)
                elif timepoint is None:
                    raise ValueError(
                        f"Data has {data.shape[i]} timepoints. "
                        f"Must specify 'timepoint' parameter to select a single timepoint."
                    )
                elif timepoint >= data.shape[i]:
                    raise ValueError(
                        f"Timepoint {timepoint} is out of range (data has {data.shape[i]} timepoints)"
                    )
                else:
                    remaining_dims.append(dim)
            elif dim == "c":
                channel_dim_size = data.shape[i]
                if data.shape[i] == 1:
                    squeeze_axes.append(i)
                elif channel is None:
                    raise ValueError(
                        f"Data has {data.shape[i]} channels. "
                        f"Must specify 'channel' parameter to select a single channel."
                    )
                elif channel >= data.shape[i]:
                    raise ValueError(
                        f"Channel {channel} is out of range (data has {data.shape[i]} channels)"
                    )
                else:
                    remaining_dims.append(dim)
            else:
                remaining_dims.append(dim)

        # Select specific timepoint if needed
        if time_dim_size > 1 and timepoint is not None:
            time_axis = dims.index("t")
            data = np.take(data, timepoint, axis=time_axis)
            # Update dims list
            dims = [d for i, d in enumerate(dims) if i != time_axis]

        # Select specific channel if needed
        if channel_dim_size > 1 and channel is not None:
            channel_axis = dims.index("c")
            data = np.take(data, channel, axis=channel_axis)
            # Update dims list
            dims = [d for i, d in enumerate(dims) if i != channel_axis]

        # Squeeze singleton dimensions
        if squeeze_axes:
            # Recalculate squeeze axes after potential dimension removal
            current_squeeze_axes = []
            for axis in squeeze_axes:
                # Count how many axes were removed before this one
                removed_before = sum(
                    1
                    for removed_axis in [
                        (
                            dims.index("t")
                            if time_dim_size > 1 and timepoint is not None
                            else -1
                        ),
                        (
                            dims.index("c")
                            if channel_dim_size > 1 and channel is not None
                            else -1
                        ),
                    ]
                    if removed_axis != -1 and removed_axis < axis
                )
                current_squeeze_axes.append(axis - removed_before)

            data = np.squeeze(data, axis=tuple(current_squeeze_axes))
            dims = [dim for i, dim in enumerate(dims) if i not in current_squeeze_axes]

        # Find Z dimension for stacking
        if "z" not in dims:
            raise ValueError("Data must have a Z dimension for TIFF stack export")

        z_axis = dims.index("z")
        z_size = data.shape[z_axis]

        # Check filename pattern contains format specifier
        if "{z" not in filename_pattern:
            # Add default z format before extension
            name, ext = os.path.splitext(filename_pattern)
            filename_pattern = f"{name}_{{z:04d}}{ext}"

        # Move Z axis to first position for easy iteration
        axes_order = list(range(data.ndim))
        axes_order[0], axes_order[z_axis] = axes_order[z_axis], axes_order[0]
        data = data.transpose(axes_order)

        # Handle data type conversion and rescaling
        supported_dtypes = {
            "uint8": np.uint8,
            "uint16": np.uint16,
            "int16": np.int16,
            "float32": np.float32,
        }

        if dtype not in supported_dtypes:
            raise ValueError(
                f"Unsupported dtype '{dtype}'. Supported types: {list(supported_dtypes.keys())}"
            )

        target_dtype = supported_dtypes[dtype]

        if rescale and dtype != "float32":
            # Get the data range
            data_min = np.min(data)
            data_max = np.max(data)

            if data_min == data_max:
                # Handle constant data case
                data_scaled = np.zeros_like(data, dtype=target_dtype)
            else:
                # Get target range for the dtype
                if dtype == "uint8":
                    target_min, target_max = 0, 255
                elif dtype == "uint16":
                    target_min, target_max = 0, 65535
                elif dtype == "int16":
                    target_min, target_max = -32768, 32767

                # Linear rescaling: new_value = (value - data_min) * (target_max - target_min) / (data_max - data_min) + target_min
                data_scaled = (
                    (data - data_min)
                    * (target_max - target_min)
                    / (data_max - data_min)
                    + target_min
                ).astype(target_dtype)

            print(
                f"Rescaled data from [{data_min:.3f}, {data_max:.3f}] to {dtype} range"
            )
        else:
            # No rescaling - just clip and convert
            if dtype == "uint8":
                data_scaled = np.clip(data, 0, 255).astype(target_dtype)
            elif dtype == "uint16":
                data_scaled = np.clip(data, 0, 65535).astype(target_dtype)
            elif dtype == "int16":
                data_scaled = np.clip(data, -32768, 32767).astype(target_dtype)
            else:  # float32
                data_scaled = data.astype(target_dtype)

            if dtype != "float32":
                print(f"Converted data to {dtype} with clipping (no rescaling)")

        data = data_scaled

        # Save each Z-slice as a separate TIFF file
        compression = "lzw" if compress else None
        saved_files = []

        for z_idx in range(z_size):
            slice_data = data[z_idx]

            # Generate filename for this slice
            filename = filename_pattern.format(z=z_idx)

            # Save the 2D slice
            tifffile.imwrite(filename, slice_data, compression=compression)
            saved_files.append(filename)

        print(f"Saved {len(saved_files)} TIFF files to {output_dir or '.'}")
        print(
            f"Files: {os.path.basename(saved_files[0])} ... {os.path.basename(saved_files[-1])}"
        )

        return output_dir or "."

    @classmethod
    def from_imaris(
        cls,
        path: str,
        level: int = 0,
        timepoint: int = 0,
        channel: int = 0,
        chunks: str = "auto",
        axes_order: str = "ZYX",
        orientation: str = "RAS",
    ) -> ZarrNii:
        """
        Load from Imaris (.ims) file format.

        Imaris files use HDF5 format with specific dataset structure.
        This method requires the 'imaris' extra dependency (h5py).

        Args:
            path: Path to Imaris (.ims) file
            level: Resolution level to load (0 = full resolution)
            timepoint: Time point to load (default: 0)
            channel: Channel to load (default: 0)
            chunks: Chunking strategy for dask array
            axes_order: Spatial axes order for compatibility (default: "ZYX")
            orientation: Default orientation (default: "RAS")

        Returns:
            ZarrNii instance

        Raises:
            ImportError: If h5py is not available
            ValueError: If the file is not a valid Imaris file
        """
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for Imaris support. "
                "Install with: pip install zarrnii[imaris] or pip install h5py"
            )

        # Open Imaris file
        with h5py.File(path, "r") as f:
            # Verify it's an Imaris file by checking for standard structure
            if "DataSet" not in f:
                raise ValueError(
                    f"File {path} does not appear to be a valid Imaris file (missing DataSet group)"
                )

            # Navigate to the specific dataset
            dataset_group = f["DataSet"]

            # Find available resolution levels
            resolution_levels = [
                key for key in dataset_group.keys() if key.startswith("ResolutionLevel")
            ]
            if not resolution_levels:
                raise ValueError("No resolution levels found in Imaris file")

            # Validate level parameter
            if level >= len(resolution_levels):
                raise ValueError(
                    f"Level {level} not available. Available levels: 0-{len(resolution_levels) - 1}"
                )

            # Navigate to specified resolution level
            res_level_key = f"ResolutionLevel {level}"
            if res_level_key not in dataset_group:
                raise ValueError(f"Resolution level {level} not found")

            res_group = dataset_group[res_level_key]

            # Find available timepoints
            timepoints = [
                key for key in res_group.keys() if key.startswith("TimePoint")
            ]
            if not timepoints:
                raise ValueError("No timepoints found in Imaris file")

            # Validate timepoint parameter
            if timepoint >= len(timepoints):
                raise ValueError(
                    f"Timepoint {timepoint} not available. Available timepoints: 0-{len(timepoints) - 1}"
                )

            # Navigate to specified timepoint
            time_key = f"TimePoint {timepoint}"
            if time_key not in res_group:
                raise ValueError(f"Timepoint {timepoint} not found")

            time_group = res_group[time_key]

            # Find available channels
            channels = [key for key in time_group.keys() if key.startswith("Channel")]
            if not channels:
                raise ValueError("No channels found in Imaris file")

            # Validate channel parameter
            if channel >= len(channels):
                raise ValueError(
                    f"Channel {channel} not available. Available channels: 0-{len(channels) - 1}"
                )

            # Navigate to specified channel
            channel_key = f"Channel {channel}"
            if channel_key not in time_group:
                raise ValueError(f"Channel {channel} not found")

            channel_group = time_group[channel_key]

            # Load the actual data
            if "Data" not in channel_group:
                raise ValueError("No Data dataset found in channel group")

            data_dataset = channel_group["Data"]

            # Load data into memory first (necessary because HDF5 file will be closed)
            data_numpy = data_dataset[:]

            # Create dask array from numpy array
            data_array = da.from_array(data_numpy, chunks=chunks)

            # Add channel dimension if not present
            if len(data_array.shape) == 3:
                data_array = data_array[np.newaxis, ...]

            # Extract spatial metadata
            # Try to get spacing information from Imaris metadata
            spacing = [1.0, 1.0, 1.0]  # Default spacing
            origin = [0.0, 0.0, 0.0]  # Default origin

            # Look for ImageSizeX, ImageSizeY, ImageSizeZ attributes
            try:
                # Navigate back to get image info
                if "ImageSizeX" in f.attrs:
                    x_size = f.attrs["ImageSizeX"]
                    y_size = f.attrs["ImageSizeY"]
                    z_size = f.attrs["ImageSizeZ"]

                    # Calculate spacing based on physical size and voxel count
                    if data_array.shape[-1] > 0:  # X dimension
                        spacing[0] = x_size / data_array.shape[-1]
                    if data_array.shape[-2] > 0:  # Y dimension
                        spacing[1] = y_size / data_array.shape[-2]
                    if data_array.shape[-3] > 0:  # Z dimension
                        spacing[2] = z_size / data_array.shape[-3]
            except (KeyError, IndexError):
                # Use default spacing if metadata is not available
                pass

            # Create dimensions
            dims = ["c"] + list(axes_order.lower())

            # Create scale and translation dictionaries
            scale_dict = {}
            translation_dict = {}
            spatial_dims = ["z", "y", "x"] if axes_order == "ZYX" else ["x", "y", "z"]

            for i, dim in enumerate(spatial_dims):
                scale_dict[dim] = spacing[i]
                translation_dict[dim] = origin[i]

            # Create NgffImage
            ngff_image = nz.NgffImage(
                data=data_array,
                dims=dims,
                scale=scale_dict,
                translation=translation_dict,
                name=f"imaris_image_{path}_{level}_{timepoint}_{channel}",
            )

        # Create and return ZarrNii instance
        return cls(
            ngff_image=ngff_image,
            axes_order=axes_order,
            xyz_orientation=orientation,
            _omero=None,
        )

    def to_imaris(
        self, path: str, compression: str = "gzip", compression_opts: int = 6
    ) -> str:
        """
        Save to Imaris (.ims) file format using HDF5.

        This method creates Imaris files compatible with Imaris software by
        following the exact HDF5 structure from correctly-formed reference files.
        All attributes use byte-array encoding as required by Imaris.

        Args:
            path: Output path for Imaris (.ims) file
            compression: HDF5 compression method (default: "gzip")
            compression_opts: Compression level (default: 6)

        Returns:
            str: Path to the saved file

        Raises:
            ImportError: If h5py is not available
        """
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for Imaris support. "
                "Install with: pip install zarrnii[imaris] or pip install h5py"
            )

        # Ensure path has .ims extension
        if not path.endswith(".ims"):
            path = path + ".ims"

        def _string_to_byte_array(s: str) -> np.ndarray:
            """Convert string to byte array as required by Imaris."""
            return np.array([c.encode() for c in s])

        # Get data and metadata
        if hasattr(self.darr, "compute"):
            data = self.darr.compute()  # Convert Dask array to numpy array
        else:
            data = np.asarray(self.darr)  # Handle numpy arrays directly

        # Handle dimensions: expect ZYX or CZYX
        if len(data.shape) == 4:
            # CZYX format
            n_channels = data.shape[0]
            z, y, x = data.shape[1:]
        elif len(data.shape) == 3:
            # ZYX format - single channel
            n_channels = 1
            z, y, x = data.shape
            data = data[np.newaxis, ...]  # Add channel dimension
        else:
            raise ValueError(
                f"Unsupported data shape: {data.shape}. Expected 3D (ZYX) or 4D (CZYX)"
            )

        # Create Imaris file structure exactly matching reference file
        with h5py.File(path, "w") as f:
            # Root attributes - use exact byte array format from reference
            f.attrs["DataSetDirectoryName"] = _string_to_byte_array("DataSet")
            f.attrs["DataSetInfoDirectoryName"] = _string_to_byte_array("DataSetInfo")
            f.attrs["ImarisDataSet"] = _string_to_byte_array("ImarisDataSet")
            f.attrs["ImarisVersion"] = _string_to_byte_array("5.5.0")
            f.attrs["NumberOfDataSets"] = np.array([1], dtype=np.uint32)
            f.attrs["ThumbnailDirectoryName"] = _string_to_byte_array("Thumbnail")

            # Create main DataSet group structure
            dataset_group = f.create_group("DataSet")
            res_group = dataset_group.create_group("ResolutionLevel 0")
            time_group = res_group.create_group("TimePoint 0")

            # Create channels with proper attributes
            for c in range(n_channels):
                channel_group = time_group.create_group(f"Channel {c}")
                channel_data = data[c]  # (Z, Y, X)

                # Channel attributes - use byte array format exactly like reference
                channel_group.attrs["ImageSizeX"] = _string_to_byte_array(str(x))
                channel_group.attrs["ImageSizeY"] = _string_to_byte_array(str(y))
                channel_group.attrs["ImageSizeZ"] = _string_to_byte_array(str(z))
                channel_group.attrs["ImageBlockSizeX"] = _string_to_byte_array(str(x))
                channel_group.attrs["ImageBlockSizeY"] = _string_to_byte_array(str(y))
                channel_group.attrs["ImageBlockSizeZ"] = _string_to_byte_array(
                    str(min(z, 16))
                )

                # Histogram range attributes
                data_min, data_max = (
                    float(channel_data.min()),
                    float(channel_data.max()),
                )
                channel_group.attrs["HistogramMin"] = _string_to_byte_array(
                    f"{data_min:.3f}"
                )
                channel_group.attrs["HistogramMax"] = _string_to_byte_array(
                    f"{data_max:.3f}"
                )

                # Create data dataset with proper compression
                # Preserve original data type but ensure it's compatible with Imaris
                if channel_data.dtype == np.float32 or channel_data.dtype == np.float64:
                    # Keep float data as is for round-trip compatibility
                    data_for_storage = channel_data.astype(np.float32)
                elif channel_data.dtype in [np.uint16, np.int16]:
                    # Keep 16-bit data as is
                    data_for_storage = channel_data
                else:
                    # Convert other types to uint8
                    data_for_storage = channel_data.astype(np.uint8)

                channel_group.create_dataset(
                    "Data",
                    data=data_for_storage,
                    compression=compression,
                    compression_opts=compression_opts,
                    chunks=True,
                )

                # Create histogram
                hist_data, _ = np.histogram(
                    channel_data.flatten(), bins=256, range=(data_min, data_max)
                )
                channel_group.create_dataset(
                    "Histogram", data=hist_data.astype(np.uint64)
                )

            # Get spacing directly from scale dictionary with proper XYZ order
            try:
                # Extract voxel sizes directly from ngff_image scale dictionary
                # This ensures we get X, Y, Z in the correct order regardless of axes_order
                sx = self.ngff_image.scale.get("x", 1.0)
                sy = self.ngff_image.scale.get("y", 1.0)
                sz = self.ngff_image.scale.get("z", 1.0)
            except:
                sx = sy = sz = 1.0

            # Calculate extents (physical coordinates)
            ext_x = sx * x
            ext_y = sy * y
            ext_z = sz * z

            # Create comprehensive DataSetInfo structure matching reference
            info_group = f.create_group("DataSetInfo")

            # Create channel info groups
            for c in range(n_channels):
                channel_info = info_group.create_group(f"Channel {c}")

                # Essential channel attributes in byte array format
                channel_info.attrs["Color"] = _string_to_byte_array(
                    "1.000 0.000 0.000"
                    if c == 0
                    else f"0.000 {1.0 if c == 1 else 0.0:.3f} {1.0 if c == 2 else 0.0:.3f}"
                )
                channel_info.attrs["Name"] = _string_to_byte_array(f"Channel {c}")
                channel_info.attrs["ColorMode"] = _string_to_byte_array("BaseColor")
                channel_info.attrs["ColorOpacity"] = _string_to_byte_array("1.000")
                channel_info.attrs["ColorRange"] = _string_to_byte_array("0 255")
                channel_info.attrs["GammaCorrection"] = _string_to_byte_array("1.000")
                channel_info.attrs["LSMEmissionWavelength"] = _string_to_byte_array(
                    "500"
                )
                channel_info.attrs["LSMExcitationWavelength"] = _string_to_byte_array(
                    "500"
                )
                channel_info.attrs["LSMPhotons"] = _string_to_byte_array("1")
                channel_info.attrs["LSMPinhole"] = _string_to_byte_array("0")

                # Add description
                description = f"Channel {c} created by ZarrNii"
                channel_info.attrs["Description"] = _string_to_byte_array(description)

            # Create CRITICAL Image group with voxel size information (this was missing!)
            image_info = info_group.create_group("Image")

            # Add essential image metadata with proper voxel size information
            image_info.attrs["X"] = _string_to_byte_array(str(x))
            image_info.attrs["Y"] = _string_to_byte_array(str(y))
            image_info.attrs["Z"] = _string_to_byte_array(str(z))
            image_info.attrs["Unit"] = _string_to_byte_array("um")
            image_info.attrs["Noc"] = _string_to_byte_array(str(n_channels))

            # CRITICAL: Set proper physical extents that define voxel size
            # Imaris reads voxel size from these extent values
            image_info.attrs["ExtMin0"] = _string_to_byte_array(f"{-ext_x / 2:.3f}")
            image_info.attrs["ExtMax0"] = _string_to_byte_array(f"{ext_x / 2:.3f}")
            image_info.attrs["ExtMin1"] = _string_to_byte_array(f"{-ext_y / 2:.3f}")
            image_info.attrs["ExtMax1"] = _string_to_byte_array(f"{ext_y / 2:.3f}")
            image_info.attrs["ExtMin2"] = _string_to_byte_array(f"{-ext_z / 2:.3f}")
            image_info.attrs["ExtMax2"] = _string_to_byte_array(f"{ext_z / 2:.3f}")

            # Add device/acquisition metadata
            image_info.attrs["ManufactorString"] = _string_to_byte_array("ZarrNii")
            image_info.attrs["ManufactorType"] = _string_to_byte_array("Generic")
            image_info.attrs["LensPower"] = _string_to_byte_array("")
            image_info.attrs["NumericalAperture"] = _string_to_byte_array("")
            image_info.attrs["RecordingDate"] = _string_to_byte_array(
                "2024-01-01 00:00:00.000"
            )
            image_info.attrs["Filename"] = _string_to_byte_array(path.split("/")[-1])
            image_info.attrs["Name"] = _string_to_byte_array("ZarrNii Export")
            image_info.attrs["Compression"] = _string_to_byte_array("5794")

            # Add description
            description = (
                f"Imaris file created by ZarrNii from {self.axes_order} format data. "
                f"Original shape: {self.darr.shape}. Converted to Imaris format "
                f"with {n_channels} channel(s) and dimensions {z}x{y}x{x}. "
                f"Voxel size: {sx:.3f} x {sy:.3f} x {sz:.3f} um."
            )
            image_info.attrs["Description"] = _string_to_byte_array(description)

            # Create Imaris metadata group
            imaris_info = info_group.create_group("Imaris")
            imaris_info.attrs["Version"] = _string_to_byte_array("7.0")
            imaris_info.attrs["ThumbnailMode"] = _string_to_byte_array("thumbnailMIP")
            imaris_info.attrs["ThumbnailSize"] = _string_to_byte_array("256")

            # Create ImarisDataSet metadata
            dataset_info = info_group.create_group("ImarisDataSet")
            dataset_info.attrs["Creator"] = _string_to_byte_array("Imaris")
            dataset_info.attrs["Version"] = _string_to_byte_array("7.0")
            dataset_info.attrs["NumberOfImages"] = _string_to_byte_array("1")

            # Add version-specific groups as seen in reference
            dataset_info_ver = info_group.create_group("ImarisDataSet       0.0.0")
            dataset_info_ver.attrs["NumberOfImages"] = _string_to_byte_array("1")
            dataset_info_ver2 = info_group.create_group("ImarisDataSet      0.0.0")
            dataset_info_ver2.attrs["NumberOfImages"] = _string_to_byte_array("1")

            # Create TimeInfo group
            time_info = info_group.create_group("TimeInfo")
            time_info.attrs["DatasetTimePoints"] = _string_to_byte_array("1")
            time_info.attrs["FileTimePoints"] = _string_to_byte_array("1")
            time_info.attrs["TimePoint1"] = _string_to_byte_array(
                "2024-01-01 00:00:00.000"
            )

            # Create Log group (basic processing log)
            log_group = info_group.create_group("Log")
            log_group.attrs["Entries"] = _string_to_byte_array("1")
            log_group.attrs["Entry0"] = _string_to_byte_array(
                f'<ZarrNiiExport channels="{" ".join(["on"] * n_channels)}"/>'
            )

            # Create thumbnail group with proper multi-channel thumbnail
            thumbnail_group = f.create_group("Thumbnail")

            # Create a combined thumbnail (256x1024 for multi-channel as in reference)
            if n_channels > 1:
                # Multi-channel thumbnail: concatenate channels horizontally
                thumb_width = 256 * n_channels
                thumbnail_data = np.zeros((256, thumb_width), dtype=np.uint8)

                for c in range(n_channels):
                    # Downsample each channel to 256x256
                    channel_data = data[c]
                    # Take MIP (Maximum Intensity Projection) along Z
                    mip = np.max(channel_data, axis=0)
                    # Resize to 256x256 (simple decimation)
                    step_y = max(1, mip.shape[0] // 256)
                    step_x = max(1, mip.shape[1] // 256)
                    thumb_channel = mip[::step_y, ::step_x]

                    # Pad or crop to exactly 256x256
                    if thumb_channel.shape[0] < 256 or thumb_channel.shape[1] < 256:
                        padded = np.zeros((256, 256), dtype=thumb_channel.dtype)
                        h, w = thumb_channel.shape
                        padded[:h, :w] = thumb_channel
                        thumb_channel = padded
                    else:
                        thumb_channel = thumb_channel[:256, :256]

                    # Place in thumbnail
                    thumbnail_data[:, c * 256 : (c + 1) * 256] = thumb_channel
            else:
                # Single channel: 256x256 thumbnail
                channel_data = data[0]
                mip = np.max(channel_data, axis=0)
                step_y = max(1, mip.shape[0] // 256)
                step_x = max(1, mip.shape[1] // 256)
                thumbnail_data = mip[::step_y, ::step_x]

                if thumbnail_data.shape[0] < 256 or thumbnail_data.shape[1] < 256:
                    padded = np.zeros((256, 256), dtype=thumbnail_data.dtype)
                    h, w = thumbnail_data.shape
                    padded[:h, :w] = thumbnail_data
                    thumbnail_data = padded
                else:
                    thumbnail_data = thumbnail_data[:256, :256]

            thumbnail_group.create_dataset("Data", data=thumbnail_data.astype(np.uint8))

        return path

    def _create_zyx_ngff_image(self) -> nz.NgffImage:
        """
        Create a new NgffImage with data reordered from XYZ to ZYX.

        This is used when saving to OME-Zarr format which expects ZYX ordering.
        The data array is transposed and scale/translation are reordered accordingly.

        Returns:
            NgffImage: New image with ZYX-ordered data and metadata
        """
        if self.axes_order != "XYZ":
            raise ValueError("This method should only be called when axes_order is XYZ")

        # Transpose data from XYZ to ZYX (reverse the spatial dimensions)
        # Assuming data shape is [C, X, Y, Z] -> [C, Z, Y, X]
        data = self.ngff_image.data

        # Find spatial dimension indices
        spatial_axes = []
        channel_axes = []
        for i, dim_name in enumerate(self.ngff_image.dims):
            if dim_name.lower() in ["x", "y", "z"]:
                spatial_axes.append(i)
            else:
                channel_axes.append(i)

        # Create transpose indices: reverse the spatial axes order
        transpose_indices = channel_axes + spatial_axes[::-1]
        transposed_data = data.transpose(transpose_indices)

        # Create new dims list with ZYX ordering
        new_dims = []
        for i, dim_name in enumerate(self.ngff_image.dims):
            if dim_name.lower() not in ["x", "y", "z"]:
                new_dims.append(dim_name)
        # Add spatial dims in ZYX order
        spatial_dim_names = [self.ngff_image.dims[i] for i in spatial_axes]
        new_dims.extend(spatial_dim_names[::-1])

        # Reorder scale and translation from XYZ to ZYX
        current_scale = self.ngff_image.scale
        current_translation = self.ngff_image.translation

        new_scale = {}
        new_translation = {}

        # Copy non-spatial dimensions
        for key, value in current_scale.items():
            if key.lower() not in ["x", "y", "z"]:
                new_scale[key] = value

        for key, value in current_translation.items():
            if key.lower() not in ["x", "y", "z"]:
                new_translation[key] = value

        # Reorder spatial dimensions from XYZ to ZYX
        if "x" in current_scale and "y" in current_scale and "z" in current_scale:
            new_scale["z"] = current_scale["z"]
            new_scale["y"] = current_scale["y"]
            new_scale["x"] = current_scale["x"]

        if (
            "x" in current_translation
            and "y" in current_translation
            and "z" in current_translation
        ):
            new_translation["z"] = current_translation["z"]
            new_translation["y"] = current_translation["y"]
            new_translation["x"] = current_translation["x"]

        # Create new NgffImage with ZYX ordering
        zyx_image = nz.NgffImage(
            data=transposed_data,
            dims=new_dims,
            scale=new_scale,
            translation=new_translation,
            name=self.ngff_image.name,
        )

        return zyx_image

    def copy(self) -> ZarrNii:
        """
        Create a copy of this ZarrNii.

        Returns:
            New ZarrNii with copied data
        """
        # Create a new NgffImage with the same properties
        copied_image = nz.NgffImage(
            data=self.ngff_image.data,  # Dask arrays are lazy so this is efficient
            dims=self.ngff_image.dims.copy(),
            scale=self.ngff_image.scale.copy(),
            translation=self.ngff_image.translation.copy(),
            name=self.ngff_image.name,
        )
        return ZarrNii(
            ngff_image=copied_image,
            axes_order=self.axes_order,
            xyz_orientation=self.xyz_orientation,
            _omero=self._omero,
        )

    def compute(self) -> nz.NgffImage:
        """
        Compute the dask array and return the underlying NgffImage.

        This triggers computation of any lazy operations and returns
        the NgffImage with computed data.

        Returns:
            NgffImage with computed data
        """
        computed_data = self.ngff_image.data.compute()

        # Create new NgffImage with computed data
        computed_image = nz.NgffImage(
            data=computed_data,
            dims=self.ngff_image.dims,
            scale=self.ngff_image.scale,
            translation=self.ngff_image.translation,
            name=self.ngff_image.name,
        )
        return computed_image

    def get_orientation(self) -> str:
        """
        Get the anatomical orientation of the dataset.

        This function returns the orientation string (e.g., 'RAS', 'LPI') of the dataset.

        Returns:
            str: The orientation string corresponding to the dataset's anatomical orientation.
        """
        return self.orientation

    def get_zooms(self, axes_order: str = None) -> np.ndarray:
        """
        Get voxel spacing (zooms) from NgffImage scale.

        Args:
            axes_order: Spatial axes order, defaults to self.axes_order

        Returns:
            Array of voxel spacings
        """
        if axes_order is None:
            axes_order = self.axes_order

        spatial_dims = ["z", "y", "x"] if axes_order == "ZYX" else ["x", "y", "z"]
        zooms = []

        for dim in spatial_dims:
            if dim in self.ngff_image.scale:
                zooms.append(self.ngff_image.scale[dim])
            else:
                zooms.append(1.0)

        return np.array(zooms)

    def get_origin(self, axes_order: str = None) -> np.ndarray:
        """
        Get origin (translation) from NgffImage.

        Args:
            axes_order: Spatial axes order, defaults to self.axes_order

        Returns:
            Array of origin coordinates
        """
        if axes_order is None:
            axes_order = self.axes_order

        spatial_dims = ["z", "y", "x"] if axes_order == "ZYX" else ["x", "y", "z"]
        origin = []

        for dim in spatial_dims:
            if dim in self.ngff_image.translation:
                origin.append(self.ngff_image.translation[dim])
            else:
                origin.append(0.0)

        return np.array(origin)

    def get_affine_matrix(self, axes_order: str = None) -> np.ndarray:
        """
        Construct a 4x4 affine matrix from NGFF metadata (scale/translation),
        and align it to self.orientation (if provided) using nibabel.orientations.

        Args:
            axes_order: Spatial axes order, e.g. 'ZYX' or 'XYZ'. Defaults to 'XYZ'.

        Returns:
            np.ndarray: 4x4 affine matrix.
        """
        if axes_order is None:
            axes_order = self.axes_order

        if axes_order == "ZYX":
            orientation = reverse_orientation_string(self.orientation)
        else:
            orientation = self.orientation

        # Safely pull scale/translation from metadata (dict-like expected)
        scale_meta = getattr(self.ngff_image, "scale", {}) or {}
        trans_meta = getattr(self.ngff_image, "translation", {}) or {}

        scale = np.ones(
            3,
        )
        trans = np.zeros(
            3,
        )

        for i, dim in enumerate(axes_order):
            s = scale_meta.get(dim.lower())
            if s is not None:
                scale[i] = float(s)

        for i, dim in enumerate(axes_order):
            s = trans_meta.get(dim.lower())
            if s is not None:
                trans[i] = float(s)

        affine = _axcodes2aff(orientation, scale=scale, translate=trans)

        return affine

    def apply_transform_ref_to_flo_indices(self, *transforms, ref_znimg, indices):
        """Transform indices from reference to floating space."""
        # Placeholder implementation - would need full transform logic
        return indices

    def apply_transform_flo_to_ref_indices(self, *transforms, ref_znimg, indices):
        """Transform indices from floating to reference space."""
        # Placeholder implementation - would need full transform logic
        return indices

    def list_channels(self) -> list[str]:
        """Get list of available channel labels from OMERO metadata.

        Extracts channel labels from OMERO metadata if available, providing
        human-readable names for multi-channel datasets.

        Returns:
            List of channel label strings. Empty list if no OMERO metadata
            is available or no channels are defined.

        Examples:
            >>> # Check available channels
            >>> labels = znii.list_channels()
            >>> print(f"Available channels: {labels}")
            >>> # ['DAPI', 'GFP', 'RFP', 'Cy5']

            >>> # Select specific channels by label
            >>> selected = znii.select_channels(channel_labels=['DAPI', 'GFP'])

        Notes:
            - Requires OMERO metadata to be present in the dataset
            - Returns empty list for datasets without channel metadata
            - Labels are extracted from the 'label' field of each channel
        """
        if self.omero is None or not hasattr(self.omero, "channels"):
            return []

        return [
            ch.label if hasattr(ch, "label") else ch.get("label", "")
            for ch in self.omero.channels
        ]

    def select_channels(
        self,
        channels: list[int] | None = None,
        channel_labels: list[str] | None = None,
    ) -> ZarrNii:
        """Select specific channels from multi-channel image data.

        Creates a new ZarrNii instance containing only the specified channels,
        reducing memory usage and focusing analysis on channels of interest.
        Supports selection by both numeric indices and human-readable labels.

        Args:
            channels: List of 0-based channel indices to select.
                Mutually exclusive with channel_labels
            channel_labels: List of channel names to select by label.
                Requires OMERO metadata. Mutually exclusive with channels

        Returns:
            New ZarrNii instance with selected channels and updated metadata

        Raises:
            ValueError: If both channels and channel_labels specified, or if
                channel_labels used without OMERO metadata, or if labels not found
            IndexError: If channel indices are out of range

        Examples:
            >>> # Select channels by index
            >>> selected = znii.select_channels(channels=[0, 2])

            >>> # Select channels by label (requires OMERO metadata)
            >>> selected = znii.select_channels(channel_labels=['DAPI', 'GFP'])

            >>> # Check available labels first
            >>> available = znii.list_channels()
            >>> print(f"Available: {available}")
            >>> selected = znii.select_channels(channel_labels=available[:2])

        Notes:
            - Preserves all spatial dimensions and timepoints
            - Updates OMERO metadata to reflect selected channels
            - Maintains spatial transformations and other metadata
            - Channel order in output matches selection order
        """
        if channels is not None and channel_labels is not None:
            raise ValueError("Cannot specify both 'channels' and 'channel_labels'")

        if channel_labels is not None:
            if self.omero is None:
                raise ValueError(
                    "Channel labels were specified but no omero metadata found"
                )

            available_labels = self.list_channels()
            channel_indices = []
            for label in channel_labels:
                if label not in available_labels:
                    raise ValueError(f"Channel label '{label}' not found")
                channel_indices.append(available_labels.index(label))
            channels = channel_indices

        if channels is None:
            # Return a copy with all channels
            return self.copy()

        # Check if channel dimension exists
        if "c" not in self.dims:
            raise ValueError("No channel dimension found in the data")

        # Get channel dimension index
        c_idx = self.dims.index("c")

        # Create slice objects for proper dimension indexing
        slices = [slice(None)] * len(self.data.shape)
        slices[c_idx] = channels

        # Select channels from data using proper dimension indexing
        selected_data = self.data[tuple(slices)]

        # Create new NgffImage with selected data
        new_ngff_image = nz.NgffImage(
            data=selected_data,
            dims=self.dims,
            scale=self.scale,
            translation=self.translation,
            name=self.name,
        )

        # Filter omero metadata to match selected channels
        filtered_omero = None
        if self.omero is not None and hasattr(self.omero, "channels"):

            class FilteredOmero:
                def __init__(self, channels):
                    self.channels = channels

            filtered_channels = [self.omero.channels[i] for i in channels]
            filtered_omero = FilteredOmero(filtered_channels)

        return ZarrNii(
            ngff_image=new_ngff_image,
            axes_order=self.axes_order,
            xyz_orientation=self.xyz_orientation,
            _omero=filtered_omero,
        )

    def select_timepoints(self, timepoints: list[int] | None = None) -> ZarrNii:
        """
        Select timepoints from the image data and return a new ZarrNii instance.

        Args:
            timepoints: Timepoint indices to select

        Returns:
            New ZarrNii instance with selected timepoints
        """
        if timepoints is None:
            # Return a copy with all timepoints
            return self.copy()

        # Check if time dimension exists
        if "t" not in self.dims:
            raise ValueError("No time dimension found in the data")

        # Get time dimension index
        t_idx = self.dims.index("t")

        # Create slice objects
        slices = [slice(None)] * len(self.data.shape)
        slices[t_idx] = timepoints

        # Select timepoints from data
        selected_data = self.data[tuple(slices)]

        # Create new NgffImage with selected data
        new_ngff_image = nz.NgffImage(
            data=selected_data,
            dims=self.dims,
            scale=self.scale,
            translation=self.translation,
            name=self.name,
        )

        return ZarrNii(
            ngff_image=new_ngff_image,
            axes_order=self.axes_order,
            xyz_orientation=self.xyz_orientation,
            _omero=self._omero,  # Timepoint selection doesn't affect omero metadata
        )

    def to_ngff_image(self, name: str = None) -> nz.NgffImage:
        """
        Convert to NgffImage object.

        Args:
            name: Optional name for the image

        Returns:
            NgffImage representation
        """
        if name is None:
            name = self.name

        return nz.NgffImage(
            data=self.data,
            dims=self.dims,
            scale=self.scale,
            translation=self.translation,
            name=name,
        )

    def segment(
        self, plugin, chunk_size: tuple[int, ...] | None = None, **kwargs
    ) -> ZarrNii:
        """
        Apply segmentation plugin to the image using blockwise processing.

        This method applies a segmentation plugin to the image data using dask's
        blockwise processing for efficient computation on large datasets.

        Args:
            plugin: Segmentation plugin instance or class to apply
            chunk_size: Optional chunk size for dask processing. If None, uses current chunks.
            **kwargs: Additional arguments passed to the plugin

        Returns:
            New ZarrNii instance with segmented data as labels
        """
        from .plugins.segmentation import SegmentationPlugin

        # Handle plugin instance or class
        if isinstance(plugin, type) and issubclass(plugin, SegmentationPlugin):
            plugin = plugin(**kwargs)
        elif not isinstance(plugin, SegmentationPlugin):
            raise TypeError(
                "Plugin must be an instance or subclass of SegmentationPlugin"
            )

        # Prepare chunk size
        if chunk_size is not None:
            # Rechunk the data if different chunk size requested
            data = self.data.rechunk(chunk_size)
        else:
            data = self.data

        # Create metadata dict to pass to plugin
        metadata = {
            "axes_order": self.axes_order,
            "orientation": self.xyz_orientation,
            "shape": self.shape,
            "dims": self.dims,
            "scale": self.scale,
            "translation": self.translation,
        }

        # Create a wrapper function for map_blocks
        def segment_block(block):
            """Wrapper function to apply segmentation to a single block."""
            # Handle single blocks
            return plugin.segment(block, metadata)

        # Apply segmentation using dask map_blocks
        segmented_data = da.map_blocks(
            segment_block,
            data,
            dtype=np.uint8,  # Segmentation results are typically uint8
            meta=np.array([], dtype=np.uint8),  # Provide meta information
        )

        # Create new NgffImage with segmented data
        new_ngff_image = nz.NgffImage(
            data=segmented_data,
            dims=self.dims.copy(),
            scale=self.scale.copy(),
            translation=self.translation.copy(),
            name=f"{self.name}_segmented_{plugin.name.lower().replace(' ', '_')}",
        )

        # Return new ZarrNii instance
        return ZarrNii(
            ngff_image=new_ngff_image,
            axes_order=self.axes_order,
            xyz_orientation=self.xyz_orientation,
        )

    def segment_otsu(
        self, nbins: int = 256, chunk_size: tuple[int, ...] | None = None
    ) -> ZarrNii:
        """
        Apply local Otsu thresholding segmentation to the image.

        Convenience method for local Otsu thresholding segmentation.
        This computes the threshold locally for each processing block.

        Args:
            nbins: Number of bins for histogram computation (default: 256)
            chunk_size: Optional chunk size for dask processing

        Returns:
            New ZarrNii instance with binary segmentation
        """
        from .plugins.segmentation import LocalOtsuSegmentation

        plugin = LocalOtsuSegmentation(nbins=nbins)
        return self.segment(plugin, chunk_size=chunk_size)

    def segment_threshold(
        self,
        thresholds: float | list[float],
        inclusive: bool = True,
        chunk_size: tuple[int, ...] | None = None,
    ) -> ZarrNii:
        """
        Apply threshold-based segmentation to the image.

        Convenience method for threshold-based segmentation using either
        manual threshold values or computed thresholds.

        Args:
            thresholds: Single threshold value or list of threshold values.
                For single threshold, creates binary segmentation (0/1).
                For multiple thresholds, creates multi-class segmentation (0/1/2/...).
            inclusive: Whether thresholds are inclusive (default: True).
                If True, pixels >= threshold are labeled as foreground.
                If False, pixels > threshold are labeled as foreground.
            chunk_size: Optional chunk size for dask processing

        Returns:
            New ZarrNii instance with labeled segmentation

        Examples:
            >>> # Binary threshold segmentation
            >>> segmented = znimg.segment_threshold(0.5)
            >>>
            >>> # Multi-level threshold segmentation
            >>> thresholds = znimg.compute_otsu_thresholds(classes=3)
            >>> segmented = znimg.segment_threshold(thresholds[1:-1])  # Exclude min/max values
        """
        from .plugins.segmentation import ThresholdSegmentation

        plugin = ThresholdSegmentation(thresholds=thresholds, inclusive=inclusive)
        return self.segment(plugin, chunk_size=chunk_size)

    def compute_histogram(
        self,
        bins: int = 256,
        range: tuple[float, float] | None = None,
        mask: ZarrNii | None = None,
        **kwargs: Any,
    ) -> tuple[da.Array, da.Array]:
        """
        Compute histogram of the image.

        This method computes the histogram of image intensities, optionally using
        a mask to weight the computation. The histogram is computed using dask for
        efficient processing of large datasets.

        Args:
            bins: Number of histogram bins (default: 256)
            range: Optional tuple (min, max) defining histogram range. If None,
                uses the full range of the data
            mask: Optional ZarrNii mask of same shape as image. Only pixels
                where mask > 0 are included in histogram computation
            **kwargs: Additional arguments passed to dask.array.histogram

        Returns:
            Tuple of (histogram_counts, bin_edges) where:
            - histogram_counts: dask array of histogram bin counts
            - bin_edges: dask array of bin edge values (length = bins + 1)

        Examples:
            >>> # Compute histogram
            >>> hist, bin_edges = znimg.compute_histogram(bins=128)
            >>>
            >>> # Compute histogram with mask
            >>> mask = znimg > 0.5
            >>> hist_masked, _ = znimg.compute_histogram(mask=mask)
        """
        from .analysis import compute_histogram

        mask_data = mask.darr if mask is not None else None
        return compute_histogram(
            self.darr, bins=bins, range=range, mask=mask_data, **kwargs
        )

    def compute_otsu_thresholds(
        self,
        classes: int = 2,
        bins: int = 256,
        range: tuple[float, float] | None = None,
        mask: ZarrNii | None = None,
        return_figure: bool = False,
    ) -> list[float] | tuple[list[float], Any]:
        """
        Compute Otsu multi-level thresholds for the image.

        This method first computes the histogram of the image, then uses
        scikit-image's threshold_multiotsu to compute optimal threshold values.

        Args:
            classes: Number of classes to separate data into (default: 2).
                Must be >= 2. For classes=2, returns 1 threshold. For classes=k,
                returns k-1 thresholds.
            bins: Number of histogram bins (default: 256)
            range: Optional tuple (min, max) defining histogram range. If None,
                uses the full range of the data
            mask: Optional ZarrNii mask of same shape as image. Only pixels
                where mask > 0 are included in histogram computation
            return_figure: If True, returns a tuple containing thresholds and a
                matplotlib figure with the histogram and annotated threshold lines
                (default: False).

        Returns:
            If return_figure is False (default):
                List of threshold values. For classes=k, returns k+1 values:
                [0, threshold1, threshold2, ..., threshold_k-1, max_intensity]
                where 0 represents the minimum and max_intensity represents the maximum.

            If return_figure is True:
                Tuple of (thresholds, figure) where figure is a matplotlib Figure
                object showing the histogram with annotated threshold lines.

        Examples:
            >>> # Compute binary threshold (2 classes)
            >>> thresholds = znimg.compute_otsu_thresholds(classes=2)
            >>> print(f"Binary thresholds: {thresholds}")
            >>>
            >>> # Compute multi-level thresholds (3 classes)
            >>> thresholds = znimg.compute_otsu_thresholds(classes=3)
            >>> print(f"Multi-level thresholds: {thresholds}")
            >>>
            >>> # Get histogram data along with thresholds
            >>> thresholds, (hist, bin_edges) = znimg.compute_otsu_thresholds(
            ...     classes=2, return_histogram=True
            ... )
            >>>
            >>> # Generate a figure with annotated thresholds
            >>> thresholds, fig = znimg.compute_otsu_thresholds(
            ...     classes=2, return_figure=True
            ... )
            >>> fig.savefig('otsu_thresholds.png')
        """
        from .analysis import compute_otsu_thresholds

        # First compute histogram
        hist, bin_edges = self.compute_histogram(bins=bins, range=range, mask=mask)

        # Then compute thresholds with optional returns
        return compute_otsu_thresholds(
            hist,
            classes=classes,
            bin_edges=bin_edges,
            return_figure=return_figure,
        )

    def apply_scaled_processing(
        self,
        plugin,
        downsample_factor: int = 4,
        chunk_size: tuple[int, ...] | None = None,
        upsampled_ome_zarr_path: str | None = None,
        **kwargs,
    ) -> ZarrNii:
        """
        Apply scaled processing plugin using multi-resolution approach.

        This method implements a multi-resolution processing pipeline where:
        1. The image is downsampled for efficient computation
        2. The plugin's lowres_func is applied to the downsampled data
        3. The result is upsampled using dask-based upsampling
        4. The plugin's highres_func applies the result to full-resolution data

        Args:
            plugin: ScaledProcessingPlugin instance or class to apply
            downsample_factor: Factor for downsampling (default: 4)
            chunk_size: Optional chunk size for low-res processing. If None, uses (1, 10, 10, 10).
            upsampled_ome_zarr_path: Path to save intermediate OME-Zarr, default saved in system temp directory.
            **kwargs: Additional arguments passed to the plugin

        Returns:
            New ZarrNii instance with processed data
        """
        from .plugins.scaled_processing import ScaledProcessingPlugin

        # Handle plugin instance or class
        if isinstance(plugin, type) and issubclass(plugin, ScaledProcessingPlugin):
            plugin = plugin(**kwargs)
        elif not isinstance(plugin, ScaledProcessingPlugin):
            raise TypeError(
                "Plugin must be an instance or subclass of ScaledProcessingPlugin"
            )

        # Step 1: Downsample the data for low-resolution processing
        lowres_znimg = self.downsample(level=int(np.log2(downsample_factor)))

        # Convert to numpy array for lowres processing
        lowres_array = lowres_znimg.data.compute()

        # Step 2: Apply low-resolution function and prepare for upsampling
        # Use chunk_size parameter for the low-res processing chunks
        lowres_chunks = chunk_size if chunk_size is not None else (1, 10, 10, 10)
        lowres_znimg.data = da.from_array(
            plugin.lowres_func(lowres_array), chunks=lowres_chunks
        )

        # Use temporary OME-Zarr to break up dask graph for performance
        import tempfile

        if upsampled_ome_zarr_path is None:
            upsampled_ome_zarr_path = tempfile.mkdtemp(suffix="_SPIM.ome.zarr")

        # Step 3: Upsample using dask-based upsampling, save to ome zarr
        lowres_znimg.upsample(to_shape=self.shape).to_ome_zarr(
            upsampled_ome_zarr_path, max_layer=1
        )

        upsampled_znimg = ZarrNii.from_ome_zarr(upsampled_ome_zarr_path)

        corrected_znimg = self.copy()

        # Step 4: Apply high-resolution function
        # rechunk original data to use same chunksize as upsampled_data, before multiplying
        corrected_znimg.data = plugin.highres_func(
            self.data.rechunk(upsampled_znimg.data.chunks), upsampled_znimg.data
        )

        return corrected_znimg

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ZarrNii(\n"
            f"  name='{self.name}', shape={self.shape}, dims={self.dims},\n"
            f"  axes_order='{self.axes_order}', xyz_orientation='{self.xyz_orientation}',\n"
            f"  scale={self.scale},\n"
            f"  dtype={self.data.dtype}, chunksize={self.data.chunksize}\n"
            f")"
        )

    def _is_metadata_valid(self, result: Any) -> bool:
        """Check if resulting array preserves metadata integrity.

        Args:
            result: The result from a Dask operation

        Returns:
            True if metadata remains valid, False otherwise
        """
        # If it's not a dask array, we can't wrap it
        if not isinstance(result, da.Array):
            return False

        # Simplest heuristic: same shape and dtype = safe
        if (
            result.shape == self.ngff_image.data.shape
            and result.dtype == self.ngff_image.data.dtype
        ):
            return True

        return False

    def _wrap_result(self, result: Any, op_name: str) -> ZarrNii:
        """Wrap Dask array result as a new ZarrNii, enforcing metadata validity.

        Args:
            result: The result from a Dask operation
            op_name: Name of the operation for error messages

        Returns:
            New ZarrNii instance with updated data

        Raises:
            MetadataInvalidError: If operation changes shape or dtype
        """
        # If result is not a dask array, return it as-is
        if not isinstance(result, da.Array):
            return result

        # Check if metadata is still valid
        if not self._is_metadata_valid(result):
            raise MetadataInvalidError(
                f"Operation '{op_name}' changes shape or dtype — metadata invalid. "
                "Use explicit spatial operations (e.g., resample, reorder_axes)."
            )

        # Create a shallow copy of self and update the data
        newobj = copy.copy(self)
        # Also copy the ngff_image to avoid mutating the original
        newobj.ngff_image = copy.copy(self.ngff_image)
        newobj.ngff_image.data = result

        return newobj

    # Arithmetic operator overloading
    def __add__(self, other):
        """Add operation."""
        return self._wrap_result(self.ngff_image.data.__add__(other), "add")

    def __radd__(self, other):
        """Reverse add operation."""
        return self._wrap_result(self.ngff_image.data.__radd__(other), "add")

    def __sub__(self, other):
        """Subtract operation."""
        return self._wrap_result(self.ngff_image.data.__sub__(other), "subtract")

    def __rsub__(self, other):
        """Reverse subtract operation."""
        return self._wrap_result(self.ngff_image.data.__rsub__(other), "subtract")

    def __mul__(self, other):
        """Multiply operation."""
        return self._wrap_result(self.ngff_image.data.__mul__(other), "multiply")

    def __rmul__(self, other):
        """Reverse multiply operation."""
        return self._wrap_result(self.ngff_image.data.__rmul__(other), "multiply")

    def __truediv__(self, other):
        """True division operation."""
        return self._wrap_result(self.ngff_image.data.__truediv__(other), "true_divide")

    def __rtruediv__(self, other):
        """Reverse true division operation."""
        return self._wrap_result(
            self.ngff_image.data.__rtruediv__(other), "true_divide"
        )

    def __floordiv__(self, other):
        """Floor division operation."""
        return self._wrap_result(
            self.ngff_image.data.__floordiv__(other), "floor_divide"
        )

    def __rfloordiv__(self, other):
        """Reverse floor division operation."""
        return self._wrap_result(
            self.ngff_image.data.__rfloordiv__(other), "floor_divide"
        )

    def __mod__(self, other):
        """Modulo operation."""
        return self._wrap_result(self.ngff_image.data.__mod__(other), "mod")

    def __rmod__(self, other):
        """Reverse modulo operation."""
        return self._wrap_result(self.ngff_image.data.__rmod__(other), "mod")

    def __pow__(self, other):
        """Power operation."""
        return self._wrap_result(self.ngff_image.data.__pow__(other), "power")

    def __rpow__(self, other):
        """Reverse power operation."""
        return self._wrap_result(self.ngff_image.data.__rpow__(other), "power")

    def __neg__(self):
        """Negation operation."""
        return self._wrap_result(self.ngff_image.data.__neg__(), "negative")

    def __pos__(self):
        """Positive operation."""
        return self._wrap_result(self.ngff_image.data.__pos__(), "positive")

    def __abs__(self):
        """Absolute value operation."""
        return self._wrap_result(self.ngff_image.data.__abs__(), "absolute")

    def __array_ufunc__(self, ufunc, method: str, *inputs, out=None, **kwargs):
        """Enable NumPy universal function support.

        This allows using NumPy ufuncs (e.g., np.sqrt, np.add) on ZarrNii objects
        while preserving metadata.

        Args:
            ufunc: The ufunc object
            method: String indicating how the ufunc was called
            inputs: Input arrays
            out: Optional output array
            **kwargs: Additional keyword arguments

        Returns:
            ZarrNii with the result of the ufunc operation
        """
        # Handle out parameter
        if out is not None:
            return NotImplemented

        # Extract dask arrays from ZarrNii objects
        inputs = [i.ngff_image.data if isinstance(i, ZarrNii) else i for i in inputs]

        # Apply the ufunc
        result = getattr(ufunc, method)(*inputs, **kwargs)

        # Wrap and return
        return self._wrap_result(result, ufunc.__name__)

    def __array_function__(self, func, types, args, kwargs):
        """Enable NumPy function protocol.

        This allows using NumPy functions (e.g., np.mean, np.concatenate) on
        ZarrNii objects while preserving metadata when safe.

        Args:
            func: The NumPy function
            types: Types involved in the operation
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            ZarrNii with the result of the function, or NotImplemented
        """
        # Only handle if at least one argument is ZarrNii
        if not any(issubclass(t, ZarrNii) for t in types):
            return NotImplemented

        # Extract dask arrays from ZarrNii objects
        args = [a.ngff_image.data if isinstance(a, ZarrNii) else a for a in args]

        # Apply the function
        result = func(*args, **kwargs)

        # Wrap and return
        return self._wrap_result(result, func.__name__)

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks.

        Combines ZarrNii metadata with the dask array visualization.
        """
        # Create HTML for ZarrNii metadata
        metadata_html = f"""
        <div style="margin-bottom: 10px;">
            <strong>ZarrNii Object</strong>
            <table style="margin-left: 20px; border-collapse: collapse;">
                <tr><td style="padding: 2px;"><strong>name:</strong></td><td style="padding: 2px;">{self.name}</td></tr>
                <tr><td style="padding: 2px;"><strong>shape:</strong></td><td style="padding: 2px;">{self.shape}</td></tr>
                <tr><td style="padding: 2px;"><strong>dims:</strong></td><td style="padding: 2px;">{self.dims}</td></tr>
                <tr><td style="padding: 2px;"><strong>axes_order:</strong></td><td style="padding: 2px;">{self.axes_order}</td></tr>
                <tr><td style="padding: 2px;"><strong>xyz_orientation:</strong></td><td style="padding: 2px;">{self.xyz_orientation}</td></tr>
                <tr><td style="padding: 2px;"><strong>scale:</strong></td><td style="padding: 2px;">{self.scale}</td></tr>
                <tr><td style="padding: 2px;"><strong>dtype:</strong></td><td style="padding: 2px;">{self.data.dtype}</td></tr>
                <tr><td style="padding: 2px;"><strong>chunksize:</strong></td><td style="padding: 2px;">{self.data.chunksize}</td></tr>
            </table>
        </div>
        """

        # Get dask array's HTML representation
        dask_html = ""
        if hasattr(self.data, "_repr_html_"):
            dask_html = f"""
            <div style="margin-top: 10px;">
                <strong>Dask Array:</strong>
                {self.data._repr_html_()}
            </div>
            """

        return metadata_html + dask_html


# Helper functions for backward compatibility
def reverse_orientation_string(orientation_str):
    """
    Reverse an orientation string to convert between ZYX and XYZ axis orders.

    This function reverses the character order of an orientation string to convert
    between ZYX-based and XYZ-based orientation encoding. For example:
    'RAS' (ZYX order) becomes 'SAR' (XYZ order).

    Args:
        orientation_str (str): Three-character orientation string (e.g., 'RAS', 'LPI')

    Returns:
        str: Reversed orientation string

    Examples:
        >>> reverse_orientation_string('RAS')
        'SAR'
        >>> reverse_orientation_string('LPI')
        'IPL'
    """

    if len(orientation_str) != 3:
        raise ValueError(
            f"Orientation string must be exactly 3 characters, got: {orientation_str}"
        )

    return orientation_str[::-1]


def _affine_to_orientation(affine):
    """
    Convert an affine matrix to an anatomical orientation string (e.g., 'RAS').

    Parameters:
        affine (numpy.ndarray): Affine matrix from voxel to world coordinates.

    Returns:
        str: Anatomical orientation (e.g., 'RAS', 'LPI').
    """
    from nibabel.orientations import io_orientation

    # Get voxel-to-world mapping
    orient = io_orientation(affine)

    # Maps for axis labels
    axis_labels = ["R", "A", "S"]
    flipped_labels = ["L", "P", "I"]

    orientation = []
    for axis, direction in orient:
        axis = int(axis)
        if direction == 1:
            orientation.append(axis_labels[axis])
        else:
            orientation.append(flipped_labels[axis])

    return "".join(orientation)


from nibabel.orientations import (
    axcodes2ornt,
    inv_ornt_aff,
    io_orientation,
    ornt_transform,
)


def _infer_spatial_shape_from_data(self, axes_order: str) -> tuple[int, int, int]:
    """
    Infer the (Z, Y, X) or (X, Y, Z) spatial shape from self.data.shape and axes_order.
    Assumes the last 3 dims of self.data are spatial (common in NGFF / OME-Zarr workflows).
    """
    shape = getattr(self, "data", None)
    if shape is None:
        raise ValueError("self.data is not set; cannot infer spatial shape.")
    shape = self.data.shape
    if len(shape) < 3:
        raise ValueError(f"Expected at least 3D data; got shape={shape}")

    tail3 = shape[-3:]  # assume trailing 3 dims are spatial
    ao = (axes_order or getattr(self, "axes_order", "XYZ")).upper()
    if ao == "ZYX":
        # trailing dims are (Z, Y, X)
        return tuple(int(d) for d in tail3)  # (Z, Y, X)
    else:
        # treat everything else as XYZ
        x, y, z = tail3
        return (int(x), int(y), int(z))  # (X, Y, Z)


def _make_affine_aligned_to_orientation(
    affine: np.ndarray, orientation: str, spatial_shape: tuple[int, int, int]
) -> np.ndarray:
    """
    Align affine to `orientation` using nibabel.orientations, with the correct spatial shape.
    """
    if not isinstance(orientation, str) or len(orientation) != 3:
        raise ValueError(
            f"orientation must be a 3-letter code like 'RAS', got {orientation!r}"
        )

    current_ornt = io_orientation(affine)
    target_ornt = axcodes2ornt(tuple(orientation.upper()))
    transform = ornt_transform(current_ornt, target_ornt)
    # IMPORTANT: pass the SPATIAL SHAPE (not affine.shape)
    return affine @ inv_ornt_aff(transform, spatial_shape)


def _axcodes2aff(axcodes, scale, translate, labels=None):
    """Create a homogeneous affine from axis codes.

    Uses the provided scale and translate to set diag and offset.

    Parameters
    ----------
    axcodes : sequence of length p
        Axis codes, e.g. ('R','A','S') or (None, 'L', 'S').
    scale: (3,) list of scaling values for X Y Z
    trans: (3,) list of translation values for X Y Z
    labels : sequence of (2,) label tuples, optional
        Same semantics as for axcodes2ornt / ornt2axcodes.  If None, defaults
        to (('L','R'), ('P','A'), ('I','S')).

    Returns
    -------
    aff : (p+1, p+1) ndarray
        Homogeneous affine implementing the permutation and flips implied by
        `axcodes`, with provided translation and scaling.

    Notes
    -----
    - If an axis code is None (a dropped axis), the corresponding column in
      the linear part is left all zeros.
    """
    ornt = axcodes2ornt(axcodes, labels)
    p = ornt.shape[0]
    aff = np.zeros((p + 1, p + 1), dtype=float)
    # Fill linear part: for each input axis (column), put a 1 or -1 in the
    # output-axis row indicated by ornt[:,0]
    for in_idx, (out_ax, flip) in enumerate(ornt):
        if np.isnan(out_ax):
            # dropped axis -> leave column zero
            continue
        out_idx = int(out_ax)
        aff[out_idx, in_idx] = float(flip) * scale[in_idx]
        aff[out_idx, p] = translate[in_idx]
    aff[p, p] = 1.0
    return aff


def _axcodes2flips(axcodes, labels=None):
    """
    Make a dict mapping dimensions ('x','y,'z') to flips (+1,-1) based on  xyz orientation string (or axcodes), e.g. RAS

    Parameters:
    axcodes : sequence of length p
        Axis codes, e.g. ('R','A','S') or (None, 'L', 'S').
    labels : sequence of (2,) label tuples, optional
        Same semantics as for axcodes2ornt / ornt2axcodes.  If None, defaults
        to (('L','R'), ('P','A'), ('I','S')).


    Returns:
        dict: Anatomical orientation (e.g., 'RAS', 'LPI').
    """

    ornt = axcodes2ornt(axcodes, labels)
    dims = ["x", "y", "z"]

    dim_flips = {}
    for i, (out_ax, flip) in enumerate(ornt):
        dim_flips[dims[i]] = float(flip)

    return dim_flips
