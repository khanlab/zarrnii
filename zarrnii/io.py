"""I/O utilities for loading and saving OME-Zarr and NIfTI data.

This module contains functions for:
- Loading NgffImage objects from OME-Zarr stores
- Saving NgffImage objects to OME-Zarr stores with pyramid generation
- Multiscale object access and manipulation
- Channel and timepoint selection utilities
- Metadata extraction and conversion

The functions in this module operate directly on ngff_zarr.NgffImage objects
and provide the foundation for the higher-level ZarrNii class interface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import ngff_zarr as nz


def load_ngff_image(
    store_or_path: Union[str, Any],
    level: int = 0,
    channels: Optional[List[int]] = None,
    channel_labels: Optional[List[str]] = None,
    timepoints: Optional[List[int]] = None,
    storage_options: Optional[Dict[str, Any]] = None,
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
    store_or_path: Union[str, Any],
    max_layer: int = 4,
    scale_factors: Optional[List[int]] = None,
    orientation: Optional[str] = None,
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
        ...                 scale_factors=[2, 4], orientation="RAS")
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

            # Add orientation metadata to the temporary zarr store if provided
            if orientation:
                try:
                    group = zarr.open_group(temp_zarr_path, mode="r+")
                    group.attrs["orientation"] = orientation
                except Exception:
                    # If we can't write orientation metadata, that's not critical
                    pass

            # Create ZIP file from the directory
            zip_base_path = store_or_path.replace(".zip", "")
            shutil.make_archive(zip_base_path, "zip", temp_zarr_path)
    else:
        # Write to zarr store directly
        nz.to_ngff_zarr(store_or_path, multiscales, **kwargs)

        # Add orientation metadata if provided
        if orientation:
            try:
                if isinstance(store_or_path, str):
                    group = zarr.open_group(store_or_path, mode="r+")
                else:
                    group = zarr.open_group(store_or_path, mode="r+")
                group.attrs["orientation"] = orientation
            except Exception:
                # If we can't write orientation metadata, that's not critical
                pass


def get_multiscales(
    store_or_path,
    storage_options: Optional[Dict] = None,
) -> nz.Multiscales:
    """
    Load the full multiscales object from an OME-Zarr store.

    This provides access to all pyramid levels and metadata.

    Args:
        store_or_path: Store or path to the OME-Zarr file
        storage_options: Additional options passed to zarr storage backend

    Returns:
        Multiscales object containing all pyramid levels and metadata

    Examples:
        >>> multiscales = get_multiscales("/path/to/data.zarr")
        >>> print(f"Number of levels: {len(multiscales.images)}")
    """
    return nz.from_ngff_zarr(store_or_path, storage_options=storage_options)


def _select_dimensions_from_image(
    image: nz.NgffImage,
    multiscales: nz.Multiscales,
    channels: Optional[List[int]] = None,
    channel_labels: Optional[List[str]] = None,
    timepoints: Optional[List[int]] = None,
) -> nz.NgffImage:
    """
    Select specific dimensions (channels/timepoints) from an NgffImage.

    This function handles dimension selection while preserving metadata,
    including proper OMERO channel metadata filtering.

    Args:
        image: Input NgffImage
        multiscales: Parent multiscales object for metadata access
        channels: List of channel indices to select
        channel_labels: List of channel names to select (alternative to channels)
        timepoints: List of timepoint indices to select

    Returns:
        New NgffImage with selected dimensions

    Raises:
        ValueError: If both channels and channel_labels are specified,
                   or if specified channels/timepoints are invalid
        KeyError: If channel_labels are not found in metadata
    """
    if channels is not None and channel_labels is not None:
        raise ValueError("Cannot specify both channels and channel_labels")

    # Check if we have omero metadata in the multiscales
    omero_metadata = getattr(multiscales, "omero", None)

    # Use the omero-aware function if we have omero metadata or channel_labels
    if omero_metadata is not None or channel_labels is not None:
        selected_image, filtered_omero = _select_dimensions_from_image_with_omero(
            image, multiscales, channels, channel_labels, timepoints, omero_metadata
        )
        return selected_image

    # Handle basic dimension selection without omero metadata
    if channels is None and timepoints is None:
        return image

    data = image.data
    dims = image.dims

    # Select timepoints if specified
    if timepoints is not None and "t" in dims:
        t_idx = dims.index("t")
        slices = [slice(None)] * len(data.shape)
        slices[t_idx] = timepoints
        data = data[tuple(slices)]

    # Select channels if specified
    if channels is not None and "c" in dims:
        c_idx = dims.index("c")
        slices = [slice(None)] * len(data.shape)
        slices[c_idx] = channels
        data = data[tuple(slices)]

    return nz.NgffImage(
        data=data,
        dims=dims,
        scale=image.scale,
        translation=image.translation,
        name=image.name,
    )


def _select_channels_from_image(
    image: nz.NgffImage,
    multiscales: nz.Multiscales,
    channels: Optional[List[int]] = None,
    channel_labels: Optional[List[str]] = None,
) -> nz.NgffImage:
    """
    Select specific channels from an NgffImage.

    Simplified channel selection function that delegates to the more
    comprehensive dimension selection function.

    Args:
        image: Input NgffImage
        multiscales: Parent multiscales object for metadata access
        channels: List of channel indices to select
        channel_labels: List of channel names to select

    Returns:
        New NgffImage with selected channels
    """
    return _select_dimensions_from_image(
        image, multiscales, channels, channel_labels, None
    )


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
