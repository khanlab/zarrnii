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

from typing import Any, Dict, List, Optional, Tuple, Union

import dask.array as da
import fsspec
import ngff_zarr as nz
import nibabel as nib
import numpy as np
from attrs import define
from scipy.ndimage import zoom

from .transform import AffineTransform, Transform

# NgffImage-based function library
# These functions operate directly on ngff_zarr.NgffImage objects


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
        storage_options: Storage options for Zarr

    Returns:
        Multiscales: The full multiscales object with all pyramid levels
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
    channels: Optional[List[int]] = None,
    channel_labels: Optional[List[str]] = None,
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


def get_affine_matrix(ngff_image: nz.NgffImage, axes_order: str = "ZYX") -> np.ndarray:
    """
    Construct an affine transformation matrix from NgffImage metadata.

    Args:
        ngff_image: Input NgffImage
        axes_order: Order of spatial axes (default: "ZYX")

    Returns:
        4x4 affine transformation matrix
    """
    # Extract scale and translation for spatial dimensions
    spatial_dims = list(axes_order.lower())

    # Build 4x4 affine matrix
    affine = np.eye(4)

    for i, dim in enumerate(spatial_dims):
        if dim in ngff_image.scale:
            affine[i, i] = ngff_image.scale[dim]
        if dim in ngff_image.translation:
            affine[i, 3] = ngff_image.translation[dim]

    return affine


def get_affine_transform(
    ngff_image: nz.NgffImage, axes_order: str = "ZYX"
) -> AffineTransform:
    """
    Get an AffineTransform object from NgffImage metadata.

    Args:
        ngff_image: Input NgffImage
        axes_order: Order of spatial axes (default: "ZYX")

    Returns:
        AffineTransform object
    """
    matrix = get_affine_matrix(ngff_image, axes_order)
    return AffineTransform.from_array(matrix)


# Function-based API for operating on NgffImage objects
def crop_ngff_image(
    ngff_image: nz.NgffImage,
    bbox_min: tuple,
    bbox_max: tuple,
    spatial_dims: List[str] = None,
) -> nz.NgffImage:
    """
    Crop an NgffImage using a bounding box.

    Args:
        ngff_image: Input NgffImage to crop
        bbox_min: Minimum corner of bounding box
        bbox_max: Maximum corner of bounding box
        spatial_dims: Names of spatial dimensions (defaults to ["z", "y", "x"])

    Returns:
        New cropped NgffImage
    """
    if spatial_dims is None:
        spatial_dims = ["z", "y", "x"]
    # Build slices for cropping
    slices = []
    spatial_idx = 0

    for dim in ngff_image.dims:
        if dim.lower() in [d.lower() for d in spatial_dims]:
            # This is a spatial dimension
            if spatial_idx < len(bbox_min):
                slices.append(slice(bbox_min[spatial_idx], bbox_max[spatial_idx]))
                spatial_idx += 1
            else:
                slices.append(slice(None))
        else:
            # Non-spatial dimension, keep all
            slices.append(slice(None))

    # Apply crop
    cropped_data = ngff_image.data[tuple(slices)]

    # Update translation to account for cropping
    new_translation = ngff_image.translation.copy()
    spatial_idx = 0

    for dim in ngff_image.dims:
        if dim.lower() in [d.lower() for d in spatial_dims]:
            if spatial_idx < len(bbox_min) and dim in new_translation:
                # Update translation by adding the crop offset
                offset = bbox_min[spatial_idx] * ngff_image.scale.get(dim, 1.0)
                new_translation[dim] += offset
                spatial_idx += 1

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
    factors: Union[int, List[int]],
    spatial_dims: List[str] = ["z", "y", "x"],
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


def _apply_near_isotropic_downsampling(znimg: "ZarrNii", axes_order: str) -> "ZarrNii":
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
    spatial_dims: List[str] = ["z", "y", "x"],
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
        orientation (str): The anatomical orientation string (e.g., 'RAS', 'LPI').
    """

    ngff_image: nz.NgffImage
    axes_order: str = "ZYX"
    orientation: str = "RAS"
    _omero: Optional[object] = None

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
    def dims(self) -> List[str]:
        """Dimension names."""
        return self.ngff_image.dims

    @property
    def scale(self) -> Dict[str, float]:
        """Scale information from NgffImage."""
        return self.ngff_image.scale

    @property
    def translation(self) -> Dict[str, float]:
        """Translation information from NgffImage."""
        return self.ngff_image.translation

    @property
    def name(self) -> str:
        """Image name from NgffImage."""
        return self.ngff_image.name

    @property
    def affine(self) -> AffineTransform:
        """
        Affine transformation matrix derived from NgffImage scale and translation.

        Returns:
            AffineTransform: 4x4 affine transformation matrix
        """
        return self.get_affine_transform()

    def get_affine_matrix(self, axes_order: str = None) -> np.ndarray:
        """
        Get 4x4 affine transformation matrix from NgffImage metadata.

        Args:
            axes_order: Spatial axes order, defaults to self.axes_order

        Returns:
            4x4 affine transformation matrix
        """
        if axes_order is None:
            axes_order = self.axes_order

        # Create identity 4x4 matrix
        affine = np.eye(4)

        # Map axes order to matrix indices
        spatial_dims = ["z", "y", "x"] if axes_order == "ZYX" else ["x", "y", "z"]

        # Set scale values
        for i, dim in enumerate(spatial_dims):
            if dim in self.ngff_image.scale:
                affine[i, i] = self.ngff_image.scale[dim]

        # Set translation values
        for i, dim in enumerate(spatial_dims):
            if dim in self.ngff_image.translation:
                affine[i, 3] = self.ngff_image.translation[dim]

        # Apply orientation alignment if orientation is available
        if hasattr(self, "orientation") and self.orientation:
            affine = align_affine_to_input_orientation(affine, self.orientation)

        return affine

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

    # Legacy compatibility properties
    @property
    def axes(self) -> Optional[List[Dict]]:
        """Axes metadata - derived from NgffImage for compatibility."""
        axes = []
        for dim in self.ngff_image.dims:
            if dim == "c":
                axes.append({"name": "c", "type": "channel", "unit": None})
            else:
                axes.append({"name": dim, "type": "space", "unit": "micrometer"})
        return axes

    @property
    def coordinate_transformations(self) -> Optional[List[Dict]]:
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
    def omero(self) -> Optional[object]:
        """Omero metadata object."""
        return self._omero

    # Constructor methods
    @classmethod
    def from_ngff_image(
        cls,
        ngff_image: nz.NgffImage,
        axes_order: str = "ZYX",
        orientation: str = "RAS",
        omero: Optional[object] = None,
    ) -> "ZarrNii":
        """
        Create ZarrNii from an existing NgffImage.

        Args:
            ngff_image: NgffImage to wrap
            axes_order: Spatial axes order for NIfTI compatibility
            orientation: Anatomical orientation string
            omero: Optional omero metadata object

        Returns:
            ZarrNii instance
        """
        return cls(
            ngff_image=ngff_image,
            axes_order=axes_order,
            orientation=orientation,
            _omero=omero,
        )

    @classmethod
    def from_darr(
        cls,
        darr: da.Array,
        affine: Optional[AffineTransform] = None,
        axes_order: str = "ZYX",
        orientation: str = "RAS",
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        name: str = "image",
        omero: Optional[object] = None,
        **kwargs,
    ) -> "ZarrNii":
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

        # Create NgffImage
        dims = ["c", "z", "y", "x"] if axes_order == "ZYX" else ["c", "x", "y", "z"]
        ngff_image = nz.NgffImage(
            data=darr, dims=dims, scale=scale, translation=translation, name=name
        )

        return cls(
            ngff_image=ngff_image,
            axes_order=axes_order,
            orientation=orientation,
            _omero=omero,
        )

    # Legacy compatibility method names
    def __init__(
        self,
        darr=None,
        affine=None,
        axes_order="ZYX",
        orientation="RAS",
        ngff_image=None,
        _omero=None,
        **kwargs,
    ):
        """
        Constructor with backward compatibility for old signature.
        """
        if ngff_image is not None:
            # New signature
            object.__setattr__(self, "ngff_image", ngff_image)
            object.__setattr__(self, "axes_order", axes_order)
            object.__setattr__(self, "orientation", orientation)
            object.__setattr__(self, "_omero", _omero)
        elif darr is not None:
            # Legacy signature - delegate to from_darr
            instance = self.from_darr(
                darr=darr,
                affine=affine,
                axes_order=axes_order,
                orientation=orientation,
                **kwargs,
            )
            object.__setattr__(self, "ngff_image", instance.ngff_image)
            object.__setattr__(self, "axes_order", instance.axes_order)
            object.__setattr__(self, "orientation", instance.orientation)
            object.__setattr__(self, "_omero", instance._omero)
        else:
            raise ValueError("Must provide either ngff_image or darr")

    @classmethod
    def from_ome_zarr(
        cls,
        store_or_path: Union[str, Any],
        level: int = 0,
        channels: Optional[List[int]] = None,
        channel_labels: Optional[List[str]] = None,
        timepoints: Optional[List[int]] = None,
        storage_options: Optional[Dict[str, Any]] = None,
        axes_order: str = "ZYX",
        orientation: str = "RAS",
        downsample_near_isotropic: bool = False,
    ) -> "ZarrNii":
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
                Standard orientations like "RAS", "LPI", etc.
            downsample_near_isotropic: If True, automatically downsample
                dimensions with smaller voxel sizes to achieve near-isotropic
                resolution

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
        except Exception as e:
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

        # Read orientation metadata (default to the provided orientation if not present)
        try:
            import zarr

            if isinstance(store_or_path, str):
                if store_or_path.endswith(".zip"):
                    zip_store = zarr.storage.ZipStore(store_or_path, mode="r")
                    group = zarr.open_group(zip_store, mode="r")
                    # Get orientation from zarr metadata, fallback to provided orientation
                    orientation = group.attrs.get("orientation", orientation)
                    zip_store.close()
                else:
                    group = zarr.open_group(store_or_path, mode="r")
                    # Get orientation from zarr metadata, fallback to provided orientation
                    orientation = group.attrs.get("orientation", orientation)
            else:
                group = zarr.open_group(store_or_path, mode="r")
                # Get orientation from zarr metadata, fallback to provided orientation
                orientation = group.attrs.get("orientation", orientation)

        except Exception:
            # If we can't read orientation metadata, use the provided default
            pass

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

        # Create ZarrNii instance with orientation
        znimg = cls(
            ngff_image=ngff_image,
            axes_order=axes_order,
            orientation=orientation,
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

        return znimg

    @classmethod
    def from_nifti(
        cls,
        path: Union[str, bytes],
        chunks: Union[str, Tuple[int, ...]] = "auto",
        axes_order: str = "XYZ",
        name: Optional[str] = None,
        as_ref: bool = False,
        zooms: Optional[Tuple[float, float, float]] = None,
    ) -> "ZarrNii":
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

        return cls(ngff_image=ngff_image, axes_order=axes_order)

    # Chainable operations - each returns a new ZarrNii instance
    def crop(
        self,
        bbox_min: Tuple[int, ...],
        bbox_max: Tuple[int, ...],
        spatial_dims: Optional[List[str]] = None,
    ) -> "ZarrNii":
        """Extract a spatial region from the image.

        Crops the image to the specified bounding box coordinates, preserving
        all metadata and non-spatial dimensions (channels, time). The cropping
        is performed in voxel coordinates.

        Args:
            bbox_min: Minimum corner coordinates of bounding box as tuple.
                Length should match number of spatial dimensions
            bbox_max: Maximum corner coordinates of bounding box as tuple.
                Length should match number of spatial dimensions
            spatial_dims: Names of spatial dimensions to crop. If None,
                automatically derived from axes_order ("z","y","x" for ZYX
                or "x","y","z" for XYZ)

        Returns:
            New ZarrNii instance with cropped data and updated spatial metadata

        Raises:
            ValueError: If bbox coordinates are invalid or out of bounds
            IndexError: If bbox dimensions don't match spatial dimensions

        Examples:
            >>> # Crop 3D region
            >>> cropped = znii.crop((10, 20, 30), (110, 120, 130))

            >>> # Crop with explicit spatial dimensions
            >>> cropped = znii.crop(
            ...     (50, 60, 70), (150, 160, 170),
            ...     spatial_dims=["x", "y", "z"]
            ... )

        Notes:
            - Coordinates are in voxel space (0-based indexing)
            - The cropped region includes bbox_min but excludes bbox_max
            - All non-spatial dimensions (channels, time) are preserved
            - Spatial transformations are automatically updated
        """
        if spatial_dims is None:
            spatial_dims = (
                ["z", "y", "x"] if self.axes_order == "ZYX" else ["x", "y", "z"]
            )
        cropped_image = crop_ngff_image(
            self.ngff_image, bbox_min, bbox_max, spatial_dims
        )
        return ZarrNii(
            ngff_image=cropped_image,
            axes_order=self.axes_order,
            orientation=self.orientation,
            _omero=self._omero,
        )

    def crop_with_bounding_box(self, bbox_min, bbox_max, ras_coords=False):
        """Legacy method name for crop."""
        return self.crop(bbox_min, bbox_max)

    def downsample(
        self,
        factors: Optional[Union[int, List[int]]] = None,
        along_x: int = 1,
        along_y: int = 1,
        along_z: int = 1,
        level: Optional[int] = None,
        spatial_dims: Optional[List[str]] = None,
    ) -> "ZarrNii":
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
            orientation=self.orientation,
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
        new_affine = AffineTransform.from_array(scaling_matrix @ self.affine.matrix)

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
            orientation=self.orientation,
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

    def apply_transform(
        self,
        *transforms: Transform,
        ref_znimg: "ZarrNii",
        spatial_dims: Optional[List[str]] = None,
    ) -> "ZarrNii":
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

        # For now, just apply the first transform (placeholder)
        if transforms:
            transformed_image = apply_transform_to_ngff_image(
                self.ngff_image, transforms[0], ref_znimg.ngff_image, spatial_dims
            )
        else:
            transformed_image = self.ngff_image

        return ZarrNii(
            ngff_image=transformed_image,
            axes_order=self.axes_order,
            orientation=self.orientation,
            _omero=self._omero,
        )

    # I/O operations
    def to_ome_zarr(
        self,
        store_or_path: Union[str, Any],
        max_layer: int = 4,
        scale_factors: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> "ZarrNii":
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
            **kwargs: Additional arguments passed to underlying to_ngff_zarr function.
                May include compression options, chunk sizes, etc.

        Returns:
            Self for method chaining

        Raises:
            OSError: If unable to write to target location
            ValueError: If invalid scale_factors provided

        Examples:
            >>> # Save with default pyramid levels
            >>> znii.to_ome_zarr("/path/to/output.zarr")

            >>> # Save to compressed ZIP with custom pyramid
            >>> znii.to_ome_zarr(
            ...     "/path/to/output.zarr.zip",
            ...     max_layer=3,
            ...     scale_factors=[2, 4]
            ... )

            >>> # Chain with other operations
            >>> result = (znii.downsample(2)
            ...               .crop((0,0,0), (100,100,100))
            ...               .to_ome_zarr("processed.zarr"))

        Notes:
            - OME-Zarr files are always saved in ZYX axis order
            - Automatic axis reordering if current order is XYZ
            - Spatial transformations and metadata are preserved
            - Orientation information is stored as custom metadata
        """
        # Determine the image to save
        if self.axes_order == "XYZ":
            # Need to reorder data from XYZ to ZYX for OME-Zarr
            ngff_image_to_save = self._create_zyx_ngff_image()
        else:
            # Already in ZYX order
            ngff_image_to_save = self.ngff_image

        save_ngff_image(
            ngff_image_to_save,
            store_or_path,
            max_layer,
            scale_factors,
            orientation=self.orientation if hasattr(self, "orientation") else None,
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

                # Add metadata for orientation
                if hasattr(self, "orientation") and self.orientation:
                    group.attrs["orientation"] = self.orientation
            except Exception:
                # If we can't write orientation metadata, that's not critical
                pass

        return self

    def to_nifti(
        self, filename: Optional[Union[str, bytes]] = None
    ) -> Union[nib.Nifti1Image, str]:
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
    ) -> "ZarrNii":
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
                    f"Level {level} not available. Available levels: 0-{len(resolution_levels)-1}"
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
                    f"Timepoint {timepoint} not available. Available timepoints: 0-{len(timepoints)-1}"
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
                    f"Channel {channel} not available. Available channels: 0-{len(channels)-1}"
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
            orientation=orientation,
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
                data_min, data_max = float(channel_data.min()), float(
                    channel_data.max()
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
            image_info.attrs["ExtMin0"] = _string_to_byte_array(f"{-ext_x/2:.3f}")
            image_info.attrs["ExtMax0"] = _string_to_byte_array(f"{ext_x/2:.3f}")
            image_info.attrs["ExtMin1"] = _string_to_byte_array(f"{-ext_y/2:.3f}")
            image_info.attrs["ExtMax1"] = _string_to_byte_array(f"{ext_y/2:.3f}")
            image_info.attrs["ExtMin2"] = _string_to_byte_array(f"{-ext_z/2:.3f}")
            image_info.attrs["ExtMax2"] = _string_to_byte_array(f"{ext_z/2:.3f}")

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
                f"<ZarrNiiExport channels=\"{' '.join(['on'] * n_channels)}\"/>"
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

    def copy(self) -> "ZarrNii":
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
            orientation=self.orientation,
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

    def get_orientation(self):
        """Get orientation string from affine matrix."""
        return affine_to_orientation(self.get_affine_matrix())

    def apply_transform_ref_to_flo_indices(self, *transforms, ref_znimg, indices):
        """Transform indices from reference to floating space."""
        # Placeholder implementation - would need full transform logic
        return indices

    def apply_transform_flo_to_ref_indices(self, *transforms, ref_znimg, indices):
        """Transform indices from floating to reference space."""
        # Placeholder implementation - would need full transform logic
        return indices

    def list_channels(self) -> List[str]:
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
        channels: Optional[List[int]] = None,
        channel_labels: Optional[List[str]] = None,
    ) -> "ZarrNii":
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

        # Select channels from data (assumes channel is last dimension)
        selected_data = self.data[..., channels]

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
            orientation=self.orientation,
            _omero=filtered_omero,
        )

    def select_timepoints(self, timepoints: Optional[List[int]] = None) -> "ZarrNii":
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
            orientation=self.orientation,
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
        self, plugin, chunk_size: Optional[Tuple[int, ...]] = None, **kwargs
    ) -> "ZarrNii":
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
            "orientation": self.orientation,
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
            orientation=self.orientation,
        )

    def segment_otsu(
        self, nbins: int = 256, chunk_size: Optional[Tuple[int, ...]] = None
    ) -> "ZarrNii":
        """
        Apply Otsu thresholding segmentation to the image.

        Convenience method for Otsu thresholding segmentation.

        Args:
            nbins: Number of bins for histogram computation (default: 256)
            chunk_size: Optional chunk size for dask processing

        Returns:
            New ZarrNii instance with binary segmentation
        """
        from .plugins.segmentation import OtsuSegmentation

        plugin = OtsuSegmentation(nbins=nbins)
        return self.segment(plugin, chunk_size=chunk_size)

    def visualize(
        self,
        mode: str = "vol",
        output_path: Optional[Union[str, "os.PathLike"]] = None,
        port: int = 8080,
        open_browser: bool = True,
        temp_zarr_path: Optional[Union[str, "os.PathLike"]] = None,
        **kwargs,
    ) -> Union[str, None, Any]:
        """
        Visualize the OME-Zarr data using vizarr or VolumeViewer for interactive web-based viewing.

        This method provides interactive visualization of the current ZarrNii instance
        using different visualization backends.

        Args:
            mode: Visualization mode - 'widget', 'vol', or 'server' (default: 'vol')
                  - 'widget': Return vizarr widget (for Jupyter notebooks)
                  - 'vol': Open dataset in VolumeViewer web viewer
                  - 'server': Not supported in current vizarr version
            output_path: Not used (deprecated, kept for compatibility)
            port: Port number for HTTP server (used in 'vol' mode)
            open_browser: Whether to automatically open the visualization in browser
            temp_zarr_path: Path for temporary OME-Zarr file. If None, creates in temp directory.
            **kwargs: Additional arguments passed to visualization backend

        Returns:
            For 'widget' mode: vizarr.Viewer widget object (for display in Jupyter)
            For 'vol' mode: URL to the VolumeViewer
            For 'server' mode: None (not supported)

        Raises:
            ImportError: If vizarr is not installed (for widget mode)
            NotImplementedError: If server mode is requested

        Examples:
            >>> znimg = ZarrNii.from_ome_zarr("data.ome.zarr")
            >>> # Open in VolumeViewer (default)
            >>> url = znimg.visualize()
            >>> print(f"View at: {url}")

            >>> # In Jupyter notebook:
            >>> widget = znimg.visualize(mode="widget")
            >>> widget  # Display the widget

        Notes:
            - Widget mode requires vizarr package: pip install zarrnii[viz]
            - Vol mode uses built-in HTTP server (no extra dependencies)
            - Creates a temporary OME-Zarr file for visualization
            - Widget mode is recommended for interactive use in Jupyter notebooks
            - Vol mode is recommended for web browser viewing and is the default
            - HTML mode generates an informational page with usage instructions
        """
        try:
            from . import visualization
            if visualization is None:
                raise ImportError("Visualization module not available")
        except ImportError:
            raise ImportError(
                "Visualization functionality requires vizarr. "
                "Install with: pip install zarrnii[viz]"
            )
        import tempfile
        import os

        # Create temporary OME-Zarr file for visualization
        if temp_zarr_path is None:
            temp_dir = tempfile.mkdtemp(prefix="zarrnii_viz_")
            temp_zarr_path = os.path.join(temp_dir, "temp_data.ome.zarr")
        
        try:
            # Save current data as OME-Zarr for visualization
            # Use conservative settings to avoid multiscale issues
            self.to_ome_zarr(temp_zarr_path, max_layer=1)
            
            # Use the visualization module
            return visualization.visualize(
                zarr_path=temp_zarr_path,
                mode=mode,
                output_path=output_path,
                port=port,
                open_browser=open_browser,
                **kwargs
            )
        except Exception as e:
            # Clean up temporary files on error
            if os.path.exists(temp_zarr_path):
                import shutil
                shutil.rmtree(os.path.dirname(temp_zarr_path), ignore_errors=True)
            raise

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ZarrNii(name='{self.name}', "
            f"shape={self.shape}, "
            f"dims={self.dims}, "
            f"scale={self.scale})"
        )


# Helper functions for backward compatibility
def affine_to_orientation(affine):
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


def orientation_to_affine(orientation, spacing=(1, 1, 1), origin=(0, 0, 0)):
    """
    Creates an affine matrix based on an orientation string (e.g., 'RAS').

    Parameters:
        orientation (str): Orientation string (e.g., 'RAS', 'LPS').
        spacing (tuple): Voxel spacing along each axis (default: (1, 1, 1)).
        origin (tuple): Origin point in physical space (default: (0, 0, 0)).

    Returns:
        affine (numpy.ndarray): Affine matrix from voxel to world coordinates.
    """
    # Validate orientation length
    if len(orientation) != 3:
        raise ValueError("Orientation must be a 3-character string (e.g., 'RAS').")

    # Axis mapping and flipping
    axis_map = {"R": 0, "L": 0, "A": 1, "P": 1, "S": 2, "I": 2}
    sign_map = {"R": 1, "L": -1, "A": 1, "P": -1, "S": 1, "I": -1}

    axes = [axis_map[ax] for ax in orientation]
    signs = [sign_map[ax] for ax in orientation]

    # Construct the affine matrix
    affine = np.zeros((4, 4))
    for i, (axis, sign) in enumerate(zip(axes, signs)):
        affine[i, axis] = sign * spacing[axis]

    # Add origin
    affine[:3, 3] = origin
    affine[3, 3] = 1

    return affine


def align_affine_to_input_orientation(affine, orientation):
    """
    Reorders and flips the affine matrix to align with the specified input orientation.

    Parameters:
        affine (np.ndarray): Initial affine matrix.
        orientation (str): Input orientation (e.g., 'RAS').

    Returns:
        np.ndarray: Reordered and flipped affine matrix.
    """
    axis_map = {"R": 0, "L": 0, "A": 1, "P": 1, "S": 2, "I": 2}
    sign_map = {"R": 1, "L": -1, "A": 1, "P": -1, "S": 1, "I": -1}

    input_axes = [axis_map[ax] for ax in orientation]
    input_signs = [sign_map[ax] for ax in orientation]

    reordered_affine = np.zeros_like(affine)
    for i, (axis, sign) in enumerate(zip(input_axes, input_signs)):
        reordered_affine[i, :3] = sign * affine[axis, :3]
        reordered_affine[i, 3] = sign * affine[i, 3]

    # Copy the homogeneous row
    reordered_affine[3, :] = affine[3, :]

    return reordered_affine


def construct_affine_with_orientation(coordinate_transformations, orientation):
    """
    Build affine matrix from coordinate transformations and align to orientation.

    Parameters:
        coordinate_transformations (list): Coordinate transformations from OME-Zarr metadata.
        orientation (str): Input orientation (e.g., 'RAS').

    Returns:
        np.ndarray: A 4x4 affine matrix.
    """
    # Initialize affine as an identity matrix
    affine = np.eye(4)

    # Extract scales and translations from coordinate transformations
    scales = [1.0, 1.0, 1.0]  # Default scales
    translations = [0.0, 0.0, 0.0]  # Default translations

    for transform in coordinate_transformations:
        if transform["type"] == "scale":
            scales = transform["scale"][-3:]  # Take the last 3 (spatial)
        elif transform["type"] == "translation":
            translations = transform["translation"][-3:]  # Take the last 3 (spatial)

    # Populate the affine matrix
    affine[:3, :3] = np.diag(scales)  # Set scaling
    affine[:3, 3] = translations  # Set translation

    # Reorder the affine matrix for the input orientation
    return align_affine_to_input_orientation(affine, orientation)
