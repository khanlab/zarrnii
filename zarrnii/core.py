"""
Unified ZarrNii implementation using NgffImage internally.

This provides a single API that maintains chainable functionality while using
NgffImage objects under the hood directly without duplicate metadata attributes.
All core functions are implemented directly in this module.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

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
    store_or_path,
    level: int = 0,
    channels: Optional[List[int]] = None,
    channel_labels: Optional[List[str]] = None,
    timepoints: Optional[List[int]] = None,
    storage_options: Optional[Dict] = None,
) -> nz.NgffImage:
    """
    Load an NgffImage from an OME-Zarr store.

    Args:
        store_or_path: Store or path to the OME-Zarr file
        level: Pyramid level to load (default: 0)
        channels: Channels to load by index (default: None, loads all channels)
        channel_labels: Channels to load by label name (default: None)
        timepoints: Timepoints to load by index (default: None, loads all timepoints)
        storage_options: Storage options for Zarr

    Returns:
        NgffImage: The loaded image at the specified level
    """
    # Load the multiscales object
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
    store_or_path,
    max_layer: int = 4,
    scale_factors: Optional[List[int]] = None,
    **kwargs,
):
    """
    Save an NgffImage to an OME-Zarr store with multiscale pyramid.

    Args:
        ngff_image: NgffImage to save
        store_or_path: Target store or path
        max_layer: Maximum number of pyramid levels
        scale_factors: Custom scale factors for pyramid levels
        **kwargs: Additional arguments for to_ngff_zarr
    """
    if scale_factors is None:
        scale_factors = [2**i for i in range(1, max_layer)]

    # Create multiscales from the image
    multiscales = nz.to_multiscales(ngff_image, scale_factors=scale_factors)

    # Write to zarr store
    nz.to_ngff_zarr(store_or_path, multiscales, **kwargs)


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
        store_or_path,
        level: int = 0,
        channels: Optional[List[int]] = None,
        channel_labels: Optional[List[str]] = None,
        timepoints: Optional[List[int]] = None,
        storage_options: Optional[Dict] = None,
        axes_order: str = "ZYX",
        orientation: str = "RAS",
        downsample_near_isotropic: bool = False,
    ) -> "ZarrNii":
        """
        Load from OME-Zarr store.

        Args:
            store_or_path: Store or path to OME-Zarr file
            level: Pyramid level to load (if beyond available levels, lazy downsampling is applied)
            channels: Channel indices to load
            channel_labels: Channel labels to load
            timepoints: Timepoint indices to load
            storage_options: Storage options for Zarr
            axes_order: Spatial axes order for NIfTI compatibility
            orientation: Default input orientation if none is specified in metadata (default: 'RAS')
            downsample_near_isotropic: If True, downsample dimensions with smaller pixel sizes
                                      to make the resulting image nearly isotropic (default: False)

        Returns:
            ZarrNii instance
        """
        # Validate channel and timepoint selection arguments
        if channels is not None and channel_labels is not None:
            raise ValueError("Cannot specify both 'channels' and 'channel_labels'")

        # Load the multiscales object
        try:
            if isinstance(store_or_path, str):
                multiscales = nz.from_ngff_zarr(
                    store_or_path, storage_options=storage_options or {}
                )
            else:
                multiscales = nz.from_ngff_zarr(store_or_path)
        except Exception as e:
            # Fallback for older zarr/ngff_zarr versions
            if isinstance(store_or_path, str):
                store = fsspec.get_mapper(store_or_path, **storage_options or {})
            else:
                store = store_or_path
            multiscales = nz.from_ngff_zarr(store)

        # Extract omero metadata if available
        omero_metadata = None
        try:
            import zarr

            if isinstance(store_or_path, str):
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
                group = zarr.open_group(store_or_path, mode="r")
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
        cls, path, chunks="auto", axes_order="XYZ", name=None, as_ref=False, zooms=None
    ):
        """
        Load from NIfTI file.

        Args:
            path: Path to NIfTI file
            chunks: Chunking strategy for dask array
            axes_order: Spatial axes order
            name: Name for the NgffImage
            as_ref: If True, creates an empty dask array with the correct shape instead of loading data
            zooms: Target voxel spacing in xyz (only valid if as_ref=True)

        Returns:
            ZarrNii instance
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
        self, bbox_min: tuple, bbox_max: tuple, spatial_dims: List[str] = None
    ) -> "ZarrNii":
        """
        Crop the image and return a new ZarrNii instance.

        Args:
            bbox_min: Minimum corner of bounding box
            bbox_max: Maximum corner of bounding box
            spatial_dims: Names of spatial dimensions (derived from axes_order if None)

        Returns:
            New ZarrNii with cropped data
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
        factors: Union[int, List[int]] = None,
        along_x: int = 1,
        along_y: int = 1,
        along_z: int = 1,
        level: int = None,
        spatial_dims: List[str] = None,
    ) -> "ZarrNii":
        """
        Downsample the image and return a new ZarrNii instance.

        Args:
            factors: Downsampling factors (int for isotropic, list for per-dimension)
            along_x: Legacy parameter for X downsampling
            along_y: Legacy parameter for Y downsampling
            along_z: Legacy parameter for Z downsampling
            level: Legacy parameter for level-based downsampling (2^level)
            spatial_dims: Names of spatial dimensions (derived from axes_order if None)

        Returns:
            New ZarrNii with downsampled data
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
        spatial_dims: List[str] = None,
    ) -> "ZarrNii":
        """
        Apply spatial transformation and return a new ZarrNii instance.

        Args:
            transforms: Transformations to apply
            ref_znimg: Reference ZarrNii defining output space
            spatial_dims: Names of spatial dimensions (derived from axes_order if None)

        Returns:
            New ZarrNii with transformed data
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
        store_or_path,
        max_layer: int = 4,
        scale_factors: Optional[List[int]] = None,
        **kwargs,
    ) -> "ZarrNii":
        """
        Save to OME-Zarr store and return self for continued chaining.

        OME-Zarr files are always written in ZYX order. If the current axes_order is XYZ,
        the data will be reordered to ZYX before writing.

        Args:
            store_or_path: Target store or path
            max_layer: Maximum number of pyramid levels
            scale_factors: Custom scale factors for pyramid levels
            **kwargs: Additional arguments for to_ngff_zarr

        Returns:
            Self for continued chaining
        """
        # Determine the image to save
        if self.axes_order == "XYZ":
            # Need to reorder data from XYZ to ZYX for OME-Zarr
            ngff_image_to_save = self._create_zyx_ngff_image()
        else:
            # Already in ZYX order
            ngff_image_to_save = self.ngff_image

        save_ngff_image(
            ngff_image_to_save, store_or_path, max_layer, scale_factors, **kwargs
        )

        # Add orientation metadata to the zarr store
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

    def to_nifti(self, filename=None):
        """
        Convert to NIfTI format.

        NIfTI files are always written in XYZ order. If the current axes_order is ZYX,
        the data will be reordered to XYZ and the affine matrix adjusted accordingly.

        
        For 5D data (T,C,Z,Y,X), singleton dimensions are removed automatically.
        Non-singleton time and channel dimensions will raise an error as NIfTI doesn't 
        support more than 4D data.

        Args:
            filename: Output filename, if None return nibabel image

        Returns:
            nibabel.Nifti1Image or path if filename provided
        """
        # Get data and dimensions
        data = self.data.compute()

        dims = self.dims
        
        # Handle dimensional reduction for NIfTI compatibility
        # NIfTI supports up to 4D, so we need to remove singleton dimensions
        squeeze_axes = []
        remaining_dims = []
        
        for i, dim in enumerate(dims):
            if dim in ['t', 'c'] and data.shape[i] == 1:
                # Remove singleton time or channel dimensions
                squeeze_axes.append(i)
            elif dim in ['t', 'c'] and data.shape[i] > 1:
                # Non-singleton time or channel dimensions - NIfTI can't handle this
                raise ValueError(f"NIfTI format doesn't support non-singleton {dim} dimension. "
                               f"Dimension '{dim}' has size {data.shape[i]}. "
                               f"Consider selecting specific timepoints/channels first.")
            else:
                remaining_dims.append(dim)
        
        # Squeeze out singleton dimensions
        if squeeze_axes:
            data = np.squeeze(data, axis=tuple(squeeze_axes))
        
        # Check final dimensionality
        if data.ndim > 4:
            raise ValueError(f"Resulting data has {data.ndim} dimensions, but NIfTI supports maximum 4D")
        
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
        """
        List available channel labels from omero metadata.

        Returns:
            List of channel labels, or empty list if no omero metadata
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
        """
        Select channels from the image data and return a new ZarrNii instance.

        Args:
            channels: Channel indices to select
            channel_labels: Channel labels to select

        Returns:
            New ZarrNii instance with selected channels
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
