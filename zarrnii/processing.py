"""Processing utilities for OME-Zarr data transformations.

This module contains functions for:
- Cropping NgffImage objects using bounding boxes
- Downsampling and upsampling with proper metadata updates
- Spatial transformations and interpolation
- Filtering and normalization operations
- Near-isotropic downsampling for anisotropic datasets

The functions in this module operate on ngff_zarr.NgffImage objects
and maintain proper scale and translation metadata throughout processing.
"""

from __future__ import annotations

from typing import List, Union

import ngff_zarr as nz
import numpy as np

from .transform import Transform


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


def apply_near_isotropic_downsampling(znimg: "ZarrNii", axes_order: str) -> "ZarrNii":
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
