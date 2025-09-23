"""Utility functions for spatial transformations and metadata handling.

This module contains helper functions for:
- Affine matrix construction and manipulation
- Anatomical orientation handling and conversion
- Coordinate system alignment and reorientation
- Metadata extraction from NgffImage objects

These utilities support the core functionality of ZarrNii by providing
low-level operations for spatial transformations and coordinate systems.
"""

from __future__ import annotations

import ngff_zarr as nz
import numpy as np

from .transform import AffineTransform


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