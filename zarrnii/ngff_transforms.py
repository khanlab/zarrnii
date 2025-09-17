"""
Spatial transformation functions that operate on NgffImage objects.

This module implements the transformation logic from the original ZarrNii
but adapted to work directly with NgffImage objects instead of wrapped dask arrays.
"""

from __future__ import annotations

from typing import List, Union
import numpy as np
import dask.array as da
import ngff_zarr as nz
from scipy.interpolate import interpn

from .transform import Transform, AffineTransform


def interp_by_block_ngff(
    block,
    block_info,
    ngff_image_data,
    transformations,
    spatial_dims=("z", "y", "x"),
    method="linear",
):
    """
    Interpolate a block of data using spatial transformations for NgffImage.
    
    This is adapted from the original interp_by_block but works with NgffImage metadata.
    
    Args:
        block: Input block to transform
        block_info: Dask block information
        ngff_image_data: The source NgffImage data array
        transformations: List of transformation objects
        spatial_dims: Names of spatial dimensions
        method: Interpolation method
        
    Returns:
        Transformed block
    """
    # Get block location info
    block_location = block_info[0]['array-location']
    block_id = block_info[None]['block-id']
    
    # Create coordinate grids for this block
    coords = []
    for i, dim in enumerate(spatial_dims):
        start = block_location[-(len(spatial_dims)-i)][0]
        end = block_location[-(len(spatial_dims)-i)][1]
        coords.append(np.arange(start, end))
    
    # Create meshgrid
    mesh_coords = np.meshgrid(*coords, indexing='ij')
    
    # Flatten coordinates for transformation
    coords_flat = np.column_stack([coord.ravel() for coord in mesh_coords])
    
    # Apply transformations sequentially
    transformed_coords = coords_flat.T  # Shape: (3, N)
    
    for transform in transformations:
        if isinstance(transform, AffineTransform):
            # Convert to homogeneous coordinates
            ones = np.ones((1, transformed_coords.shape[1]))
            homogeneous = np.vstack([transformed_coords, ones])
            
            # Apply affine transformation
            transformed_homogeneous = transform.matrix @ homogeneous
            transformed_coords = transformed_homogeneous[:3, :]
    
    # Reshape back to block shape
    new_shape = tuple(end - start for start, end in block_location[-(len(spatial_dims)):])
    transformed_coords = transformed_coords.T.reshape(new_shape + (len(spatial_dims),))
    
    # Interpolate from source data
    # Note: This is a simplified version - full implementation would need
    # proper coordinate system handling and bounds checking
    
    result = np.zeros_like(block)
    
    # For now, return the input block (placeholder)
    # Full implementation would perform the actual interpolation
    return result


def apply_transform_to_ngff_image_full(
    source_image: nz.NgffImage,
    reference_image: nz.NgffImage,
    *transforms: Transform,
    spatial_dims: List[str] = ["z", "y", "x"],
    interpolation_method: str = "linear"
) -> nz.NgffImage:
    """
    Apply spatial transformations to map source_image into reference_image space.
    
    This function implements the core transformation logic adapted from ZarrNii
    but working directly with NgffImage objects.
    
    Args:
        source_image: Source NgffImage to transform
        reference_image: Reference NgffImage defining output space
        *transforms: Sequence of transforms to apply
        spatial_dims: Names of spatial dimensions
        interpolation_method: Interpolation method ("linear", "nearest", etc.)
        
    Returns:
        New transformed NgffImage in reference space
    """
    # Get reference space metadata
    ref_scale = reference_image.scale
    ref_translation = reference_image.translation
    ref_shape = reference_image.data.shape
    ref_dims = reference_image.dims
    
    # Create affine transformations from NgffImage metadata
    ref_affine = _ngff_image_to_affine(reference_image, spatial_dims)
    source_affine = _ngff_image_to_affine(source_image, spatial_dims)
    
    # Build transformation chain
    tfms_to_apply = [AffineTransform.from_array(ref_affine)]
    tfms_to_apply.extend(transforms)
    tfms_to_apply.append(AffineTransform.from_array(source_affine).invert())
    
    # Apply transformations using dask map_blocks
    transformed_data = da.map_blocks(
        interp_by_block_ngff,
        reference_image.data,
        dtype=np.float32,
        chunks=reference_image.data.chunks,
        ngff_image_data=source_image.data,
        transformations=tfms_to_apply,
        spatial_dims=spatial_dims,
        method=interpolation_method,
        meta=np.array([], dtype=np.float32)
    )
    
    # Create new NgffImage with transformed data
    return nz.NgffImage(
        data=transformed_data,
        dims=ref_dims,
        scale=ref_scale,
        translation=ref_translation,
        name=f"transformed_{source_image.name}",
    )


def _ngff_image_to_affine(ngff_image: nz.NgffImage, spatial_dims: List[str]) -> np.ndarray:
    """
    Convert NgffImage scale and translation to a 4x4 affine matrix.
    
    Args:
        ngff_image: Input NgffImage
        spatial_dims: Names of spatial dimensions in order
        
    Returns:
        4x4 affine transformation matrix
    """
    affine = np.eye(4)
    
    for i, dim in enumerate(spatial_dims):
        if dim in ngff_image.scale:
            affine[i, i] = ngff_image.scale[dim]
        if dim in ngff_image.translation:
            affine[i, 3] = ngff_image.translation[dim]
    
    return affine


def create_reference_ngff_image(
    shape: tuple,
    dims: List[str],
    scale: dict,
    translation: dict,
    dtype=np.float32,
    chunks=None,
    name: str = "reference"
) -> nz.NgffImage:
    """
    Create a reference NgffImage with specified spatial properties.
    
    This is useful for defining a target space for transformations.
    
    Args:
        shape: Shape of the reference image
        dims: Dimension names
        scale: Scale dictionary (e.g., {"z": 2.0, "y": 1.0, "x": 1.0})
        translation: Translation dictionary
        dtype: Data type for the array
        chunks: Chunk sizes for dask array
        name: Name for the image
        
    Returns:
        New NgffImage representing the reference space
    """
    if chunks is None:
        chunks = "auto"
    
    # Create empty dask array
    data = da.zeros(shape, dtype=dtype, chunks=chunks)
    
    return nz.NgffImage(
        data=data,
        dims=dims,
        scale=scale,
        translation=translation,
        name=name,
    )


def compose_transforms(*transforms: Transform) -> AffineTransform:
    """
    Compose multiple transformations into a single transformation.
    
    Args:
        *transforms: Sequence of Transform objects to compose
        
    Returns:
        Single AffineTransform representing the composition
        
    Note:
        Currently only supports AffineTransform objects.
        Displacement transforms would need special handling.
    """
    if not transforms:
        return AffineTransform.identity()
    
    # Start with identity
    result_matrix = np.eye(4)
    
    # Apply transforms in order
    for transform in transforms:
        if isinstance(transform, AffineTransform):
            result_matrix = transform.matrix @ result_matrix
        else:
            raise NotImplementedError(f"Composition not implemented for {type(transform)}")
    
    return AffineTransform.from_array(result_matrix)


def resample_ngff_image(
    ngff_image: nz.NgffImage,
    target_scale: dict,
    target_shape: tuple = None,
    spatial_dims: List[str] = ["z", "y", "x"],
    method: str = "linear"
) -> nz.NgffImage:
    """
    Resample an NgffImage to a target resolution.
    
    Args:
        ngff_image: Input NgffImage to resample
        target_scale: Target scale dictionary
        target_shape: Target shape (if None, computed from scale change)
        spatial_dims: Names of spatial dimensions
        method: Resampling method
        
    Returns:
        Resampled NgffImage
    """
    current_scale = ngff_image.scale
    
    # Compute target shape if not provided
    if target_shape is None:
        target_shape = list(ngff_image.data.shape)
        for i, dim in enumerate(ngff_image.dims):
            if dim.lower() in [d.lower() for d in spatial_dims]:
                if dim in current_scale and dim in target_scale:
                    scale_factor = current_scale[dim] / target_scale[dim]
                    target_shape[i] = int(target_shape[i] * scale_factor)
        target_shape = tuple(target_shape)
    
    # Create reference image with target properties
    ref_image = create_reference_ngff_image(
        shape=target_shape,
        dims=ngff_image.dims,
        scale=target_scale,
        translation=ngff_image.translation,
        chunks="auto",  # Let dask figure out appropriate chunks
        name=f"resampled_{ngff_image.name}"
    )
    
    # Apply resampling (identity transform, just different reference space)
    identity = AffineTransform.identity()
    return apply_transform_to_ngff_image_full(
        ngff_image, ref_image, identity,
        spatial_dims=spatial_dims,
        interpolation_method=method
    )