"""
Chainable wrapper for NgffImage operations.

This module provides a chainable interface similar to the original ZarrNii
but working with NgffImage objects internally. This combines the benefits
of direct NgffImage usage with the ergonomic chainable API.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import dask.array as da
import ngff_zarr as nz

from .transform import AffineTransform, Transform
from .ngff_core import (
    crop_ngff_image, downsample_ngff_image, apply_transform_to_ngff_image,
    get_affine_matrix, get_affine_transform, load_ngff_image, save_ngff_image,
    get_multiscales, _select_channels_from_image
)


class ChainableNgffImage:
    """
    Chainable wrapper around NgffImage that provides ergonomic method chaining.
    
    This class wraps an NgffImage and provides methods that return new 
    ChainableNgffImage instances, enabling fluent method chaining similar
    to the original ZarrNii API.
    
    Attributes:
        ngff_image (nz.NgffImage): The wrapped NgffImage object
    """
    
    def __init__(self, ngff_image: nz.NgffImage):
        """
        Initialize ChainableNgffImage with an NgffImage.
        
        Args:
            ngff_image: The NgffImage to wrap
        """
        self.ngff_image = ngff_image
    
    @classmethod
    def from_ome_zarr(
        cls,
        store_or_path,
        level: int = 0,
        channels: Optional[List[int]] = None,
        channel_labels: Optional[List[str]] = None,
        storage_options: Optional[Dict] = None,
    ) -> "ChainableNgffImage":
        """
        Load from OME-Zarr store and wrap in chainable interface.
        
        Args:
            store_or_path: Store or path to OME-Zarr file
            level: Pyramid level to load
            channels: Channel indices to load
            channel_labels: Channel labels to load
            storage_options: Storage options for Zarr
            
        Returns:
            ChainableNgffImage wrapping the loaded NgffImage
        """
        # Load the multiscales object first
        multiscales = get_multiscales(store_or_path, storage_options)
        
        # Get the specified level
        ngff_image = multiscales.images[level]
        
        # Handle channel selection if specified
        if channels is not None or channel_labels is not None:
            ngff_image = _select_channels_from_image(
                ngff_image, multiscales, channels, channel_labels
            )
        
        return cls(ngff_image)
    
    @classmethod 
    def from_ngff_image(cls, ngff_image: nz.NgffImage) -> "ChainableNgffImage":
        """
        Wrap an existing NgffImage in the chainable interface.
        
        Args:
            ngff_image: NgffImage to wrap
            
        Returns:
            ChainableNgffImage wrapping the input
        """
        return cls(ngff_image)
    
    # Properties for easy access to underlying NgffImage attributes
    @property
    def data(self) -> da.Array:
        """Access the image data."""
        return self.ngff_image.data
    
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
        """Scale information."""
        return self.ngff_image.scale
    
    @property
    def translation(self) -> Dict[str, float]:
        """Translation information."""
        return self.ngff_image.translation
    
    @property
    def name(self) -> str:
        """Image name."""
        return self.ngff_image.name
    
    def get_affine_matrix(self, axes_order: str = "ZYX") -> np.ndarray:
        """Get 4x4 affine transformation matrix."""
        return get_affine_matrix(self.ngff_image, axes_order)
    
    def get_affine_transform(self, axes_order: str = "ZYX") -> AffineTransform:
        """Get AffineTransform object."""
        return get_affine_transform(self.ngff_image, axes_order)
    
    # Chainable operations - each returns a new ChainableNgffImage
    def crop(
        self, 
        bbox_min: tuple, 
        bbox_max: tuple,
        spatial_dims: List[str] = ["z", "y", "x"]
    ) -> "ChainableNgffImage":
        """
        Crop the image and return a new chainable instance.
        
        Args:
            bbox_min: Minimum corner of bounding box
            bbox_max: Maximum corner of bounding box
            spatial_dims: Names of spatial dimensions
            
        Returns:
            New ChainableNgffImage with cropped data
        """
        cropped_image = crop_ngff_image(self.ngff_image, bbox_min, bbox_max, spatial_dims)
        return ChainableNgffImage(cropped_image)
    
    def downsample(
        self, 
        factors: Union[int, List[int]],
        spatial_dims: List[str] = ["z", "y", "x"]
    ) -> "ChainableNgffImage":
        """
        Downsample the image and return a new chainable instance.
        
        Args:
            factors: Downsampling factors (int for isotropic, list for per-dimension)
            spatial_dims: Names of spatial dimensions  
            
        Returns:
            New ChainableNgffImage with downsampled data
        """
        downsampled_image = downsample_ngff_image(self.ngff_image, factors, spatial_dims)
        return ChainableNgffImage(downsampled_image)
    
    def apply_transform(
        self,
        transform: Transform,
        reference_image: Union["ChainableNgffImage", nz.NgffImage],
        spatial_dims: List[str] = ["z", "y", "x"]
    ) -> "ChainableNgffImage":
        """
        Apply spatial transformation and return a new chainable instance.
        
        Args:
            transform: Transformation to apply
            reference_image: Reference image defining output space
            spatial_dims: Names of spatial dimensions
            
        Returns:
            New ChainableNgffImage with transformed data
        """
        # Handle both ChainableNgffImage and NgffImage reference
        if isinstance(reference_image, ChainableNgffImage):
            ref_ngff = reference_image.ngff_image
        else:
            ref_ngff = reference_image
            
        transformed_image = apply_transform_to_ngff_image(
            self.ngff_image, transform, ref_ngff, spatial_dims
        )
        return ChainableNgffImage(transformed_image)
    
    def resample(
        self,
        target_scale: Dict[str, float],
        target_shape: Optional[tuple] = None,
        spatial_dims: List[str] = ["z", "y", "x"],
        method: str = "linear"
    ) -> "ChainableNgffImage":
        """
        Resample to target resolution and return a new chainable instance.
        
        Args:
            target_scale: Target scale dictionary
            target_shape: Target shape (computed from scale if None)
            spatial_dims: Names of spatial dimensions
            method: Resampling method
            
        Returns:
            New ChainableNgffImage with resampled data
        """
        from .ngff_transforms import resample_ngff_image
        resampled_image = resample_ngff_image(
            self.ngff_image, target_scale, target_shape, spatial_dims, method
        )
        return ChainableNgffImage(resampled_image)
    
    # I/O operations
    def save(
        self,
        store_or_path,
        max_layer: int = 4,
        scale_factors: Optional[List[int]] = None,
        **kwargs
    ) -> "ChainableNgffImage":
        """
        Save to OME-Zarr store and return self for continued chaining.
        
        Args:
            store_or_path: Target store or path
            max_layer: Maximum number of pyramid levels
            scale_factors: Custom scale factors for pyramid levels
            **kwargs: Additional arguments for to_ngff_zarr
            
        Returns:
            Self for continued chaining
        """
        save_ngff_image(self.ngff_image, store_or_path, max_layer, scale_factors, **kwargs)
        return self
    
    def copy(self) -> "ChainableNgffImage":
        """
        Create a copy of this chainable image.
        
        Returns:
            New ChainableNgffImage with copied data
        """
        # Create a new NgffImage with the same properties
        copied_image = nz.NgffImage(
            data=self.ngff_image.data,  # Dask arrays are lazy so this is efficient
            dims=self.ngff_image.dims.copy(),
            scale=self.ngff_image.scale.copy(),
            translation=self.ngff_image.translation.copy(),
            name=self.ngff_image.name,
        )
        return ChainableNgffImage(copied_image)
    
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
    
    def to_zarrnii(self, axes_order: str = "ZYX"):
        """
        Convert to legacy ZarrNii for compatibility.
        
        Args:
            axes_order: Spatial axes order for ZarrNii
            
        Returns:
            ZarrNii instance
        """
        from .core import ZarrNii
        return ZarrNii.from_ngff_image(self.ngff_image, axes_order)
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"ChainableNgffImage(name='{self.name}', "
                f"shape={self.shape}, "
                f"dims={self.dims}, "
                f"scale={self.scale})")


# Convenience function for creating chainable instances
def chainable_from_zarrnii(zarrnii) -> ChainableNgffImage:
    """
    Create a ChainableNgffImage from a legacy ZarrNii instance.
    
    Args:
        zarrnii: ZarrNii instance to convert
        
    Returns:
        ChainableNgffImage wrapping the converted NgffImage
    """
    ngff_image = zarrnii.to_ngff_image()
    return ChainableNgffImage(ngff_image)