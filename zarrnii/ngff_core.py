"""
New NgffImage-based core classes for ZarrNii.

This module contains the new architecture that works directly with 
ngff_zarr.NgffImage and ngff_zarr.Multiscales objects instead of 
wrapping single-scale dask arrays.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union
import numpy as np
import dask.array as da
import ngff_zarr as nz
from attrs import define, field

from .transform import AffineTransform, Transform


@define
class NgffZarrNii:
    """
    ZarrNii implementation that works directly with NgffImage objects.
    
    This class represents a more direct wrapper around ngff_zarr objects,
    providing spatial transformation and processing capabilities while
    maintaining the multiscale structure.
    
    Attributes:
        ngff_image (nz.NgffImage): The primary NgffImage object
        multiscales (nz.Multiscales, optional): The multiscales container if available
    """
    
    ngff_image: nz.NgffImage
    multiscales: Optional[nz.Multiscales] = None
    
    @classmethod
    def from_ome_zarr(
        cls,
        store_or_path,
        level: int = 0,
        channels: Optional[List[int]] = None,
        channel_labels: Optional[List[str]] = None,
        storage_options: Optional[Dict] = None,
    ):
        """
        Load an NgffZarrNii from an OME-Zarr store.
        
        Args:
            store_or_path: Store or path to the OME-Zarr file
            level: Pyramid level to primary load (default: 0)
            channels: Channels to load by index (default: None, loads all channels)
            channel_labels: Channels to load by label name (default: None)
            storage_options: Storage options for Zarr
            
        Returns:
            NgffZarrNii: A new instance wrapping the NgffImage
        """
        # Load the multiscales object
        multiscales = nz.from_ngff_zarr(store_or_path, storage_options=storage_options)
        
        # For now, we'll use the specified level as the primary image
        # In the future, we might want to maintain the full multiscale structure
        primary_image = multiscales.images[level]
        
        # Handle channel selection if specified
        if channels is not None or channel_labels is not None:
            primary_image = cls._select_channels_from_image(
                primary_image, multiscales, channels, channel_labels
            )
        
        return cls(ngff_image=primary_image, multiscales=multiscales)
    
    @classmethod
    def _select_channels_from_image(
        cls,
        image: nz.NgffImage,
        multiscales: nz.Multiscales,
        channels: Optional[List[int]] = None,
        channel_labels: Optional[List[str]] = None,
    ) -> nz.NgffImage:
        """
        Create a new NgffImage with selected channels.
        
        This is a helper method to handle channel selection.
        """
        # Get axis names
        axis_names = [axis.name for axis in multiscales.metadata.axes]
        
        # Handle channel label resolution
        if channel_labels is not None:
            if multiscales.metadata.omero is None or not hasattr(multiscales.metadata.omero, 'channels'):
                raise ValueError("Channel labels specified but no omero metadata found")
            
            # Extract available labels
            omero_channels = multiscales.metadata.omero.channels
            available_labels = []
            for ch in omero_channels:
                if hasattr(ch, 'label'):
                    available_labels.append(ch.label)
                elif isinstance(ch, dict):
                    available_labels.append(ch.get('label', ''))
                else:
                    available_labels.append(str(getattr(ch, 'label', '')))
            
            # Resolve labels to indices
            resolved_channels = []
            for label in channel_labels:
                try:
                    idx = available_labels.index(label)
                    resolved_channels.append(idx)
                except ValueError:
                    raise ValueError(f"Channel label '{label}' not found. Available: {available_labels}")
            
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
        # We need to preserve the coordinate transformations and other metadata
        new_image = nz.NgffImage(
            data=new_data,
            dims=image.dims,
            scale=image.scale,
            translation=image.translation,
            name=image.name,
        )
        
        return new_image
    
    @property
    def data(self) -> da.Array:
        """Access the primary image data as a dask array."""
        return self.ngff_image.data
    
    @property
    def shape(self) -> tuple:
        """Shape of the primary image data."""
        return self.ngff_image.data.shape
    
    @property
    def dims(self) -> List[str]:
        """Dimension names of the primary image."""
        return self.ngff_image.dims
    
    @property
    def scale(self) -> Dict[str, float]:
        """Scale information from the primary image."""
        return self.ngff_image.scale
    
    @property
    def translation(self) -> Dict[str, float]:
        """Translation information from the primary image.""" 
        return self.ngff_image.translation
    
    def get_affine_matrix(self, axes_order: str = "ZYX") -> np.ndarray:
        """
        Construct an affine transformation matrix from the NgffImage metadata.
        
        Args:
            axes_order: Order of spatial axes (default: "ZYX")
            
        Returns:
            4x4 affine transformation matrix
        """
        # Extract scale and translation for spatial dimensions
        spatial_dims = list(axes_order.lower())
        
        # Build 4x4 affine matrix
        affine = np.eye(4)
        
        for i, dim in enumerate(spatial_dims):
            if dim in self.scale:
                affine[i, i] = self.scale[dim]
            if dim in self.translation:
                affine[i, 3] = self.translation[dim]
        
        return affine
    
    def get_affine_transform(self, axes_order: str = "ZYX") -> AffineTransform:
        """
        Get an AffineTransform object from the NgffImage metadata.
        
        Args:
            axes_order: Order of spatial axes (default: "ZYX")
            
        Returns:
            AffineTransform object
        """
        matrix = self.get_affine_matrix(axes_order)
        return AffineTransform.from_array(matrix)
    
    def to_ome_zarr(
        self, 
        store_or_path,
        max_layer: int = 4,
        scale_factors: Optional[List[int]] = None,
        **kwargs
    ):
        """
        Write the NgffZarrNii to an OME-Zarr store.
        
        Args:
            store_or_path: Target store or path
            max_layer: Maximum number of pyramid levels
            scale_factors: Custom scale factors for pyramid levels
            **kwargs: Additional arguments for to_ngff_zarr
        """
        if scale_factors is None:
            scale_factors = [2**i for i in range(1, max_layer)]
        
        # Create multiscales from the primary image
        multiscales = nz.to_multiscales(
            self.ngff_image,
            scale_factors=scale_factors
        )
        
        # Preserve original omero metadata if available
        if self.multiscales and self.multiscales.metadata.omero:
            multiscales.metadata.omero = self.multiscales.metadata.omero
        
        # Write to zarr store
        nz.to_ngff_zarr(store_or_path, multiscales, **kwargs)


# Function-based API for operating on NgffImage objects
def crop_ngff_image(
    ngff_image: nz.NgffImage,
    bbox_min: tuple,
    bbox_max: tuple,
    spatial_dims: List[str] = ["z", "y", "x"]
) -> nz.NgffImage:
    """
    Crop an NgffImage using a bounding box.
    
    Args:
        ngff_image: Input NgffImage to crop
        bbox_min: Minimum corner of bounding box
        bbox_max: Maximum corner of bounding box  
        spatial_dims: Names of spatial dimensions
        
    Returns:
        New cropped NgffImage
    """
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
    spatial_dims: List[str] = ["z", "y", "x"]
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
    spatial_dims: List[str] = ["z", "y", "x"]
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
    # Import the full implementation to avoid circular imports
    from .ngff_transforms import apply_transform_to_ngff_image_full
    
    return apply_transform_to_ngff_image_full(
        ngff_image, reference_image, transform,
        spatial_dims=spatial_dims
    )