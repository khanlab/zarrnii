"""
New unified ZarrNii implementation using NgffImage internally.

This provides a single API that maintains chainable functionality while using 
NgffImage objects under the hood directly without duplicate metadata attributes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import dask.array as da
import ngff_zarr as nz
import nibabel as nib
import fsspec
from attrs import define

from .transform import AffineTransform, Transform


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


def _select_channels_from_image(ngff_image, multiscales, channels, channel_labels):
    """Select specific channels from an NgffImage."""
    # Implementation would go here - for now, return the image as-is
    return ngff_image


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
    """

    ngff_image: nz.NgffImage
    axes_order: str = "ZYX"

    # Properties that delegate to the internal NgffImage
    @property
    def data(self) -> da.Array:
        """Access the image data (dask array)."""
        return self.ngff_image.data
    
    @property
    def darr(self) -> da.Array:
        """Legacy property name for image data."""
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
            if dim == 'c':
                axes.append({"name": "c", "type": "channel", "unit": None})
            else:
                axes.append({"name": dim, "type": "space", "unit": "micrometer"})
        return axes
    
    @property
    def coordinate_transformations(self) -> Optional[List[Dict]]:
        """Coordinate transformations - derived from NgffImage scale/translation."""
        transforms = []
        
        # Scale transform
        scale_list = [self.ngff_image.scale.get(dim, 1.0) for dim in self.ngff_image.dims]
        transforms.append({"type": "scale", "scale": scale_list})
        
        # Translation transform
        translation_list = [self.ngff_image.translation.get(dim, 0.0) for dim in self.ngff_image.dims]
        if any(v != 0.0 for v in translation_list):
            transforms.append({"type": "translation", "translation": translation_list})
        
        return transforms if transforms else None
    
    @property
    def omero(self) -> Optional[Dict]:
        """Omero metadata - currently not supported in NgffImage directly."""
        return None

    # Constructor methods
    @classmethod
    def from_ngff_image(cls, ngff_image: nz.NgffImage, axes_order: str = "ZYX") -> "ZarrNii":
        """
        Create ZarrNii from an existing NgffImage.
        
        Args:
            ngff_image: NgffImage to wrap
            axes_order: Spatial axes order for NIfTI compatibility
            
        Returns:
            ZarrNii instance
        """
        return cls(ngff_image=ngff_image, axes_order=axes_order)

    @classmethod
    def from_ome_zarr(
        cls,
        store_or_path,
        level: int = 0,
        channels: Optional[List[int]] = None,
        channel_labels: Optional[List[str]] = None,
        storage_options: Optional[Dict] = None,
        axes_order: str = "ZYX",
    ) -> "ZarrNii":
        """
        Load from OME-Zarr store.
        
        Args:
            store_or_path: Store or path to OME-Zarr file
            level: Pyramid level to load
            channels: Channel indices to load
            channel_labels: Channel labels to load
            storage_options: Storage options for Zarr
            axes_order: Spatial axes order for NIfTI compatibility
            
        Returns:
            ZarrNii instance
        """
        # Load the multiscales object
        try:
            if isinstance(store_or_path, str):
                multiscales = nz.from_ngff_zarr(store_or_path, storage_options=storage_options or {})
            else:
                multiscales = nz.from_ngff_zarr(store_or_path)
        except Exception as e:
            # Fallback for older zarr/ngff_zarr versions
            if isinstance(store_or_path, str):
                store = fsspec.get_mapper(store_or_path, **storage_options or {})
            else:
                store = store_or_path
            multiscales = nz.from_ngff_zarr(store)
        
        # Get the specified level
        ngff_image = multiscales.images[level]
        
        # Handle channel selection if specified
        if channels is not None or channel_labels is not None:
            ngff_image = _select_channels_from_image(
                ngff_image, multiscales, channels, channel_labels
            )
        
        return cls(ngff_image=ngff_image, axes_order=axes_order)

    @classmethod
    def from_nifti(cls, path, chunks="auto", axes_order="ZYX", name=None):
        """
        Load from NIfTI file.
        
        Args:
            path: Path to NIfTI file
            chunks: Chunking strategy for dask array
            axes_order: Spatial axes order
            name: Name for the NgffImage
            
        Returns:
            ZarrNii instance
        """
        # Load NIfTI file
        nifti_img = nib.load(path)
        array = nifti_img.get_fdata()
        affine_matrix = nifti_img.affine

        # Convert to dask array
        darr = da.from_array(array, chunks=chunks)
        
        # Add channel dimension if not present
        if len(darr.shape) == 3:
            darr = darr[np.newaxis, ...]
        
        # Create dimensions
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
            data=darr,
            dims=dims,
            scale=scale,
            translation=translation,
            name=name
        )

        return cls(ngff_image=ngff_image, axes_order=axes_order)

    # Chainable operations - each returns a new ZarrNii instance
    def crop(
        self, 
        bbox_min: tuple, 
        bbox_max: tuple,
        spatial_dims: List[str] = ["z", "y", "x"]
    ) -> "ZarrNii":
        """
        Crop the image and return a new ZarrNii instance.
        
        Args:
            bbox_min: Minimum corner of bounding box
            bbox_max: Maximum corner of bounding box
            spatial_dims: Names of spatial dimensions
            
        Returns:
            New ZarrNii with cropped data
        """
        from .ngff_core import crop_ngff_image
        cropped_image = crop_ngff_image(self.ngff_image, bbox_min, bbox_max, spatial_dims)
        return ZarrNii(ngff_image=cropped_image, axes_order=self.axes_order)

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
        spatial_dims: List[str] = ["z", "y", "x"]
    ) -> "ZarrNii":
        """
        Downsample the image and return a new ZarrNii instance.
        
        Args:
            factors: Downsampling factors (int for isotropic, list for per-dimension)
            along_x: Legacy parameter for X downsampling
            along_y: Legacy parameter for Y downsampling
            along_z: Legacy parameter for Z downsampling
            level: Legacy parameter for level-based downsampling (2^level)
            spatial_dims: Names of spatial dimensions  
            
        Returns:
            New ZarrNii with downsampled data
        """
        from .ngff_core import downsample_ngff_image
        
        # Handle legacy parameters
        if factors is None:
            if level is not None:
                factors = 2 ** level
            else:
                factors = [along_z, along_y, along_x]
        
        downsampled_image = downsample_ngff_image(self.ngff_image, factors, spatial_dims)
        return ZarrNii(ngff_image=downsampled_image, axes_order=self.axes_order)

    def apply_transform(
        self,
        *transforms: Transform,
        ref_znimg: "ZarrNii",
        spatial_dims: List[str] = ["z", "y", "x"]
    ) -> "ZarrNii":
        """
        Apply spatial transformation and return a new ZarrNii instance.
        
        Args:
            transforms: Transformations to apply
            ref_znimg: Reference ZarrNii defining output space
            spatial_dims: Names of spatial dimensions
            
        Returns:
            New ZarrNii with transformed data
        """
        from .ngff_core import apply_transform_to_ngff_image
        
        # For now, just apply the first transform (placeholder)
        if transforms:
            transformed_image = apply_transform_to_ngff_image(
                self.ngff_image, transforms[0], ref_znimg.ngff_image, spatial_dims
            )
        else:
            transformed_image = self.ngff_image
            
        return ZarrNii(ngff_image=transformed_image, axes_order=self.axes_order)

    # I/O operations
    def to_ome_zarr(
        self,
        store_or_path,
        max_layer: int = 4,
        scale_factors: Optional[List[int]] = None,
        **kwargs
    ) -> "ZarrNii":
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
        from .ngff_core import save_ngff_image
        save_ngff_image(self.ngff_image, store_or_path, max_layer, scale_factors, **kwargs)
        return self

    def to_nifti(self, filename=None):
        """
        Convert to NIfTI format.
        
        Args:
            filename: Output filename, if None return nibabel image
            
        Returns:
            nibabel.Nifti1Image or path if filename provided
        """
        # Get data and affine
        data = self.data.compute()
        affine_matrix = self.get_affine_matrix()
        
        # Remove channel dimension for NIfTI if it's size 1
        if data.shape[0] == 1:
            data = data[0]
        
        # Create NIfTI image
        nifti_img = nib.Nifti1Image(data, affine_matrix)
        
        if filename is not None:
            nib.save(nifti_img, filename)
            return filename
        else:
            return nifti_img

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
        return ZarrNii(ngff_image=copied_image, axes_order=self.axes_order)
    
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

    def __repr__(self) -> str:
        """String representation."""
        return (f"ZarrNii(name='{self.name}', "
                f"shape={self.shape}, "
                f"dims={self.dims}, "
                f"scale={self.scale})")


# Helper functions for backward compatibility
def affine_to_orientation(affine):
    """Convert affine matrix to orientation string."""
    # Simplified implementation - could be more robust
    return "RAS"


def orientation_to_affine(orientation, spacing=(1, 1, 1), origin=(0, 0, 0)):
    """Convert orientation string to affine matrix."""
    affine = np.eye(4)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1] 
    affine[2, 2] = spacing[2]
    affine[0, 3] = origin[0]
    affine[1, 3] = origin[1]
    affine[2, 3] = origin[2]
    return affine