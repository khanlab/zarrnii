"""Main ZarrNii class for unified OME-Zarr and NIfTI workflows.

This module contains the core ZarrNii class which provides:
- Chainable operations for image processing (crop, downsample, upsample)
- I/O support for OME-Zarr, NIfTI, and Imaris formats
- Spatial transformation capabilities
- Multi-channel and temporal data handling
- Plugin-based segmentation and processing

The ZarrNii class maintains compatibility with legacy workflows while
using NgffImage objects internally for better metadata preservation
and multiscale support.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import dask.array as da
import fsspec
import ngff_zarr as nz
import nibabel as nib
import numpy as np
from attrs import define
from scipy.ndimage import zoom

from .enums import ImageType, TransformType
from .io import (
    load_ngff_image,
    save_ngff_image,
    _select_dimensions_from_image_with_omero,
)
from .processing import (
    crop_ngff_image,
    downsample_ngff_image,
    apply_transform_to_ngff_image,
    apply_near_isotropic_downsampling,
)
from .transform import AffineTransform, Transform
from .utils import (
    affine_to_orientation,
    align_affine_to_input_orientation,
)


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
        chunks: tuple[int, ...] | Literal["auto"] = "auto",
        rechunk: bool = False,
    ) -> "ZarrNii":
        """Load ZarrNii from OME-Zarr store with flexible options."""
        # Load NgffImage using the io module
        ngff_image = load_ngff_image(
            store_or_path, level, channels, channel_labels, timepoints, storage_options
        )
        
        # Create ZarrNii instance
        znimg = cls.from_ngff_image(ngff_image, axes_order, orientation)
        
        # Apply near-isotropic downsampling if requested
        if downsample_near_isotropic:
            znimg = apply_near_isotropic_downsampling(znimg, axes_order)
            
        if rechunk:
            znimg.data = znimg.data.rechunk(chunks)
            
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
        """Load ZarrNii from NIfTI file with flexible loading options."""
        if not as_ref and zooms is not None:
            raise ValueError("`zooms` can only be used when `as_ref=True`.")

        # Load NIfTI file
        nifti_img = nib.load(path)
        shape = nifti_img.header.get_data_shape()
        affine_matrix = nifti_img.affine.copy()

        if as_ref:
            # Create an empty dask array
            darr = da.empty((1, *shape), chunks=chunks, dtype="float32")
        else:
            # Load the NIfTI data and convert to a dask array
            array = nifti_img.get_fdata()
            darr = da.from_array(array, chunks=chunks)

        # Add channel dimension if not present
        if len(darr.shape) == 3:
            darr = darr[np.newaxis, ...]

        # Create dimensions based on axes_order
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

    # Chainable operations
    def crop(
        self,
        bbox_min: Tuple[int, ...],
        bbox_max: Tuple[int, ...],
        spatial_dims: Optional[List[str]] = None,
    ) -> "ZarrNii":
        """Extract a spatial region from the image."""
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

    def downsample(
        self,
        factors: Optional[Union[int, List[int]]] = None,
        along_x: int = 1,
        along_y: int = 1,
        along_z: int = 1,
        level: Optional[int] = None,
        spatial_dims: Optional[List[str]] = None,
    ) -> "ZarrNii":
        """Reduce image resolution by downsampling."""
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

    def to_nifti(
        self, filename: Optional[Union[str, bytes]] = None
    ) -> Union[nib.Nifti1Image, str]:
        """Convert to NIfTI format with automatic dimension handling."""
        # Get data and dimensions
        data = self.data.compute()
        dims = self.dims

        # Handle dimensional reduction for NIfTI compatibility
        squeeze_axes = []
        for i, dim in enumerate(dims):
            if dim in ["t", "c"] and data.shape[i] == 1:
                squeeze_axes.append(i)
            elif dim in ["t", "c"] and data.shape[i] > 1:
                raise ValueError(
                    f"NIfTI format doesn't support non-singleton {dim} dimension. "
                    f"Dimension '{dim}' has size {data.shape[i]}. "
                    f"Consider selecting specific timepoints/channels first."
                )

        # Squeeze out singleton dimensions
        if squeeze_axes:
            data = np.squeeze(data, axis=tuple(squeeze_axes))

        # Check final dimensionality
        if data.ndim > 4:
            raise ValueError(
                f"Resulting data has {data.ndim} dimensions, but NIfTI supports maximum 4D"
            )

        # Handle spatial reordering based on axes_order
        if self.axes_order == "ZYX":
            # Data spatial dimensions are in ZYX order, need to transpose to XYZ
            if data.ndim == 3:
                data = data.transpose(2, 1, 0)
            elif data.ndim == 4:
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

    def to_ome_zarr(
        self,
        store_or_path: Union[str, Any],
        max_layer: int = 4,
        scale_factors: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> "ZarrNii":
        """Save to OME-Zarr store with multiscale pyramid."""
        save_ngff_image(
            self.ngff_image,
            store_or_path,
            max_layer,
            scale_factors,
            orientation=getattr(self, "orientation", None),
            **kwargs,
        )
        return self

    def apply_transform(
        self,
        *transforms: Transform,
        ref_znimg: "ZarrNii",
        spatial_dims: Optional[List[str]] = None,
    ) -> "ZarrNii":
        """Apply spatial transformations to image data."""
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

    def apply_transform_ref_to_flo_indices(self, *transforms, ref_znimg, indices):
        """Transform indices from reference to floating space."""
        # Placeholder implementation - would need full transform logic
        return indices

    def apply_transform_flo_to_ref_indices(self, *transforms, ref_znimg, indices):
        """Transform indices from floating to reference space."""
        # Placeholder implementation - would need full transform logic
        return indices

    def copy(self) -> "ZarrNii":
        """Create a copy of this ZarrNii."""
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

    def get_orientation(self) -> str:
        """Get the anatomical orientation of the dataset."""
        return self.orientation

    def get_zooms(self, axes_order: str = None) -> np.ndarray:
        """Get voxel spacing (zooms) from NgffImage scale."""
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
        """Get origin (translation) from NgffImage."""
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

    def list_channels(self) -> List[str]:
        """Get list of available channel labels from OMERO metadata."""
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
        """Select specific channels from multi-channel image data."""
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
            orientation=self.orientation,
            _omero=filtered_omero,
        )

    def select_timepoints(self, timepoints: Optional[List[int]] = None) -> "ZarrNii":
        """Select timepoints from the image data and return a new ZarrNii instance."""
        if timepoints is None:
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
            _omero=self._omero,
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ZarrNii(name='{self.name}', "
            f"shape={self.shape}, "
            f"dims={self.dims}, "
            f"scale={self.scale})"
        )