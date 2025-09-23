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
    _select_dimensions_from_image_with_omero,
    load_ngff_image,
    save_ngff_image,
)
from .processing import (
    apply_near_isotropic_downsampling,
    apply_transform_to_ngff_image,
    crop_ngff_image,
    downsample_ngff_image,
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
        # Validate channel and timepoint selection arguments
        if channels is not None and channel_labels is not None:
            raise ValueError("Cannot specify both 'channels' and 'channel_labels'")

        # Load the multiscales object
        import ngff_zarr as nz

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
            # If we can't extract omero metadata, continue without it
            pass

        # Load NgffImage and select dimensions
        ngff_image = load_ngff_image(
            store_or_path, level, None, None, None, storage_options
        )

        # Apply channel and timepoint selection with OMERO metadata
        if channels is not None or channel_labels is not None or timepoints is not None:
            ngff_image, filtered_omero = _select_dimensions_from_image_with_omero(
                ngff_image,
                multiscales,
                channels,
                channel_labels,
                timepoints,
                omero_metadata,
            )
        else:
            filtered_omero = omero_metadata

        # Create ZarrNii instance
        znimg = cls(
            ngff_image=ngff_image,
            axes_order=axes_order,
            orientation=orientation,
            _omero=filtered_omero,
        )

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
            """Scales blocks to the desired size using `scipy.ndimage.zoom`."""
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

        upsampled_ngff = nz.NgffImage(
            data=darr_scaled,
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
        """Calculate new chunk sizes for a dask array to match a target shape."""
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

    def to_imaris(
        self, path: str, compression: str = "gzip", compression_opts: int = 6
    ) -> str:
        """
        Save to Imaris (.ims) file format using HDF5.

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

        # Create basic Imaris file structure
        with h5py.File(path, "w") as f:
            # Root attributes
            f.attrs["DataSetDirectoryName"] = _string_to_byte_array("DataSet")
            f.attrs["DataSetInfoDirectoryName"] = _string_to_byte_array("DataSetInfo")
            f.attrs["ImarisDataSet"] = _string_to_byte_array("ImarisDataSet")
            f.attrs["ImarisVersion"] = _string_to_byte_array("5.5.0")
            f.attrs["NumberOfDataSets"] = np.array([1], dtype=np.uint32)

            # Create main DataSet group structure
            dataset_group = f.create_group("DataSet")
            res_group = dataset_group.create_group("ResolutionLevel 0")
            time_group = res_group.create_group("TimePoint 0")

            # Create channels with data
            for c in range(n_channels):
                channel_group = time_group.create_group(f"Channel {c}")
                channel_data = data[c]  # (Z, Y, X)

                # Channel attributes
                channel_group.attrs["ImageSizeX"] = _string_to_byte_array(str(x))
                channel_group.attrs["ImageSizeY"] = _string_to_byte_array(str(y))
                channel_group.attrs["ImageSizeZ"] = _string_to_byte_array(str(z))

                # Create data dataset
                channel_group.create_dataset(
                    "Data",
                    data=channel_data.astype(np.float32),
                    compression=compression,
                    compression_opts=compression_opts,
                    chunks=True,
                )

            # Create basic DataSetInfo structure
            info_group = f.create_group("DataSetInfo")
            image_info = info_group.create_group("Image")
            image_info.attrs["X"] = _string_to_byte_array(str(x))
            image_info.attrs["Y"] = _string_to_byte_array(str(y))
            image_info.attrs["Z"] = _string_to_byte_array(str(z))
            image_info.attrs["Unit"] = _string_to_byte_array("um")
            image_info.attrs["Noc"] = _string_to_byte_array(str(n_channels))

        return path

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

    def apply_scaled_processing(
        self,
        plugin,
        downsample_factor: int = 4,
        chunk_size: Optional[Tuple[int, ...]] = None,
        use_temp_zarr: bool = True,
        temp_zarr_path: Optional[str] = None,
        **kwargs,
    ) -> "ZarrNii":
        """
        Apply scaled processing plugin using multi-resolution approach.

        This method implements a multi-resolution processing pipeline where:
        1. The image is downsampled for efficient computation
        2. The plugin's lowres_func is applied to the downsampled data
        3. The result is upsampled using dask-based upsampling
        4. The plugin's highres_func applies the result to full-resolution data

        Args:
            plugin: ScaledProcessingPlugin instance or class to apply
            downsample_factor: Factor for downsampling (default: 4)
            chunk_size: Optional chunk size for low-res processing. If None, uses (1, 10, 10, 10).
            use_temp_zarr: Whether to use temporary OME-Zarr for breaking up dask graph (default: True)
            temp_zarr_path: Path for temporary OME-Zarr file. If None, uses temp directory.
            **kwargs: Additional arguments passed to the plugin

        Returns:
            New ZarrNii instance with processed data
        """
        from .plugins.scaled_processing import ScaledProcessingPlugin

        # Handle plugin instance or class
        if isinstance(plugin, type) and issubclass(plugin, ScaledProcessingPlugin):
            plugin = plugin(**kwargs)
        elif not isinstance(plugin, ScaledProcessingPlugin):
            raise TypeError(
                "Plugin must be an instance or subclass of ScaledProcessingPlugin"
            )

        # Step 1: Downsample the data for low-resolution processing
        lowres_znimg = self.downsample(level=int(np.log2(downsample_factor)))

        # Convert to numpy array for lowres processing
        lowres_array = lowres_znimg.data.compute()

        # Step 2: Apply low-resolution function and prepare for upsampling
        # Use chunk_size parameter for the low-res processing chunks
        lowres_chunks = chunk_size if chunk_size is not None else (1, 10, 10, 10)
        lowres_znimg.data = da.from_array(
            plugin.lowres_func(lowres_array), chunks=lowres_chunks
        )

        # Step 3: Upsample using dask-based upsampling
        upsampled_znimg = lowres_znimg.upsample(to_shape=self.shape)

        if use_temp_zarr:
            # Use temporary OME-Zarr to break up dask graph for performance
            import os
            import tempfile

            if temp_zarr_path is None:
                # Create temp file in system temp directory
                temp_dir = tempfile.gettempdir()
                temp_name = (
                    f"zarrnii_scaled_processing_{os.getpid()}_{id(self)}.ome.zarr"
                )
                temp_zarr_path = os.path.join(temp_dir, temp_name)

            try:
                upsampled_znimg.to_ome_zarr(temp_zarr_path)
                # Reload with the same axes_order to avoid shape reordering
                reloaded_znimg = ZarrNii.from_ome_zarr(temp_zarr_path)
                # Ensure the reloaded data matches the expected shape by preserving axes order
                if reloaded_znimg.axes_order != self.axes_order:
                    # If axes order changed, we need to transpose the data back
                    if (
                        self.axes_order == "XYZ" and reloaded_znimg.axes_order == "ZYX"
                    ) or (
                        self.axes_order == "ZYX" and reloaded_znimg.axes_order == "XYZ"
                    ):
                        # Simple case: just transpose the spatial dimensions (skip channel dim)
                        upsampled_data = da.transpose(reloaded_znimg.data, (0, 3, 2, 1))
                    else:
                        # More complex reordering - fallback to direct data
                        upsampled_data = upsampled_znimg.data
                else:
                    upsampled_data = reloaded_znimg.data
            finally:
                # Clean up temp file after loading data
                if os.path.exists(temp_zarr_path):
                    import shutil

                    shutil.rmtree(temp_zarr_path, ignore_errors=True)
        else:
            # Use the upsampled data directly without temp file
            upsampled_data = upsampled_znimg.data

        # Step 4: Apply high-resolution function
        processed_data = plugin.highres_func(self.data, upsampled_data)

        # Create new NgffImage with processed data
        new_ngff_image = nz.NgffImage(
            data=processed_data,
            dims=self.dims.copy(),
            scale=self.scale.copy(),
            translation=self.translation.copy(),
            name=f"{self.name}_{plugin.name.lower().replace(' ', '_')}",
        )

        # Return new ZarrNii instance
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
