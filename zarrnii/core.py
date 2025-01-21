from __future__ import annotations

from collections.abc import MutableMapping
from typing import Dict, List, Optional

import dask.array as da
import fsspec
import nibabel as nib
import numpy as np
import zarr
from attrs import define, field
from dask.diagnostics import ProgressBar
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image
from scipy.interpolate import interpn
from scipy.ndimage import zoom

from .transform import AffineTransform, Transform


@define
class ZarrNii:
    """
    Represents a Zarr-based image with NIfTI compatibility and OME-Zarr metadata.

    Attributes:
        darr (da.Array): The main dask array holding image data.
        affine (AffineTransform, optional): The affine transformation matrix.
        axes_order (str): The order of the axes in the data array ('ZYX' or 'XYZ').
        axes (Optional[List[Dict]], optional): Metadata about the axes (from OME-Zarr).
        coordinate_transformations (Optional[List[Dict]], optional): Transformations applied to the data
            (from OME-Zarr metadata).
        omero (Optional[Dict], optional): Metadata related to visualization and channels (from OME-Zarr).
    """

    darr: da.Array
    affine: Optional[AffineTransform] = None
    axes_order: str = "ZYX"

    # Metadata for OME-Zarr
    axes: Optional[List[Dict]] = field(
        default=None, metadata={"description": "Metadata about the axes"}
    )
    coordinate_transformations: Optional[List[Dict]] = field(
        default=None, metadata={"description": "OME-Zarr coordinate transformations"}
    )
    omero: Optional[Dict] = field(
        default=None, metadata={"description": "OME-Zarr Omero metadata"}
    )

    @classmethod
    def from_darr(
        cls,
        darr,
        affine=None,
        orientation="RAS",
        axes_order="XYZ",
        axes=None,
        coordinate_transformations=None,
        omero=None,
        spacing=(1, 1, 1),
        origin=(0, 0, 0),
    ):
        """
        Creates a ZarrNii instance from an existing Dask array.

        Parameters:
            darr (da.Array): Input Dask array.
            affine (AffineTransform or np.ndarray, optional): Affine transform to associate with the array.
                If None, an affine will be created based on the orientation, spacing, and origin.
            orientation (str, optional): Orientation string used to generate an affine matrix (default: "RAS").
            axes_order (str): The axes order of the input array (default: "XYZ").
            axes (list, optional): Axes metadata for OME-Zarr. If None, default axes are generated.
            coordinate_transformations (list, optional): Coordinate transformations for OME-Zarr metadata.
            omero (dict, optional): Omero metadata for OME-Zarr.
            spacing (tuple, optional): Voxel spacing along each axis (default: (1, 1, 1)).
            origin (tuple, optional): Origin point in physical space (default: (0, 0, 0)).

        Returns:
            ZarrNii: A populated ZarrNii instance.
        """

        # Generate affine from orientation if not explicitly provided
        if affine is None:
            affine = orientation_to_affine(orientation, spacing, origin)

        # Generate default axes if none are provided
        if axes is None:
            axes = [{"name": "c", "type": "channel", "unit": None}] + [
                {"name": ax, "type": "space", "unit": "micrometer"} for ax in axes_order
            ]

        # Generate default coordinate transformations if none are provided
        if coordinate_transformations is None:
            # Derive scale and translation from the affine
            scale = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))  # Diagonal scales
            translation = affine[:3, 3]  # Translation vector
            coordinate_transformations = [
                {"type": "scale", "scale": [1] + scale.tolist()},  # Add channel scale
                {
                    "type": "translation",
                    "translation": [0] + translation.tolist(),
                },  # Add channel translation
            ]

        # Generate default omero metadata if none is provided
        if omero is None:
            omero = {
                "channels": [{"label": f"Channel-{i}"} for i in range(darr.shape[0])],
                "rdefs": {"model": "color"},
            }

        # Create and return the ZarrNii instance
        return cls(
            darr,
            affine=AffineTransform.from_array(affine),
            axes_order=axes_order,
            axes=axes,
            coordinate_transformations=coordinate_transformations,
            omero=omero,
        )

    @classmethod
    def from_array(
        cls,
        array,
        affine=None,
        chunks="auto",
        orientation="RAS",
        axes_order="XYZ",
        axes=None,
        coordinate_transformations=None,
        omero=None,
        spacing=(1, 1, 1),
        origin=(0, 0, 0),
    ):
        """
        Creates a ZarrNii instance from an existing numpy array.

        Parameters:
            array (np.ndarray): Input numpy array.
            affine (AffineTransform or np.ndarray, optional): Affine transform to associate with the array.
                If None, an affine will be created based on the orientation, spacing, and origin.
            orientation (str, optional): Orientation string used to generate an affine matrix (default: "RAS").
            chunks (str or tuple): Chunk size for dask array (default: "auto").
            axes_order (str): The axes order of the input array (default: "ZYX").
            axes (list, optional): Axes metadata for OME-Zarr. If None, default axes are generated.
            coordinate_transformations (list, optional): Coordinate transformations for OME-Zarr metadata.
            omero (dict, optional): Omero metadata for OME-Zarr.
            spacing (tuple, optional): Voxel spacing along each axis (default: (1, 1, 1)).
            origin (tuple, optional): Origin point in physical space (default: (0, 0, 0)).


        Returns:
            ZarrNii: A populated ZarrNii instance.

        """

        return cls.from_darr(
            da.from_array(array, chunks=chunks),
            affine=affine,
            orientation=orientation,
            axes_order=axes_order,
            axes=axes,
            coordinate_transformations=coordinate_transformations,
            omero=omero,
            spacing=spacing,
            origin=origin,
        )

    @classmethod
    def from_nifti(cls, path, chunks="auto", as_ref=False, zooms=None):
        """
        Creates a ZarrNii instance from a NIfTI file. Populates OME-Zarr metadata
        based on the NIfTI affine matrix.

        Parameters:
            path (str): Path to the NIfTI file.
            chunks (str or tuple): Chunk size for dask array (default: "auto").
            as_ref (bool): If True, creates an empty dask array with the correct shape instead of loading data.
            zooms (list or np.ndarray): Target voxel spacing in xyz (only valid if as_ref=True).

        Returns:
            ZarrNii: A populated ZarrNii instance.

        Raises:
            ValueError: If `zooms` is specified when `as_ref=False`.
        """
        if not as_ref and zooms is not None:
            raise ValueError("`zooms` can only be used when `as_ref=True`.")

        # Load the NIfTI file and extract metadata
        nii = nib.load(path)
        shape = nii.header.get_data_shape()
        affine = nii.affine

        # Adjust shape and affine if zooms are provided
        if zooms is not None:
            in_zooms = np.sqrt(
                (affine[:3, :3] ** 2).sum(axis=0)
            )  # Current voxel spacing
            scaling_factor = in_zooms / zooms
            new_shape = [
                int(np.floor(shape[0] * scaling_factor[2])),  # Z
                int(np.floor(shape[1] * scaling_factor[1])),  # Y
                int(np.floor(shape[2] * scaling_factor[0])),  # X
            ]
            np.fill_diagonal(affine[:3, :3], zooms)
        else:
            new_shape = shape

        if as_ref:
            # Create an empty dask array with the adjusted shape
            darr = da.empty((1, *new_shape), chunks=chunks, dtype="float32")
        else:
            # Load the NIfTI data and convert to a dask array
            data = np.expand_dims(nii.get_fdata(), axis=0)  # Add a channel dimension
            darr = da.from_array(data, chunks=chunks)

        # Define axes order and metadata
        axes_order = "XYZ"
        axes = [
            {"name": "channel", "type": "channel", "unit": None},
            {"name": "x", "type": "space", "unit": "millimeter"},
            {"name": "y", "type": "space", "unit": "millimeter"},
            {"name": "z", "type": "space", "unit": "millimeter"},
        ]

        # Extract coordinate transformations from the affine matrix
        scale = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))  # Diagonal scales
        translation = affine[:3, 3]  # Translation vector
        coordinate_transformations = [
            {"type": "scale", "scale": [1] + scale.tolist()},  # Add channel scale
            {
                "type": "translation",
                "translation": [0] + translation.tolist(),
            },  # Add channel translation
        ]

        # Define basic Omero metadata
        omero = {
            "channels": [{"label": "Channel-0"}],  # Placeholder channel information
            "rdefs": {"model": "color"},
        }

        # Create and return the ZarrNii instance
        return cls(
            darr,
            affine=AffineTransform.from_array(affine),
            axes_order=axes_order,
            axes=axes,
            coordinate_transformations=coordinate_transformations,
            omero=omero,
        )

    @classmethod
    def from_ome_zarr(
        cls,
        path,
        level=0,
        channels=[0],
        chunks="auto",
        z_level_offset=-2,
        rechunk=False,
        storage_options=None,
        orientation="IPL",
        as_ref=False,
        zooms=None,
    ):
        """
        Reads in an OME-Zarr file as a ZarrNii image, optionally as a reference.

        Parameters:
            path (str): Path to the OME-Zarr file.
            level (int): Pyramid level to load (default: 0).
            channels (list): Channels to load (default: [0]).
            chunks (str or tuple): Chunk size for dask array (default: "auto").
            z_level_offset (int): Offset for Z downsampling level (default: -2).
            rechunk (bool): Whether to rechunk the data (default: False).
            storage_options (dict): Storage options for Zarr.
            orientation (str): Default input orientation if none is specified in metadata (default: 'IPL').
            as_ref (bool): If True, creates an empty dask array with the correct shape instead of loading data.
            zooms (list or np.ndarray): Target voxel spacing in xyz (only valid if as_ref=True).

        Returns:
            ZarrNii: A populated ZarrNii instance.

        Raises:
            ValueError: If `zooms` is specified when `as_ref=False`.
        """

        if not as_ref and zooms is not None:
            raise ValueError("`zooms` can only be used when `as_ref=True`.")

        # Open the Zarr metadata
        store = zarr.open(path, mode="r")
        multiscales = store.attrs.get("multiscales", [{}])
        datasets = multiscales[0].get("datasets", [{}])
        coordinate_transformations = datasets[level].get(
            "coordinateTransformations", []
        )
        axes = multiscales[0].get("axes", [])
        omero = store.attrs.get("omero", {})

        # Read orientation metadata (default to `orientation` if not present)
        orientation = store.attrs.get("orientation", orientation)

        # Determine the level and whether downsampling is required
        if not as_ref:
            (
                level,
                do_downsample,
                downsampling_kwargs,
            ) = cls.get_level_and_downsampling_kwargs(
                path, level, z_level_offset, storage_options=storage_options
            )
        else:
            do_downsample = False

        # Load data or metadata as needed
        darr_base = da.from_zarr(
            path, component=f"/{level}", storage_options=storage_options
        )[channels, :, :, :]
        shape = darr_base.shape

        affine = cls.construct_affine(coordinate_transformations, orientation)

        if zooms is not None:
            # Handle zoom adjustments
            in_zooms = np.sqrt(
                (affine[:3, :3] ** 2).sum(axis=0)
            )  # Current voxel spacing
            scaling_factor = in_zooms / zooms
            new_shape = [
                shape[0],
                int(np.floor(shape[1] * scaling_factor[2])),  # Z
                int(np.floor(shape[2] * scaling_factor[1])),  # Y
                int(np.floor(shape[3] * scaling_factor[0])),  # X
            ]
            np.fill_diagonal(affine[:3, :3], zooms)
        else:
            new_shape = shape

        if as_ref:
            # Create an empty array with the updated shape
            darr = da.empty(new_shape, chunks=chunks, dtype=darr_base.dtype)
        else:
            darr = darr_base
            if rechunk:
                darr = darr.rechunk(chunks)

        if rechunk:
            darr = darr.rechunk(chunks)

        if do_downsample:
            return cls(
                darr,
                affine=AffineTransform.from_array(affine),
                axes_order="ZYX",
                axes=axes,
                coordinate_transformations=coordinate_transformations,
                omero=omero,
            ).downsample(**downsampling_kwargs)
        else:
            return cls(
                darr,
                affine=AffineTransform.from_array(affine),
                axes_order="ZYX",
                axes=axes,
                coordinate_transformations=coordinate_transformations,
                omero=omero,
            )

    @staticmethod
    def align_affine_to_input_orientation(affine, orientation):
        """
        Reorders and flips the affine matrix to align with the specified input orientation.

        Parameters:
            affine (np.ndarray): Initial affine matrix.
            in_orientation (str): Input orientation (e.g., 'RAS').

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
        reordered_affine[3, :] = affine[3, :]  # Preserve homogeneous row

        return reordered_affine

    @staticmethod
    def construct_affine(coordinate_transformations, orientation):
        """
        Constructs the affine matrix based on OME-Zarr coordinate transformations
        and adjusts it for the input orientation.

        Parameters:
            coordinate_transformations (list): Coordinate transformations from OME-Zarr metadata.
            orientation (str): Input orientation (e.g., 'RAS').

        Returns:
            np.ndarray: A 4x4 affine matrix.
        """
        # Initialize affine as an identity matrix
        affine = np.eye(4)

        # Parse scales and translations
        scales = [1.0, 1.0, 1.0]
        translations = [0.0, 0.0, 0.0]

        for transform in coordinate_transformations:
            if transform["type"] == "scale":
                scales = transform["scale"][1:]  # Ignore the channel/time dimension
            elif transform["type"] == "translation":
                translations = transform["translation"][
                    1:
                ]  # Ignore the channel/time dimension

        # Populate the affine matrix
        affine[:3, :3] = np.diag(scales)  # Set scaling
        affine[:3, 3] = translations  # Set translation

        # Reorder the affine matrix for the input orientation
        return ZarrNii.align_affine_to_input_orientation(affine, orientation)

    @staticmethod
    def reorder_affine_for_xyz(affine):
        """
        Reorders the affine matrix from ZYX to XYZ axes order and adjusts the translation.

        Parameters:
            affine (np.ndarray): Affine matrix in ZYX order.

        Returns:
            np.ndarray: Affine matrix reordered to XYZ order.
        """
        # Reordering matrix to go from ZYX to XYZ
        reorder_xfm = np.array(
            [
                [0, 0, 1, 0],  # Z -> X
                [0, 1, 0, 0],  # Y -> Y
                [1, 0, 0, 0],  # X -> Z
                [0, 0, 0, 1],  # Homogeneous row
            ]
        )

        # Apply reordering to the affine matrix
        affine_reordered = affine @ reorder_xfm

        # Adjust translation (last column)
        translation_zyx = affine[:3, 3]
        reorder_perm = [2, 1, 0]  # Map ZYX -> XYZ
        translation_xyz = translation_zyx[reorder_perm]

        # Update reordered affine with adjusted translation
        affine_reordered[:3, 3] = translation_xyz
        return affine_reordered

    def apply_transform(self, *tfms, ref_znimg):
        """
        Apply a sequence of transformations to the current ZarrNii instance
        to align it with the reference ZarrNii instance (`ref_znimg`).

        This is a lazy operation and doesn't perform computations until
        `.compute()` is called on the returned dask array.

        Parameters:
            *tfms: Transformations to apply. Each transformation should be a
                   Transform (or subclass) object.
            ref_znimg (ZarrNii): The reference ZarrNii instance to align with.

        Returns:
            ZarrNii: A new ZarrNii instance with the transformations applied.

        Notes:
            - The transformations are applied in the following order:
              1. The affine transformation of the reference image.
              2. The transformations passed as `*tfms`.
              3. The inverse affine transformation of the current image.
            - The data in the returned ZarrNii is lazily interpolated using
              `dask.array.map_blocks`.

        Example:
            transformed_znimg = znimg.apply_transform(
                transform1, transform2, ref_znimg=ref_image
            )
        """
        # Initialize the list of transformations to apply
        tfms_to_apply = [ref_znimg.affine]  # Start with the reference image affine

        # Append all transformations passed as arguments
        tfms_to_apply.extend(tfms)

        # Append the inverse of the current image's affine
        tfms_to_apply.append(self.affine.invert())

        # Create a new ZarrNii instance for the interpolated image
        interp_znimg = ref_znimg

        # Lazily apply the transformations using dask
        interp_znimg.darr = da.map_blocks(
            interp_by_block,  # Function to interpolate each block
            ref_znimg.darr,  # Reference image data
            dtype=np.float32,  # Output data type
            transforms=tfms_to_apply,  # Transformations to apply
            flo_znimg=self,  # Floating image to align
        )

        return interp_znimg

    def apply_transform_ref_to_flo_indices(self, *tfms, ref_znimg, indices):
        """
        Transforms indices from the reference image space to the floating image space
        by applying a sequence of transformations.

        Parameters:
            *tfms: Transform objects to apply. These can be `AffineTransform`,
                   `DisplacementTransform`, or other subclasses of `Transform`.
            ref_znimg (ZarrNii): The reference ZarrNii instance defining the source space.
            indices (np.ndarray): 3xN array of indices in the reference space.

        Returns:
            np.ndarray: 3xN array of transformed indices in the floating image space.

        Notes:
            - Indices are treated as vectors in homogeneous coordinates, enabling
              transformation via matrix multiplication.
            - Transformations are applied in the following order:
              1. The affine transformation of the reference image.
              2. The transformations passed as `*tfms`.
              3. The inverse affine transformation of the floating image.

        Example:
            transformed_indices = flo_znimg.apply_transform_ref_to_flo_indices(
                transform1, transform2, ref_znimg=ref_image, indices=indices_in_ref
            )
        """
        # Initialize the list of transformations to apply
        tfms_to_apply = [ref_znimg.affine]  # Start with the reference image affine

        # Append all provided transformations
        tfms_to_apply.extend(tfms)

        # Append the inverse affine transformation of the current image
        tfms_to_apply.append(self.affine.invert())

        # Ensure indices are in homogeneous coordinates (4xN matrix)
        homog = np.ones((1, indices.shape[1]))
        xfm_vecs = np.vstack((indices, homog))

        # Sequentially apply transformations
        for tfm in tfms_to_apply:
            xfm_vecs = tfm.apply_transform(xfm_vecs)

        # Return the transformed indices in non-homogeneous coordinates
        return xfm_vecs[:3, :]

    def apply_transform_flo_to_ref_indices(self, *tfms, ref_znimg, indices):
        """
        Transforms indices from the floating image space to the reference image space
        by applying a sequence of transformations.

        Parameters:
            *tfms: Transform objects to apply. These can be `AffineTransform`,
                   `DisplacementTransform`, or other subclasses of `Transform`.
            ref_znimg (ZarrNii): The reference ZarrNii instance defining the target space.
            indices (np.ndarray): 3xN array of indices in the floating image space.

        Returns:
            np.ndarray: 3xN array of transformed indices in the reference image space.

        Notes:
            - Indices are treated as vectors in homogeneous coordinates, enabling
              transformation via matrix multiplication.
            - Transformations are applied in the following order:
              1. The affine transformation of the floating image.
              2. The transformations passed as `*tfms`.
              3. The inverse affine transformation of the reference image.

        Example:
            transformed_indices = flo_znimg.apply_transform_flo_to_ref_indices(
                transform1, transform2, ref_znimg=ref_image, indices=indices_in_flo
            )
        """
        # Initialize the list of transformations to apply
        tfms_to_apply = [self.affine]  # Start with the floating image affine

        # Append all provided transformations
        tfms_to_apply.extend(tfms)

        # Append the inverse affine transformation of the reference image
        tfms_to_apply.append(ref_znimg.affine.invert())

        # Ensure indices are in homogeneous coordinates (4xN matrix)
        homog = np.ones((1, indices.shape[1]))
        xfm_vecs = np.vstack((indices, homog))

        # Sequentially apply transformations
        for tfm in tfms_to_apply:
            xfm_vecs = tfm.apply_transform(xfm_vecs)

        # Return the transformed indices in non-homogeneous coordinates
        return xfm_vecs[:3, :]

    def get_bounded_subregion(self, points: np.ndarray):
        """
        Extracts a bounded subregion of the dask array containing the specified points,
        along with the grid points for interpolation.

        If the points extend beyond the domain of the dask array, the extent is capped
        at the boundaries. If all points are outside the domain, the function returns
        `(None, None)`.

        Parameters:
            points (np.ndarray): Nx3 or Nx4 array of coordinates in the array's space.
                                 If Nx4, the last column is assumed to be the homogeneous
                                 coordinate and is ignored.

        Returns:
            tuple:
                grid_points (tuple): A tuple of three 1D arrays representing the grid
                                     points along each axis (X, Y, Z) in the subregion.
                subvol (np.ndarray or None): The extracted subregion as a NumPy array.
                                             Returns `None` if all points are outside
                                             the array domain.

        Notes:
            - The function uses `compute()` on the dask array to immediately load the
              subregion, as Dask doesn't support the type of indexing required for
              interpolation.
            - A padding of 1 voxel is applied around the extent of the points.

        Example:
            grid_points, subvol = znimg.get_bounded_subregion(points)
            if subvol is not None:
                print("Subvolume shape:", subvol.shape)
        """
        pad = 1  # Padding around the extent of the points

        # Compute the extent of the points in the array's coordinate space
        min_extent = np.floor(points.min(axis=1)[:3] - pad).astype("int")
        max_extent = np.ceil(points.max(axis=1)[:3] + pad).astype("int")

        # Clip the extents to ensure they stay within the bounds of the array
        clip_min = np.zeros_like(min_extent)
        clip_max = np.array(self.darr.shape[-3:])  # Z, Y, X dimensions

        min_extent = np.clip(min_extent, clip_min, clip_max)
        max_extent = np.clip(max_extent, clip_min, clip_max)

        # Check if all points are outside the domain
        if np.any(max_extent <= min_extent):
            return None, None

        # Extract the subvolume using the computed extents
        subvol = self.darr[
            :,
            min_extent[0] : max_extent[0],
            min_extent[1] : max_extent[1],
            min_extent[2] : max_extent[2],
        ].compute()

        # Generate grid points for interpolation
        grid_points = (
            np.arange(min_extent[0], max_extent[0]),  # Z
            np.arange(min_extent[1], max_extent[1]),  # Y
            np.arange(min_extent[2], max_extent[2]),  # X
        )

        return grid_points, subvol

    def to_nifti(self, filename=None):
        """
        Convert the current ZarrNii instance to a NIfTI-1 image (Nifti1Image)
        and optionally save it to a file.

        Parameters:
            filename (str, optional): Output path for the NIfTI file. If None,
                                      the function returns the NIfTI object.

        Returns:
            nib.Nifti1Image: The NIfTI-1 image representation of the ZarrNii instance
                             if `filename` is not provided.

        Notes:
            - Reorders data to XYZ order if the current `axes_order` is ZYX.
            - Adjusts the affine matrix accordingly to match the reordered data.
        """

        # Reorder data to match NIfTI's expected XYZ order if necessary
        if self.axes_order == "ZYX":
            data = da.moveaxis(
                self.darr, (0, 1, 2, 3), (0, 3, 2, 1)
            ).compute()  # Reorder to XYZ
            affine = self.reorder_affine_for_xyz(
                self.affine.matrix
            )  # Reorder affine to match
        else:
            data = self.darr.compute()
            affine = self.affine.matrix  # No reordering needed
        # Create the NIfTI-1 image
        nii_img = nib.Nifti1Image(
            data[0], affine
        )  # Remove the channel dimension for NIfTI

        # Save the NIfTI file if a filename is provided
        if filename:
            nib.save(nii_img, filename)
        else:
            return nii_img

    def to_ome_zarr(self, filename, max_layer=4, scaling_method="local_mean", **kwargs):
        """
        Save the current ZarrNii instance to an OME-Zarr file, always writing
        axes in ZYX order.

        Parameters:
            filename (str): Output path for the OME-Zarr file.
            max_layer (int): Maximum number of downsampling layers (default: 4).
            scaling_method (str): Method for downsampling (default: "local_mean").
            **kwargs: Additional arguments for `write_image`.
        """
        # Always write OME-Zarr axes as ZYX
        axes = [
            {"name": "c", "type": "channel", "unit": None},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]

        # Reorder data if the axes order is XYZ (NIfTI-like)
        if self.axes_order == "XYZ":
            out_darr = da.moveaxis(
                self.darr, (0, 1, 2, 3), (0, 3, 2, 1)
            )  # Reorder to ZYX
            #   flip_xfm = np.diag((-1, -1, -1, 1))  # Apply flips for consistency
            # out_affine = flip_xfm @ self.affine
            out_affine = self.reorder_affine_for_xyz(self.affine.matrix)
        #  voxdim = np.flip(voxdim)  # Adjust voxel dimensions to ZYX
        else:
            out_darr = self.darr
            out_affine = self.affine

        # Extract offset and voxel dimensions from the affine matrix
        offset = out_affine[:3, 3]
        voxdim = np.sqrt((out_affine[:3, :3] ** 2).sum(axis=0))  # Extract scales

        # Prepare coordinate transformations
        coordinate_transformations = []
        for layer in range(max_layer + 1):
            scale = [
                1,
                voxdim[0],
                (2**layer) * voxdim[1],  # Downsampling in Y
                (2**layer) * voxdim[2],  # Downsampling in X
            ]
            translation = [
                0,
                offset[0],
                offset[1] / (2**layer),
                offset[2] / (2**layer),
            ]
            coordinate_transformations.append(
                [
                    {"type": "scale", "scale": scale},
                    {"type": "translation", "translation": translation},
                ]
            )

        # Set up Zarr store
        store = zarr.storage.FSStore(filename, dimension_separator="/", mode="w")
        group = zarr.group(store, overwrite=True)

        # Add metadata for orientation
        group.attrs["orientation"] = affine_to_orientation(
            out_affine
        )  # Write current orientation

        # Set up scaler for multi-resolution pyramid
        if max_layer == 0:
            scaler = None
        else:
            scaler = Scaler(max_layer=max_layer, method=scaling_method)

        # Write the data to OME-Zarr
        with ProgressBar():
            write_image(
                image=out_darr,
                group=group,
                scaler=scaler,
                coordinate_transformations=coordinate_transformations,
                axes=axes,
                **kwargs,
            )

    def crop_with_bounding_box(self, bbox_min, bbox_max, ras_coords=False):
        """
        Crops the ZarrNii instance using a bounding box and returns a new cropped instance.

        Parameters:
            bbox_min (tuple): Minimum corner of the bounding box (Z, Y, X) in voxel coordinates.
                             If `ras_coords=True`, this should be in RAS space.
            bbox_max (tuple): Maximum corner of the bounding box (Z, Y, X) in voxel coordinates.
                             If `ras_coords=True`, this should be in RAS space.
            ras_coords (bool): Whether the bounding box coordinates are in RAS space.
                               If True, they will be converted to voxel coordinates using the affine.

        Returns:
            ZarrNii: A new ZarrNii instance representing the cropped subregion.

        Notes:
            - When `ras_coords=True`, the bounding box coordinates are transformed from RAS to voxel space
              using the inverse of the affine transformation.
            - The affine transformation is updated to reflect the cropped region, ensuring spatial consistency.

        Example:
            # Define a bounding box in voxel space
            bbox_min = (10, 20, 30)
            bbox_max = (50, 60, 70)

            # Crop the ZarrNii instance
            cropped_znimg = znimg.crop_with_bounding_box(bbox_min, bbox_max)

            # Define a bounding box in RAS space
            ras_min = (-10, -20, -30)
            ras_max = (10, 20, 30)

            # Crop using RAS coordinates
            cropped_znimg_ras = znimg.crop_with_bounding_box(ras_min, ras_max, ras_coords=True)
        """
        # Convert RAS coordinates to voxel coordinates if needed
        if ras_coords:
            bbox_min = np.round(self.affine.invert() @ np.array(bbox_min)).astype(int)
            bbox_max = np.round(self.affine.invert() @ np.array(bbox_max)).astype(int)
            bbox_min = tuple(bbox_min[:3].flatten())
            bbox_max = tuple(bbox_max[:3].flatten())

        # Slice the dask array based on the bounding box
        darr_cropped = self.darr[
            :,
            bbox_min[0] : bbox_max[0],  # Z
            bbox_min[1] : bbox_max[1],  # Y
            bbox_min[2] : bbox_max[2],  # X
        ]

        # Update the affine to reflect the cropped region
        trans_vox = np.eye(4, 4)
        trans_vox[:3, 3] = bbox_min  # Translation for the cropped region
        new_affine = self.affine @ trans_vox

        # Create and return a new ZarrNii instance for the cropped region
        return ZarrNii.from_darr(
            darr_cropped, affine=new_affine, axes_order=self.axes_order
        )

    def get_bounding_box_around_label(self, label_number, padding=0, ras_coords=False):
        """
        Calculates the bounding box around a given label in the ZarrNii instance.

        Parameters:
            label_number (int): The label value for which the bounding box is computed.
            padding (int, optional): Extra padding added around the bounding box in voxel units (default: 0).
            ras_coords (bool, optional): If True, returns the bounding box coordinates in RAS space.
                                         Otherwise, returns voxel coordinates (default: False).

        Returns:
            tuple:
                bbox_min (np.ndarray): Minimum corner of the bounding box (Z, Y, X) as a (3, 1) array.
                bbox_max (np.ndarray): Maximum corner of the bounding box (Z, Y, X) as a (3, 1) array.

        Notes:
            - The function uses `da.argwhere` to locate the indices of the specified label lazily.
            - Padding is added symmetrically around the bounding box, and the result is clipped to
              ensure it remains within the array bounds.
            - If `ras_coords=True`, the bounding box coordinates are transformed to RAS space using
              the affine transformation.

        Example:
            # Compute bounding box for label 1 with 5-voxel padding
            bbox_min, bbox_max = znimg.get_bounding_box_around_label(1, padding=5)

            # Get the bounding box in RAS space
            bbox_min_ras, bbox_max_ras = znimg.get_bounding_box_around_label(1, padding=5, ras_coords=True)
        """
        # Locate the indices of the specified label
        indices = da.argwhere(self.darr == label_number).compute()

        if indices.size == 0:
            raise ValueError(f"Label {label_number} not found in the array.")

        # Compute the minimum and maximum extents in each dimension
        bbox_min = (
            indices.min(axis=0).reshape((4, 1))[1:] - padding
        )  # Exclude channel axis
        bbox_max = indices.max(axis=0).reshape((4, 1))[1:] + 1 + padding

        # Clip the extents to ensure they stay within bounds
        bbox_min = np.clip(bbox_min, 0, np.array(self.darr.shape[1:]).reshape(3, 1))
        bbox_max = np.clip(bbox_max, 0, np.array(self.darr.shape[1:]).reshape(3, 1))

        # Convert to RAS coordinates if requested
        if ras_coords:
            bbox_min = self.affine @ bbox_min
            bbox_max = self.affine @ bbox_max

        return bbox_min, bbox_max

    def downsample(
        self, along_x=1, along_y=1, along_z=1, level=None, z_level_offset=-2
    ):
        """
        Downsamples the ZarrNii instance by local mean reduction.

        Parameters:
            along_x (int, optional): Downsampling factor along the X-axis (default: 1).
            along_y (int, optional): Downsampling factor along the Y-axis (default: 1).
            along_z (int, optional): Downsampling factor along the Z-axis (default: 1).
            level (int, optional): If specified, calculates downsampling factors based on the level,
                                   with Z-axis adjusted by `z_level_offset`.
            z_level_offset (int, optional): Offset for the Z-axis downsampling factor when using `level`
                                            (default: -2).

        Returns:
            ZarrNii: A new ZarrNii instance with the downsampled data and updated affine.

        Notes:
            - If `level` is provided, downsampling factors are calculated as:
                - `along_x = along_y = 2**level`
                - `along_z = 2**max(level + z_level_offset, 0)`
            - Updates the affine matrix to reflect the new voxel size after downsampling.
            - Uses `dask.array.coarsen` for efficient reduction along specified axes.

        Example:
            # Downsample by specific factors
            downsampled_znimg = znimg.downsample(along_x=2, along_y=2, along_z=1)

            # Downsample using a pyramid level
            downsampled_znimg = znimg.downsample(level=2)
        """
        # Calculate downsampling factors if level is specified
        if level is not None:
            along_x = 2**level
            along_y = 2**level
            level_z = max(level + z_level_offset, 0)
            along_z = 2**level_z

        # Determine axes mapping based on axes_order
        if self.axes_order == "XYZ":
            axes = {0: 1, 1: along_x, 2: along_y, 3: along_z}  # (C, X, Y, Z)
        else:
            axes = {0: 1, 1: along_z, 2: along_y, 3: along_x}  # (C, Z, Y, X)

        # Perform local mean reduction using coarsen
        agg_func = np.mean
        darr_scaled = da.coarsen(agg_func, x=self.darr, axes=axes, trim_excess=True)

        # Update the affine matrix to reflect downsampling
        scaling_matrix = np.diag((along_x, along_y, along_z, 1))
        new_affine = AffineTransform.from_array(scaling_matrix @ self.affine.matrix)

        # Create and return a new ZarrNii instance
        return ZarrNii.from_darr(
            darr_scaled, affine=new_affine, axes_order=self.axes_order
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
            zip(self.darr.shape, self.darr.chunks, target_shape)
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

    def divide_by_downsampled(self, znimg_ds):
        """
        Divides the current dask array by another dask array (`znimg_ds`),
        which is assumed to be a downsampled version of the current array.

        This method upscales the downsampled array to match the resolution
        of the current array before performing element-wise division.

        Parameters:
            znimg_ds (ZarrNii): A ZarrNii instance representing the downsampled array.

        Returns:
            ZarrNii: A new ZarrNii instance containing the result of the division.

        Notes:
            - The chunking of the current array is adjusted to ensure 1:1 correspondence
              with the chunks of the downsampled array after upscaling.
            - The division operation is performed block-wise using `dask.array.map_blocks`.

        Example:
            znimg_divided = znimg.divide_by_downsampled(downsampled_znimg)
            print("Result shape:", znimg_divided.darr.shape)
        """
        # Calculate upsampled chunks for the downsampled array
        target_chunks = znimg_ds.__get_upsampled_chunks(
            self.darr.shape, return_scaling=False
        )

        # Rechunk the current high-resolution array to match the target chunks
        darr_rechunk = self.darr.rechunk(chunks=target_chunks)

        # Define the block-wise operation for zooming and division
        def block_zoom_and_divide_by(x1, x2, block_info=None):
            """
            Zooms x2 to match the size of x1 and performs element-wise division.

            Parameters:
                x1 (np.ndarray): High-resolution block from the current array.
                x2 (np.ndarray): Downsampled block from `znimg_ds`.
                block_info (dict, optional): Metadata about the current block.

            Returns:
                np.ndarray: The result of `x1 / zoom(x2, scaling)`.
            """
            # Calculate the scaling factors for zooming
            scaling = tuple(n1 / n2 for n1, n2 in zip(x1.shape, x2.shape))
            return x1 / zoom(x2, scaling, order=1, prefilter=False)

        # Perform block-wise division
        darr_div = da.map_blocks(
            block_zoom_and_divide_by,
            darr_rechunk,
            znimg_ds.darr,
            dtype=self.darr.dtype,
        )

        # Return the result as a new ZarrNii instance
        return ZarrNii(darr_div, self.affine, self.axes_order)

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
            - Currently, the method assumes `axes_order != 'XYZ'` for proper affine scaling.
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
                for chunks_i, scale in zip(self.darr.chunks, scaling)
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
            zoom_blocks, self.darr, dtype=self.darr.dtype, chunks=chunks_out
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

        # Return a new ZarrNii instance with the upsampled data
        return ZarrNii.from_darr(
            darr_scaled.rechunk(), affine=new_affine, axes_order=self.axes_order
        )

    @staticmethod
    def get_max_level(path, storage_options=None):
        """
        Retrieves the maximum level of multiscale downsampling in an OME-Zarr dataset.

        Parameters:
            path (str or MutableMapping): Path to the OME-Zarr dataset or a `MutableMapping` store.
            storage_options (dict, optional): Storage options for accessing remote or custom storage.

        Returns:
            int: The maximum level of multiscale downsampling (zero-based index).
            None: If no multiscale levels are found.

        Notes:
            - The function assumes that the Zarr dataset follows the OME-Zarr specification
              with multiscale metadata.
        """
        # Determine the store type
        if isinstance(path, MutableMapping):
            store = path
        else:
            store = fsspec.get_mapper(path, storage_options=storage_options)

        # Open the Zarr group
        group = zarr.open(store, mode="r")

        # Access the multiscale metadata
        multiscales = group.attrs.get("multiscales", [])

        # Determine the maximum level
        if multiscales:
            max_level = len(multiscales[0]["datasets"]) - 1
            return max_level
        else:
            print("No multiscale levels found.")
            return None

    @staticmethod
    def get_level_and_downsampling_kwargs(
        ome_zarr_path, level, z_level_offset=-2, storage_options=None
    ):
        """
        Determines the appropriate pyramid level and additional downsampling factors for an OME-Zarr dataset.

        Parameters:
            ome_zarr_path (str or MutableMapping): Path to the OME-Zarr dataset or a `MutableMapping` store.
            level (int): Desired downsampling level.
            z_level_offset (int, optional): Offset to adjust the Z-axis downsampling level (default: -2).
            storage_options (dict, optional): Storage options for accessing remote or custom storage.

        Returns:
            tuple:
                - level (int): The selected pyramid level (capped by the maximum level).
                - do_downsample (bool): Whether additional downsampling is required.
                - downsampling_kwargs (dict): Factors for downsampling along X, Y, and Z.

        Notes:
            - If the requested level exceeds the available pyramid levels, the function calculates
              additional downsampling factors (`level_xy`, `level_z`) for XY and Z axes.
        """
        max_level = ZarrNii.get_max_level(
            ome_zarr_path, storage_options=storage_options
        )

        # Determine the pyramid level and additional downsampling factors
        if level > max_level:  # Requested level exceeds pyramid levels
            level_xy = level - max_level
            level_z = max(level + z_level_offset, 0)
            level = max_level
        else:
            level_xy = 0
            level_z = max(level + z_level_offset, 0)

        print(
            f"level: {level}, level_xy: {level_xy}, level_z: {level_z}, max_level: {max_level}"
        )

        # Determine if additional downsampling is needed
        do_downsample = level_xy > 0 or level_z > 0

        # Return the level, downsampling flag, and downsampling parameters
        return (
            level,
            do_downsample,
            {
                "along_x": 2**level_xy,
                "along_y": 2**level_xy,
                "along_z": 2**level_z,
            },
        )


# -- inline functions
def interp_by_block(
    x,
    transforms: list[Transform],
    flo_znimg: ZarrNii,
    block_info=None,
    interp_method="linear",
):
    """
    Interpolates the floating image (`flo_znimg`) onto the reference image block (`x`)
    using the provided transformations.

    This function extracts the necessary subset of the floating image for each block
    of the reference image, applies the transformations, and interpolates the floating
    image intensities onto the reference image grid.

    Parameters:
        x (np.ndarray): The reference image block to interpolate onto.
        transforms (list[Transform]): A list of `Transform` objects to apply to the
                                       reference image coordinates.
        flo_znimg (ZarrNii): The floating ZarrNii instance to interpolate from.
        block_info (dict, optional): Metadata about the current block being processed.
        interp_method (str, optional): Interpolation method. Defaults to "linear".

    Returns:
        np.ndarray: The interpolated block of the reference image.

    Notes:
        - The function transforms the reference image block coordinates to the floating
          image space, extracts the required subregion from the floating image, and
          performs interpolation.
        - If the transformed coordinates are completely outside the bounds of the floating
          image, a zero-filled array is returned.

    Example:
        interpolated_block = interp_by_block(
            x=ref_block,
            transforms=[transform1, transform2],
            flo_znimg=floating_image,
            block_info=block_metadata,
        )
    """
    # Extract the array location (block bounds) from block_info
    arr_location = block_info[0]["array-location"]

    # Generate coordinate grids for the reference image block
    xv, yv, zv = np.meshgrid(
        np.arange(arr_location[-3][0], arr_location[-3][1]),
        np.arange(arr_location[-2][0], arr_location[-2][1]),
        np.arange(arr_location[-1][0], arr_location[-1][1]),
        indexing="ij",
    )

    # Reshape grids into vectors for matrix multiplication
    xvf = xv.reshape((1, np.product(xv.shape)))
    yvf = yv.reshape((1, np.product(yv.shape)))
    zvf = zv.reshape((1, np.product(zv.shape)))
    homog = np.ones(xvf.shape)

    xfm_vecs = np.vstack((xvf, yvf, zvf, homog))

    # Apply transformations sequentially
    for tfm in transforms:
        xfm_vecs = tfm.apply_transform(xfm_vecs)

    # Initialize the output array for interpolated values
    interpolated = np.zeros(x.shape)

    # Determine the required subregion of the floating image
    grid_points, flo_vol = flo_znimg.get_bounded_subregion(xfm_vecs)
    if grid_points is None and flo_vol is None:
        # Points are fully outside the floating image; return zeros
        return interpolated

    # Interpolate each channel of the floating image
    for c in range(flo_vol.shape[0]):
        interpolated[c, :, :, :] = (
            interpn(
                grid_points,
                flo_vol[c, :, :, :],
                xfm_vecs[:3, :].T,  # Transformed coordinates
                method=interp_method,
                bounds_error=False,
                fill_value=0,
            )
            .reshape((x.shape[-3], x.shape[-2], x.shape[-1]))
            .astype(block_info[None]["dtype"])
        )

    return interpolated


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
        affine[i, 3] = origin[axis]
    affine[3, 3] = 1  # set homog coord

    return affine
