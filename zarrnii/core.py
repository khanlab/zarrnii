from __future__ import annotations
import fsspec

from typing import TYPE_CHECKING


from collections.abc import MutableMapping
from pathlib import Path

import dask.array as da
import nibabel as nib
import numpy as np
import zarr
from attrs import define
from dask.diagnostics import ProgressBar
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image
from scipy.ndimage import zoom
from scipy.interpolate import interpn

from .enums import ImageType
from .transform import Transform, AffineTransform

@define
class ZarrNii:
    darr: da.Array
    affine: AffineTransform = None
    axes_order: str = 'ZYX'
  
    # for maintaining ome_zarr metadata:
    axes: dict = None # from ome_zarr
    coordinate_transformations: list(dict) = None  # from ome_zarr
    omero: dict = None # from ome_zarr
  

    @classmethod
    def from_darr(
        cls,
        darr,
        affine=None,
        axes_order="ZYX",
        axes=None,
        coordinate_transformations=None,
        omero=None,
    ):
        """
        Creates a ZarrNii instance from an existing Dask array.

        Parameters:
            darr (da.Array): Input Dask array.
            affine (AffineTransform or np.ndarray, optional): Affine transform to associate with the array.
                If None, an identity affine transform is used.
            axes_order (str): The axes order of the input array (default: "ZYX").
            axes (list, optional): Axes metadata for OME-Zarr. If None, default axes are generated.
            coordinate_transformations (list, optional): Coordinate transformations for OME-Zarr metadata.
            omero (dict, optional): Omero metadata for OME-Zarr.

        Returns:
            ZarrNii: A populated ZarrNii instance.
        """
        # Validate affine input and convert if necessary
        if affine is None:
            affine = AffineTransform.identity()  # Default to identity transform
        elif isinstance(affine, np.ndarray):
            affine = AffineTransform.from_array(affine)

        # Generate default axes if none are provided
        if axes is None:
            axes = [{"name": "c", "type": "channel", "unit": None}] + [
                {"name": ax, "type": "space", "unit": "micrometer"} for ax in axes_order
            ]

        # Generate default coordinate transformations if none are provided
        if coordinate_transformations is None:
            # Derive scale and translation from the affine
            scale = np.sqrt((affine.matrix[:3, :3] ** 2).sum(axis=0))  # Diagonal scales
            translation = affine.matrix[:3, 3]  # Translation vector
            coordinate_transformations = [
                {"type": "scale", "scale": [1] + scale.tolist()},  # Add channel scale
                {"type": "translation", "translation": [0] + translation.tolist()},  # Add channel translation
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
            affine=affine,
            axes_order=axes_order,
            axes=axes,
            coordinate_transformations=coordinate_transformations,
            omero=omero,
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
            in_zooms = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))  # Current voxel spacing
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
            {"type": "translation", "translation": [0] + translation.tolist()},  # Add channel translation
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
        coordinate_transformations = datasets[level].get("coordinateTransformations", [])
        axes = multiscales[0].get("axes", [])
        omero = store.attrs.get("omero", {})


        # Read orientation metadata (default to `orientation` if not present)
        orientation = store.attrs.get("orientation", orientation)
    
        
        # Determine the level and whether downsampling is required
        if not as_ref:
            level, do_downsample, downsampling_kwargs = cls.get_level_and_downsampling_kwargs(
                path, level, z_level_offset, storage_options=storage_options
            )
        else:
            do_downsample = False

        # Load data or metadata as needed
        darr_base = da.from_zarr(path, component=f"/{level}", storage_options=storage_options)[
            channels, :, :, :
        ]
        shape = darr_base.shape

        
        affine = cls.construct_affine(coordinate_transformations, orientation)

        if zooms is not None:
            # Handle zoom adjustments
            in_zooms = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))  # Current voxel spacing
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
        axis_map = {'R': 0, 'L': 0, 'A': 1, 'P': 1, 'S': 2, 'I': 2}
        sign_map = {'R': 1, 'L': -1, 'A': 1, 'P': -1, 'S': 1, 'I': -1}
        
        input_axes = [axis_map[ax] for ax in orientation]
        input_signs = [sign_map[ax] for ax in orientation]
            
        reordered_affine = np.zeros_like(affine) 
        for i, (axis, sign) in enumerate(zip(input_axes, input_signs)):
            reordered_affine[i, :3] = sign * affine[axis, :3]
            reordered_affine[i, 3] = sign * affine[axis, 3]
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
                translations = transform["translation"][1:]  # Ignore the channel/time dimension

        # Populate the affine matrix
        affine[:3, :3] = np.diag(scales)  # Set scaling
        affine[:3, 3] = translations     # Set translation

        # Reorder the affine matrix for the input orientation
        return ZarrNii.align_affine_to_input_orientation(affine, orientation)



    @staticmethod
    def reorder_affine_for_xyz(affine):
        """
        Reorders the affine matrix from ZYX to XYZ axes order.

        Parameters:
            affine (np.ndarray): Affine matrix in ZYX order.

        Returns:
            np.ndarray: Affine matrix reordered to XYZ order.
        """
        # Reordering matrix to go from ZYX to XYZ
        reorder_xfm = np.array([
            [0, 0, 1, 0],  # Z -> X
            [0, 1, 0, 0],  # Y -> Y
            [1, 0, 0, 0],  # X -> Z
            [0, 0, 0, 1],  # Homogeneous row
        ])
        return affine @ reorder_xfm #use right-multiply so columns get reordered


    @staticmethod
    def check_img_type(path) -> ImageType:
        if isinstance(path,MutableMapping):
            return ImageType.OME_ZARR
        suffixes = Path(path).suffixes
        if len(suffixes) > 2:
            if suffixes[-3] == ".ome" and suffixes[-2] == ".zarr" and suffixes[-1] == ".zip":
                return ImageType.OME_ZARR

        if len(suffixes) > 1:
            if suffixes[-2] == ".ome" and suffixes[-1] == ".zarr":
                return ImageType.OME_ZARR
            if suffixes[-2] == ".nii" and suffixes[-1] == ".gz":
                return ImageType.NIFTI
        if suffixes[-1] == ".zarr":
            return ImageType.ZARR
        elif suffixes[-1] == ".nii":
            return ImageType.NIFTI
        else:
            return ImageType.UNKNOWN

    def apply_transform(self, *tfms, ref_znimg):
        """return ZarrNii applying transform to floating image.
        this is a lazy function, doesn't do any work until you compute()
        on the returned dask array.
        """

        # tfms already has the transformations to apply,
        # just need the conversion to/from vox/ras at start and end
        tfms_to_apply = []
        tfms_to_apply.append(ref_znimg.affine)
        for tfm in tfms:
            tfms_to_apply.append(tfm)
        tfms_to_apply.append(self.affine.invert())

        # out image in space of ref
        interp_znimg = ref_znimg

        # perform interpolation on each block in parallel
        interp_znimg.darr = da.map_blocks(
            interp_by_block,
            ref_znimg.darr,
            dtype=np.float32,
            transforms=tfms_to_apply,
            flo_znimg=self,
        )

        return interp_znimg

    def apply_transform_ref_to_flo_indices(
        self, *tfms, ref_znimg, indices
    ):
        """takes indices in ref space, transforms, and provides
        indices in the flo space."""

        # tfms already has the transformations to apply, just
        # need the conversion to/from vox/ras at start and end
        tfms_to_apply = []
        tfms_to_apply.append(ref_znimg.affine)
        for tfm in tfms:
            tfms_to_apply.append(tfm)
        tfms_to_apply.append(self.affine.invert())

        # here we use indices as vectors (indices should be 3xN array),
        # we add ones to make 4xN so we can matrix multiply

        homog = np.ones((1, indices.shape[1]))
        xfm_vecs = np.vstack((indices, homog))

        # apply tfms_to_apply one at a time (will need to edit this for warps)
        for tfm in tfms_to_apply:
            xfm_vecs = tfm.apply_transform(xfm_vecs)

        # now we should have vecs in space of ref
        return xfm_vecs[:3, :]

    def apply_transform_flo_to_ref_indices(
        self, *tfms, ref_znimg, indices
    ):
        """takes indices in flo space, transforms, and
        provides indices in the ref space."""

        # tfms already has the transformations to apply,
        # just need the conversion to/from vox/ras at start and end
        tfms_to_apply = []
        tfms_to_apply.append(self.affine)
        for tfm in tfms:
            tfms_to_apply.append(tfm)
        tfms_to_apply.append(ref_znimg.affine.invert())

        homog = np.ones((1, indices.shape[1]))
        xfm_vecs = np.vstack((indices, homog))

        # apply tfms_to_apply one at a time
        for tfm in tfms_to_apply:
            xfm_vecs = tfm.apply_transform(xfm_vecs)

        # now we should have vecs in space of ref
        return xfm_vecs[:3, :]

    def get_bounded_subregion(self, points: np.array):
        """
        Uses the extent of points, along with the shape of the dask array,
        to return a subregion of the dask array that contains the points,
        along with the grid points corresponding to the indices from the
        uncropped dask array (to be used for interpolation).

        if the points go outside the domain of the dask array, then it is
        capped off with floor or ceil.

        points are Nx3 or Nx4 coordinates (with/without homog coord)
        darr is the floating image dask array

        We use compute() on the floating dask array to immediately get it
        since dask doesn't support nd fancy indexing yet that
        interpn seems to use
        """

        pad = 1
        min_extent = np.floor(points.min(axis=1)[:3] - pad).astype("int")
        max_extent = np.ceil(points.max(axis=1)[:3] + pad).astype("int")

        clip_min = np.zeros(min_extent.shape)
        clip_max = np.array(
            [self.darr.shape[-3], self.darr.shape[-2], self.darr.shape[-1]]
        )

        min_extent = np.clip(min_extent, clip_min, clip_max)
        max_extent = np.clip(max_extent, clip_min, clip_max)

        # problematic if all points are outside the domain --
        # - if so then no need to interpolate,
        # just return None to indicate this block should be all zeros
        if (max_extent == min_extent).sum() > 0:
            return (None, None)

        subvol = self.darr[
            :,
            min_extent[0] : max_extent[0],
            min_extent[1] : max_extent[1],
            min_extent[2] : max_extent[2],
        ].compute()

        # along with grid points for interpolation
        grid_points = (
            np.arange(min_extent[0], max_extent[0]),
            np.arange(min_extent[1], max_extent[1]),
            np.arange(min_extent[2], max_extent[2]),
        )

        return (grid_points, subvol)

    def as_Nifti1Image(self, filename, **kwargs):

        return nib.Nifti1Image(self.darr.squeeze(), matrix=self.affine)

    def to_nifti(self, filename):
        """
        Save the current ZarrNii instance to a NIfTI file.

        Parameters:
            filename (str): Output path for the NIfTI file.
        """
        # Reorder data to match NIfTI's expected XYZ order if necessary
        if self.axes_order == "ZYX":
            data = da.moveaxis(self.darr, (0, 1, 2, 3), (0, 3, 2, 1)).compute()  # Reorder to XYZ
            affine = self.reorder_affine_for_xyz(self.affine.matrix)  # Reorder affine to match
        else:
            data = self.darr.compute()
            affine = self.affine.matrix  # No reordering needed


        # Create the NIfTI image
        nii_img = nib.Nifti1Image(data[0], affine)  # Remove the channel dimension for NIfTI

        # Save the NIfTI file
        nib.save(nii_img, filename)



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
            out_darr = da.moveaxis(self.darr, (0, 1, 2, 3), (0, 3, 2, 1))  # Reorder to ZYX
         #   flip_xfm = np.diag((-1, -1, -1, 1))  # Apply flips for consistency
            #out_affine = flip_xfm @ self.affine
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
        group.attrs["orientation"] = affine_to_orientation(out_affine) # Write current orientation


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
        # adjust darr using slicing with bbox indices
        # bbox_min and bbox_max are tuples

        if ras_coords:
            #get vox coords by using ras2vox
            bbox_min = np.round(self.affine.invert() @ np.array(bbox_min))
            bbox_max = np.round(self.affine.invert() @ np.array(bbox_max))
            bbox_min = tuple(bbox_min[:3].flatten())
            bbox_max = tuple(bbox_max[:3].flatten())

        darr_cropped = self.darr[:,
                bbox_min[0]:bbox_max[0],
                bbox_min[1]:bbox_max[1],
                bbox_min[2]:bbox_max[2]]

        trans_vox = np.eye(4,4)
        trans_vox[:3,3] = bbox_min

        new_affine = self.affine @ trans_vox

        return ZarrNii.from_darr(darr_cropped,affine=new_affine,axes_order=self.axes_order)


    def get_bounding_box_around_label(self,label_number, padding=0, ras_coords=False):

        indices = da.argwhere(self.darr==label_number).compute()

        # Compute the minimum and maximum extents in each dimension
        bbox_min = indices.min(axis=0).reshape((4,1))[1:] - padding
        bbox_max = indices.max(axis=0).reshape((4,1))[1:] + 1 + padding

        #clip the data in case padding puts it out of bounds
        bbox_min = np.clip(bbox_min,np.zeros((3,1)),np.array(self.darr.shape[1:]).reshape(3,1))
        bbox_max = np.clip(bbox_max,np.zeros((3,1)),np.array(self.darr.shape[1:]).reshape(3,1))

        if ras_coords:
            bbox_min = self.affine @ np.array(bbox_min)
            bbox_max = self.affine @ np.array(bbox_max)

        return (bbox_min,bbox_max)




    def downsample(self,along_x=1,along_y=1,along_z=1,level=None,z_level_offset=-2):
        """ Downsamples by local mean. Can either specify along_x,along_y,along_z, 
            or specify the level, and the downsampling factors will be calculated, 
            taking into account the z_level_offset."""


        if level is not None:
            along_x = 2**level
            along_y = 2**level
            level_z = max(level+z_level_offset,0)
            along_z = 2**level_z
        

        if self.axes_order=='XYZ':
            axes ={0:1,
                                                        1: along_x,
                                                        2: along_y,
                                                        3: along_z}
        else:
            axes ={0:1,
                                                        1: along_z,
                                                        2: along_y,
                                                        3: along_x}

        agg_func = np.mean

        #coarsen performs a reduction in a local neighbourhood, defined by axes
        #TODO: check if astype is needed..
#        darr_scaled = da.coarsen(agg_func,x=self.darr.astype(dtype),axes=axes, trim_excess=True)
        darr_scaled = da.coarsen(agg_func,x=self.darr,axes=axes, trim_excess=True)

        #we need to also update the affine, scaling by the ds_factor
        scaling_matrix = np.diag((along_x,along_y,along_z,1))
        new_affine = scaling_matrix @ self.affine

        return ZarrNii.from_darr(darr_scaled,affine=new_affine,axes_order=self.axes_order)


    def __get_upsampled_chunks(self, target_shape,return_scaling=True):
        """ given a target shape, this calculates new chunking 
        by scaling the input chunks by a scaling factor, adjusting the
        last chunk if needed. This can be used for upsampling the data, 
        or ensuring 1-1 correspondence between downsampled and upsampled arrays"""

        new_chunks = []
        scaling = []

        for dim, (orig_shape, orig_chunks, new_shape) in enumerate(zip(self.darr.shape, self.darr.chunks, target_shape)):
            # Calculate the scaling factor for this dimension
            scaling_factor = new_shape / orig_shape

            # Scale each chunk size and round to get an initial estimate
            scaled_chunks = [int(round(chunk * scaling_factor)) for chunk in orig_chunks]
            total = sum(scaled_chunks)
            
            # Adjust the chunks to ensure they sum up to the target shape exactly
            diff = new_shape - total
            if diff != 0:
                # Correct rounding errors by adjusting the last chunk size in the dimension
                scaled_chunks[-1] += diff

            new_chunks.append(tuple(scaled_chunks))
            scaling.append(scaling_factor)
        if return_scaling:
            return (tuple(new_chunks),scaling)
        else:
            return new_chunks


    def divide_by_downsampled(self,znimg_ds):
        """takes current darr and divides by another darr (which is a downsampled version)"""

        #first, given the chunks of the downsampled darr and the high-res shape, 
        # we get the upsampled chunks (ie these chunks then have 1-1 correspondence
        # with downsampled
        target_chunks = znimg_ds.__get_upsampled_chunks(self.darr.shape,return_scaling=False)

        #we rechunk our original high-res image to that chunking 
        darr_rechunk = self.darr.rechunk(chunks=target_chunks)
        
        #now, self.darr and znimg_ds.darr have matching blocks
        #so we can use map_blocks to perform operation

        def block_zoom_and_divide_by(x1,x2,block_info):
            """ this zooms x2 to the size of x1, then does x1 / x2"""
            # get desired scaling by x2 to x1

            scaling = tuple(n_1 / n_2 for n_1,n_2 in zip(x1.shape,x2.shape))

            return x1 / zoom(x2,scaling,order=1,prefilter=False)

        darr_div = da.map_blocks(
                    block_zoom_and_divide_by,
                    darr_rechunk,
                    znimg_ds.darr,
                    dtype=self.darr.dtype)


        return ZarrNii(darr_div,self.affine,self.axes_order)


    def upsample(self,along_x=1,along_y=1,along_z=1,to_shape=None):
        """ upsamples with scipy.ndimage.zoom
            specify either along_x/along_y/along_z, or the target
            shape with to_shape=(c,z,y,x) or (c,x,y,z) if nifti axes
            
            note: this doesn't work yet if self has axes_order=='XYZ'"""
        
        
        #we run map_blocks with chunk sizes modulated by upsampling rate
        if to_shape == None:
            if self.axes_order == 'XYZ':
                scaling = (1,along_x, along_y, along_z)
            else:
                scaling = (1,along_z, along_y, along_x)

            chunks_out = tuple(tuple(c * scale for c in chunks_i) for chunks_i, scale in zip(self.darr.chunks, scaling))

        else:
            chunks_out,scaling = self.__get_upsampled_chunks(to_shape)

        def zoom_blocks(x,block_info):
            
            # get desired scaling by comparing input shape to output shape
            # block_info[None] is the output block info
            scaling = tuple(out_n / in_n for out_n,in_n in zip(block_info[None]['chunk-shape'],x.shape))

            return zoom(x,scaling,order=1,prefilter=False)


        darr_scaled = da.map_blocks(zoom_blocks,
                    self.darr,
                    dtype=self.darr.dtype,
                    chunks=chunks_out)


        #we need to also update the affine, scaling by the ds_factor
        if self.axes_order == 'XYZ':
            scaling_matrix = np.diag((1/scaling[1],1/scaling[2],1/scaling[3],1))
        else:
            scaling_matrix = np.diag((1/scaling[-1],1/scaling[-2],1/scaling[-3],1))
        new_affine = scaling_matrix @ self.affine

        return ZarrNii.from_darr(darr_scaled.rechunk(),affine=new_affine,axes_order=self.axes_order)



    @staticmethod
    def get_max_level(path,storage_options=None):
        
        if isinstance(path,MutableMapping):
            store = path
        else:
            store = fsspec.get_mapper(path,storage_options=storage_options)

        # Open the Zarr group
        group = zarr.open(store, mode='r')

        # Access the multiscale metadata
        multiscales = group.attrs.get('multiscales', [])

        # Get the number of levels (number of arrays in the multiscale hierarchy)
        if multiscales:
            max_level = len(multiscales[0]['datasets'])-1
            return max_level
        else:
            print("No multiscale levels found.")
            return None


    @staticmethod
    def get_level_and_downsampling_kwargs(ome_zarr_path,level,z_level_offset=-2,storage_options=None):

        max_level = ZarrNii.get_max_level(ome_zarr_path,storage_options=storage_options)
        if level > max_level: #if we want to ds more than ds_levels in pyramid
            level_xy = level-max_level
            level_z = max(level+z_level_offset,0)
            level = max_level
        else:
            level_xy=0
            level_z=max(level+z_level_offset,0)

        print(f'level: {level}, level_xy: {level_xy}, level_z: {level_z}, max_level: {max_level}')
        if level_xy>0 or level_z>0:
            do_downsample=True
        else:
            do_downsample=False
        return (level,do_downsample,{'along_x':2**level_xy,'along_y':2**level_xy,'along_z':2**level_z})

        
#-- inline functions

def interp_by_block(
    x,
    transforms: list[Transform],
    flo_znimg: ZarrNii,
    block_info=None,
    interp_method="linear",
):
    """
    main idea here is we take coordinates from the current block (ref image)
    transform them, then in that transformed space, then interpolate the
    floating image intensities

    since the floating image is a dask array, we need to just load a
    subset of it -- we use the bounds of the points in the transformed
     space to define what range of the floating image we load in
    """
    arr_location = block_info[0]["array-location"]
    xv, yv, zv = np.meshgrid(
        np.arange(arr_location[-3][0], arr_location[-3][1]),
        np.arange(arr_location[-2][0], arr_location[-2][1]),
        np.arange(arr_location[-1][0], arr_location[-1][1]),
        indexing="ij",
    )

    # reshape them into a vectors (x,y,z,1) for each point, so we can
    # matrix multiply
    xvf = xv.reshape((1, np.product(xv.shape)))
    yvf = yv.reshape((1, np.product(yv.shape)))
    zvf = zv.reshape((1, np.product(zv.shape)))
    homog = np.ones(xvf.shape)

    xfm_vecs = np.vstack((xvf, yvf, zvf, homog))

    # apply transforms one at a time (will need to edit this for warps)
    for tfm in transforms:
        xfm_vecs = tfm.apply_transform(xfm_vecs)

    # then finally interpolate those points on the template dseg volume
    # need to interpolate for each channel

    interpolated = np.zeros(x.shape)

    # find bounding box required for flo vol
    (grid_points, flo_vol) = flo_znimg.get_bounded_subregion(xfm_vecs)
    if grid_points is None and flo_vol is None:
        # points were fully outside the floating image, so just return zeros
        return interpolated
    else:
        for c in range(flo_vol.shape[0]):
            interpolated[c, :, :, :] = (
                interpn(
                    grid_points,
                    flo_vol[c, :, :, :],
                    xfm_vecs[:3, :].T,  #
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
    axis_labels = ['R', 'A', 'S']
    flipped_labels = ['L', 'P', 'I']

    orientation = []
    for axis, direction in orient:
        axis = int(axis)
        if direction == 1:
            orientation.append(axis_labels[axis])
        else:
            orientation.append(flipped_labels[axis])

    return ''.join(orientation)

