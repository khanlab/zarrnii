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
    def from_path_as_ref(
        cls, path, level=0, channels=[0], chunks='auto', zooms=None
    ):
        """ref image dont need data,  just the shape and affine"""


        img_type = cls.check_img_type(path)
        if img_type is ImageType.OME_ZARR:
            darr_base = da.from_zarr(path, component=f"/{level}")[
                channels, :, :, :
            ]
            axes_order = 'ZYX'
            affine = affine_from_zarr(path)

        elif img_type is ImageType.NIFTI:
            darr_base = da.from_array(nib.load(path).get_fdata())
            axes_order = 'XYZ'
            affine = affine_from_nii(path)
        else:
            raise TypeError(f"Unsupported image type for ZarrNii: {path}")


        out_shape = [
            len(channels),
            darr_base.shape[-3],
            darr_base.shape[-2],
            darr_base.shape[-1],
        ]

        if zooms is not None:
            # zooms sets the target spacing in xyz

            in_zooms = np.diag(affine.matrix)[:3]
            nvox_scaling_factor = in_zooms / zooms

            out_shape[1:] = np.floor(out_shape[1:] * nvox_scaling_factor)
            # adjust matrix too
            np.fill_diagonal(affine.matrix[:3, :3], zooms)


        darr_empty = da.empty(
            darr_base,
            shape=out_shape,
            chunks=chunks,
        )

        return cls(
            darr_empty, affine=affine, axes_order=axes_order
        )

  
    @classmethod
    def from_darr(cls, darr, affine=np.eye(4), axes_order='ZYX'):

        affine = AffineTransform.from_array(affine)

        return cls(
            darr,
            affine=affine,
            axes_order=axes_order,
        )

    @classmethod
    def from_nifti(cls, path, chunks="auto" ):

        darr = da.from_array(
            np.expand_dims(nib.load(path).get_fdata(), axis=0),
            chunks=chunks,
        )
        axes_order = 'XYZ'
        affine = affine_from_nii(path)

        return cls(
                darr,
                affine=affine,
                axes_order=axes_order,
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
        in_orientation="IPL",
    ):
        """
        Reads in an OME-Zarr file as a ZarrNii image. Performs lazy downsampling if the
        specified level isn't stored in the pyramid. Also downsamples Z, adjusted by
        z_level_offset (since Z typically has lower resolution than XY).

        Populates OME-Zarr metadata fields and constructs the affine matrix based on
        the input orientation.
        """
        # Determine the level and whether downsampling is required
        level, do_downsample, downsampling_kwargs = cls.get_level_and_downsampling_kwargs(
            path, level, z_level_offset, storage_options=storage_options
        )

        # Load the data array for the specified level and channels
        darr = da.from_zarr(path, component=f"/{level}", storage_options=storage_options)[
            channels, :, :, :
        ]

        if rechunk:
            darr = darr.rechunk(chunks)

        # Open the Zarr metadata
        store = zarr.open(path, mode="r")
        multiscales = store.attrs.get("multiscales", [{}])
        datasets = multiscales[0].get("datasets", [{}])
        coordinate_transformations = datasets[level].get("coordinateTransformations", [])
        axes = multiscales[0].get("axes", [])
        omero = store.attrs.get("omero", {})

        # Default axes order for OME-Zarr
        axes_order = "ZYX"

        # Construct the affine matrix and update it for the input orientation
        affine = cls.construct_affine(coordinate_transformations, in_orientation)
        print('affine created in from_ome_zarr()')
        print(affine)
        if do_downsample:
            # Return downsampled instance
            return cls(
                darr,
                affine=AffineTransform.from_array(affine),
                axes_order=axes_order,
                axes=axes,
                coordinate_transformations=coordinate_transformations,
                omero=omero,
            ).downsample(**downsampling_kwargs)
        else:
            # Return instance without downsampling
            return cls(
                darr,
                affine=AffineTransform.from_array(affine),
                axes_order=axes_order,
                axes=axes,
                coordinate_transformations=coordinate_transformations,
                omero=omero,
            )

    @staticmethod
    def reorder_affine(affine, in_orientation):
        """
        Reorders and flips the affine matrix based on the input orientation.

        Parameters:
            affine (np.ndarray): Initial affine matrix.
            in_orientation (str): Input orientation (e.g., 'RAS').

        Returns:
            np.ndarray: Reordered and flipped affine matrix.
        """
        axis_map = {'R': 0, 'L': 0, 'A': 1, 'P': 1, 'S': 2, 'I': 2}
        sign_map = {'R': 1, 'L': -1, 'A': 1, 'P': -1, 'S': 1, 'I': -1}

        input_axes = [axis_map[ax] for ax in in_orientation]
        input_signs = [sign_map[ax] for ax in in_orientation]

        reordered_affine = np.zeros_like(affine)
        for i, (axis, sign) in enumerate(zip(input_axes, input_signs)):
            reordered_affine[i, :3] = sign * affine[axis, :3]
            reordered_affine[i, 3] = sign * affine[axis, 3]
        reordered_affine[3, :] = affine[3, :]  # Preserve homogeneous row

        return reordered_affine


    @staticmethod
    def construct_affine(coordinate_transformations, in_orientation):
        """
        Constructs the affine matrix based on OME-Zarr coordinate transformations
        and adjusts it for the input orientation.

        Parameters:
            coordinate_transformations (list): Coordinate transformations from OME-Zarr metadata.
            in_orientation (str): Input orientation (e.g., 'RAS').

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

        print(f'affine constructed from {in_orientation}, and scales {scales}, translations {translations}')
        print(affine)

        # Reorder the affine matrix for the input orientation
        return ZarrNii.reorder_affine(affine, in_orientation)



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

        return nib.Nifti1Image(self.darr.squeeze(), matrix=self.affine.matrix)


    def to_nifti(self, filename, **kwargs):

        if self.axes_order == 'XYZ': #this means it was read-in as a NIFTI, so we write out as normal
            out_darr = self.darr.squeeze()
            affine = self.affine.matrix

        else: #was read-in as a Zarr, so we reorder/flip data, and adjust affine accordingly

            out_darr = da.moveaxis(self.darr, (0, 1, 2, 3), (0, 3, 2, 1)).squeeze()
            #adjust affine accordingly

            # reorder_xfm -- changes from z,y,x to x,y,z ordering
            reorder_xfm = np.eye(4)
            reorder_xfm[:3, :3] = np.flip(
                reorder_xfm[:3, :3], axis=0
            )  # reorders z-y-x to x-y-z and vice versa

            flip_xfm = np.diag((-1,-1,-1,1))

            #right-multiply to reorder cols
            affine = self.affine.matrix @ reorder_xfm

        out_nib = nib.Nifti1Image(out_darr, affine=affine)
        with ProgressBar():
            out_nib.to_filename(filename)

    def to_ome_zarr(
        self, filename, max_layer=4, scaling_method="local_mean", **kwargs
    ):
        #the affine specifies how to go from the darr to nifti RAS space

        #in creating affine for zarr, we had to apply scale, translation
        # then reorder, flip

        # we can apply steps


        offset=self.affine.matrix[:3,3]

        if self.axes_order == 'XYZ':
            #we have a nifti image -- need to apply the transformations (reorder, flip) to the
            # data in the OME_Zarr, so it is consistent.
            out_darr=da.moveaxis(self.darr, (0, 1, 2, 3), (0, 3, 2, 1))

            reorder_xfm = np.eye(4)
            reorder_xfm[:3, :3] = np.flip(
                reorder_xfm[:3, :3], axis=0
            )  # reorders z-y-x to x-y-z and vice versa

            flip_xfm = np.diag((-1,-1,-1,1))
            out_affine = flip_xfm @ self.affine.matrix
            # voxdim needs to be Z Y Z
            voxdim=np.flip(np.diag(out_affine)[:3])
        else:
            out_darr = self.darr

            # adjust affine accordingly
            # reorder_xfm -- changes from z,y,x to x,y,z ordering
            reorder_xfm = np.eye(4)
            reorder_xfm[:3, :3] = np.flip(
                reorder_xfm[:3, :3], axis=0
            )  # reorders z-y-x to x-y-z and vice versa

            flip_xfm = np.diag((-1,-1,-1,1))

            out_affine = flip_xfm @ reorder_xfm @ self.affine.matrix

            voxdim=np.diag(out_affine)[:3]

        coordinate_transformations = []
        # for each resolution (dataset), we have a list of dicts,
        # transformations to apply..
        # in this case just a single one (scaling by voxel size)

        for layer in range(max_layer + 1):
            coord_transform_layer=[]
            #add scale
            coord_transform_layer.append(
                    {
                        "scale": [
                            1,
                            voxdim[0],
                            (2**layer) * voxdim[1],
                            (2**layer) * voxdim[2],
                        ],  # image-pyramids in XY only
                        "type": "scale",
                    }
                    )
            #add translation
            coord_transform_layer.append({
                        "translation": [
                            0,
                            offset[0],
                            offset[1] / (2**layer),
                            offset[2] / (2**layer),
                        ],  # image-pyramids in XY only
                        "type": "translation",
                    }
                    )

            coordinate_transformations.append(coord_transform_layer)

        axes = [{"name": "c", "type": "channel"}] + [
            {"name": ax, "type": "space", "unit": "micrometer"}
            for ax in ["z", "y", "x"]
        ]

        store = zarr.storage.FSStore(
            filename, dimension_separator="/", mode="w"
        )
        group = zarr.group(store, overwrite=True)

        if max_layer ==0:
            scaler = None
        else:
            scaler = Scaler(max_layer=max_layer, method=scaling_method)

        with ProgressBar():
            write_image(
                image=out_darr,
                group=group,
                scaler=scaler,
                coordinate_transformations=coordinate_transformations,
                axes=axes,
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

        new_affine = self.affine.matrix @ trans_vox

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
        new_affine = scaling_matrix @ self.affine.matrix

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
        new_affine = scaling_matrix @ self.affine.matrix

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

        

# -- inline helper functions

def affine_from_zarr(in_zarr_path: str, level=0) -> AffineTransform:
    # read coordinate transform from ome-zarr
    zi = zarr.open(in_zarr_path, mode='r')
    attrs = zi["/"].attrs.asdict()
    multiscale = 0  # first multiscale image
    transforms = attrs["multiscales"][multiscale]["datasets"][level][
        "coordinateTransformations"
    ]

    affine = np.eye(4)

    #apply each ome zarr transform sequentially
    for transform in transforms:
    # (for now, we just assume the transformations will be called scale and translation)
        if transform['type'] == "scale":
            scaling_zyx = transform["scale"][-3:]
            scaling_xfm = np.diag(np.hstack((scaling_zyx,1)))
            affine = scaling_xfm @ affine
        elif transform['type'] == "translation":
            translation_xfm = np.eye(4)
            translation_xfm[:3,3] = transform["translation"][-3:]
            affine = translation_xfm @ affine
        
    # reorder_xfm -- changes from z,y,x to x,y,z ordering
    reorder_xfm = np.eye(4)
    reorder_xfm[:3, :3] = np.flip(
        reorder_xfm[:3, :3], axis=0
    )  # reorders z-y-x to x-y-z and vice versa

    affine = reorder_xfm @ affine
    
    flip_xfm = np.diag((-1,-1,-1,1))
    affine = flip_xfm @ affine
    
    return AffineTransform.from_array(affine)

def affine_from_nii(in_nii_path: str) -> AffineTransform:
    return AffineTransform.from_array(nib.load(in_nii_path).affine)




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

