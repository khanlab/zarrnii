from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .transform import Transform

from pathlib import Path

import dask.array as da
import nibabel as nib
import numpy as np
import zarr
from attrs import define
from dask.diagnostics import ProgressBar
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image

from .enums import ImageType
from .transform import interp_by_block


@define
class ZarrNii:
    darr: da.Array
    ras2vox: Transform = None
    vox2ras: Transform = None
    axes_nifti: bool = (
        False  # set to true if the axes are ordered for NIFTI (X,Y,Z,C,T)
    )


    @classmethod
    def from_path_as_ref(
        cls, path, level=0, channels=[0], chunks='auto', zooms=None
    ):
        """ref image dont need data,  just the shape and affine"""

        from .transform import Transform

        img_type = cls.check_img_type(path)
        if img_type is ImageType.OME_ZARR:
            darr_base = da.from_zarr(path, component=f"/{level}")[
                channels, :, :, :
            ]
            axes_nifti = False
        elif img_type is ImageType.NIFTI:
            darr_base = da.from_array(nib.load(path).get_fdata())
            axes_nifti = True
        else:
            raise TypeError("Unsupported image type for ZarrNii")

        vox2ras = Transform.vox2ras_from_image(path)
        ras2vox = Transform.ras2vox_from_image(path)

        out_shape = [
            len(channels),
            darr_base.shape[-3],
            darr_base.shape[-2],
            darr_base.shape[-1],
        ]

        if zooms is not None:
            # zooms sets the target spacing in xyz

            in_zooms = np.diag(vox2ras.affine)[:3]
            nvox_scaling_factor = in_zooms / zooms

            out_shape[1:] = np.floor(out_shape[1:] * nvox_scaling_factor)
            # adjust affine too
            np.fill_diagonal(vox2ras.affine[:3, :3], zooms)

        ras2vox.affine = np.linalg.inv(vox2ras.affine)

        darr_empty = da.empty(
            darr_base,
            shape=out_shape,
            chunks=chunks,
        )

        return cls(
            darr_empty, ras2vox=ras2vox, vox2ras=vox2ras, axes_nifti=axes_nifti
        )

    @classmethod
    def from_path(cls, path, level=0, channels=[0], chunks="auto", rechunk=False):
        """returns a dask array whether a nifti or ome_zarr is provided"""
        from .transform import Transform

        img_type = cls.check_img_type(path)
        if img_type is ImageType.OME_ZARR:
            darr = da.from_zarr(path, component=f"/{level}")[channels, :, :, :]
            if rechunk:
                darr = darr.rechunk(chunks)
            zi = zarr
            axes_nifti = False
        elif img_type is ImageType.NIFTI:
            darr = da.from_array(
                np.expand_dims(nib.load(path).get_fdata(), axis=0),
                chunks=chunks,
            )
            axes_nifti = True
        else:
            raise TypeError("Unsupported image type for ZarrNii")

        return cls(
            darr,
            ras2vox=Transform.ras2vox_from_image(path,level=level),
            vox2ras=Transform.vox2ras_from_image(path,level=level),
            axes_nifti=axes_nifti,
        )

    @classmethod
    def from_darr(cls, darr, vox2ras=np.eye(4), axes_nifti=False):
        from .transform import Transform

        ras2vox = np.linalg.inv(vox2ras)

        return cls(
            darr,
            ras2vox=Transform.affine_ras_from_array(ras2vox),
            vox2ras=Transform.affine_ras_from_array(vox2ras),
            axes_nifti=axes_nifti,
        )

    @staticmethod
    def check_img_type(path) -> ImageType:
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
        tfms_to_apply.append(ref_znimg.vox2ras)
        for tfm in tfms:
            tfms_to_apply.append(tfm)
        tfms_to_apply.append(self.ras2vox)

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
        tfms_to_apply.append(ref_znimg.vox2ras)
        for tfm in tfms:
            tfms_to_apply.append(tfm)
        tfms_to_apply.append(self.ras2vox)

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
        tfms_to_apply.append(self.vox2ras)
        for tfm in tfms:
            tfms_to_apply.append(tfm)
        tfms_to_apply.append(ref_znimg.ras2vox)

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

        return nib.Nifti1Image(self.darr.squeeze(), affine=self.vox2ras.affine)


    def to_nifti(self, filename, **kwargs):

        if self.axes_nifti: #this means it was read-in as a NIFTI, so we write out as normal
            out_darr = self.darr.squeeze()
            affine = self.vox2ras.affine

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
            affine = self.vox2ras.affine @ reorder_xfm

        out_nib = nib.Nifti1Image(out_darr, affine=affine)
        out_nib.to_filename(filename)

    def to_ome_zarr(
        self, filename, max_layer=4, scaling_method="local_mean", **kwargs
    ):
        #the affine specifies how to go from the darr to nifti RAS space

        #in creating affine for zarr, we had to apply scale, translation
        # then reorder, flip

        # we can apply steps


        offset=self.vox2ras.affine[:3,3]

        if self.axes_nifti:
            #we have a nifti image -- need to apply the transformations (reorder, flip) to the
            # data in the OME_Zarr, so it is consistent.
            out_darr=da.moveaxis(self.darr, (0, 1, 2, 3), (0, 3, 2, 1))

            reorder_xfm = np.eye(4)
            reorder_xfm[:3, :3] = np.flip(
                reorder_xfm[:3, :3], axis=0
            )  # reorders z-y-x to x-y-z and vice versa

            #out_affine = self.vox2ras.affine
            flip_xfm = np.diag((-1,-1,-1,1))
            out_affine = flip_xfm @ self.vox2ras.affine
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

            out_affine = flip_xfm @ reorder_xfm @ self.vox2ras.affine

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

        scaler = Scaler(max_layer=max_layer, method=scaling_method)

        with ProgressBar():
            write_image(
                image=out_darr.rechunk(),
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
            bbox_min = np.round(self.ras2vox @ np.array(bbox_min))
            bbox_max = np.round(self.ras2vox @ np.array(bbox_max))
            bbox_min = tuple(bbox_min[:3].flatten())
            bbox_max = tuple(bbox_max[:3].flatten())

        darr_cropped = self.darr[:,
                bbox_min[0]:bbox_max[0],
                bbox_min[1]:bbox_max[1],
                bbox_min[2]:bbox_max[2]]

        trans_vox = np.eye(4,4)
        trans_vox[:3,3] = bbox_min

        new_vox2ras = self.vox2ras.affine @ trans_vox

        return ZarrNii.from_darr(darr_cropped,vox2ras=new_vox2ras,axes_nifti=self.axes_nifti)


    def get_bounding_box_around_label(self,label_number, padding=0, ras_coords=False):

        indices = da.argwhere(self.darr==label_number).compute()

        # Compute the minimum and maximum extents in each dimension
        bbox_min = indices.min(axis=0).reshape((4,1))[1:] - padding
        bbox_max = indices.max(axis=0).reshape((4,1))[1:] + 1 + padding

        #clip the data in case padding puts it out of bounds
        bbox_min = np.clip(bbox_min,np.zeros((3,1)),np.array(self.darr.shape[1:]).reshape(3,1))
        bbox_max = np.clip(bbox_max,np.zeros((3,1)),np.array(self.darr.shape[1:]).reshape(3,1))

        if ras_coords:
            bbox_min = self.vox2ras @ np.array(bbox_min)
            bbox_max = self.vox2ras @ np.array(bbox_max)

        return (bbox_min,bbox_max)




    def downsample(self,along_x=1,along_y=1,along_z=1):
        """ downsamples by local mean"""

        if self.axes_nifti:
            axes ={0:1,
                                                        1: along_x,
                                                        2: along_y,
                                                        3: along_z}
        else:
            axes ={0:1,
                                                        1: along_z,
                                                        2: along_y,
                                                        3: along_x}

        #coarsen performs a reduction in a local neighbourhood, defined by axes
        darr_scaled = da.coarsen(np.mean,x=self.darr,axes=axes, trim_excess=True)


        #we need to also update the affine, scaling by the ds_factor
        scaling_matrix = np.diag((along_x,along_y,along_z,1))
        new_vox2ras = scaling_matrix @ self.vox2ras.affine

        return ZarrNii.from_darr(darr_scaled,vox2ras=new_vox2ras,axes_nifti=self.axes_nifti)


    """ WIP
    def set_zooms(self,along_x=1,along_y=1,along_z=1):

        if self.axes_nifti:
            new_shape = [self.darr.shape[1]

    new_shape = tuple(int(dim * zoom) for dim, zoom in zip(dask_array.shape, zoom_factors))


        #we need to also update the affine, scaling by the ds_factor
        scaling_matrix = np.diag((],1))
        new_vox2ras = scaling_matrix @ self.vox2ras.affine


        return ZarrNii.from_darr(darr_scaled,vox2ras=new_vox2ras,axes_nifti=self.axes_nifti)
        """



