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
        cls, path, level=0, channels=[0], chunks=(50, 50, 50), zooms=None
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
            print("unknown image type")
            return None

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
            chunks=(len(channels), chunks[0], chunks[1], chunks[2]),
        )

        return cls(
            darr_empty, ras2vox=ras2vox, vox2ras=vox2ras, axes_nifti=axes_nifti
        )

    @classmethod
    def from_path(cls, path, level=0, channels=[0], chunks="auto"):
        """returns a dask array whether a nifti or ome_zarr is provided"""
        from .transform import Transform

        img_type = cls.check_img_type(path)
        if img_type is ImageType.OME_ZARR:
            darr = da.from_zarr(path, component=f"/{level}")[channels, :, :, :]
            zi = zarr
            axes_nifti = False
        elif img_type is ImageType.NIFTI:
            darr = da.from_array(
                np.expand_dims(nib.load(path).get_fdata(), axis=0),
                chunks=chunks,
            )
            axes_nifti = True
        else:
            print("unknown image type")
            return None

        return cls(
            darr,
            ras2vox=Transform.ras2vox_from_image(path),
            vox2ras=Transform.vox2ras_from_image(path),
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

    def apply_transform(self, *tfms, ref_dimg):
        """return ZarrNii applying transform to floating image.
        this is a lazy function, doesn't do any work until you compute()
        on the returned dask array.
        """

        # tfms already has the transformations to apply,
        # just need the conversion to/from vox/ras at start and end
        tfms_to_apply = []
        tfms_to_apply.append(ref_dimg.vox2ras)
        for tfm in tfms:
            tfms_to_apply.append(tfm)
        tfms_to_apply.append(self.ras2vox)

        # out image in space of ref
        interp_dimg = ref_dimg

        # perform interpolation on each block in parallel
        interp_dimg.darr = da.map_blocks(
            interp_by_block,
            ref_dimg.darr,
            dtype=np.float32,
            transforms=tfms_to_apply,
            flo_dimg=self,
        )

        return interp_dimg

    def apply_transform_ref_to_flo_indices(
        self, *tfms, ref_dimg, indices
    ):
        """takes indices in ref space, transforms, and provides
        indices in the flo space."""

        # tfms already has the transformations to apply, just
        # need the conversion to/from vox/ras at start and end
        tfms_to_apply = []
        tfms_to_apply.append(ref_dimg.vox2ras)
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
        self, *tfms, ref_dimg, indices
    ):
        """takes indices in flo space, transforms, and
        provides indices in the ref space."""

        # tfms already has the transformations to apply,
        # just need the conversion to/from vox/ras at start and end
        tfms_to_apply = []
        tfms_to_apply.append(self.vox2ras)
        for tfm in tfms:
            tfms_to_apply.append(tfm)
        tfms_to_apply.append(ref_dimg.ras2vox)

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

    def to_nifti(self, filename, **kwargs):
        if self.axes_nifti:
            out_darr = self.darr.squeeze().compute(**kwargs)
            out_affine = self.vox2ras.affine

        else:
            # we need to convert to nifti convention
            # (XYZ by reordering and negating)
            out_darr = (
                da.flip(da.moveaxis(self.darr, (0, 1, 2, 3), (0, 3, 2, 1)))
                .squeeze()
                .compute(**kwargs)
            )
            voxdim = np.diag(np.flip(self.vox2ras.affine[:3, :3], axis=0))

            voxdim = -voxdim[::-1]
            out_affine = np.diag(np.hstack((voxdim, 1)))
            #add back the offset offset 
            out_affine[:3,3] = -self.vox2ras.affine[:3,3]

        out_nib = nib.Nifti1Image(out_darr, affine=out_affine)
        out_nib.to_filename(filename)

    def to_ome_zarr(
        self, filename, max_layer=4, scaling_method="local_mean", **kwargs
    ):
        offset=self.vox2ras.affine[:3,3]

        if (
            self.axes_nifti
        ):  # double check to see if this is needed, add a test too..
            voxdim = np.diag(self.vox2ras.affine)[:3]

            # if the reference image came from nifti space, we need to
            # swap axes ordering and flip
            if voxdim[0] < 0:
                out_darr = da.moveaxis(self.darr, (0, 1, 2, 3), (0, 3, 2, 1))
                voxdim = -voxdim[::-1]
                offset=offset[::-1]

            else:
                out_darr = da.flip(
                    da.moveaxis(self.darr, (0, 1, 2, 3), (0, 3, 2, 1))
                )
                voxdim = voxdim[::-1]
                offset=offset[::-1]
        else:
            voxdim = np.diag(np.flip(self.vox2ras.affine[:3, :3], axis=0))
            out_darr = self.darr
            offset=offset[::-1]
            voxdim = -voxdim

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

    def crop_with_bounding_box(self, bbox_min, bbox_max):
        # adjust darr using slicing with bbox indices
        # bbox_min and bbox_max are tuples 
        darr_cropped = self.darr[:,
                bbox_min[0]:bbox_max[0],
                bbox_min[1]:bbox_max[1],
                bbox_min[2]:bbox_max[2]]

        # bbox is indices (ie voxels), so need to convert to ras


        offset_indices = np.array(bbox_min).reshape(3,1)        
        homog = np.ones((1, offset_indices.shape[1]))
        vecs = np.vstack((offset_indices, homog))

        xfm_vecs = self.vox2ras.apply_transform(vecs)

        offset_vox2ras = self.vox2ras.affine
        offset_vox2ras[:3,3]=-xfm_vecs[:3,0]

        return ZarrNii.from_darr(darr_cropped,vox2ras=offset_vox2ras,axes_nifti=self.axes_nifti)
