from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .transform import TransformSpec

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
class DaskImage:
    darr: da.Array
    ras2vox: TransformSpec = None
    vox2ras: TransformSpec = None
    axes_nifti: bool = (
        False  # set to true if the axes are ordered for NIFTI (X,Y,Z,C,T)
    )

    attrs = {}  # TODO: need to store ome-zarr attributes somehow?

    @classmethod
    def from_path_as_ref(
        cls, path, level=0, channels=[0], chunks=(50, 50, 50), zooms=None
    ):
        """ref image dont need data,  just the shape and affine"""

        from .transform import TransformSpec

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

        vox2ras = TransformSpec.vox2ras_from_image(path)
        ras2vox = TransformSpec.ras2vox_from_image(path)

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
        from .transform import TransformSpec

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
            ras2vox=TransformSpec.ras2vox_from_image(path),
            vox2ras=TransformSpec.vox2ras_from_image(path),
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

    def apply_transform(self, *tfm_specs, ref_dimg):
        """return DaskImage applying transform to floating image.
        this is a lazy function, doesn't do any work until you compute()
        on the returned dask array.
        """

        # transform specs already has the transformations to apply, 
        #just need the conversion to/from vox/ras at start and end
        transforms = []
        transforms.append(ref_dimg.vox2ras)
        for tfm in tfm_specs:
            transforms.append(tfm)
        transforms.append(self.ras2vox)

        # out image in space of ref
        interp_dimg = ref_dimg

        # perform interpolation on each block in parallel
        interp_dimg.darr = da.map_blocks(
            interp_by_block,
            ref_dimg.darr,
            dtype=np.float32,
            transform_specs=transforms,
            flo_dimg=self,
        )

        return interp_dimg

    def apply_transform_ref_to_flo_indices(
        self, *tfm_specs, ref_dimg, indices
    ):
        """takes indices in ref space, transforms, and provides indices in the flo space."""

        # transform specs already has the transformations to apply, just need the conversion to/from vox/ras at start and end
        transforms = []
        transforms.append(ref_dimg.vox2ras)
        for tfm in tfm_specs:
            transforms.append(tfm)
        transforms.append(self.ras2vox)

        # --- here we use indices as vectors (indices should be 3xN array), we add ones to make 4xN
        #  so we can matrix multiply

        homog = np.ones((1, indices.shape[1]))
        xfm_vecs = np.vstack((indices, homog))

        # apply transforms one at a time (will need to edit this for warps)
        for tfm in transforms:
            xfm_vecs = tfm.apply_transform(xfm_vecs)

        # now we should have vecs in space of ref
        return xfm_vecs[:3, :]

    def apply_transform_flo_to_ref_indices(
        self, *tfm_specs, ref_dimg, indices
    ):
        """takes indices in flo space, transforms, and provides indices in the ref space."""

        # transform specs already has the transformations to apply, just need the conversion to/from vox/ras at start and end
        transforms = []
        transforms.append(self.vox2ras)
        for tfm in tfm_specs:
            transforms.append(tfm)
        transforms.append(ref_dimg.ras2vox)

        # --- here we use indices as vectors (indices should be 3xN array), we add ones to make 4xN
        #  so we can matrix multiply

        homog = np.ones((1, indices.shape[1]))
        xfm_vecs = np.vstack((indices, homog))

        # apply transforms one at a time
        for tfm in transforms:
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
        since dask doesn't support nd fancy indexing yet that interpn seems to use
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

        # problematic if all points are outside the domain -- if so then no need to interpolate,
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
            # we need to convert to nifti convention (XYZ by reordering and negating)
            out_darr = (
                da.flip(da.moveaxis(self.darr, (0, 1, 2, 3), (0, 3, 2, 1)))
                .squeeze()
                .compute(**kwargs)
            )
            voxdim = np.diag(np.flip(self.vox2ras.affine[:3, :3], axis=0))

            voxdim = -voxdim[::-1]
            out_affine = np.diag(np.hstack((voxdim, 1)))

        out_nib = nib.Nifti1Image(out_darr, affine=out_affine)
        out_nib.to_filename(filename)

    def to_ome_zarr(
        self, filename, max_layer=4, scaling_method="local_mean", **kwargs
    ):
        if (
            self.axes_nifti
        ):  # double check to see if this is needed, add a test too..
            voxdim = np.diag(self.vox2ras.affine)[:3]

            # if the reference image came from nifti space, we need to swap axes ordering and flip
            if voxdim[0] < 0:
                out_darr = da.moveaxis(self.darr, (0, 1, 2, 3), (0, 3, 2, 1))
                voxdim = -voxdim[::-1]

            else:
                out_darr = da.flip(
                    da.moveaxis(self.darr, (0, 1, 2, 3), (0, 3, 2, 1))
                )
                voxdim = voxdim[::-1]
        else:
            voxdim = np.diag(np.flip(self.vox2ras.affine[:3, :3], axis=0))
            out_darr = self.darr
            voxdim = -voxdim

        coordinate_transformations = []
        # for each resolution (dataset), we have a list of dicts, transformations to apply..
        # in this case just a single one (scaling by voxel size)

        for l in range(max_layer + 1):
            coordinate_transformations.append(
                [
                    {
                        "scale": [
                            1,
                            voxdim[0],
                            (2**l) * voxdim[1],
                            (2**l) * voxdim[2],
                        ],  # image-pyramids in XY only
                        "type": "scale",
                    }
                ]
            )

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

    def crop_with_bounding_box(self, bbox):
        # adjust darr using slicing with bbox

        # adjust vox2ras and ras2vox (add offset)
        #        self.vox2ras[:3,3] = offset

        return dimg_cropped
