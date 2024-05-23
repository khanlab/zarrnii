from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dask_image import DaskImage

import nibabel as nib
import numpy as np
import zarr
from attrs import define
from dask.diagnostics import ProgressBar
from scipy.interpolate import interpn

from .enums import ImageType, TransformType


@define
class TransformSpec:
    tfm_type: TransformType
    affine: np.array = None
    disp_xyz: np.array = None
    disp_grid: np.array = None
    disp_ras2vox: np.array = None

    @classmethod
    def affine_ras_from_txt(cls, path, invert=False):
        affine = np.loadtxt(path)
        if invert:
            affine = np.linalg.inv(self.affine)

        return cls(TransformType.AFFINE_RAS, affine=affine)

    @classmethod
    def affine_ras_from_array(cls, affine, invert=False):
        if invert:
            affine = np.linalg.inv(affine)

        return cls(TransformType.AFFINE_RAS, affine=affine)

    @classmethod
    def displacement_from_nifti(cls, path):
        disp_nib = nib.load(path)
        disp_xyz = disp_nib.get_fdata().squeeze()
        disp_ras2vox = np.linalg.inv(disp_nib.affine)

        # convert from itk transform
        disp_xyz[:, :, :, 0] = -disp_xyz[:, :, :, 0]
        disp_xyz[:, :, :, 1] = -disp_xyz[:, :, :, 1]

        disp_grid = (
            np.arange(disp_xyz.shape[0]),
            np.arange(disp_xyz.shape[1]),
            np.arange(disp_xyz.shape[2]),
        )

        return cls(
            TransformType.DISPLACEMENT_RAS,
            disp_xyz=disp_xyz,
            disp_grid=disp_grid,
            disp_ras2vox=disp_ras2vox,
        )

    @classmethod
    def vox2ras_from_image(cls, path: str, level=0):
        from .dask_image import DaskImage

        img_type = DaskImage.check_img_type(path)

        if img_type is ImageType.OME_ZARR:
            return cls(
                TransformType.AFFINE_RAS,
                affine=cls.get_vox2ras_zarr(path, level),
            )
        elif img_type is ImageType.NIFTI:
            return cls(
                TransformType.AFFINE_RAS, affine=cls.get_vox2ras_nii(path)
            )
        else:
            print("unknown image type for vox2ras")
            return None

    @classmethod
    def ras2vox_from_image(cls, path: str, level=0):
        from .dask_image import DaskImage

        img_type = DaskImage.check_img_type(path)
        if img_type is ImageType.OME_ZARR:
            return cls(
                TransformType.AFFINE_RAS,
                affine=cls.get_ras2vox_zarr(path, level),
            )
        elif img_type is ImageType.NIFTI:
            return cls(
                TransformType.AFFINE_RAS, affine=cls.get_ras2vox_nii(path)
            )
        else:
            print("unknown image type for ras2vox")
            return None

    @staticmethod
    def get_vox2ras_nii(in_nii_path: str) -> np.array:
        return nib.load(in_nii_path).affine

    @staticmethod
    def get_ras2vox_nii(in_nii_path: str) -> np.array:
        return np.linalg.inv(TransformSpec.get_vox2ras_nii(in_nii_path))

    @staticmethod
    def get_vox2ras_zarr(in_zarr_path: str, level=0) -> np.array:
        # read coordinate transform from ome-zarr
        zi = zarr.open(in_zarr_path)
        attrs = zi["/"].attrs.asdict()
        multiscale = 0  # first multiscale image
        transforms = attrs["multiscales"][multiscale]["datasets"][level][
            "coordinateTransformations"
        ]

        # reorder_xfm -- changes from z,y,x to x,y,z ordering
        reorder_xfm = np.eye(4)
        reorder_xfm[:3, :3] = np.flip(
            reorder_xfm[:3, :3], axis=0
        )  # reorders z-y-x to x-y-z and vice versa

        # scaling xfm
        scaling_xfm = np.eye(4)
        scaling_xfm[0, 0] = -transforms[0]["scale"][
            -1
        ]  # x  # 0-index in transforms is the first (and only) transform
        scaling_xfm[1, 1] = -transforms[0]["scale"][-2]  # y
        scaling_xfm[2, 2] = -transforms[0]["scale"][-3]  # z

        return scaling_xfm @ reorder_xfm

    @staticmethod
    def get_ras2vox_zarr(in_zarr_path: str, level=0) -> np.array:
        return np.linalg.inv(
            TransformSpec.get_vox2ras_zarr(in_zarr_path, level=level)
        )

    def apply_transform(self, vecs: np.array) -> np.array:
        if self.tfm_type == TransformType.AFFINE_RAS:
            return self.affine @ vecs

        elif self.tfm_type == TransformType.DISPLACEMENT_RAS:
            # we have the grid points, the volumes to interpolate displacements

            # first we need to transform points to vox space of the warp
            vox_vecs = self.disp_ras2vox @ vecs

            # then interpolate the displacement in x, y, z:
            disp_vecs = np.zeros(vox_vecs.shape)

            for ax in range(3):
                disp_vecs[ax, :] = interpn(
                    self.disp_grid,
                    self.disp_xyz[:, :, :, ax].squeeze(),
                    vox_vecs[:3, :].T,
                    method="linear",
                    bounds_error=False,
                    fill_value=0,
                )

            return vecs + disp_vecs


def interp_by_block(
    x,
    transform_specs: list[TransformSpec],
    flo_dimg: DaskImage,
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
    for tfm_spec in transform_specs:
        xfm_vecs = tfm_spec.apply_transform(xfm_vecs)

    # then finally interpolate those points on the template dseg volume
    # need to interpolate for each channel

    interpolated = np.zeros(x.shape)

    # find bounding box required for flo vol
    (grid_points, flo_vol) = flo_dimg.get_bounded_subregion(xfm_vecs)
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
