import os
import tempfile

import nibabel as nib
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from synthetic_ome_zarr import generate_synthetic_dataset

from zarrnii import AffineTransform, ZarrNii


@pytest.mark.usefixtures("cleandir")
def test_from_nifti_to_nifti(nifti_nib):
    """create a nifti with nibabel, read it with ZarrNii, then write it back as a nifti.
    ensure data, affine and header do not change."""

    nifti_nib.to_filename("test.nii")
    nib_orig = nib.load("test.nii")

    znimg = ZarrNii.from_nifti("test.nii")
    znimg.to_nifti("test_fromznimg.nii")
    nib_znimg = nib.load("test_fromznimg.nii")

    # now compare nib_orig and nib_znimg
    assert_array_equal(nib_orig.affine, nib_znimg.affine)
    assert_array_equal(nib_orig.get_fdata(), nib_znimg.get_fdata())


@pytest.mark.usefixtures("cleandir")
def test_from_nifti_to_zarr_to_nifti(nifti_nib):
    """create a nifti with nibabel, read it with ZarrNii, write it back as zarr,
    then read it again with ZarrNii,
    ensure data, affine and header do not change."""

    nifti_nib.to_filename("test.nii")
    nib_orig = nib.load("test.nii")

    znimg = ZarrNii.from_nifti("test.nii")

    # now we have znimg with axes_order == 'XYZ'

    znimg.to_ome_zarr("test_fromznimg.ome.zarr")
    znimg2 = ZarrNii.from_ome_zarr("test_fromznimg.ome.zarr")
    znimg2

    # now we have znimg2 with axes_order == 'ZYX'
    #  this means, the affine is ZYX vox dims negated

    znimg2.to_nifti("test_fromznimg_tonii.nii")
    znimg3 = ZarrNii.from_nifti("test_fromznimg_tonii.nii")

    assert_array_equal(znimg.affine, znimg3.affine)
    assert_array_equal(znimg.darr.compute(), znimg3.darr.compute())
    assert_array_equal(
        nib_orig.header.get_zooms(),
        nib.load("test_fromznimg_tonii.nii").header.get_zooms(),
    )


@pytest.mark.usefixtures("cleandir")
def test_from_nifti_to_zarr_to_zarr(nifti_nib):
    """create a nifti with nibabel, read it with ZarrNii, write it back as zarr,
    then read it again with ZarrNii,
    ensure data, affine and header do not change."""

    nifti_nib.to_filename("test.nii")

    znimg = ZarrNii.from_nifti("test.nii")

    # now we have znimg with axes_order == XYZ
    #  this means, the affine is XYZ vox dims

    print(f"znimg from nii: {znimg}")
    znimg.to_ome_zarr("test_fromznimg.ome.zarr")
    znimg2 = ZarrNii.from_ome_zarr("test_fromznimg.ome.zarr")
    znimg2

    print(f"znimg2 from zarr: {znimg2}")
    # now we have znimg2 with axes_order == ZYX
    #  this means, the affine is ZYX vox dims negated

    assert znimg2.axes_order == "ZYX"
    assert_array_equal(
        np.flip(znimg2.darr.shape[1:], axis=0), nifti_nib.get_fdata().shape
    )

    znimg2.to_ome_zarr("test_fromznimg_tozarr.ome.zarr")
    znimg3 = ZarrNii.from_ome_zarr("test_fromznimg_tozarr.ome.zarr")

    assert znimg3.axes_order == "ZYX"
    assert_array_equal(
        np.flip(znimg3.darr.shape[1:], axis=0), nifti_nib.get_fdata().shape
    )

    print(f"znimg3 from zarr: {znimg3}")
    assert_array_equal(znimg2.affine, znimg3.affine)
    assert_array_equal(znimg2.darr.compute(), znimg3.darr.compute())


@pytest.mark.usefixtures("cleandir")
def test_affine_transform_identify(nifti_nib):
    """tests affine transform from nifti to zarr using identity transformation"""

    nifti_nib.to_filename("test.nii")

    flo_znimg = ZarrNii.from_nifti("test.nii")

    ref_znimg = ZarrNii.from_nifti("test.nii")

    interp_znimg = flo_znimg.apply_transform(
        AffineTransform.from_array(np.eye(4)), ref_znimg=ref_znimg
    )

    assert_array_almost_equal(flo_znimg.darr.compute(), interp_znimg.darr.compute())


@pytest.mark.usefixtures("cleandir")
def test_transform_indices_flo_to_ref(nifti_nib):
    """tests transformation of points from floating to reference space"""
    # first try with just identity, floating as nifti, ref as zarr

    nifti_nib.to_filename("test.nii")

    ZarrNii.from_nifti("test.nii").to_ome_zarr("test_flo.ome.zarr")

    ref_znimg = ZarrNii.from_ome_zarr("test_flo.ome.zarr")
    flo_znimg = ZarrNii.from_nifti("test.nii")

    ident = AffineTransform.from_array(np.eye(4))
    flo_indices = np.array((99, 49, 199)).reshape(3, 1)
    flo_to_ref_indices = flo_znimg.apply_transform_flo_to_ref_indices(
        ident, ref_znimg=ref_znimg, indices=flo_indices
    )

    flo_to_ref_to_flo_indices = flo_znimg.apply_transform_ref_to_flo_indices(
        ident, ref_znimg=ref_znimg, indices=flo_to_ref_indices
    )

    print("flo")
    print(flo_indices)
    print("flo -> ref")
    print(flo_to_ref_indices)
    print("flo -> ref -> flo")
    print(flo_to_ref_to_flo_indices)
    assert_array_almost_equal(flo_indices, flo_to_ref_to_flo_indices)


@pytest.mark.usefixtures("cleandir")
def test_transform_indices_ref_to_flo(nifti_nib):
    """tests transformation of points from reference to floating space"""
    # first try with just identity, ref as nifti, flo as zarr

    nifti_nib.to_filename("test.nii")

    ZarrNii.from_nifti("test.nii").to_ome_zarr("test_ref.ome.zarr")

    ref_znimg = ZarrNii.from_ome_zarr("test_ref.ome.zarr")
    flo_znimg = ZarrNii.from_nifti("test.nii")

    ident = AffineTransform.from_array(np.eye(4))
    ref_indices = np.array((99, 49, 199)).reshape(3, 1)
    ref_to_flo_indices = flo_znimg.apply_transform_ref_to_flo_indices(
        ident, ref_znimg=ref_znimg, indices=ref_indices
    )

    ref_flo_to_ref_indices = flo_znimg.apply_transform_flo_to_ref_indices(
        ident, ref_znimg=ref_znimg, indices=ref_to_flo_indices
    )

    assert_array_almost_equal(ref_indices, ref_flo_to_ref_indices)


class TestOMEZarr:
    @pytest.mark.usefixtures("cleandir")
    def test_ome_zarr(self):
        # test reading and writing ome zarr
        OME_ZARR_PATH = "./test.ome.zarr"
        generate_synthetic_dataset(
            OME_ZARR_PATH, arr_sz=(1, 16, 128, 128), MAX_LAYER=1  # layer 0 and 1
        )
        arr = ZarrNii.from_ome_zarr(OME_ZARR_PATH, level=0, channels=[0]).darr
        assert arr.compute().sum() > 0


"""
    @pytest.mark.usefixtures("cleandir")
    def test_ome_zarr_zip(self):
        # test reading and writing ome zarr
        OME_ZARR_PATH = './test.ome.zarr.zip'
        generate_synthetic_dataset(
            OME_ZARR_PATH,
            arr_sz=(1, 16, 128, 128),
            MAX_LAYER=1  # layer 0 and 1
        )
        arr = ZarrNii.from_ome_zarr(OME_ZARR_PATH, level=1, channels=[0]).darr
        assert arr.compute().sum() > 0
"""
