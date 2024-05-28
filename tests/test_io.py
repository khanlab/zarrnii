import os
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from zarrnii import ZarrNii, Transform


@pytest.fixture
def nifti_nib():
    img_size = (100, 50, 200)
    pix_dims = (0.3, 0.2, 1.5, 1)

    nifti_nib = nib.Nifti1Image(
        np.random.rand(*img_size), affine=np.diag(pix_dims)
    )

    return nifti_nib


@pytest.fixture
def cleandir():
    with tempfile.TemporaryDirectory() as newpath:
        old_cwd = os.getcwd()
        os.chdir(newpath)
        yield
        os.chdir(old_cwd)


@pytest.mark.usefixtures("cleandir")
def test_from_nifti_to_nifti(nifti_nib):
    """create a nifti with nibabel, read it with ZarrNii, then write it back as a nifti.
    ensure data, affine and header do not change."""

    nifti_nib.to_filename("test.nii")
    nib_orig = nib.load("test.nii")

    dimg = ZarrNii.from_path("test.nii")
    dimg.to_nifti("test_fromdimg.nii")
    nib_dimg = nib.load("test_fromdimg.nii")

    # now compare nib_orig and nib_dimg
    assert_array_equal(nib_orig.affine, nib_dimg.affine)
    assert_array_equal(nib_orig.get_fdata(), nib_dimg.get_fdata())


@pytest.mark.usefixtures("cleandir")
def test_from_nifti_to_zarr_to_nifti(nifti_nib):
    """create a nifti with nibabel, read it with ZarrNii, write it back as zarr,
    then read it again with ZarrNii,
    ensure data, affine and header do not change."""

    nifti_nib.to_filename("test.nii")
    nib_orig = nib.load("test.nii")

    dimg = ZarrNii.from_path("test.nii")

    # now we have dimg with axes_nifti == True
    #  this means, the affine is XYZ vox dims

    dimg.to_ome_zarr("test_fromdimg.ome.zarr")
    dimg2 = ZarrNii.from_path("test_fromdimg.ome.zarr")
    dimg2

    # now we have dimg2 with axes_nifti == False
    #  this means, the affine is ZYX vox dims negated

    dimg2.to_nifti("test_fromdimg_tonii.nii")
    dimg3 = ZarrNii.from_path("test_fromdimg_tonii.nii")


    print(dimg)
    print(dimg2)
    print(dimg3)
    assert_array_equal(dimg.vox2ras.affine, dimg3.vox2ras.affine)
    assert_array_equal(dimg.darr.compute(), dimg3.darr.compute())


@pytest.mark.usefixtures("cleandir")
def test_from_nifti_to_zarr_to_zarr(nifti_nib):
    """create a nifti with nibabel, read it with ZarrNii, write it back as zarr,
    then read it again with ZarrNii,
    ensure data, affine and header do not change."""

    nifti_nib.to_filename("test.nii")
    nib_orig = nib.load("test.nii")

    dimg = ZarrNii.from_path("test.nii")

    # now we have dimg with axes_nifti == True
    #  this means, the affine is XYZ vox dims

    print(f"dimg from nii: {dimg}")
    dimg.to_ome_zarr("test_fromdimg.ome.zarr")
    dimg2 = ZarrNii.from_path("test_fromdimg.ome.zarr")
    dimg2

    print(f"dimg2 from zarr: {dimg2}")
    # now we have dimg2 with axes_nifti == False
    #  this means, the affine is ZYX vox dims negated

    assert dimg2.axes_nifti == False
    assert_array_equal(
        np.flip(dimg2.darr.shape[1:], axis=0), nifti_nib.get_fdata().shape
    )

    dimg2.to_ome_zarr("test_fromdimg_tozarr.ome.zarr")
    dimg3 = ZarrNii.from_path("test_fromdimg_tozarr.ome.zarr")

    assert dimg3.axes_nifti == False
    assert_array_equal(
        np.flip(dimg3.darr.shape[1:], axis=0), nifti_nib.get_fdata().shape
    )

    print(f"dimg3 from zarr: {dimg3}")
    assert_array_equal(dimg2.vox2ras.affine, dimg3.vox2ras.affine)
    assert_array_equal(dimg2.darr.compute(), dimg3.darr.compute())


@pytest.mark.usefixtures("cleandir")
def test_affine_transform_identify(nifti_nib):
    """tests affine transform from nifti to zarr using identity transformation"""

    nifti_nib.to_filename("test.nii")

    flo_dimg = ZarrNii.from_path("test.nii")

    ref_dimg = ZarrNii.from_path("test.nii")

    interp_dimg = flo_dimg.apply_transform(
        Transform.affine_ras_from_array(np.eye(4)), ref_dimg=ref_dimg
    )

    assert_array_almost_equal(
        flo_dimg.darr.compute(), interp_dimg.darr.compute()
    )


@pytest.mark.usefixtures("cleandir")
def test_transform_indices_flo_to_ref(nifti_nib):
    """tests transformation of points from floating to reference space"""
    # first try with just identity, floating as nifti, ref as zarr

    nifti_nib.to_filename("test.nii")

    ZarrNii.from_path("test.nii").to_ome_zarr("test_flo.ome.zarr")

    ref_dimg = ZarrNii.from_path("test_flo.ome.zarr")
    flo_dimg = ZarrNii.from_path("test.nii")

    ident = Transform.affine_ras_from_array(np.eye(4))
    flo_indices = np.array((99, 49, 199)).reshape(3, 1)
    flo_to_ref_indices = flo_dimg.apply_transform_flo_to_ref_indices(
        ident, ref_dimg=ref_dimg, indices=flo_indices
    )

    flo_to_ref_to_flo_indices = flo_dimg.apply_transform_ref_to_flo_indices(
        ident, ref_dimg=ref_dimg, indices=flo_to_ref_indices
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

    ZarrNii.from_path("test.nii").to_ome_zarr("test_ref.ome.zarr")

    ref_dimg = ZarrNii.from_path("test_ref.ome.zarr")
    flo_dimg = ZarrNii.from_path("test.nii")

    ident = Transform.affine_ras_from_array(np.eye(4))
    ref_indices = np.array((99, 49, 199)).reshape(3, 1)
    ref_to_flo_indices = flo_dimg.apply_transform_ref_to_flo_indices(
        ident, ref_dimg=ref_dimg, indices=ref_indices
    )

    ref_flo_to_ref_indices = flo_dimg.apply_transform_flo_to_ref_indices(
        ident, ref_dimg=ref_dimg, indices=ref_to_flo_indices
    )

    assert_array_almost_equal(ref_indices, ref_flo_to_ref_indices)
