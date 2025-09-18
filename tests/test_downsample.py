import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from zarrnii import ZarrNii


@pytest.mark.usefixtures("cleandir")
def test_downsample(nifti_nib):
    nifti_nib.to_filename("test.nii")
    znimg = ZarrNii.from_nifti("test.nii")

    print(znimg)
    print(znimg.axes_order)
    ds = np.array([2, 5, 8])
    znimg_downsampled = znimg.downsample(along_x=ds[0], along_y=ds[1], along_z=ds[2])
    print(znimg_downsampled)
    print(znimg_downsampled.axes_order)

    # check size is same as expected size
    assert_array_equal(znimg.darr.shape[1:], znimg_downsampled.darr.shape[1:] * ds)


@pytest.mark.usefixtures("cleandir")
def test_upsample(nifti_nib):
    """Test basic upsampling functionality."""
    nifti_nib.to_filename("test.nii")
    znimg = ZarrNii.from_nifti("test.nii")

    # Test isotropic upsampling
    us_factor = 2
    znimg_upsampled = znimg.upsample(
        along_x=us_factor, along_y=us_factor, along_z=us_factor
    )

    # Check that the upsampled array has the expected shape
    expected_shape = tuple(
        s * us_factor for s in znimg.shape[1:]
    )  # Skip channel dimension
    assert znimg_upsampled.shape[1:] == expected_shape

    # Check that axes_order is preserved
    assert znimg_upsampled.axes_order == znimg.axes_order

    # Check that the affine matrix is updated correctly (voxel size should be halved)
    orig_zooms = znimg.get_zooms(axes_order="XYZ")
    upsampled_zooms = znimg_upsampled.get_zooms(axes_order="XYZ")
    expected_zooms = orig_zooms / us_factor
    assert_array_almost_equal(upsampled_zooms, expected_zooms)


@pytest.mark.usefixtures("cleandir")
def test_upsample_anisotropic(nifti_nib):
    """Test anisotropic upsampling with different factors per axis."""
    nifti_nib.to_filename("test.nii")
    znimg = ZarrNii.from_nifti("test.nii")

    # Test anisotropic upsampling
    us_x, us_y, us_z = 2, 3, 4
    znimg_upsampled = znimg.upsample(along_x=us_x, along_y=us_y, along_z=us_z)

    # Check shape - note that for XYZ axes_order, the factors apply as expected
    if znimg.axes_order == "XYZ":
        expected_shape = (
            znimg.shape[1] * us_x,
            znimg.shape[2] * us_y,
            znimg.shape[3] * us_z,
        )
    else:  # ZYX
        expected_shape = (
            znimg.shape[1] * us_z,
            znimg.shape[2] * us_y,
            znimg.shape[3] * us_x,
        )

    assert znimg_upsampled.shape[1:] == expected_shape

    # Check that zooms are updated correctly
    orig_zooms = znimg.get_zooms(axes_order="XYZ")
    upsampled_zooms = znimg_upsampled.get_zooms(axes_order="XYZ")
    expected_zooms = orig_zooms / np.array([us_x, us_y, us_z])
    assert_array_almost_equal(upsampled_zooms, expected_zooms)


@pytest.mark.usefixtures("cleandir")
def test_upsample_to_shape(nifti_nib):
    """Test upsampling to a specific target shape."""
    nifti_nib.to_filename("test.nii")
    znimg = ZarrNii.from_nifti("test.nii")

    # Define target shape (including channel dimension)
    target_shape = (1, 128, 128, 128)
    znimg_upsampled = znimg.upsample(to_shape=target_shape)

    # Check that the upsampled array has the target shape
    assert znimg_upsampled.shape == target_shape

    # Check that axes_order is preserved
    assert znimg_upsampled.axes_order == znimg.axes_order


@pytest.mark.usefixtures("cleandir")
def test_downsample_on_read(nifti_nib):
    nifti_nib.to_filename("test.nii")
    znimg = ZarrNii.from_nifti("test.nii")

    # now we have znimg with axes_order == 'XYZ'
    znimg.to_ome_zarr("test_fromznimg.ome.zarr", max_layer=0)

    level = 2
    znimg2 = ZarrNii.from_ome_zarr("test_fromznimg.ome.zarr", level=level)
    znimg2

    print(znimg)
    print(znimg2)

    xyz_orig = znimg.darr.shape[1:]
    xyz_ds = znimg2.darr.shape[:0:-1]

    print(xyz_orig)
    print(xyz_ds)

    from math import ceil

    # x y and z are modified by 2^level
    assert_array_equal(ceil(xyz_orig[0] / (2**level)), xyz_ds[0])
    assert_array_equal(ceil(xyz_orig[1] / (2**level)), xyz_ds[1])
    assert_array_equal(ceil(xyz_orig[2] / (2**level)), xyz_ds[2])


@pytest.mark.usefixtures("cleandir")
def test_read_from_downsampled_level(nifti_nib):
    nifti_nib.to_filename("test.nii")
    znimg = ZarrNii.from_nifti("test.nii")

    # now we have znimg with axes_order == 'XYZ'
    znimg.to_ome_zarr("test_fromznimg.ome.zarr", max_layer=4)

    level = 2
    znimg2 = ZarrNii.from_ome_zarr("test_fromznimg.ome.zarr", level=level)
    znimg2

    xyz_orig = znimg.darr.shape[1:]
    xyz_ds = znimg2.darr.shape[:0:-1]

    # x y and z are modified by 2^level
    assert_array_equal(int(xyz_orig[0] / (2**level)), xyz_ds[0])
    assert_array_equal(int(xyz_orig[1] / (2**level)), xyz_ds[1])
    assert_array_equal(int(xyz_orig[2] / (2**level)), xyz_ds[2])
