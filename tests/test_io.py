import os
import tempfile

import dask.array as da
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

    assert_array_equal(nib_orig.header.get_zooms(), znimg.get_zooms(axes_order="XYZ"))
    assert_array_equal(nib_orig.affine[:3, 3], znimg.get_origin(axes_order="XYZ"))


@pytest.mark.usefixtures("cleandir")
def test_from_nifti_as_ref(nifti_nib):
    """Test loading NIfTI as reference space without loading data."""
    nifti_nib.to_filename("test.nii")
    nib_orig = nib.load("test.nii")

    # Load as reference (empty array)
    znimg_ref = ZarrNii.from_nifti("test.nii", as_ref=True)

    # Should have same shape and affine but different (empty) data
    assert znimg_ref.shape == (1,) + nib_orig.shape
    assert_array_equal(znimg_ref.get_affine_matrix(axes_order="XYZ"), nib_orig.affine)

    # Data should be empty float32 array
    assert znimg_ref.data.dtype == np.float32
    computed_data = znimg_ref.data.compute()
    assert np.all(computed_data == 0)  # Should be all zeros (empty)


@pytest.mark.usefixtures("cleandir")
def test_from_nifti_as_ref_with_zooms(nifti_nib):
    """Test loading NIfTI as reference space with custom zooms."""
    nifti_nib.to_filename("test.nii")
    nib_orig = nib.load("test.nii")

    # Define target zooms (different from original)
    target_zooms = np.array([2.0, 2.0, 2.0])

    # Load as reference with custom zooms
    znimg_ref = ZarrNii.from_nifti("test.nii", as_ref=True, zooms=target_zooms)

    # Check that zooms are updated in affine matrix
    expected_affine = nib_orig.affine.copy()
    np.fill_diagonal(expected_affine[:3, :3], target_zooms)

    assert_array_almost_equal(
        znimg_ref.get_affine_matrix(axes_order="XYZ"), expected_affine
    )
    assert_array_almost_equal(znimg_ref.get_zooms(axes_order="XYZ"), target_zooms)

    # Shape should be adjusted based on scaling factor
    orig_zooms = np.sqrt((nib_orig.affine[:3, :3] ** 2).sum(axis=0))
    scaling_factor = orig_zooms / target_zooms
    expected_shape = [
        int(np.floor(nib_orig.shape[i] * scaling_factor[2 - i])) for i in range(3)
    ]

    assert znimg_ref.shape == (1,) + tuple(expected_shape)


@pytest.mark.usefixtures("cleandir")
def test_from_nifti_zooms_without_as_ref_error(nifti_nib):
    """Test that providing zooms without as_ref=True raises an error."""
    nifti_nib.to_filename("test.nii")

    with pytest.raises(ValueError, match="`zooms` can only be used when `as_ref=True`"):
        ZarrNii.from_nifti("test.nii", as_ref=False, zooms=[1.0, 1.0, 1.0])


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

    assert_array_equal(nib_orig.header.get_zooms(), znimg.get_zooms(axes_order="XYZ"))
    assert_array_equal(nib_orig.affine[:3, 3], znimg.get_origin(axes_order="XYZ"))

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


@pytest.mark.usefixtures("cleandir")
def test_zarr_write_simple(nifti_nib):
    """Test that we can write OME-Zarr data from NIfTI."""
    # Create a simple test using the working nifti fixture
    nifti_nib.to_filename("test.nii")
    znimg = ZarrNii.from_nifti("test.nii")

    # Test writing to OME-Zarr
    output_path = "test_output.ome.zarr"
    znimg.to_ome_zarr(output_path)

    # Test that the output file exists and can be read back
    import os

    assert os.path.exists(output_path)

    # Load it back and check basic properties
    reloaded = ZarrNii.from_ome_zarr(output_path)
    assert reloaded is not None
    # Note: shapes may differ due to axis reordering between NIfTI and OME-Zarr formats
    # But the data volume should be the same
    assert np.prod(reloaded.darr.shape) == np.prod(znimg.darr.shape)


@pytest.mark.usefixtures("cleandir_fake")
@pytest.mark.skip(
    reason="Fixture znimg_from_multiscales has CRC32 checksum issue - related to synthetic data generation"
)
def test_write_ome_zarr(znimg_from_multiscales):
    """Test that we can write OME-Zarr data back to a file."""
    print(znimg_from_multiscales)

    # Test basic properties of the loaded znimg
    assert znimg_from_multiscales is not None
    assert hasattr(znimg_from_multiscales, "darr")
    assert hasattr(znimg_from_multiscales, "affine")

    # Test that we can write it to a new OME-Zarr file
    output_path = "test_output.ome.zarr"
    znimg_from_multiscales.to_ome_zarr(output_path)

    # Test that the output file exists and can be read back
    import os

    assert os.path.exists(output_path)

    # Load it back and check basic properties
    reloaded = ZarrNii.from_ome_zarr(output_path)
    assert reloaded is not None
    assert reloaded.darr.shape == znimg_from_multiscales.darr.shape


class TestOMEZarr:
    @pytest.mark.usefixtures("cleandir")
    @pytest.mark.xfail(reason="Known issue with synthetic data generation")
    def test_ome_zarr(self):
        # test reading and writing ome zarr
        OME_ZARR_PATH = "./test.ome.zarr"
        generate_synthetic_dataset(
            OME_ZARR_PATH,
            arr_sz=(1, 16, 128, 128),
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


def test_orientation_functionality(tmp_path):
    """Test orientation reading/writing functionality."""
    # Create a simple test image
    data = np.random.rand(1, 64, 64, 64).astype(np.float32)
    znimg = ZarrNii.from_darr(
        da.from_array(data, chunks=(1, 32, 32, 32)), orientation="LPI", axes_order="ZYX"
    )

    # Check that orientation is set correctly
    assert znimg.orientation == "LPI"
    assert znimg.get_orientation() == "LPI"

    # Save to OME-Zarr
    zarr_path = tmp_path / "test_orientation.zarr"
    znimg.to_ome_zarr(str(zarr_path))

    # Load back and check orientation is preserved
    loaded_znimg = ZarrNii.from_ome_zarr(str(zarr_path))
    assert loaded_znimg.orientation == "LPI"
    assert loaded_znimg.get_orientation() == "LPI"


def test_orientation_in_from_ome_zarr(tmp_path):
    """Test orientation parameter in from_ome_zarr."""
    # Create a simple test image
    data = np.random.rand(1, 64, 64, 64).astype(np.float32)
    znimg = ZarrNii.from_darr(
        da.from_array(data, chunks=(1, 32, 32, 32)), orientation="RAS", axes_order="ZYX"
    )

    # Save to OME-Zarr
    zarr_path = tmp_path / "test_from_ome_zarr_orientation.zarr"
    znimg.to_ome_zarr(str(zarr_path))

    # Load with different default orientation (should use stored orientation)
    loaded_znimg = ZarrNii.from_ome_zarr(str(zarr_path), orientation="LPI")
    assert loaded_znimg.orientation == "RAS"  # Should use the stored orientation

    # Test fallback to provided orientation when none is stored
    # (This would require a zarr file without orientation metadata)
    # For now, just test that the parameter is accepted
    loaded_znimg2 = ZarrNii.from_ome_zarr(str(zarr_path), orientation="IPL")
    assert loaded_znimg2.orientation == "RAS"  # Should still use stored orientation


def test_orientation_xyz_consistent_definition():
    """Test that orientation strings are consistently defined with respect to XYZ ordering."""
    # Create test data
    data = np.random.rand(1, 64, 64, 64).astype(np.float32)
    dask_data = da.from_array(data, chunks=(1, 32, 32, 32))

    # Test RAS orientation with both ZYX and XYZ axes_order
    znimg_zyx = ZarrNii.from_darr(dask_data, orientation="RAS", axes_order="ZYX")
    znimg_xyz = ZarrNii.from_darr(dask_data, orientation="RAS", axes_order="XYZ")

    # Both should have the same orientation string
    assert znimg_zyx.get_orientation() == "RAS"
    assert znimg_xyz.get_orientation() == "RAS"

    # Test that RAS means:
    # - R to L in X direction (axis 0 in physical space)
    # - P to A in Y direction (axis 1 in physical space)
    # - I to S in Z direction (axis 2 in physical space)

    # For XYZ axes_order, the affine should directly reflect RAS
    affine_xyz = znimg_xyz.get_affine_matrix(axes_order="XYZ")

    # For ZYX axes_order, we need to be careful about interpretation
    affine_zyx = znimg_zyx.get_affine_matrix(axes_order="ZYX")

    # The key test: regardless of axes_order, RAS should mean the same thing
    # in physical space - verify this by checking the orientation extracted from affine
    from zarrnii.core import affine_to_orientation

    # Both should give us "RAS" when we extract orientation from their affines
    # (assuming the affine is properly constructed for XYZ space)
    extracted_orientation_xyz = affine_to_orientation(affine_xyz)
    extracted_orientation_zyx = affine_to_orientation(affine_zyx)

    print(f"ZYX affine:\n{affine_zyx}")
    print(f"XYZ affine:\n{affine_xyz}")
    print(f"Extracted from ZYX: {extracted_orientation_zyx}")
    print(f"Extracted from XYZ: {extracted_orientation_xyz}")

    # Both should consistently represent RAS in physical space
    assert extracted_orientation_xyz == "RAS"
    assert extracted_orientation_zyx == "RAS"


def test_orientation_consistency_multiple_strings():
    """Test orientation consistency with different orientation strings."""
    data = np.random.rand(1, 32, 32, 32).astype(np.float32)
    dask_data = da.from_array(data, chunks=(1, 16, 16, 16))

    orientations = ["RAS", "LPI", "RAI", "LPS"]

    for orient in orientations:
        # Create with both axes orders
        znimg_zyx = ZarrNii.from_darr(dask_data, orientation=orient, axes_order="ZYX")
        znimg_xyz = ZarrNii.from_darr(dask_data, orientation=orient, axes_order="XYZ")

        # Both should report the same orientation
        assert znimg_zyx.get_orientation() == orient
        assert znimg_xyz.get_orientation() == orient

        # Extract orientation from affines - should be consistent
        from zarrnii.core import affine_to_orientation

        affine_zyx = znimg_zyx.get_affine_matrix(axes_order="ZYX")
        affine_xyz = znimg_xyz.get_affine_matrix(axes_order="XYZ")

        extracted_zyx = affine_to_orientation(affine_zyx)
        extracted_xyz = affine_to_orientation(affine_xyz)

        print(
            f"Orientation {orient}: ZYX extracted={extracted_zyx}, XYZ extracted={extracted_xyz}"
        )

        # Key requirement: orientation should be defined consistently in XYZ space
        assert extracted_xyz == orient
        assert extracted_zyx == orient


def test_orientation_xyz_definition_clarity():
    """Test that orientation strings are always defined with respect to XYZ physical coordinates.

    This test documents and verifies that orientation strings like 'RAS' always mean:
    - R/L refers to the X axis (left-right)
    - A/P refers to the Y axis (anterior-posterior)
    - S/I refers to the Z axis (superior-inferior)

    This is true regardless of whether the data is stored with axes_order="XYZ" or "ZYX".
    """
    # Create test data
    data = np.random.rand(1, 64, 64, 64).astype(np.float32)
    dask_data = da.from_array(data, chunks=(1, 32, 32, 32))

    # Test case: RAS orientation should mean the same thing for both axes orders
    znimg_xyz = ZarrNii.from_darr(dask_data, orientation="RAS", axes_order="XYZ")
    znimg_zyx = ZarrNii.from_darr(dask_data, orientation="RAS", axes_order="ZYX")

    # Both should report RAS orientation
    assert znimg_xyz.get_orientation() == "RAS"
    assert znimg_zyx.get_orientation() == "RAS"

    # Get affine matrices (in their respective coordinate systems)
    affine_xyz = znimg_xyz.get_affine_matrix(axes_order="XYZ")  # X,Y,Z order
    affine_zyx = znimg_zyx.get_affine_matrix(axes_order="ZYX")  # Z,Y,X order

    # Key test: when we extract orientation from the affines, both should give RAS
    # This confirms that RAS is consistently interpreted in XYZ physical space
    from zarrnii.core import affine_to_orientation

    # Both affines should be interpretable as RAS orientation
    extracted_xyz = affine_to_orientation(affine_xyz)
    extracted_zyx = affine_to_orientation(affine_zyx)

    assert (
        extracted_xyz == "RAS"
    ), f"XYZ affine should extract as RAS, got {extracted_xyz}"
    assert (
        extracted_zyx == "RAS"
    ), f"ZYX affine should extract as RAS, got {extracted_zyx}"

    # Document what RAS means:
    print("✓ RAS orientation confirmed to mean:")
    print("  - R→L along X axis (physical left-right)")
    print("  - A→P along Y axis (physical anterior-posterior)")
    print("  - I→S along Z axis (physical inferior-superior)")
    print("  - This definition is consistent regardless of axes_order")


def test_orientation_round_trip_preservation():
    """Test that orientation is preserved through round-trip operations."""
    data = np.random.rand(1, 32, 32, 32).astype(np.float32)
    dask_data = da.from_array(data, chunks=(1, 16, 16, 16))

    # Test with different combinations
    test_cases = [
        ("RAS", "ZYX"),
        ("RAS", "XYZ"),
        ("LPI", "ZYX"),
        ("LPI", "XYZ"),
        ("RAI", "ZYX"),
        ("RAI", "XYZ"),
    ]

    for orient, axes_order in test_cases:
        znimg = ZarrNii.from_darr(dask_data, orientation=orient, axes_order=axes_order)

        # Apply some transformations that should preserve orientation
        transformed = znimg.downsample(factors=2).crop(
            bbox_min=(0, 0, 0), bbox_max=(1, 16, 16, 16)
        )

        # Orientation should be preserved
        assert transformed.get_orientation() == orient
        assert transformed.orientation == orient
