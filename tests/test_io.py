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


def test_to_ome_zarr_backend_parameter(znimg_from_multiscales):
    """Test that to_ome_zarr works with different backend parameters."""
    import os
    import tempfile

    # Test with default backend (ngff-zarr)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_default.ome.zarr")
        znimg_from_multiscales.to_ome_zarr(output_path)
        assert os.path.exists(output_path)
        reloaded = ZarrNii.from_ome_zarr(output_path)
        assert reloaded.darr.shape == znimg_from_multiscales.darr.shape

    # Test with explicit ngff-zarr backend
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_ngff_zarr.ome.zarr")
        znimg_from_multiscales.to_ome_zarr(output_path, backend="ngff-zarr")
        assert os.path.exists(output_path)
        reloaded = ZarrNii.from_ome_zarr(output_path)
        assert reloaded.darr.shape == znimg_from_multiscales.darr.shape

    # Test with ome-zarr-py backend
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_ome_zarr_py.ome.zarr")
        znimg_from_multiscales.to_ome_zarr(output_path, backend="ome-zarr-py")
        assert os.path.exists(output_path)
        reloaded = ZarrNii.from_ome_zarr(output_path)
        assert reloaded.darr.shape == znimg_from_multiscales.darr.shape

    # Test with invalid backend (should raise ValueError)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_invalid.ome.zarr")
        try:
            znimg_from_multiscales.to_ome_zarr(output_path, backend="invalid")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Invalid backend" in str(e)
            assert "ngff-zarr" in str(e)
            assert "ome-zarr-py" in str(e)


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

    # Load with different default orientation (should override with this orientation)
    loaded_znimg = ZarrNii.from_ome_zarr(str(zarr_path), orientation="LPI")
    assert loaded_znimg.orientation == "LPI"  # Should use the stored orientation

    # Test fallback to provided orientation when none is stored
    # (This would require a zarr file without orientation metadata)
    # For now, just test that the parameter is accepted
    loaded_znimg2 = ZarrNii.from_ome_zarr(str(zarr_path), orientation=None)
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
    from zarrnii.core import _affine_to_orientation

    #  when we extract orientation from their affines, the axes ordering
    # changes the affine.. so the orienation extracted from the affine
    # will necessarily be different..
    extracted_orientation_xyz = _affine_to_orientation(affine_xyz)
    extracted_orientation_zyx = _affine_to_orientation(affine_zyx)

    print(f"ZYX affine:\n{affine_zyx}")
    print(f"XYZ affine:\n{affine_xyz}")
    print(f"Extracted from ZYX: {extracted_orientation_zyx}")
    print(f"Extracted from XYZ: {extracted_orientation_xyz}")

    # Both should consistently represent RAS in physical space
    assert extracted_orientation_xyz == "RAS"
    assert extracted_orientation_zyx == "SAR"


def test_orientation_consistency_multiple_strings():
    """Test orientation consistency with different orientation strings."""
    data = np.random.rand(1, 32, 32, 32).astype(np.float32)
    dask_data = da.from_array(data, chunks=(1, 16, 16, 16))

    orientations = ["RAS", "LPI", "RAI", "LPS"]
    orientations_rev = ["SAR", "IPL", "IAR", "SPL"]

    for orient, orient_rev in zip(orientations, orientations_rev):
        # Create with both axes orders
        znimg_zyx = ZarrNii.from_darr(dask_data, orientation=orient, axes_order="ZYX")
        znimg_xyz = ZarrNii.from_darr(dask_data, orientation=orient, axes_order="XYZ")

        # Both should report the same orientation
        assert znimg_zyx.get_orientation() == orient
        assert znimg_xyz.get_orientation() == orient

        # Extract orientation from affines - should be consistent
        from zarrnii.core import _affine_to_orientation

        affine_zyx = znimg_zyx.get_affine_matrix(axes_order="ZYX")
        affine_xyz = znimg_xyz.get_affine_matrix(axes_order="XYZ")

        extracted_zyx = _affine_to_orientation(affine_zyx)
        extracted_xyz = _affine_to_orientation(affine_xyz)

        print(
            f"Orientation {orient}: ZYX extracted={extracted_zyx}, XYZ extracted={extracted_xyz}"
        )

        # Key requirement: orientation should be defined consistently in XYZ space
        assert extracted_xyz == orient
        assert extracted_zyx == orient_rev


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

    # Key test: when we extract orientation from the affines,
    # the affine will be dependent on axes ordering, thus the orientation
    # extracted from the affine will be reversed
    from zarrnii.core import _affine_to_orientation

    # Both affines should be interpretable as RAS orientation
    extracted_xyz = _affine_to_orientation(affine_xyz)
    extracted_zyx = _affine_to_orientation(affine_zyx)

    assert extracted_xyz == "RAS", (
        f"XYZ affine should extract as RAS, got {extracted_xyz}"
    )
    assert extracted_zyx == "SAR", (
        f"ZYX affine should extract as SAR, got {extracted_zyx}"
    )

    # Document what RAS means:
    print("✓ RAS orientation confirmed to mean:")
    print("  - R→L along X axis (physical left-right)")
    print("  - A→P along Y axis (physical anterior-posterior)")
    print("  - I→S along Z axis (physical inferior-superior)")
    print(
        "  - This definition is consistent regardless of axes_order, except when derived from the affine"
    )


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


def test_xyz_orientation_backwards_compatibility(tmp_path):
    """Test that xyz_orientation is preferred over legacy orientation with reversal fallback."""
    import dask.array as da
    import numpy as np
    import zarr

    # Create test data
    data = np.random.rand(1, 32, 32, 32).astype(np.float32)
    dask_data = da.from_array(data, chunks=(1, 16, 16, 16))

    # Test 1: New xyz_orientation is preferred when both exist
    zarr_path = tmp_path / "test_new_xyz_orientation.zarr"
    znimg = ZarrNii.from_darr(dask_data, orientation="RAS", axes_order="ZYX")
    znimg.to_ome_zarr(str(zarr_path))

    # Manually add both xyz_orientation and legacy orientation to the metadata
    group = zarr.open_group(str(zarr_path), mode="r+")
    group.attrs["xyz_orientation"] = "LPI"  # New format
    group.attrs["orientation"] = "SAR"  # Legacy format (reversed for ZYX)

    # Load and verify xyz_orientation takes precedence
    loaded_znimg = ZarrNii.from_ome_zarr(str(zarr_path))
    assert loaded_znimg.orientation == "LPI"
    assert loaded_znimg.xyz_orientation == "LPI"

    # Test 2: Legacy orientation fallback with reversal
    zarr_path_legacy = tmp_path / "test_legacy_orientation.zarr"
    znimg2 = ZarrNii.from_darr(dask_data, orientation="RAS", axes_order="ZYX")
    znimg2.to_ome_zarr(str(zarr_path_legacy))

    # Remove xyz_orientation and add only legacy orientation (ZYX-based)
    group_legacy = zarr.open_group(str(zarr_path_legacy), mode="r+")
    if "xyz_orientation" in group_legacy.attrs:
        del group_legacy.attrs["xyz_orientation"]
    group_legacy.attrs["orientation"] = "SAR"  # ZYX-based, should be reversed to "RAS"

    # Load and verify legacy orientation is reversed properly
    loaded_znimg_legacy = ZarrNii.from_ome_zarr(str(zarr_path_legacy))
    assert loaded_znimg_legacy.orientation == "RAS"  # Should be reversed from "SAR"
    assert loaded_znimg_legacy.xyz_orientation == "RAS"

    # Test 3: Default orientation when neither exists
    zarr_path_default = tmp_path / "test_default_orientation.zarr"
    znimg3 = ZarrNii.from_darr(dask_data, orientation="RAS", axes_order="ZYX")
    znimg3.to_ome_zarr(str(zarr_path_default))

    # Remove both orientation attributes
    group_default = zarr.open_group(str(zarr_path_default), mode="r+")
    if "xyz_orientation" in group_default.attrs:
        del group_default.attrs["xyz_orientation"]
    if "orientation" in group_default.attrs:
        del group_default.attrs["orientation"]

    # Load with explicit default orientation
    loaded_znimg_default = ZarrNii.from_ome_zarr(
        str(zarr_path_default), orientation="LPI"
    )
    assert loaded_znimg_default.orientation == "LPI"
    assert loaded_znimg_default.xyz_orientation == "LPI"


def test_reverse_orientation_string():
    """Test the reverse_orientation_string utility function."""
    from zarrnii.core import reverse_orientation_string

    # Test basic reversals
    assert reverse_orientation_string("RAS") == "SAR"
    assert reverse_orientation_string("LPI") == "IPL"
    assert reverse_orientation_string("RAI") == "IAR"
    assert reverse_orientation_string("LPS") == "SPL"

    # Test that reversing twice returns original
    for orientation in ["RAS", "LPI", "RAI", "LPS"]:
        reversed_twice = reverse_orientation_string(
            reverse_orientation_string(orientation)
        )
        assert reversed_twice == orientation

    # Test error cases
    with pytest.raises(
        ValueError, match="Orientation string must be exactly 3 characters"
    ):
        reverse_orientation_string("RA")

    with pytest.raises(
        ValueError, match="Orientation string must be exactly 3 characters"
    ):
        reverse_orientation_string("RASA")


def test_orientation_property_compatibility():
    """Test that the orientation property provides backward compatibility."""
    import dask.array as da
    import numpy as np

    # Create test data
    data = np.random.rand(1, 32, 32, 32).astype(np.float32)
    dask_data = da.from_array(data, chunks=(1, 16, 16, 16))

    # Test that orientation property maps to xyz_orientation
    znimg = ZarrNii.from_darr(dask_data, orientation="LPI", axes_order="ZYX")

    # Both should return the same value
    assert znimg.orientation == "LPI"
    assert znimg.xyz_orientation == "LPI"
    assert znimg.get_orientation() == "LPI"

    # Test setting orientation property updates xyz_orientation
    znimg.orientation = "RAS"
    assert znimg.orientation == "RAS"
    assert znimg.xyz_orientation == "RAS"

    # Test direct xyz_orientation access
    znimg.xyz_orientation = "RAI"
    assert znimg.orientation == "RAI"
    assert znimg.xyz_orientation == "RAI"


def test_orientation_metadata_new_format_written(tmp_path):
    """Test that new files write xyz_orientation instead of orientation."""
    import dask.array as da
    import numpy as np
    import zarr

    # Create test data
    data = np.random.rand(1, 32, 32, 32).astype(np.float32)
    dask_data = da.from_array(data, chunks=(1, 16, 16, 16))

    # Create ZarrNii with specific orientation
    znimg = ZarrNii.from_darr(dask_data, orientation="LPI", axes_order="ZYX")

    # Save to OME-Zarr
    zarr_path = tmp_path / "test_new_format.zarr"
    znimg.to_ome_zarr(str(zarr_path))

    # Check that the new format metadata is written
    group = zarr.open_group(str(zarr_path), mode="r")
    assert "xyz_orientation" in group.attrs
    assert group.attrs["xyz_orientation"] == "LPI"

    # The old orientation key should not be present in new files
    assert "orientation" not in group.attrs


def test_complex_orientation_scenarios(tmp_path):
    """Test complex scenarios with orientation handling."""
    import dask.array as da
    import numpy as np
    import zarr

    # Create test data
    data = np.random.rand(1, 32, 32, 32).astype(np.float32)
    dask_data = da.from_array(data, chunks=(1, 16, 16, 16))

    # Test various orientation strings that might exist in legacy files
    legacy_orientations = [
        ("SAR", "RAS"),  # ZYX SAR -> XYZ RAS
        ("IPL", "LPI"),  # ZYX IPL -> XYZ LPI
        ("IAR", "RAI"),  # ZYX IAR -> XYZ RAI
        ("SPL", "LPS"),  # ZYX SPL -> XYZ LPS
    ]

    for legacy_orient, expected_xyz in legacy_orientations:
        zarr_path = tmp_path / f"test_legacy_{legacy_orient}.zarr"

        # Create a zarr file with only legacy orientation
        znimg = ZarrNii.from_darr(dask_data, orientation="RAS", axes_order="ZYX")
        znimg.to_ome_zarr(str(zarr_path))

        # Modify to have only legacy orientation
        group = zarr.open_group(str(zarr_path), mode="r+")
        if "xyz_orientation" in group.attrs:
            del group.attrs["xyz_orientation"]
        group.attrs["orientation"] = legacy_orient

        # Load and verify proper conversion
        loaded_znimg = ZarrNii.from_ome_zarr(str(zarr_path))
        assert loaded_znimg.orientation == expected_xyz
        assert loaded_znimg.xyz_orientation == expected_xyz


def test_orientation_round_trip_with_new_format(tmp_path):
    """Test that round-trip operations preserve orientation with new format."""
    import dask.array as da
    import numpy as np

    data = np.random.rand(1, 32, 32, 32).astype(np.float32)
    dask_data = da.from_array(data, chunks=(1, 16, 16, 16))

    test_orientations = ["RAS", "LPI", "RAI", "LPS", "LAI", "RPI"]

    for orient in test_orientations:
        zarr_path = tmp_path / f"test_roundtrip_{orient}.zarr"

        # Create, save and reload
        znimg = ZarrNii.from_darr(dask_data, orientation=orient, axes_order="ZYX")
        znimg.to_ome_zarr(str(zarr_path))
        loaded_znimg = ZarrNii.from_ome_zarr(str(zarr_path))

        # Apply transformations and verify orientation preservation
        transformed = loaded_znimg.downsample(factors=2)
        cropped = transformed.crop(bbox_min=(0, 0, 0), bbox_max=(1, 16, 16, 16))

        # All should preserve the original orientation
        assert loaded_znimg.orientation == orient
        assert transformed.orientation == orient
        assert cropped.orientation == orient

        # All should have xyz_orientation set correctly
        assert loaded_znimg.xyz_orientation == orient
        assert transformed.xyz_orientation == orient
        assert cropped.xyz_orientation == orient
