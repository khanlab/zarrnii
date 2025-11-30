import ngff_zarr as nz
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

    # x and y are modified by 2^level (ome-zarr-py downsamples in xy only)
    assert_array_equal(int(xyz_orig[0] / (2**level)), xyz_ds[0])
    assert_array_equal(int(xyz_orig[1] / (2**level)), xyz_ds[1])
    # z is NOT downsampled by ome-zarr-py (only downsamples in xy plane)
    assert_array_equal(xyz_orig[2], xyz_ds[2])


@pytest.mark.usefixtures("cleandir")
def test_near_isotropic_downsampling(nifti_nib):
    """Test near-isotropic downsampling functionality."""
    nifti_nib.to_filename("test.nii")
    znimg = ZarrNii.from_nifti("test.nii")

    # Create an OME-Zarr with anisotropic voxels
    znimg.to_ome_zarr("test_anisotropic.ome.zarr", max_layer=0)

    # Use ngff-zarr to modify the scale to create anisotropic data
    # We'll make z scale much smaller (higher resolution) than x and y
    multiscales = nz.from_ngff_zarr("test_anisotropic.ome.zarr")
    original_image = multiscales.images[0]

    # Create a new NgffImage with anisotropic scales
    # z is very fine (0.1), x and y are coarse (0.4)
    modified_image = nz.NgffImage(
        data=original_image.data,
        dims=list(original_image.dims),
        scale={"c": 1.0, "z": 0.1, "y": 0.4, "x": 0.4},
        translation=original_image.translation,
        name=original_image.name,
        axes_units=original_image.axes_units,
    )

    # Save the modified image back to the same path
    new_multiscales = nz.to_multiscales(modified_image)
    nz.to_ngff_zarr("test_anisotropic.ome.zarr", new_multiscales, overwrite=True)

    # Load without downsampling
    znimg_normal = ZarrNii.from_ome_zarr(
        "test_anisotropic.ome.zarr", downsample_near_isotropic=False
    )

    # Debug: Print scales
    print(f"Normal scales: {znimg_normal.scale}")

    # Load with near-isotropic downsampling
    znimg_isotropic = ZarrNii.from_ome_zarr(
        "test_anisotropic.ome.zarr", downsample_near_isotropic=True
    )

    # Debug: Print scales and shapes
    print(f"Isotropic scales: {znimg_isotropic.scale}")
    print(f"Normal shape: {znimg_normal.shape}")
    print(f"Isotropic shape: {znimg_isotropic.shape}")

    # The z dimension should be downsampled by a factor of 4 (since 0.4/0.1 = 4, and 2^2 = 4)
    # Check that the z dimension was downsampled
    if znimg_normal.axes_order == "ZYX":
        z_dim_idx = 1  # z is the second dimension (after channel)
    else:  # XYZ
        z_dim_idx = 3  # z is the fourth dimension (after channel, x, y)

    # The z dimension should be roughly 1/4 the size due to downsampling by factor of 4
    normal_z_size = znimg_normal.shape[z_dim_idx]
    isotropic_z_size = znimg_isotropic.shape[z_dim_idx]

    # Verify that downsampling occurred by factor of 4 (allowing for rounding)
    expected_size = normal_z_size // 4
    assert abs(isotropic_z_size - expected_size) <= 1

    # Check that the scales are more isotropic
    normal_scales = [
        znimg_normal.scale[dim] for dim in ["x", "y", "z"] if dim in znimg_normal.scale
    ]
    isotropic_scales = [
        znimg_isotropic.scale[dim]
        for dim in ["x", "y", "z"]
        if dim in znimg_isotropic.scale
    ]

    # Calculate the ratio of max to min scale
    normal_ratio = max(normal_scales) / min(normal_scales)
    isotropic_ratio = max(isotropic_scales) / min(isotropic_scales)

    # The isotropic version should have a smaller ratio (more isotropic)
    assert isotropic_ratio < normal_ratio


@pytest.mark.usefixtures("cleandir")
def test_near_isotropic_downsampling_no_effect(nifti_nib):
    """Test that near-isotropic downsampling has no effect when voxels are already isotropic."""
    nifti_nib.to_filename("test.nii")
    znimg = ZarrNii.from_nifti("test.nii")

    # Create an OME-Zarr and manually set isotropic scales
    znimg.to_ome_zarr("test_isotropic.ome.zarr", max_layer=0)

    # Use ngff-zarr to modify the scale to ensure isotropy
    multiscales = nz.from_ngff_zarr("test_isotropic.ome.zarr")
    original_image = multiscales.images[0]

    # Create a new NgffImage with isotropic scales
    modified_image = nz.NgffImage(
        data=original_image.data,
        dims=list(original_image.dims),
        scale={"c": 1.0, "z": 1.0, "y": 1.0, "x": 1.0},
        translation=original_image.translation,
        name=original_image.name,
        axes_units=original_image.axes_units,
    )

    # Save the modified image back to the same path
    new_multiscales = nz.to_multiscales(modified_image)
    nz.to_ngff_zarr("test_isotropic.ome.zarr", new_multiscales, overwrite=True)

    # Load without downsampling
    znimg_normal = ZarrNii.from_ome_zarr(
        "test_isotropic.ome.zarr", downsample_near_isotropic=False
    )

    # Load with near-isotropic downsampling (should have no effect)
    znimg_isotropic = ZarrNii.from_ome_zarr(
        "test_isotropic.ome.zarr", downsample_near_isotropic=True
    )

    # Shapes should be identical since no downsampling is needed
    assert znimg_normal.shape == znimg_isotropic.shape

    # Scales should be identical
    for dim in ["x", "y", "z"]:
        if dim in znimg_normal.scale and dim in znimg_isotropic.scale:
            assert abs(znimg_normal.scale[dim] - znimg_isotropic.scale[dim]) < 1e-6


@pytest.mark.usefixtures("cleandir")
def test_near_isotropic_downsampling_parameter_validation(nifti_nib):
    """Test that the downsample_near_isotropic parameter is properly handled."""
    nifti_nib.to_filename("test.nii")
    znimg = ZarrNii.from_nifti("test.nii")
    znimg.to_ome_zarr("test_param.ome.zarr", max_layer=0)

    # Test with explicit False
    znimg1 = ZarrNii.from_ome_zarr(
        "test_param.ome.zarr", downsample_near_isotropic=False
    )

    # Test with explicit True
    znimg2 = ZarrNii.from_ome_zarr(
        "test_param.ome.zarr", downsample_near_isotropic=True
    )

    # Test default behavior (should be same as False)
    znimg3 = ZarrNii.from_ome_zarr("test_param.ome.zarr")

    # Default and explicit False should be identical
    assert znimg1.shape == znimg3.shape
    for dim in ["x", "y", "z"]:
        if dim in znimg1.scale and dim in znimg3.scale:
            assert abs(znimg1.scale[dim] - znimg3.scale[dim]) < 1e-6


@pytest.mark.usefixtures("cleandir")
def test_near_isotropic_downsampling_skipped_when_multiple_dims(nifti_nib):
    """Test that downsampling is skipped when multiple dimensions need correction.

    When two or more dimensions require downsampling to become isotropic (e.g.,
    x=0.1, y=0.1, z=0.4), the logic should skip downsampling entirely to avoid
    incorrectly downsampling x and y to match z.
    """
    # Define test scale values for clarity
    fine_scale = 0.1  # High resolution (x and y)
    coarse_scale = 0.4  # Low resolution (z)

    nifti_nib.to_filename("test.nii")
    znimg = ZarrNii.from_nifti("test.nii")

    # Create an OME-Zarr with anisotropic voxels where TWO dimensions need correction
    znimg.to_ome_zarr("test_multi_anisotropic.ome.zarr", max_layer=0)

    # Modify the scale using ngff-zarr
    multiscales = nz.from_ngff_zarr("test_multi_anisotropic.ome.zarr")
    original_image = multiscales.images[0]

    # Create a new NgffImage where x and y are fine and z is coarse
    # This means both x and y would need downsampling to match z
    modified_image = nz.NgffImage(
        data=original_image.data,
        dims=list(original_image.dims),
        scale={"c": 1.0, "z": coarse_scale, "y": fine_scale, "x": fine_scale},
        translation=original_image.translation,
        name=original_image.name,
        axes_units=original_image.axes_units,
    )

    # Save the modified image back
    new_multiscales = nz.to_multiscales(modified_image)
    nz.to_ngff_zarr("test_multi_anisotropic.ome.zarr", new_multiscales, overwrite=True)

    # Load without downsampling
    znimg_normal = ZarrNii.from_ome_zarr(
        "test_multi_anisotropic.ome.zarr", downsample_near_isotropic=False
    )

    # Load with near-isotropic downsampling
    # Since two dimensions (x and y) would need correction, downsampling should be skipped
    znimg_isotropic = ZarrNii.from_ome_zarr(
        "test_multi_anisotropic.ome.zarr", downsample_near_isotropic=True
    )

    # Shapes should be identical since downsampling is skipped when multiple dims need correction
    assert znimg_normal.shape == znimg_isotropic.shape

    # Scales should be identical (no downsampling applied)
    for dim in ["x", "y", "z"]:
        if dim in znimg_normal.scale and dim in znimg_isotropic.scale:
            assert abs(znimg_normal.scale[dim] - znimg_isotropic.scale[dim]) < 1e-6


@pytest.mark.usefixtures("cleandir")
def test_get_upsampled_chunks_function(nifti_nib):
    """Test the __get_upsampled_chunks function ensures exact target shapes."""
    nifti_nib.to_filename("test.nii")
    znimg = ZarrNii.from_nifti("test.nii")

    # Test various target shapes to ensure chunks sum up exactly
    test_cases = [
        (1, 100, 50, 200),  # Simple case
        (1, 127, 63, 255),  # Odd numbers
        (1, 200, 100, 400),  # Double size
        (1, 75, 37, 150),  # Non-power-of-2 factors
    ]

    for target_shape in test_cases:
        # Get the upsampled chunks
        new_chunks, scaling = znimg._ZarrNii__get_upsampled_chunks(target_shape)

        # Check that chunks sum up to exactly the target shape
        for dim, (target_dim, chunks_dim) in enumerate(zip(target_shape, new_chunks)):
            chunks_sum = sum(chunks_dim)
            assert (
                chunks_sum == target_dim
            ), f"Dimension {dim}: chunks sum {chunks_sum} != target {target_dim}"

        # Check that scaling factors are reasonable
        for dim, (orig_dim, target_dim, scale) in enumerate(
            zip(znimg.shape, target_shape, scaling)
        ):
            expected_scale = target_dim / orig_dim
            assert (
                abs(scale - expected_scale) < 1e-10
            ), f"Dimension {dim}: scaling {scale} != expected {expected_scale}"


@pytest.mark.usefixtures("cleandir")
def test_upsample_to_shape_exact_match(nifti_nib):
    """Test that upsample with to_shape produces exactly the target shape."""
    nifti_nib.to_filename("test.nii")
    znimg = ZarrNii.from_nifti("test.nii")

    # Test various target shapes
    test_cases = [
        (1, 100, 50, 200),  # Simple case
        (1, 127, 63, 255),  # Odd numbers
        (1, 200, 100, 400),  # Double size
        (1, 75, 37, 150),  # Non-power-of-2 factors
    ]

    for target_shape in test_cases:
        upsampled = znimg.upsample(to_shape=target_shape)
        assert (
            upsampled.shape == target_shape
        ), f"Upsampled shape {upsampled.shape} != target {target_shape}"

        # Also verify the data shape matches (not just the ZarrNii shape property)
        assert (
            upsampled.data.shape == target_shape
        ), f"Data shape {upsampled.data.shape} != target {target_shape}"
