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


@pytest.mark.usefixtures("cleandir")
def test_near_isotropic_downsampling(nifti_nib):
    """Test near-isotropic downsampling functionality."""
    nifti_nib.to_filename("test.nii")
    znimg = ZarrNii.from_nifti("test.nii")

    # Create an OME-Zarr with anisotropic voxels
    znimg.to_ome_zarr("test_anisotropic.ome.zarr", max_layer=0)

    # Manually modify the scale to create anisotropic data
    # We'll make z scale much smaller (higher resolution) than x and y
    import zarr

    store = zarr.open("test_anisotropic.ome.zarr", mode="r+")
    multiscales = store.attrs["multiscales"]

    # Modify the scale to make z resolution 4x finer than x and y
    original_transforms = multiscales[0]["datasets"][0]["coordinateTransformations"]
    for transform in original_transforms:
        if transform["type"] == "scale":
            # Make z scale much smaller (finer resolution) than x and y
            if len(transform["scale"]) >= 4:  # [c, z, y, x] for ZYX
                # Set specific scales to ensure z is the finest
                transform["scale"][-3] = 0.1  # z dimension - very fine
                transform["scale"][-2] = 0.4  # y dimension - coarse
                transform["scale"][-1] = 0.4  # x dimension - coarse

    # Update the multiscales metadata
    store.attrs["multiscales"] = multiscales

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

    # Manually modify the scale to ensure isotropy
    import zarr

    store = zarr.open("test_isotropic.ome.zarr", mode="r+")
    multiscales = store.attrs["multiscales"]

    original_transforms = multiscales[0]["datasets"][0]["coordinateTransformations"]
    for transform in original_transforms:
        if transform["type"] == "scale":
            # Set all spatial dimensions to the same scale
            if len(transform["scale"]) >= 4:  # [c, z, y, x] for ZYX
                transform["scale"][-3] = 1.0  # z dimension
                transform["scale"][-2] = 1.0  # y dimension
                transform["scale"][-1] = 1.0  # x dimension

    # Update the multiscales metadata
    store.attrs["multiscales"] = multiscales

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


@pytest.mark.usefixtures("cleandir")
def test_isotropic_downsampling_with_level(nifti_nib):
    """Test level-constrained isotropic downsampling with the new isotropic parameter."""
    nifti_nib.to_filename("test.nii")
    znimg = ZarrNii.from_nifti("test.nii")

    # Create an OME-Zarr with anisotropic voxels (matching example in issue)
    znimg.to_ome_zarr("test_level.ome.zarr", max_layer=0)

    import zarr

    store = zarr.open("test_level.ome.zarr", mode="r+")
    multiscales = store.attrs["multiscales"]

    # Set scales to match issue example: z=2.0, y=1.0, x=1.0
    original_transforms = multiscales[0]["datasets"][0]["coordinateTransformations"]
    for transform in original_transforms:
        if transform["type"] == "scale":
            if len(transform["scale"]) >= 4:  # [c, z, y, x] for ZYX
                transform["scale"][-3] = 2.0  # z dimension - already coarser
                transform["scale"][-2] = 1.0  # y dimension - fine
                transform["scale"][-1] = 1.0  # x dimension - fine

    store.attrs["multiscales"] = multiscales

    # Load without downsampling
    znimg_normal = ZarrNii.from_ome_zarr("test_level.ome.zarr", isotropic=False)

    print(f"Normal scales: {znimg_normal.scale}")
    print(f"Normal shape: {znimg_normal.shape}")

    # Load with level=2 constraint (max factor = 4 = 2^2)
    znimg_level2 = ZarrNii.from_ome_zarr("test_level.ome.zarr", isotropic=2)

    print(f"Level 2 scales: {znimg_level2.scale}")
    print(f"Level 2 shape: {znimg_level2.shape}")

    # According to the issue:
    # x and y should be downsampled by 4x (final scale: 4.0)
    # z should be downsampled by 2x (final scale: 4.0)
    # Result: z=4.0, y=4.0, x=4.0 (isotropic)

    # Check scales are near-isotropic
    scales = [znimg_level2.scale[dim] for dim in ["x", "y", "z"]]
    print(f"Final scales: {scales}")

    # All scales should be close to 4.0
    for scale_val in scales:
        assert abs(scale_val - 4.0) < 0.1, f"Scale {scale_val} not close to 4.0"

    # Check that dimensions were downsampled appropriately
    if znimg_normal.axes_order == "ZYX":
        x_idx, y_idx, z_idx = 3, 2, 1
    else:
        x_idx, y_idx, z_idx = 1, 2, 3

    # x and y should be downsampled by 4
    assert abs(znimg_normal.shape[x_idx] / znimg_level2.shape[x_idx] - 4) < 0.5
    assert abs(znimg_normal.shape[y_idx] / znimg_level2.shape[y_idx] - 4) < 0.5

    # z should be downsampled by 2
    assert abs(znimg_normal.shape[z_idx] / znimg_level2.shape[z_idx] - 2) < 0.5


@pytest.mark.usefixtures("cleandir")
def test_isotropic_downsampling_level_constraint(nifti_nib):
    """Test that level constraint prevents over-downsampling."""
    nifti_nib.to_filename("test.nii")
    znimg = ZarrNii.from_nifti("test.nii")

    # Create an OME-Zarr with highly anisotropic voxels
    znimg.to_ome_zarr("test_constraint.ome.zarr", max_layer=0)

    import zarr

    store = zarr.open("test_constraint.ome.zarr", mode="r+")
    multiscales = store.attrs["multiscales"]

    # Set extreme anisotropy: z=0.125 (8x finer), y=1.0, x=1.0
    original_transforms = multiscales[0]["datasets"][0]["coordinateTransformations"]
    for transform in original_transforms:
        if transform["type"] == "scale":
            if len(transform["scale"]) >= 4:
                transform["scale"][-3] = 0.125  # z dimension - very fine
                transform["scale"][-2] = 1.0  # y dimension
                transform["scale"][-1] = 1.0  # x dimension

    store.attrs["multiscales"] = multiscales

    # Load without constraint
    znimg_normal = ZarrNii.from_ome_zarr("test_constraint.ome.zarr", isotropic=False)

    # With level=1 (max factor = 2), z can only be downsampled 2x
    znimg_level1 = ZarrNii.from_ome_zarr("test_constraint.ome.zarr", isotropic=1)

    # z would ideally be downsampled 8x, but level=1 caps it at 2x
    # Final z scale should be 0.125 * 2 = 0.25
    # x and y stay at 1.0 (no downsampling needed to match coarsest dimension)
    print(f"Level 1 scales: {znimg_level1.scale}")

    assert abs(znimg_level1.scale["z"] - 0.25) < 0.01
    assert abs(znimg_level1.scale["x"] - 1.0) < 0.01
    assert abs(znimg_level1.scale["y"] - 1.0) < 0.01

    # With level=3 (max factor = 8), z can be fully downsampled to match x,y
    znimg_level3 = ZarrNii.from_ome_zarr("test_constraint.ome.zarr", isotropic=3)

    print(f"Level 3 scales: {znimg_level3.scale}")

    # All should be isotropic at 1.0
    for dim in ["x", "y", "z"]:
        assert abs(znimg_level3.scale[dim] - 1.0) < 0.01


@pytest.mark.usefixtures("cleandir")
def test_isotropic_parameter_compatibility(nifti_nib):
    """Test new isotropic parameter and backward compatibility with downsample_near_isotropic."""
    nifti_nib.to_filename("test.nii")
    znimg = ZarrNii.from_nifti("test.nii")
    znimg.to_ome_zarr("test_compat.ome.zarr", max_layer=0)

    # Test with new boolean parameter (should work like old parameter)
    znimg_new_true = ZarrNii.from_ome_zarr("test_compat.ome.zarr", isotropic=True)
    znimg_new_false = ZarrNii.from_ome_zarr("test_compat.ome.zarr", isotropic=False)

    # Test with integer level parameter
    znimg_level2 = ZarrNii.from_ome_zarr("test_compat.ome.zarr", isotropic=2)

    # All should load successfully (actual behavior depends on data isotropy)
    assert znimg_new_true is not None
    assert znimg_new_false is not None
    assert znimg_level2 is not None


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
