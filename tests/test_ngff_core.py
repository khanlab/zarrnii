"""
Tests for the NgffImage-based function API (now integrated in core).
"""

import os
import tempfile

import dask.array as da
import ngff_zarr as nz
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from zarrnii.core import (
    apply_transform_to_ngff_image,
    crop_ngff_image,
    downsample_ngff_image,
    get_multiscales,
    load_ngff_image,
    save_ngff_image,
)


class TestNgffImageFunctions:
    """Test the NgffImage-based function API."""

    @pytest.fixture
    def simple_ngff_image(self):
        """Create a simple NgffImage for testing."""
        # Create test data
        data = da.random.random((1, 32, 64, 64), chunks=(1, 16, 32, 32))

        # Create NgffImage
        ngff_image = nz.NgffImage(
            data=data,
            dims=["c", "z", "y", "x"],
            scale={"z": 2.0, "y": 1.0, "x": 1.0},
            translation={"z": 0.0, "y": 0.0, "x": 0.0},
            name="test_image",
        )

        return ngff_image

    @pytest.fixture
    def temp_zarr_store(self, simple_ngff_image):
        """Create a temporary OME-Zarr store for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, "test.zarr")

            # Create multiscales and save to zarr
            multiscales = nz.to_multiscales(simple_ngff_image, scale_factors=[2, 4])
            nz.to_ngff_zarr(zarr_path, multiscales)

            yield zarr_path

    def test_load_ngff_image(self, temp_zarr_store):
        """Test loading NgffImage from OME-Zarr store."""
        ngff_image = load_ngff_image(temp_zarr_store, level=0)

        assert ngff_image is not None
        assert ngff_image.data.shape == (1, 32, 64, 64)
        assert list(ngff_image.dims) == ["c", "z", "y", "x"]
        assert ngff_image.scale["z"] == 2.0
        assert ngff_image.scale["y"] == 1.0
        assert ngff_image.scale["x"] == 1.0

    def test_load_ngff_image_different_level(self, temp_zarr_store):
        """Test loading different pyramid levels."""
        ngff_level0 = load_ngff_image(temp_zarr_store, level=0)
        ngff_level1 = load_ngff_image(temp_zarr_store, level=1)

        # Level 1 should be smaller due to downsampling
        assert ngff_level1.data.shape[1] < ngff_level0.data.shape[1]  # Z dimension
        assert ngff_level1.data.shape[2] < ngff_level0.data.shape[2]  # Y dimension
        assert ngff_level1.data.shape[3] < ngff_level0.data.shape[3]  # X dimension

    def test_save_ngff_image(self, simple_ngff_image):
        """Test saving NgffImage to OME-Zarr."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.zarr")
            save_ngff_image(simple_ngff_image, output_path, max_layer=3)

            # Verify the output exists and can be loaded
            assert os.path.exists(output_path)

            # Load back and verify
            reloaded = load_ngff_image(output_path, level=0)
            assert reloaded.data.shape == simple_ngff_image.data.shape
            assert list(reloaded.dims) == list(simple_ngff_image.dims)

    def test_get_multiscales(self, temp_zarr_store):
        """Test getting full multiscales object."""
        multiscales = get_multiscales(temp_zarr_store)

        assert multiscales is not None
        assert len(multiscales.images) == 3  # Original + 2 downsampled levels

        # Check that each level has appropriate shape
        for i, image in enumerate(multiscales.images):
            expected_scale_factor = 2**i
            if i == 0:
                assert image.data.shape == (1, 32, 64, 64)
            else:
                # Each level should be smaller
                assert image.data.shape[1] <= multiscales.images[i - 1].data.shape[1]

    def test_crop_ngff_image(self):
        """Test cropping an NgffImage."""
        # Create test image
        data = da.ones((1, 32, 64, 64), chunks=(1, 16, 32, 32))
        ngff_image = nz.NgffImage(
            data=data,
            dims=["c", "z", "y", "x"],
            scale={"z": 2.0, "y": 1.0, "x": 1.0},
            translation={"z": 10.0, "y": 20.0, "x": 30.0},
            name="test_image",
        )

        # Crop to a smaller region
        bbox_min = {"z": 5, "y": 10, "x": 15}  # Z, Y, X
        bbox_max = {"z": 15, "y": 30, "x": 35}  # Z, Y, X

        dim_flips = {
            "x": 1,
            "y": 1,
            "z": 1,
        }  # would normally get this from _axcodes2flips
        cropped = crop_ngff_image(ngff_image, bbox_min, bbox_max, dim_flips)

        # Check new shape
        expected_shape = (1, 10, 20, 20)  # C unchanged, Z, Y, X cropped
        assert cropped.data.shape == expected_shape

        # Check translation is updated
        assert cropped.translation["z"] == 10.0 + (5 * 2.0)  # original + offset * scale
        assert cropped.translation["y"] == 20.0 + (10 * 1.0)
        assert cropped.translation["x"] == 30.0 + (15 * 1.0)

        # Scale should be unchanged
        assert cropped.scale == ngff_image.scale

    def test_downsample_ngff_image_isotropic(self):
        """Test isotropic downsampling of an NgffImage."""
        # Create test image
        data = da.ones((1, 32, 64, 64), chunks=(1, 16, 32, 32))
        ngff_image = nz.NgffImage(
            data=data,
            dims=["c", "z", "y", "x"],
            scale={"z": 2.0, "y": 1.0, "x": 1.0},
            translation={"z": 10.0, "y": 20.0, "x": 30.0},
            name="test_image",
        )

        downsampled = downsample_ngff_image(ngff_image, factors=2)

        # Check new shape (every spatial dimension should be halved)
        expected_shape = (1, 16, 32, 32)  # C unchanged, Z, Y, X downsampled by 2
        assert downsampled.data.shape == expected_shape

        # Check scale is updated
        assert downsampled.scale["z"] == 4.0  # 2.0 * 2
        assert downsampled.scale["y"] == 2.0  # 1.0 * 2
        assert downsampled.scale["x"] == 2.0  # 1.0 * 2

        # Translation should be unchanged
        assert downsampled.translation == ngff_image.translation

    def test_downsample_ngff_image_anisotropic(self):
        """Test anisotropic downsampling of an NgffImage."""
        # Create test image
        data = da.ones((1, 32, 64, 64), chunks=(1, 16, 32, 32))
        ngff_image = nz.NgffImage(
            data=data,
            dims=["c", "z", "y", "x"],
            scale={"z": 2.0, "y": 1.0, "x": 1.0},
            translation={"z": 10.0, "y": 20.0, "x": 30.0},
            name="test_image",
        )

        factors = [1, 2, 4]  # Different factors for Z, Y, X
        downsampled = downsample_ngff_image(ngff_image, factors=factors)

        # Check new shape
        expected_shape = (1, 32, 32, 16)  # C unchanged, Z same, Y/2, X/4
        assert downsampled.data.shape == expected_shape

        # Check scale is updated appropriately
        assert downsampled.scale["z"] == 2.0  # 2.0 * 1
        assert downsampled.scale["y"] == 2.0  # 1.0 * 2
        assert downsampled.scale["x"] == 4.0  # 1.0 * 4

    def test_apply_transform_to_ngff_image_placeholder(self):
        """Test the transform function placeholder."""
        from zarrnii.transform import AffineTransform

        # Create test images
        data = da.ones((1, 32, 64, 64), chunks=(1, 16, 32, 32))
        test_ngff_image = nz.NgffImage(
            data=data,
            dims=["c", "z", "y", "x"],
            scale={"z": 2.0, "y": 1.0, "x": 1.0},
            translation={"z": 10.0, "y": 20.0, "x": 30.0},
            name="test_image",
        )

        # Create a simple identity transform
        identity_transform = AffineTransform.identity()

        # Create a reference image (same as test image for now)
        reference = test_ngff_image

        # Apply transform (currently a placeholder)
        transformed = apply_transform_to_ngff_image(
            test_ngff_image, identity_transform, reference
        )

        # For now, this is just a placeholder that returns reference data
        assert transformed.data.shape == reference.data.shape
        assert transformed.dims == reference.dims
        assert transformed.scale == reference.scale


class TestOmeZarrWriter:
    """Test the ome-zarr-py based writer."""

    @pytest.fixture
    def simple_ngff_image(self):
        """Create a simple NgffImage for testing."""
        # Create test data
        data = da.random.random((1, 32, 64, 64), chunks=(1, 16, 32, 32))

        # Create NgffImage
        ngff_image = nz.NgffImage(
            data=data,
            dims=["c", "z", "y", "x"],
            scale={"z": 2.0, "y": 1.0, "x": 1.0},
            translation={"z": 0.0, "y": 0.0, "x": 0.0},
            name="test_image",
        )

        return ngff_image

    def test_save_ngff_image_with_ome_zarr(self, simple_ngff_image):
        """Test saving NgffImage using ome-zarr-py library."""
        from zarrnii.core import save_ngff_image_with_ome_zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.zarr")
            save_ngff_image_with_ome_zarr(simple_ngff_image, output_path, max_layer=3)

            # Verify the output exists and can be loaded
            assert os.path.exists(output_path)

            # Load back and verify with load_ngff_image
            reloaded = load_ngff_image(output_path, level=0)
            assert reloaded.data.shape == simple_ngff_image.data.shape
            assert list(reloaded.dims) == list(simple_ngff_image.dims)

    def test_save_ngff_image_with_ome_zarr_custom_scale_factors(
        self, simple_ngff_image
    ):
        """Test saving with custom scale factors."""
        from zarrnii.core import save_ngff_image_with_ome_zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.zarr")
            save_ngff_image_with_ome_zarr(
                simple_ngff_image, output_path, scale_factors=[2, 4], max_layer=3
            )

            # Verify the output exists
            assert os.path.exists(output_path)

            # Load back and verify pyramid levels
            multiscales = get_multiscales(output_path)
            assert len(multiscales.images) == 3  # Original + 2 downsampled levels

    def test_save_ngff_image_with_ome_zarr_orientation(self, simple_ngff_image):
        """Test saving with xyz_orientation metadata."""
        import zarr

        from zarrnii.core import save_ngff_image_with_ome_zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.zarr")
            save_ngff_image_with_ome_zarr(
                simple_ngff_image,
                output_path,
                max_layer=2,
                xyz_orientation="RAS",
            )

            # Verify the output exists
            assert os.path.exists(output_path)

            # Check that orientation metadata was saved
            group = zarr.open_group(output_path, mode="r")
            assert "xyz_orientation" in group.attrs
            assert group.attrs["xyz_orientation"] == "RAS"

    def test_save_ngff_image_with_ome_zarr_zip(self, simple_ngff_image):
        """Test saving to ZIP file."""
        from zarrnii.core import save_ngff_image_with_ome_zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.zarr.zip")
            save_ngff_image_with_ome_zarr(simple_ngff_image, output_path, max_layer=2)

            # Verify the ZIP file was created
            assert os.path.exists(output_path)

            # Load back from ZIP and verify
            reloaded = load_ngff_image(output_path, level=0)
            assert reloaded.data.shape == simple_ngff_image.data.shape
            assert list(reloaded.dims) == list(simple_ngff_image.dims)

    def test_save_ngff_image_with_ome_zarr_no_pyramid(self, simple_ngff_image):
        """Test saving without pyramid levels (max_layer=0)."""
        from zarrnii.core import save_ngff_image_with_ome_zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.zarr")
            save_ngff_image_with_ome_zarr(simple_ngff_image, output_path, max_layer=0)

            # Verify the output exists
            assert os.path.exists(output_path)

            # Load back and verify - should only have 1 level
            multiscales = get_multiscales(output_path)
            assert len(multiscales.images) == 1  # Only original level

    def test_save_ngff_image_with_ome_zarr_z_axis_downsampling(self, simple_ngff_image):
        """Test that ome-zarr-py backend uses isotropic-aware scale factors by default.

        simple_ngff_image has anisotropic voxels (z=2, y=1, x=1).  The default
        behaviour should correct isotropy at level 1 (y and x downsampled 2×,
        z unchanged) and then downsample all dimensions uniformly at level 2.
        """
        from zarrnii.core import save_ngff_image_with_ome_zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.zarr")
            save_ngff_image_with_ome_zarr(simple_ngff_image, output_path, max_layer=3)

            # Verify the output exists
            assert os.path.exists(output_path)

            # Load back and check scales
            multiscales = get_multiscales(output_path)
            assert len(multiscales.images) == 3  # 3 pyramid levels

            z_scales = [img.scale.get("z", 1.0) for img in multiscales.images]
            x_scales = [img.scale.get("x", 1.0) for img in multiscales.images]
            y_scales = [img.scale.get("y", 1.0) for img in multiscales.images]

            # Level 1 achieves isotropy: z stays at 2.0, y and x go from 1.0 → 2.0
            assert (
                z_scales[0] == z_scales[1]
            ), f"Z should be unchanged at level 1 (already coarsest): {z_scales}"
            assert (
                y_scales[1] == 2 * y_scales[0]
            ), f"Y should double at level 1: {y_scales}"
            assert (
                x_scales[1] == 2 * x_scales[0]
            ), f"X should double at level 1: {x_scales}"

            # All dimensions should be isotropic at level 1
            assert z_scales[1] == y_scales[1] == x_scales[1], (
                f"All scales should be equal at level 1: z={z_scales[1]}, "
                f"y={y_scales[1]}, x={x_scales[1]}"
            )

            # Level 2 doubles all dimensions
            assert (
                z_scales[2] == 2 * z_scales[1]
            ), f"Z should double from level 1 to 2: {z_scales}"
            assert (
                x_scales[0] < x_scales[1] < x_scales[2]
            ), f"X scales should increase: {x_scales}"
            assert (
                y_scales[0] < y_scales[1] < y_scales[2]
            ), f"Y scales should increase: {y_scales}"

    def test_save_ngff_image_with_ome_zarr_xy_only_downsampling(
        self, simple_ngff_image
    ):
        """Test that passing integer scale_factors only downsamples xy, not z."""
        from zarrnii.core import save_ngff_image_with_ome_zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.zarr")
            # Integer scale_factors → xy-only downsampling (ome-zarr-py behaviour)
            save_ngff_image_with_ome_zarr(
                simple_ngff_image,
                output_path,
                max_layer=3,
                scale_factors=[2, 4],
            )

            assert os.path.exists(output_path)

            multiscales = get_multiscales(output_path)
            assert len(multiscales.images) == 3

            z_scales = [img.scale.get("z", 1.0) for img in multiscales.images]
            x_scales = [img.scale.get("x", 1.0) for img in multiscales.images]
            y_scales = [img.scale.get("y", 1.0) for img in multiscales.images]

            # Z scales should be constant (integer factors → xy-only)
            assert all(
                z == z_scales[0] for z in z_scales
            ), f"Z scales should be constant across pyramid levels, but got: {z_scales}"

            # X and Y scales should increase with pyramid level
            assert (
                x_scales[0] < x_scales[1] < x_scales[2]
            ), f"X scales should increase: {x_scales}"
            assert (
                y_scales[0] < y_scales[1] < y_scales[2]
            ), f"Y scales should increase: {y_scales}"

    def test_compute_isotropic_scale_factors_anisotropic(self):
        """Test _compute_isotropic_scale_factors for anisotropic input."""
        import dask.array as da

        from zarrnii.core import _compute_isotropic_scale_factors

        # z=4, y=2, x=2: z is coarsest; y and x need 2× correction
        data = da.zeros((1, 32, 64, 64), chunks=(1, 16, 32, 32))
        ngff_image = nz.NgffImage(
            data=data,
            dims=["c", "z", "y", "x"],
            scale={"z": 4.0, "y": 2.0, "x": 2.0},
            translation={"z": 0.0, "y": 0.0, "x": 0.0},
        )

        factors = _compute_isotropic_scale_factors(ngff_image, max_layer=4)

        assert len(factors) == 3, f"Expected 3 factor dicts, got {len(factors)}"

        # Level 1: z=1 (no change), y=2, x=2
        assert factors[0] == {
            "z": 1,
            "y": 2,
            "x": 2,
        }, f"Unexpected level-1 factors: {factors[0]}"
        # Level 2: z=2, y=4, x=4
        assert factors[1] == {
            "z": 2,
            "y": 4,
            "x": 4,
        }, f"Unexpected level-2 factors: {factors[1]}"
        # Level 3: z=4, y=8, x=8
        assert factors[2] == {
            "z": 4,
            "y": 8,
            "x": 8,
        }, f"Unexpected level-3 factors: {factors[2]}"

    def test_compute_isotropic_scale_factors_isotropic(self):
        """Test _compute_isotropic_scale_factors falls back to uniform 2× for isotropic input."""
        import dask.array as da

        from zarrnii.core import _compute_isotropic_scale_factors

        # All scales equal → uniform downsampling
        data = da.zeros((1, 32, 64, 64), chunks=(1, 16, 32, 32))
        ngff_image = nz.NgffImage(
            data=data,
            dims=["c", "z", "y", "x"],
            scale={"z": 1.0, "y": 1.0, "x": 1.0},
            translation={"z": 0.0, "y": 0.0, "x": 0.0},
        )

        factors = _compute_isotropic_scale_factors(ngff_image, max_layer=4)

        assert len(factors) == 3
        assert factors[0] == {
            "z": 2,
            "y": 2,
            "x": 2,
        }, f"Expected uniform 2× factors, got: {factors[0]}"
        assert factors[1] == {
            "z": 4,
            "y": 4,
            "x": 4,
        }, f"Expected uniform 4× factors, got: {factors[1]}"
        assert factors[2] == {
            "z": 8,
            "y": 8,
            "x": 8,
        }, f"Expected uniform 8× factors, got: {factors[2]}"

    def test_to_ome_zarr_isotropic_scale_factors(self):
        """Test that to_ome_zarr produces isotropic-aware pyramid for anisotropic data."""
        import dask.array as da

        from zarrnii import ZarrNii

        # Build a ZarrNii with anisotropic voxels z=4, y=2, x=2
        data = da.zeros((1, 16, 32, 32), chunks=(1, 8, 16, 16))
        ngff_image = nz.NgffImage(
            data=data,
            dims=["c", "z", "y", "x"],
            scale={"c": 1.0, "z": 4.0, "y": 2.0, "x": 2.0},
            translation={"c": 0.0, "z": 0.0, "y": 0.0, "x": 0.0},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save source zarr then load as ZarrNii
            src_path = os.path.join(tmpdir, "src.zarr")
            multiscales = nz.to_multiscales(ngff_image)
            nz.to_ngff_zarr(src_path, multiscales)
            znimg = ZarrNii.from_ome_zarr(src_path)

            # Write pyramid with default (isotropic-aware) scale factors
            out_path = os.path.join(tmpdir, "out.zarr")
            znimg.to_ome_zarr(out_path, max_layer=3)

            # Load resulting pyramid and inspect scales
            ms = get_multiscales(out_path)
            assert len(ms.images) == 3

            z_scales = [img.scale.get("z", 1.0) for img in ms.images]
            y_scales = [img.scale.get("y", 1.0) for img in ms.images]
            x_scales = [img.scale.get("x", 1.0) for img in ms.images]

            # At level 1 all spatial scales should be equal (isotropic)
            assert y_scales[1] == z_scales[1] == x_scales[1], (
                f"Level 1 should be isotropic: z={z_scales[1]}, "
                f"y={y_scales[1]}, x={x_scales[1]}"
            )
            # z stays the same from level 0 to level 1 (it was already coarsest)
            assert (
                z_scales[1] == z_scales[0]
            ), f"Z should be unchanged at level 1: {z_scales}"
            # y and x double at level 1
            assert y_scales[1] == 2 * y_scales[0]
            assert x_scales[1] == 2 * x_scales[0]
