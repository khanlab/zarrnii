"""
Tests for scaled processing plugins functionality.

This module tests the multi-resolution plugin architecture and the bias field
correction implementation to ensure they work correctly with ZarrNii images.
"""

import dask.array as da
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from zarrnii import ZarrNii
from zarrnii.plugins import GaussianBiasFieldCorrection, hookimpl

# Import HAS_ANTSPYX for conditional tests
try:
    from zarrnii.plugins.scaled_processing.n4_biasfield import HAS_ANTSPYX
except ImportError:
    HAS_ANTSPYX = False


class TestScaledProcessingPlugin:
    """Test the pluggy-based scaled processing plugin interface."""

    def test_plugin_interface(self):
        """Test that plain pluggy-style plugins implement the required interface."""

        # Create a minimal implementation using @hookimpl only (no inheritance)
        class TestPlugin:
            @hookimpl
            def lowres_func(self, lowres_array):
                # Simple identity operation
                return lowres_array

            @hookimpl
            def highres_func(self, fullres_array, upsampled_output):
                # Simple multiplication by 2
                return fullres_array * 2

            @hookimpl
            def scaled_processing_plugin_name(self):
                return "Test Plugin"

            @hookimpl
            def scaled_processing_plugin_description(self):
                return "A test plugin"

        plugin = TestPlugin()

        # Test methods directly (no properties required)
        assert plugin.scaled_processing_plugin_name() == "Test Plugin"
        assert plugin.scaled_processing_plugin_description() == "A test plugin"

        # Test lowres function
        test_lowres = np.random.rand(10, 10).astype(np.float32)
        lowres_result = plugin.lowres_func(test_lowres)
        assert lowres_result.shape == test_lowres.shape
        np.testing.assert_array_equal(lowres_result, test_lowres)

        # Test highres function - now both arrays are NumPy arrays (block-level)
        test_fullres = np.random.rand(20, 20).astype(np.float32)
        # Create upsampled version (same size as fullres for new interface)
        upsampled_result = lowres_result
        highres_result = plugin.highres_func(test_fullres, upsampled_result)
        assert highres_result.shape == test_fullres.shape
        # Result should be double the original
        np.testing.assert_array_almost_equal(highres_result, test_fullres * 2)


class TestGaussianBiasFieldCorrection:
    """Test the GaussianBiasFieldCorrection plugin."""

    def test_bias_field_plugin_initialization(self):
        """Test GaussianBiasFieldCorrection plugin initialization."""
        plugin = GaussianBiasFieldCorrection(sigma=3.0, mode="constant")

        assert (
            plugin.scaled_processing_plugin_name() == "Gaussian Bias Field Correction"
        )
        assert (
            "Multi-resolution bias field correction"
            in plugin.scaled_processing_plugin_description()
        )
        assert plugin.sigma == 3.0
        assert plugin.mode == "constant"

    def test_lowres_func_basic(self):
        """Test the lowres_func with basic input."""
        plugin = GaussianBiasFieldCorrection(sigma=1.0)

        # Create test data with a simple gradient (simulating bias field)
        test_data = np.ones((20, 20), dtype=np.float32)
        test_data[:, :10] = 0.5  # Create intensity variation

        result = plugin.lowres_func(test_data)

        # Result should have the same shape
        assert result.shape == test_data.shape
        # Result should be smoothed version
        assert np.all(result > 0)  # No zeros due to smoothing

    def test_lowres_func_3d(self):
        """Test the lowres_func with 3D input."""
        plugin = GaussianBiasFieldCorrection(sigma=1.0)

        # Create 3D test data
        test_data = np.ones((10, 20, 20), dtype=np.float32)
        test_data[:, :, :10] = 0.5

        result = plugin.lowres_func(test_data)

        assert result.shape == test_data.shape
        assert np.all(result > 0)

    def test_lowres_func_edge_cases(self):
        """Test lowres_func with edge cases."""
        plugin = GaussianBiasFieldCorrection()

        # Test empty array
        with pytest.raises(ValueError, match="Input array is empty"):
            plugin.lowres_func(np.array([]))

        # Test 1D array
        with pytest.raises(ValueError, match="Input array must be at least 2D"):
            plugin.lowres_func(np.array([1, 2, 3]))

    def test_highres_func_application(self):
        """Test the highres_func with upsampled bias field."""
        plugin = GaussianBiasFieldCorrection()

        # Create test data - both arrays are NumPy arrays (block-level)
        fullres_data = np.ones((40, 40), dtype=np.float32) * 100  # Original image
        upsampled_bias = np.ones((40, 40), dtype=np.float32) * 2  # Upsampled bias field

        result = plugin.highres_func(fullres_data, upsampled_bias)

        # Result should have same shape as full-res data
        assert result.shape == fullres_data.shape

        # Result should be approximately original divided by bias (100/2 = 50)
        expected = 50.0
        assert_array_almost_equal(result, np.full_like(result, expected), decimal=1)

    def test_highres_func_division_by_zero_protection(self):
        """Test highres_func protects against division by zero."""
        plugin = GaussianBiasFieldCorrection()

        # Test with some zero values in bias field
        fullres_data = np.ones((20, 20), dtype=np.float32) * 100
        upsampled_bias = np.zeros((20, 20), dtype=np.float32)  # All zeros

        result = plugin.highres_func(fullres_data, upsampled_bias)

        assert result.shape == fullres_data.shape
        # Should not result in inf values
        assert np.all(np.isfinite(result))


class TestZarrNiiScaledProcessingIntegration:
    """Test integration of scaled processing with ZarrNii."""

    @pytest.mark.usefixtures("cleandir")
    def test_apply_scaled_processing_method_plugin_instance(self, nifti_nib):
        """Test apply_scaled_processing method with plugin instance."""
        nifti_nib.to_filename("test.nii")
        znimg = ZarrNii.from_nifti("test.nii", axes_order="ZYX")

        plugin = GaussianBiasFieldCorrection(sigma=2.0)
        result = znimg.apply_scaled_processing(plugin, downsample_factor=2)

        # Check that we get a new ZarrNii instance
        assert isinstance(result, ZarrNii)
        assert result.shape == znimg.shape

    @pytest.mark.usefixtures("cleandir")
    def test_apply_scaled_processing_method_plugin_class(self, nifti_nib):
        """Test apply_scaled_processing method with plugin class."""
        nifti_nib.to_filename("test.nii")
        znimg = ZarrNii.from_nifti("test.nii", axes_order="ZYX")

        result = znimg.apply_scaled_processing(
            GaussianBiasFieldCorrection, sigma=1.0, downsample_factor=2
        )

        assert isinstance(result, ZarrNii)
        assert result.shape == znimg.shape

    @pytest.mark.skipif(not HAS_ANTSPYX, reason="antspyx not available")
    @pytest.mark.usefixtures("cleandir")
    def test_apply_scaled_processing_n4_plugin_instance(self, nifti_nib):
        """Test apply_scaled_processing method with N4 plugin instance."""
        from zarrnii.plugins.scaled_processing.n4_biasfield import N4BiasFieldCorrection

        nifti_nib.to_filename("test.nii")
        znimg = ZarrNii.from_nifti("test.nii", axes_order="ZYX")

        plugin = N4BiasFieldCorrection(
            convergence={"iters": [10], "tol": 0.01},
            shrink_factor=2,
        )
        result = znimg.apply_scaled_processing(plugin, downsample_factor=2)

        # Check that we get a new ZarrNii instance
        assert isinstance(result, ZarrNii)
        assert result.shape == znimg.shape

    @pytest.mark.skipif(not HAS_ANTSPYX, reason="antspyx not available")
    @pytest.mark.usefixtures("cleandir")
    def test_apply_scaled_processing_n4_plugin_class(self, nifti_nib):
        """Test apply_scaled_processing method with N4 plugin class."""
        from zarrnii.plugins.scaled_processing.n4_biasfield import N4BiasFieldCorrection

        nifti_nib.to_filename("test.nii")
        znimg = ZarrNii.from_nifti("test.nii", axes_order="ZYX")

        result = znimg.apply_scaled_processing(
            N4BiasFieldCorrection,
            convergence={"iters": [5], "tol": 0.01},
            downsample_factor=2,
        )

        assert isinstance(result, ZarrNii)
        assert result.shape == znimg.shape

    @pytest.mark.usefixtures("cleandir")
    def test_apply_scaled_processing_invalid_plugin(self, nifti_nib):
        """Test apply_scaled_processing with invalid plugin."""
        nifti_nib.to_filename("test.nii")
        znimg = ZarrNii.from_nifti("test.nii", axes_order="ZYX")

        with pytest.raises(
            TypeError,
            match="Plugin must have callable 'lowres_func' and 'highres_func' methods",
        ):
            znimg.apply_scaled_processing("not_a_plugin")

    @pytest.mark.usefixtures("cleandir")
    def test_apply_scaled_processing_basic(self, nifti_nib):
        """Test apply_scaled_processing basic usage with a plugin instance."""
        nifti_nib.to_filename("test.nii")
        znimg = ZarrNii.from_nifti("test.nii", axes_order="ZYX")

        plugin = GaussianBiasFieldCorrection(sigma=1.0)
        # map_blocks approach does not need a chunk_size parameter
        result = znimg.apply_scaled_processing(plugin, downsample_factor=2)

        assert isinstance(result, ZarrNii)
        assert result.shape == znimg.shape

    @pytest.mark.usefixtures("cleandir")
    def test_apply_scaled_processing_unknown_kwargs_with_instance_raises(
        self, nifti_nib
    ):
        """Test apply_scaled_processing raises TypeError for unknown kwargs with instance."""
        nifti_nib.to_filename("test.nii")
        znimg = ZarrNii.from_nifti("test.nii", axes_order="ZYX")

        plugin = GaussianBiasFieldCorrection(sigma=1.0)

        # Passing unknown kwargs with an instance should raise TypeError
        with pytest.raises(TypeError, match="unexpected keyword arguments"):
            znimg.apply_scaled_processing(
                plugin, downsample_factor=2, use_temp_zarr=False
            )

        with pytest.raises(TypeError, match="unexpected keyword arguments"):
            znimg.apply_scaled_processing(
                plugin, downsample_factor=2, temp_zarr_path="/tmp/test.zarr"
            )

    @pytest.mark.usefixtures("cleandir")
    def test_apply_scaled_processing_5d_data(self):
        """Test apply_scaled_processing with 5D data (T, C, Z, Y, X)."""
        # Create 5D test data with singleton time and channel dimensions
        data_5d = np.random.rand(1, 1, 20, 20, 20).astype(np.float32) * 100
        dask_data = da.from_array(data_5d, chunks=(1, 1, 10, 10, 10))

        znimg_5d = ZarrNii.from_darr(
            dask_data, spacing=(1.0, 1.0, 1.0), dims=["t", "c", "z", "y", "x"]
        )

        plugin = GaussianBiasFieldCorrection(sigma=1.0)

        # Test with default chunk size (should be spatial only)
        result = znimg_5d.apply_scaled_processing(plugin, downsample_factor=2)

        assert isinstance(result, ZarrNii)
        assert result.shape == znimg_5d.shape
        assert len(result.dims) == 5

    @pytest.mark.usefixtures("cleandir")
    def test_apply_scaled_processing_5d_data_plugin_instance_with_invalid_kwargs(self):
        """Test apply_scaled_processing raises TypeError when kwargs passed with instance."""
        # Create 5D test data with singleton time and channel dimensions
        data_5d = np.random.rand(1, 1, 20, 20, 20).astype(np.float32) * 100
        dask_data = da.from_array(data_5d, chunks=(1, 1, 10, 10, 10))

        znimg_5d = ZarrNii.from_darr(
            dask_data, spacing=(1.0, 1.0, 1.0), dims=["t", "c", "z", "y", "x"]
        )

        plugin = GaussianBiasFieldCorrection(sigma=1.0)

        # Passing extra kwargs with a plugin instance should raise TypeError
        with pytest.raises(TypeError, match="unexpected keyword arguments"):
            znimg_5d.apply_scaled_processing(
                plugin, downsample_factor=2, chunk_size=(16, 16, 16)
            )


class TestScaledProcessingWorkflow:
    """Test complete scaled processing workflows."""

    @pytest.mark.usefixtures("cleandir")
    def test_bias_correction_workflow(self, nifti_nib):
        """Test complete bias field correction workflow."""
        nifti_nib.to_filename("test.nii")
        znimg = ZarrNii.from_nifti("test.nii", axes_order="ZYX")

        # Apply bias field correction
        corrected = znimg.apply_scaled_processing(
            GaussianBiasFieldCorrection(sigma=2.0), downsample_factor=4
        )

        # Check properties are preserved
        assert corrected.axes_order == znimg.axes_order
        assert corrected.orientation == znimg.orientation
        assert corrected.shape == znimg.shape

        # Check that data values are reasonable (not all zeros or infinities)
        corrected_data = corrected.data.compute()
        assert not np.all(corrected_data == 0)
        assert np.all(np.isfinite(corrected_data))

    def test_plugin_repr(self):
        """Test plugin string representation."""
        plugin = GaussianBiasFieldCorrection(sigma=3.0, mode="reflect")
        repr_str = repr(plugin)

        assert "GaussianBiasFieldCorrection" in repr_str
        assert "sigma=3.0" in repr_str
        assert "mode=reflect" in repr_str


class TestN4BiasFieldCorrection:
    """Test the N4BiasFieldCorrection plugin."""

    def test_n4_plugin_import_without_antspyx(self):
        """Test N4BiasFieldCorrection import behavior without antspyx."""
        # This test would need a way to mock the import failure
        # For now, we'll test that the plugin can be imported when antspyx is available
        from zarrnii.plugins.scaled_processing.n4_biasfield import HAS_ANTSPYX

        if HAS_ANTSPYX:
            from zarrnii.plugins.scaled_processing.n4_biasfield import (
                N4BiasFieldCorrection,
            )

            plugin = N4BiasFieldCorrection()
            assert plugin.scaled_processing_plugin_name() == "N4 Bias Field Correction"

    @pytest.mark.skipif(not HAS_ANTSPYX, reason="antspyx not available")
    def test_n4_plugin_initialization(self):
        """Test N4BiasFieldCorrection plugin initialization."""
        from zarrnii.plugins.scaled_processing.n4_biasfield import N4BiasFieldCorrection

        plugin = N4BiasFieldCorrection(
            convergence={"iters": [25], "tol": 0.002},
            shrink_factor=2,
        )

        assert plugin.scaled_processing_plugin_name() == "N4 Bias Field Correction"
        assert (
            "Multi-resolution N4 bias field correction"
            in plugin.scaled_processing_plugin_description()
        )
        assert plugin.convergence == {"iters": [25], "tol": 0.002}
        assert plugin.shrink_factor == 2

    @pytest.mark.skipif(not HAS_ANTSPYX, reason="antspyx not available")
    def test_n4_lowres_func_basic(self):
        """Test the N4 lowres_func with basic input."""
        from zarrnii.plugins.scaled_processing.n4_biasfield import N4BiasFieldCorrection

        plugin = N4BiasFieldCorrection(convergence={"iters": [10], "tol": 0.01})

        # Create test data with intensity variation (simulating bias field)
        test_data = np.ones((20, 20), dtype=np.float32) * 100
        test_data[:, :10] = 50  # Create intensity variation

        result = plugin.lowres_func(test_data)

        # Result should have the same shape
        assert result.shape == test_data.shape
        # Result should be positive (bias field)
        assert np.all(result > 0)
        # Result should be finite
        assert np.all(np.isfinite(result))

    @pytest.mark.skipif(not HAS_ANTSPYX, reason="antspyx not available")
    def test_n4_lowres_func_3d(self):
        """Test the N4 lowres_func with 3D input."""
        from zarrnii.plugins.scaled_processing.n4_biasfield import N4BiasFieldCorrection

        plugin = N4BiasFieldCorrection(convergence={"iters": [5], "tol": 0.01})

        # Create 3D test data
        test_data = np.ones((10, 20, 20), dtype=np.float32) * 100
        test_data[:, :, :10] = 50

        result = plugin.lowres_func(test_data)

        assert result.shape == test_data.shape
        assert np.all(result > 0)
        assert np.all(np.isfinite(result))

    @pytest.mark.skipif(not HAS_ANTSPYX, reason="antspyx not available")
    def test_n4_lowres_func_edge_cases(self):
        """Test N4 lowres_func with edge cases."""
        from zarrnii.plugins.scaled_processing.n4_biasfield import N4BiasFieldCorrection

        plugin = N4BiasFieldCorrection()

        # Test empty array
        with pytest.raises(ValueError, match="Input array is empty"):
            plugin.lowres_func(np.array([]))

        # Test 1D array
        with pytest.raises(ValueError, match="Input array must be at least 2D"):
            plugin.lowres_func(np.array([1, 2, 3]))

    @pytest.mark.skipif(not HAS_ANTSPYX, reason="antspyx not available")
    def test_n4_highres_func_application(self):
        """Test the N4 highres_func with upsampled bias field."""
        from zarrnii.plugins.scaled_processing.n4_biasfield import N4BiasFieldCorrection

        plugin = N4BiasFieldCorrection()

        # Create test data - both arrays are NumPy arrays (block-level)
        fullres_data = np.ones((40, 40), dtype=np.float32) * 100
        upsampled_bias = np.ones((40, 40), dtype=np.float32) * 2

        result = plugin.highres_func(fullres_data, upsampled_bias)

        # Result should have same shape as full-res data
        assert result.shape == fullres_data.shape

        # Result should be approximately original divided by bias (100/2 = 50)
        expected = 50.0
        assert_array_almost_equal(result, np.full_like(result, expected), decimal=1)

    @pytest.mark.skipif(not HAS_ANTSPYX, reason="antspyx not available")
    def test_n4_highres_func_division_by_zero_protection(self):
        """Test N4 highres_func protects against division by zero."""
        from zarrnii.plugins.scaled_processing.n4_biasfield import N4BiasFieldCorrection

        plugin = N4BiasFieldCorrection()

        # Test with some zero values in bias field
        fullres_data = np.ones((20, 20), dtype=np.float32) * 100
        upsampled_bias = np.zeros((20, 20), dtype=np.float32)

        result = plugin.highres_func(fullres_data, upsampled_bias)

        assert result.shape == fullres_data.shape
        # Should not result in inf values
        assert np.all(np.isfinite(result))

    def test_n4_plugin_repr(self):
        """Test N4 plugin string representation."""
        from zarrnii.plugins.scaled_processing.n4_biasfield import HAS_ANTSPYX

        if HAS_ANTSPYX:
            from zarrnii.plugins.scaled_processing.n4_biasfield import (
                N4BiasFieldCorrection,
            )

            plugin = N4BiasFieldCorrection(
                convergence={"iters": [30], "tol": 0.005},
                shrink_factor=3,
            )
            repr_str = repr(plugin)

            assert "N4BiasFieldCorrection" in repr_str
            assert "shrink_factor=3" in repr_str


class TestSegmentationCleaner:
    """Test the SegmentationCleaner plugin."""

    def test_segmentation_cleaner_plugin_initialization(self):
        """Test SegmentationCleaner plugin initialization."""
        from zarrnii.plugins.scaled_processing.segmentation_cleaner import (
            SegmentationCleaner,
        )

        plugin = SegmentationCleaner(
            mask_threshold=60, max_extent=0.2, exclusion_threshold=40
        )

        assert plugin.scaled_processing_plugin_name() == "Segmentation Cleaner"
        assert (
            "Multi-resolution segmentation cleaning"
            in plugin.scaled_processing_plugin_description()
        )
        assert plugin.mask_threshold == 60
        assert plugin.max_extent == 0.2
        assert plugin.exclusion_threshold == 40

    def test_lowres_func_basic(self):
        """Test the lowres_func with basic segmentation input."""
        from zarrnii.plugins.scaled_processing.segmentation_cleaner import (
            SegmentationCleaner,
        )

        plugin = SegmentationCleaner(mask_threshold=50, max_extent=0.15)

        # Create test data: one LARGE object with low extent (sparse coverage)
        # and one compact object with high extent
        test_data = np.zeros((100, 100), dtype=np.float32)

        # Create a large sparse object with low extent
        # Draw a thin boundary box - bounding box 40x40=1600, but only ~160 pixels
        # extent ~ 0.1
        for i in range(20, 60):
            test_data[i, 20] = 100  # Left edge
            test_data[i, 59] = 100  # Right edge
        for j in range(20, 60):
            test_data[20, j] = 100  # Top edge
            test_data[59, j] = 100  # Bottom edge

        # Create a compact object (extent = 1.0, should NOT be excluded)
        test_data[70:80, 70:80] = 100  # 10x10 solid block

        result = plugin.lowres_func(test_data)

        # Result should have the same shape
        assert result.shape == test_data.shape
        # Result should be uint8
        assert result.dtype == np.uint8
        # Result should only contain 0 or 100
        assert np.all(np.isin(result, [0, 100]))
        # The sparse object should be marked for exclusion
        assert np.any(result > 0)

    def test_lowres_func_3d(self):
        """Test the lowres_func with 3D segmentation input."""
        from zarrnii.plugins.scaled_processing.segmentation_cleaner import (
            SegmentationCleaner,
        )

        plugin = SegmentationCleaner(mask_threshold=50, max_extent=0.15)

        # Create 3D test data with a large sparse object
        test_data = np.zeros((30, 40, 40), dtype=np.float32)

        # Draw hollow 3D box
        for i in range(10, 25):
            test_data[i, 10, 10] = 100
            test_data[i, 10, 30] = 100
            test_data[i, 30, 10] = 100
            test_data[i, 30, 30] = 100
        for j in range(10, 31):
            test_data[10, j, 10] = 100
            test_data[10, j, 30] = 100
            test_data[25, j, 10] = 100
            test_data[25, j, 30] = 100
        for k in range(10, 31):
            test_data[10, 10, k] = 100
            test_data[10, 30, k] = 100
            test_data[25, 10, k] = 100
            test_data[25, 30, k] = 100

        result = plugin.lowres_func(test_data)

        assert result.shape == test_data.shape
        assert result.dtype == np.uint8
        assert np.all(np.isin(result, [0, 100]))

    def test_lowres_func_4d(self):
        """Test the lowres_func with 4D (batched) segmentation input."""
        from zarrnii.plugins.scaled_processing.segmentation_cleaner import (
            SegmentationCleaner,
        )

        plugin = SegmentationCleaner(mask_threshold=50, max_extent=0.15)

        # Create 4D test data (e.g., channel dimension + 3D spatial)
        test_data = np.zeros((2, 30, 40, 40), dtype=np.float32)

        # Add sparse object to first channel - hollow box with many edges
        # Large bounding box but sparse coverage (low extent)
        for i in range(10, 25):
            for j in [10, 30]:
                for k in [10, 30]:
                    test_data[0, i, j, k] = 100
        for j in range(10, 31):
            for i in [10, 25]:
                for k in [10, 30]:
                    test_data[0, i, j, k] = 100
        for k in range(10, 31):
            for i in [10, 25]:
                for j in [10, 30]:
                    test_data[0, i, j, k] = 100

        # Add compact object to second channel
        test_data[1, 15:20, 15:20, 15:20] = 100

        result = plugin.lowres_func(test_data)

        assert result.shape == test_data.shape
        assert result.dtype == np.uint8
        assert np.all(np.isin(result, [0, 100]))
        # Second channel should have no exclusions (compact object)
        assert np.all(result[1] == 0)

    def test_lowres_func_edge_cases(self):
        """Test lowres_func with edge cases."""
        from zarrnii.plugins.scaled_processing.segmentation_cleaner import (
            SegmentationCleaner,
        )

        plugin = SegmentationCleaner()

        # Test empty array
        with pytest.raises(ValueError, match="Input array is empty"):
            plugin.lowres_func(np.array([]))

        # Test 1D array
        with pytest.raises(ValueError, match="Input array must be at least 2D"):
            plugin.lowres_func(np.array([1, 2, 3]))

    def test_highres_func_application(self):
        """Test the highres_func with upsampled exclusion mask."""
        from zarrnii.plugins.scaled_processing.segmentation_cleaner import (
            SegmentationCleaner,
        )

        plugin = SegmentationCleaner(exclusion_threshold=50)

        # Create full-resolution segmentation data as NumPy arrays (block-level)
        fullres_data = np.ones((40, 40), dtype=np.float32) * 100

        # Create upsampled exclusion mask: top-left quadrant is excluded (value 100)
        upsampled_mask = np.zeros((40, 40), dtype=np.uint8)
        upsampled_mask[:20, :20] = 100

        result = plugin.highres_func(fullres_data, upsampled_mask)

        # Result should have same shape as full-res data
        assert result.shape == fullres_data.shape

        # Check that excluded regions are zeroed
        # Top-left quadrant should be zero (excluded)
        assert np.all(result[:20, :20] == 0)
        # Other regions should remain at 100
        assert np.all(result[20:, :] == 100)
        assert np.all(result[:, 20:] == 100)

    def test_highres_func_threshold_boundary(self):
        """Test highres_func threshold behavior."""
        from zarrnii.plugins.scaled_processing.segmentation_cleaner import (
            SegmentationCleaner,
        )

        plugin = SegmentationCleaner(exclusion_threshold=50)

        # Create full-resolution data as NumPy arrays (block-level)
        fullres_data = np.ones((20, 20), dtype=np.float32) * 100

        # Create exclusion mask with values at threshold boundary
        upsampled_mask = np.full((20, 20), 49, dtype=np.uint8)
        upsampled_mask[:10, :] = 50  # Set top half to exactly 50

        result = plugin.highres_func(fullres_data, upsampled_mask)

        # Values >= 50 should be excluded (zeroed)
        assert np.all(result[:10, :] == 0)
        # Values < 50 should be preserved
        assert np.all(result[10:, :] == 100)

    def test_segmentation_cleaner_integration(self):
        """Test complete segmentation cleaning workflow."""
        from zarrnii.plugins.scaled_processing.segmentation_cleaner import (
            SegmentationCleaner,
        )

        # Create synthetic segmentation with large sparse artifact
        test_data = np.zeros((100, 100), dtype=np.float32)

        # Add compact objects (high extent, should be kept)
        test_data[10:20, 10:20] = 100  # Solid 10x10 block
        test_data[30:40, 30:40] = 100  # Another solid block

        # Add large sparse artifact (low extent ~0.1, should be removed)
        # Draw hollow rectangle - large bounding box but sparse coverage
        for i in range(50, 90):
            test_data[i, 50] = 100  # Left edge
            test_data[i, 89] = 100  # Right edge
        for j in range(50, 90):
            test_data[50, j] = 100  # Top edge
            test_data[89, j] = 100  # Bottom edge

        plugin = SegmentationCleaner(mask_threshold=50, max_extent=0.15)

        # Test lowres function
        lowres_result = plugin.lowres_func(test_data)
        assert lowres_result.shape == test_data.shape
        # Should have some exclusion mask
        assert np.any(lowres_result > 0)

        # Test highres function - now accepts/returns NumPy arrays
        highres_result = plugin.highres_func(test_data, lowres_result)
        cleaned = highres_result

        # Verify that data was cleaned
        assert cleaned.shape == test_data.shape
        # Some regions should have been zeroed out
        assert np.sum(cleaned == 0) > np.sum(test_data == 0)

    def test_plugin_repr(self):
        """Test plugin string representation."""
        from zarrnii.plugins.scaled_processing.segmentation_cleaner import (
            SegmentationCleaner,
        )

        plugin = SegmentationCleaner(
            mask_threshold=60, max_extent=0.2, exclusion_threshold=40
        )
        repr_str = repr(plugin)

        assert "SegmentationCleaner" in repr_str
        assert "mask_threshold=60" in repr_str
        assert "max_extent=0.2" in repr_str
        assert "exclusion_threshold=40" in repr_str
