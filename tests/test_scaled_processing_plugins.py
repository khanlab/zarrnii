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
from zarrnii.plugins import GaussianBiasFieldCorrection, ScaledProcessingPlugin

# Import HAS_ANTSPYX for conditional tests
try:
    from zarrnii.plugins.scaled_processing.n4_biasfield import HAS_ANTSPYX
except ImportError:
    HAS_ANTSPYX = False


class TestScaledProcessingPlugin:
    """Test the base ScaledProcessingPlugin interface."""

    def test_abstract_plugin_cannot_be_instantiated(self):
        """Test that abstract ScaledProcessingPlugin cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ScaledProcessingPlugin()

    def test_plugin_interface(self):
        """Test that plugins implement the required interface."""

        # Create a minimal implementation for testing
        class TestPlugin(ScaledProcessingPlugin):
            def lowres_func(self, lowres_array):
                # Simple identity operation
                return lowres_array

            def highres_func(self, fullres_array, lowres_output):
                # Simple multiplication by 2
                return fullres_array * 2

            @property
            def name(self):
                return "Test Plugin"

            @property
            def description(self):
                return "A test plugin"

        plugin = TestPlugin(param1=10, param2="test")

        # Test basic properties
        assert plugin.name == "Test Plugin"
        assert plugin.description == "A test plugin"
        assert plugin.params == {"param1": 10, "param2": "test"}

        # Test lowres function
        test_lowres = np.random.rand(10, 10).astype(np.float32)
        lowres_result = plugin.lowres_func(test_lowres)
        assert lowres_result.shape == test_lowres.shape
        np.testing.assert_array_equal(lowres_result, test_lowres)

        # Test highres function - now both arrays should be same size
        test_fullres = da.from_array(
            np.random.rand(20, 20).astype(np.float32), chunks=(10, 10)
        )
        # Create upsampled version (same size as fullres for new interface)
        upsampled_result = da.from_array(lowres_result, chunks=(10, 10))
        highres_result = plugin.highres_func(test_fullres, upsampled_result)
        assert highres_result.shape == test_fullres.shape
        # Result should be double the original
        np.testing.assert_array_almost_equal(
            highres_result.compute(), test_fullres.compute() * 2
        )


class TestGaussianBiasFieldCorrection:
    """Test the GaussianBiasFieldCorrection plugin."""

    def test_bias_field_plugin_initialization(self):
        """Test GaussianBiasFieldCorrection plugin initialization."""
        plugin = GaussianBiasFieldCorrection(sigma=3.0, mode="constant")

        assert plugin.name == "Gaussian Bias Field Correction"
        assert "Multi-resolution bias field correction" in plugin.description
        assert plugin.sigma == 3.0
        assert plugin.mode == "constant"
        assert plugin.params == {"sigma": 3.0, "mode": "constant"}

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

        # Create test data - both arrays same size now
        fullres_data = (
            da.ones((40, 40), chunks=(20, 20), dtype=np.float32) * 100
        )  # Original image
        upsampled_bias = (
            da.ones((40, 40), chunks=(20, 20), dtype=np.float32) * 2
        )  # Upsampled bias field (same size as fullres)

        result = plugin.highres_func(fullres_data, upsampled_bias)

        # Result should have same shape as full-res data
        assert result.shape == fullres_data.shape

        # Result should be approximately original divided by bias (100/2 = 50)
        result_computed = result.compute()
        expected = 50.0
        assert_array_almost_equal(
            result_computed, np.full_like(result_computed, expected), decimal=1
        )

    def test_highres_func_division_by_zero_protection(self):
        """Test highres_func protects against division by zero."""
        plugin = GaussianBiasFieldCorrection()

        # Test with some zero values in bias field
        fullres_data = da.ones((20, 20), chunks=(10, 10), dtype=np.float32) * 100
        upsampled_bias = da.zeros(
            (20, 20), chunks=(10, 10), dtype=np.float32
        )  # All zeros

        result = plugin.highres_func(fullres_data, upsampled_bias)

        assert result.shape == fullres_data.shape
        # Should not result in inf values
        result_computed = result.compute()
        assert np.all(np.isfinite(result_computed))


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
            match="Plugin must be an instance or subclass of ScaledProcessingPlugin",
        ):
            znimg.apply_scaled_processing("not_a_plugin")

    @pytest.mark.usefixtures("cleandir")
    def test_apply_scaled_processing_with_custom_chunks(self, nifti_nib):
        """Test apply_scaled_processing with custom chunk size."""
        nifti_nib.to_filename("test.nii")
        znimg = ZarrNii.from_nifti("test.nii", axes_order="ZYX")

        plugin = GaussianBiasFieldCorrection(sigma=1.0)
        result = znimg.apply_scaled_processing(
            plugin, downsample_factor=2, chunk_size=(1, 32, 32, 32)
        )

        assert isinstance(result, ZarrNii)
        assert result.shape == znimg.shape

    @pytest.mark.usefixtures("cleandir")
    def test_apply_scaled_processing_temp_zarr_options(self, nifti_nib):
        """Test apply_scaled_processing with temp zarr options."""
        nifti_nib.to_filename("test.nii")
        znimg = ZarrNii.from_nifti("test.nii", axes_order="ZYX")

        plugin = GaussianBiasFieldCorrection(sigma=1.0)

        # Test with custom zarr store paths
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            upsampled_path = os.path.join(temp_dir, "upsampled.ome.zarr")
            original_path = os.path.join(temp_dir, "original.zarr")
            rechunked_path = os.path.join(temp_dir, "rechunked.zarr")

            result = znimg.apply_scaled_processing(
                plugin,
                downsample_factor=2,
                upsampled_ome_zarr_path=upsampled_path,
                original_zarr_path=original_path,
                rechunked_zarr_path=rechunked_path,
            )
            assert isinstance(result, ZarrNii)
            assert result.shape == znimg.shape

            # Verify that the stores were created
            assert os.path.exists(upsampled_path)
            assert os.path.exists(original_path)
            assert os.path.exists(rechunked_path)


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
            assert plugin.name == "N4 Bias Field Correction"

    @pytest.mark.skipif(not HAS_ANTSPYX, reason="antspyx not available")
    def test_n4_plugin_initialization(self):
        """Test N4BiasFieldCorrection plugin initialization."""
        from zarrnii.plugins.scaled_processing.n4_biasfield import N4BiasFieldCorrection

        plugin = N4BiasFieldCorrection(
            convergence={"iters": [25], "tol": 0.002},
            shrink_factor=2,
        )

        assert plugin.name == "N4 Bias Field Correction"
        assert "Multi-resolution N4 bias field correction" in plugin.description
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

        # Create test data - both arrays same size
        fullres_data = da.ones((40, 40), chunks=(20, 20), dtype=np.float32) * 100
        upsampled_bias = da.ones((40, 40), chunks=(20, 20), dtype=np.float32) * 2

        result = plugin.highres_func(fullres_data, upsampled_bias)

        # Result should have same shape as full-res data
        assert result.shape == fullres_data.shape

        # Result should be approximately original divided by bias (100/2 = 50)
        result_computed = result.compute()
        expected = 50.0
        assert_array_almost_equal(
            result_computed, np.full_like(result_computed, expected), decimal=1
        )

    @pytest.mark.skipif(not HAS_ANTSPYX, reason="antspyx not available")
    def test_n4_highres_func_division_by_zero_protection(self):
        """Test N4 highres_func protects against division by zero."""
        from zarrnii.plugins.scaled_processing.n4_biasfield import N4BiasFieldCorrection

        plugin = N4BiasFieldCorrection()

        # Test with some zero values in bias field
        fullres_data = da.ones((20, 20), chunks=(10, 10), dtype=np.float32) * 100
        upsampled_bias = da.zeros((20, 20), chunks=(10, 10), dtype=np.float32)

        result = plugin.highres_func(fullres_data, upsampled_bias)

        assert result.shape == fullres_data.shape
        # Should not result in inf values
        result_computed = result.compute()
        assert np.all(np.isfinite(result_computed))

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
