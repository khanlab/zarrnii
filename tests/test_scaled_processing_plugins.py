"""
Tests for scaled processing plugins functionality.

This module tests the multi-resolution plugin architecture and the bias field
correction implementation to ensure they work correctly with ZarrNii images.
"""

import dask.array as da
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from zarrnii import BiasFieldCorrection, ScaledProcessingPlugin, ZarrNii


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

        # Test highres function
        test_fullres = da.from_array(
            np.random.rand(20, 20).astype(np.float32), chunks=(10, 10)
        )
        highres_result = plugin.highres_func(test_fullres, lowres_result)
        assert highres_result.shape == test_fullres.shape
        # Result should be double the original
        np.testing.assert_array_almost_equal(
            highres_result.compute(), test_fullres.compute() * 2
        )


class TestBiasFieldCorrection:
    """Test the BiasFieldCorrection plugin."""

    def test_bias_field_plugin_initialization(self):
        """Test BiasFieldCorrection plugin initialization."""
        plugin = BiasFieldCorrection(sigma=3.0, mode="constant")

        assert plugin.name == "Bias Field Correction"
        assert "Multi-resolution bias field correction" in plugin.description
        assert plugin.sigma == 3.0
        assert plugin.mode == "constant"
        assert plugin.params == {"sigma": 3.0, "mode": "constant"}

    def test_lowres_func_basic(self):
        """Test the lowres_func with basic input."""
        plugin = BiasFieldCorrection(sigma=1.0)

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
        plugin = BiasFieldCorrection(sigma=1.0)

        # Create 3D test data
        test_data = np.ones((10, 20, 20), dtype=np.float32)
        test_data[:, :, :10] = 0.5

        result = plugin.lowres_func(test_data)

        assert result.shape == test_data.shape
        assert np.all(result > 0)

    def test_lowres_func_edge_cases(self):
        """Test lowres_func with edge cases."""
        plugin = BiasFieldCorrection()

        # Test empty array
        with pytest.raises(ValueError, match="Input array is empty"):
            plugin.lowres_func(np.array([]))

        # Test 1D array
        with pytest.raises(ValueError, match="Input array must be at least 2D"):
            plugin.lowres_func(np.array([1, 2, 3]))

    def test_highres_func_upsampling(self):
        """Test the highres_func upsampling and application."""
        plugin = BiasFieldCorrection()

        # Create test data
        fullres_data = (
            da.ones((40, 40), chunks=(20, 20), dtype=np.float32) * 100
        )  # Original image
        lowres_bias = (
            np.ones((20, 20), dtype=np.float32) * 2
        )  # 2x downsampled bias field

        result = plugin.highres_func(fullres_data, lowres_bias)

        # Result should have same shape as full-res data
        assert result.shape == fullres_data.shape

        # Result should be approximately original divided by bias (100/2 = 50)
        result_computed = result.compute()
        expected = 50.0
        assert_array_almost_equal(
            result_computed, np.full_like(result_computed, expected), decimal=1
        )

    def test_highres_func_shape_mismatch(self):
        """Test highres_func with different dimensional inputs."""
        plugin = BiasFieldCorrection()

        # Test with different aspect ratios
        fullres_data = da.ones((60, 40), chunks=(30, 20), dtype=np.float32) * 100
        lowres_bias = np.ones((15, 10), dtype=np.float32) * 2  # 4x downsampled

        result = plugin.highres_func(fullres_data, lowres_bias)

        assert result.shape == fullres_data.shape
        # Should still work despite shape differences due to resizing


class TestZarrNiiScaledProcessingIntegration:
    """Test integration of scaled processing with ZarrNii."""

    @pytest.mark.usefixtures("cleandir")
    def test_apply_scaled_processing_method_plugin_instance(self, nifti_nib):
        """Test apply_scaled_processing method with plugin instance."""
        nifti_nib.to_filename("test.nii")
        znimg = ZarrNii.from_nifti("test.nii")

        plugin = BiasFieldCorrection(sigma=2.0)
        result = znimg.apply_scaled_processing(plugin, downsample_factor=2)

        # Check that we get a new ZarrNii instance
        assert isinstance(result, ZarrNii)
        assert result.shape == znimg.shape
        assert "bias_field_correction" in result.name

    @pytest.mark.usefixtures("cleandir")
    def test_apply_scaled_processing_method_plugin_class(self, nifti_nib):
        """Test apply_scaled_processing method with plugin class."""
        nifti_nib.to_filename("test.nii")
        znimg = ZarrNii.from_nifti("test.nii")

        result = znimg.apply_scaled_processing(
            BiasFieldCorrection, sigma=1.0, downsample_factor=2
        )

        assert isinstance(result, ZarrNii)
        assert result.shape == znimg.shape

    @pytest.mark.usefixtures("cleandir")
    def test_apply_scaled_processing_invalid_plugin(self, nifti_nib):
        """Test apply_scaled_processing with invalid plugin."""
        nifti_nib.to_filename("test.nii")
        znimg = ZarrNii.from_nifti("test.nii")

        with pytest.raises(
            TypeError,
            match="Plugin must be an instance or subclass of ScaledProcessingPlugin",
        ):
            znimg.apply_scaled_processing("not_a_plugin")

    @pytest.mark.usefixtures("cleandir")
    def test_apply_scaled_processing_with_custom_chunks(self, nifti_nib):
        """Test apply_scaled_processing with custom chunk size."""
        nifti_nib.to_filename("test.nii")
        znimg = ZarrNii.from_nifti("test.nii")

        plugin = BiasFieldCorrection(sigma=1.0)
        result = znimg.apply_scaled_processing(
            plugin, downsample_factor=2, chunk_size=(1, 32, 32, 32)
        )

        assert isinstance(result, ZarrNii)
        assert result.shape == znimg.shape


class TestScaledProcessingWorkflow:
    """Test complete scaled processing workflows."""

    @pytest.mark.usefixtures("cleandir")
    def test_bias_correction_workflow(self, nifti_nib):
        """Test complete bias field correction workflow."""
        nifti_nib.to_filename("test.nii")
        znimg = ZarrNii.from_nifti("test.nii")

        # Apply bias field correction
        corrected = znimg.apply_scaled_processing(
            BiasFieldCorrection(sigma=2.0), downsample_factor=4
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
        plugin = BiasFieldCorrection(sigma=3.0, mode="reflect")
        repr_str = repr(plugin)

        assert "BiasFieldCorrection" in repr_str
        assert "sigma=3.0" in repr_str
        assert "mode=reflect" in repr_str
