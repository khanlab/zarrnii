"""
Tests for threshold segmentation plugin.

This module tests the ThresholdSegmentation plugin and LocalOtsuSegmentation
to ensure they work correctly.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from zarrnii import LocalOtsuSegmentation, ThresholdSegmentation


class TestThresholdSegmentation:
    """Test the ThresholdSegmentation plugin."""

    def test_binary_threshold_basic(self):
        """Test basic binary threshold segmentation."""
        plugin = ThresholdSegmentation(thresholds=0.5)

        # Create test image with known values
        image = np.array([[0.1, 0.6], [0.4, 0.8]])
        result = plugin.segment(image)

        expected = np.array([[0, 1], [0, 1]], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_binary_threshold_inclusive(self):
        """Test binary threshold with inclusive parameter."""
        # Test inclusive (>=)
        plugin_inclusive = ThresholdSegmentation(thresholds=0.5, inclusive=True)

        # Test exclusive (>)
        plugin_exclusive = ThresholdSegmentation(thresholds=0.5, inclusive=False)

        # Image where 0.5 is exactly on the boundary
        image = np.array([0.4, 0.5, 0.6])

        result_inclusive = plugin_inclusive.segment(image)
        result_exclusive = plugin_exclusive.segment(image)

        # Inclusive: 0.5 should be labeled as 1
        assert_array_equal(result_inclusive, [0, 1, 1])

        # Exclusive: 0.5 should be labeled as 0
        assert_array_equal(result_exclusive, [0, 0, 1])

    def test_multi_level_threshold(self):
        """Test multi-level threshold segmentation."""
        plugin = ThresholdSegmentation(thresholds=[0.3, 0.7])

        # Values: 0.1 < 0.3, 0.5 in [0.3, 0.7), 0.9 >= 0.7
        image = np.array([0.1, 0.5, 0.9])
        result = plugin.segment(image)

        # 0.1 -> 0, 0.5 -> 1, 0.9 -> 2
        expected = np.array([0, 1, 2], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_multi_level_threshold_complex(self):
        """Test complex multi-level threshold segmentation."""
        plugin = ThresholdSegmentation(thresholds=[0.2, 0.5, 0.8])

        image = np.array([[0.1, 0.3, 0.6, 0.9], [0.15, 0.45, 0.75, 0.95]])
        result = plugin.segment(image)

        # 0.1, 0.15 -> 0 (< 0.2)
        # 0.3, 0.45 -> 1 (>= 0.2, < 0.5)
        # 0.6, 0.75 -> 2 (>= 0.5, < 0.8)
        # 0.9, 0.95 -> 3 (>= 0.8)
        expected = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_single_threshold_as_list(self):
        """Test that single threshold can be provided as list."""
        plugin = ThresholdSegmentation(thresholds=[0.5])

        image = np.array([0.3, 0.7])
        result = plugin.segment(image)

        expected = np.array([0, 1], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_threshold_sorting(self):
        """Test that thresholds are automatically sorted."""
        plugin = ThresholdSegmentation(thresholds=[0.8, 0.2, 0.5])

        # Should be sorted to [0.2, 0.5, 0.8]
        assert plugin.thresholds == [0.2, 0.5, 0.8]

        image = np.array([0.1, 0.3, 0.6, 0.9])
        result = plugin.segment(image)

        # 0.1 -> 0, 0.3 -> 1, 0.6 -> 2, 0.9 -> 3
        expected = np.array([0, 1, 2, 3], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_3d_image(self):
        """Test threshold segmentation on 3D image."""
        plugin = ThresholdSegmentation(thresholds=0.5)

        image = np.array([[[0.2, 0.8], [0.4, 0.6]], [[0.3, 0.9], [0.1, 0.7]]])
        result = plugin.segment(image)

        expected = np.array([[[0, 1], [0, 1]], [[0, 1], [0, 1]]], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_empty_image_error(self):
        """Test error handling for empty image."""
        plugin = ThresholdSegmentation(thresholds=0.5)

        empty_image = np.array([])
        with pytest.raises(ValueError, match="Input image is empty"):
            plugin.segment(empty_image)

    def test_1d_image_works(self):
        """Test that 1D images work correctly."""
        plugin = ThresholdSegmentation(thresholds=0.5)

        image_1d = np.array([0.1, 0.5, 0.9])
        result = plugin.segment(image_1d)

        expected = np.array([0, 1, 1], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_plugin_properties(self):
        """Test plugin name and description properties."""
        # Binary threshold
        plugin_binary = ThresholdSegmentation(thresholds=0.5)
        assert plugin_binary.name == "Binary Threshold"
        assert "Binary threshold segmentation" in plugin_binary.description
        assert "0.5" in plugin_binary.description

        # Multi-level threshold
        plugin_multi = ThresholdSegmentation(thresholds=[0.3, 0.7])
        assert plugin_multi.name == "Multi-level Threshold"
        assert "Multi-level threshold segmentation" in plugin_multi.description
        assert "[0.3, 0.7]" in plugin_multi.description

    def test_get_thresholds_method(self):
        """Test get_thresholds method."""
        thresholds = [0.2, 0.5, 0.8]
        plugin = ThresholdSegmentation(thresholds=thresholds)

        returned_thresholds = plugin.get_thresholds()

        assert returned_thresholds == thresholds
        # Should return a copy, not the original
        returned_thresholds.append(1.0)
        assert plugin.thresholds != returned_thresholds

    def test_parameters_storage(self):
        """Test that parameters are stored correctly."""
        plugin = ThresholdSegmentation(thresholds=[0.3, 0.7], inclusive=False)

        assert plugin.thresholds == [0.3, 0.7]
        assert plugin.inclusive is False

        # Check that parameters are stored in parent class
        assert plugin.params["thresholds"] == [0.3, 0.7]
        assert plugin.params["inclusive"] is False


class TestLocalOtsuSegmentation:
    """Test the LocalOtsuSegmentation plugin (renamed from OtsuSegmentation)."""

    def test_basic_functionality(self):
        """Test basic local Otsu segmentation."""
        plugin = LocalOtsuSegmentation(nbins=256)

        # Create bimodal test image
        image = np.zeros((10, 10))
        image[:5, :] = 0.2  # Dark region
        image[5:, :] = 0.8  # Bright region

        result = plugin.segment(image)

        assert result.shape == image.shape
        assert result.dtype == np.uint8
        assert set(np.unique(result)).issubset({0, 1})

    def test_plugin_properties(self):
        """Test plugin name and description."""
        plugin = LocalOtsuSegmentation()

        assert plugin.name == "Local Otsu Thresholding"
        assert "Local Otsu" in plugin.description
        assert "locally" in plugin.description

    def test_get_threshold_method(self):
        """Test get_threshold method."""
        plugin = LocalOtsuSegmentation(nbins=128)

        # Create test image with known threshold
        image = np.zeros((10, 10))
        image[:5, :] = 0.3
        image[5:, :] = 0.7

        threshold = plugin.get_threshold(image)

        assert isinstance(threshold, float)
        assert 0.3 < threshold < 0.7

    def test_backward_compatibility_alias(self):
        """Test that OtsuSegmentation is still available as alias."""
        from zarrnii.plugins.segmentation.local_otsu import OtsuSegmentation

        # Should be the same class
        assert OtsuSegmentation is LocalOtsuSegmentation

        # Should work the same way
        plugin = OtsuSegmentation(nbins=64)
        assert isinstance(plugin, LocalOtsuSegmentation)
        assert plugin.name == "Local Otsu Thresholding"

    def test_constant_image_handling(self):
        """Test handling of constant images."""
        plugin = LocalOtsuSegmentation()

        # Constant image
        constant_image = np.ones((5, 5)) * 0.5
        result = plugin.segment(constant_image)

        # Should return all zeros for constant image
        assert_array_equal(result, np.zeros((5, 5), dtype=np.uint8))

    def test_empty_image_error(self):
        """Test error handling for empty image."""
        plugin = LocalOtsuSegmentation()

        empty_image = np.array([])
        with pytest.raises(ValueError, match="Input image is empty"):
            plugin.segment(empty_image)

    def test_1d_image_error(self):
        """Test error handling for 1D image."""
        plugin = LocalOtsuSegmentation()

        image_1d = np.array([0.1, 0.5, 0.9])
        with pytest.raises(ValueError, match="Input image must be at least 2D"):
            plugin.segment(image_1d)

    def test_parameters_storage(self):
        """Test that parameters are stored correctly."""
        plugin = LocalOtsuSegmentation(nbins=128)

        assert plugin.nbins == 128
        assert plugin.params["nbins"] == 128


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_otsu_segmentation_import(self):
        """Test that OtsuSegmentation can still be imported."""
        from zarrnii import OtsuSegmentation

        # Should be the LocalOtsuSegmentation class
        assert OtsuSegmentation is LocalOtsuSegmentation

        plugin = OtsuSegmentation()
        assert isinstance(plugin, LocalOtsuSegmentation)

    def test_existing_segment_otsu_method(self):
        """Test that existing segment_otsu method uses LocalOtsuSegmentation."""
        import dask.array as da
        import ngff_zarr as nz

        from zarrnii import ZarrNii

        # Create test ZarrNii
        data = da.ones((1, 10, 10, 10), chunks=(1, 5, 5, 5))
        dims = ["c", "z", "y", "x"]
        scale = {"z": 1.0, "y": 1.0, "x": 1.0}
        translation = {"z": 0.0, "y": 0.0, "x": 0.0}

        ngff_image = nz.NgffImage(
            data=data, dims=dims, scale=scale, translation=translation, name="test"
        )

        znimg = ZarrNii(ngff_image=ngff_image, axes_order="ZYX", orientation="RAS")

        # This should still work and use LocalOtsuSegmentation internally
        result = znimg.segment_otsu(nbins=64)

        assert result.shape == znimg.shape
        assert result.darr.dtype == np.uint8
