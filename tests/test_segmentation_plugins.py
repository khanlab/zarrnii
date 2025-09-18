"""
Tests for segmentation plugins functionality.

This module tests the plugin architecture and the Otsu segmentation implementation
to ensure they work correctly with ZarrNii images.
"""

import dask.array as da
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from zarrnii import OtsuSegmentation, SegmentationPlugin, ZarrNii


class TestSegmentationPlugin:
    """Test the base SegmentationPlugin interface."""

    def test_abstract_plugin_cannot_be_instantiated(self):
        """Test that abstract SegmentationPlugin cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SegmentationPlugin()

    def test_plugin_interface(self):
        """Test that plugins implement the required interface."""

        # Create a minimal implementation for testing
        class TestPlugin(SegmentationPlugin):
            def segment(self, image, metadata=None):
                return np.ones_like(image, dtype=np.uint8)

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

        # Test segmentation
        test_image = np.random.rand(10, 10).astype(np.float32)
        result = plugin.segment(test_image)
        assert result.shape == test_image.shape
        assert result.dtype == np.uint8
        assert np.all(result == 1)


class TestOtsuSegmentation:
    """Test the Otsu thresholding segmentation plugin."""

    def test_otsu_plugin_initialization(self):
        """Test Otsu plugin can be initialized with different parameters."""
        # Default initialization
        plugin = OtsuSegmentation()
        assert plugin.nbins == 256
        assert plugin.name == "Otsu Thresholding"
        assert "Otsu's automatic threshold" in plugin.description

        # Custom initialization
        plugin_custom = OtsuSegmentation(nbins=128)
        assert plugin_custom.nbins == 128

    def test_otsu_segmentation_bimodal_image(self):
        """Test Otsu segmentation on a bimodal synthetic image."""
        # Create a synthetic bimodal image
        np.random.seed(42)
        background = np.random.normal(0.2, 0.05, (50, 50))
        foreground = np.random.normal(0.8, 0.05, (30, 30))

        # Create image with clear foreground and background
        image = np.zeros((100, 100))
        image[:50, :50] = background
        image[35:65, 35:65] = foreground

        plugin = OtsuSegmentation()
        result = plugin.segment(image)

        # Check result properties
        assert result.shape == image.shape
        assert result.dtype == np.uint8
        assert np.all(np.isin(result, [0, 1]))  # Binary result

        # The center region should mostly be foreground (1)
        center_region = result[45:55, 45:55]
        assert np.mean(center_region) > 0.7  # Most pixels should be foreground

    def test_otsu_constant_image(self):
        """Test Otsu segmentation on a constant image."""
        # Create constant image
        image = np.ones((20, 20)) * 0.5

        plugin = OtsuSegmentation()
        result = plugin.segment(image)

        # Should return all zeros for constant image
        assert result.shape == image.shape
        assert result.dtype == np.uint8
        assert np.all(result == 0)

    def test_otsu_binary_image(self):
        """Test Otsu segmentation on already binary image."""
        # Create binary image
        image = np.zeros((20, 20), dtype=bool)
        image[5:15, 5:15] = True

        plugin = OtsuSegmentation()
        result = plugin.segment(image)

        # Should preserve the binary pattern
        assert result.shape == image.shape
        assert result.dtype == np.uint8
        expected = image.astype(np.uint8)
        assert_array_equal(result, expected)

    def test_otsu_multichannel_image(self):
        """Test Otsu segmentation on multi-channel image."""
        # Create multi-channel image (channels first)
        np.random.seed(42)
        image = np.random.rand(3, 20, 20)
        image[0, 5:15, 5:15] += 0.5  # Make first channel have higher values in center

        plugin = OtsuSegmentation()
        result = plugin.segment(image)

        # Should preserve input shape and process based on first channel
        assert result.shape == image.shape  # (3, 20, 20)
        assert result.dtype == np.uint8
        assert np.all(np.isin(result, [0, 1]))

    def test_otsu_empty_image(self):
        """Test Otsu segmentation with empty image."""
        image = np.array([])

        plugin = OtsuSegmentation()
        with pytest.raises(ValueError, match="Input image is empty"):
            plugin.segment(image)

    def test_otsu_1d_image(self):
        """Test Otsu segmentation with 1D image."""
        image = np.array([1, 2, 3, 4, 5])

        plugin = OtsuSegmentation()
        with pytest.raises(ValueError, match="Input image must be at least 2D"):
            plugin.segment(image)

    def test_get_threshold_method(self):
        """Test the get_threshold method."""
        # Create bimodal image
        np.random.seed(42)
        image = np.concatenate(
            [np.random.normal(0.2, 0.05, 100), np.random.normal(0.8, 0.05, 100)]
        ).reshape(10, 20)

        plugin = OtsuSegmentation()
        threshold = plugin.get_threshold(image)

        # Threshold should be between the two modes (relaxed bounds for randomness)
        assert 0.25 < threshold < 0.75
        assert isinstance(threshold, float)


class TestZarrNiiSegmentationIntegration:
    """Test segmentation plugin integration with ZarrNii."""

    def create_test_zarrnii(self):
        """Create a test ZarrNii instance with synthetic bimodal data."""
        np.random.seed(42)

        # Create synthetic bimodal 3D image
        background = np.random.normal(0.2, 0.05, (1, 20, 30, 30))
        foreground_mask = np.zeros((1, 20, 30, 30), dtype=bool)
        foreground_mask[0, 8:12, 10:20, 10:20] = True

        image_data = background.copy()
        image_data[foreground_mask] = np.random.normal(
            0.8, 0.05, np.sum(foreground_mask)
        )

        # Create dask array
        darr = da.from_array(image_data, chunks=(1, 10, 15, 15))

        # Create ZarrNii instance using the legacy constructor approach
        znimg = ZarrNii.from_darr(darr, axes_order="ZYX", orientation="RAS")
        return znimg

    def test_segment_method_with_plugin_instance(self):
        """Test the segment method with a plugin instance."""
        znimg = self.create_test_zarrnii()
        plugin = OtsuSegmentation(nbins=128)

        # Apply segmentation
        segmented = znimg.segment(plugin)

        # Check result properties
        assert isinstance(segmented, ZarrNii)
        assert segmented.shape == znimg.shape
        assert segmented.data.dtype == np.uint8
        assert segmented.axes_order == znimg.axes_order
        assert segmented.orientation == znimg.orientation
        assert "segmented_otsu_thresholding" in segmented.name

        # Check that segmentation worked
        result_data = segmented.data.compute()
        assert np.all(np.isin(result_data, [0, 1]))
        assert np.sum(result_data) > 0  # Should have some foreground pixels

    def test_segment_method_with_plugin_class(self):
        """Test the segment method with a plugin class and kwargs."""
        znimg = self.create_test_zarrnii()

        # Apply segmentation using plugin class
        segmented = znimg.segment(OtsuSegmentation, nbins=64)

        # Check result properties
        assert isinstance(segmented, ZarrNii)
        assert segmented.shape == znimg.shape
        assert segmented.data.dtype == np.uint8

        # Check that segmentation worked
        result_data = segmented.data.compute()
        assert np.all(np.isin(result_data, [0, 1]))

    def test_segment_method_with_custom_chunks(self):
        """Test the segment method with custom chunk size."""
        znimg = self.create_test_zarrnii()
        plugin = OtsuSegmentation()

        # Apply segmentation with custom chunk size
        custom_chunks = (1, 5, 10, 10)
        segmented = znimg.segment(plugin, chunk_size=custom_chunks)

        # Check that chunks were applied
        assert segmented.data.chunks[0] == (1,)
        assert segmented.data.chunks[1] == (5, 5, 5, 5)
        assert segmented.data.chunks[2] == (10, 10, 10)
        assert segmented.data.chunks[3] == (10, 10, 10)

    def test_segment_method_invalid_plugin(self):
        """Test the segment method with invalid plugin."""
        znimg = self.create_test_zarrnii()

        # Test with invalid plugin type
        with pytest.raises(TypeError, match="Plugin must be an instance or subclass"):
            znimg.segment("not_a_plugin")

        with pytest.raises(TypeError, match="Plugin must be an instance or subclass"):
            znimg.segment(42)

    def test_segment_otsu_convenience_method(self):
        """Test the segment_otsu convenience method."""
        znimg = self.create_test_zarrnii()

        # Apply Otsu segmentation using convenience method
        segmented = znimg.segment_otsu(nbins=128)

        # Check result properties
        assert isinstance(segmented, ZarrNii)
        assert segmented.shape == znimg.shape
        assert segmented.data.dtype == np.uint8
        assert "segmented_otsu_thresholding" in segmented.name

        # Check that segmentation worked
        result_data = segmented.data.compute()
        assert np.all(np.isin(result_data, [0, 1]))
        assert np.sum(result_data) > 0  # Should have some foreground pixels

    def test_segment_otsu_with_custom_chunks(self):
        """Test the segment_otsu method with custom chunk size."""
        znimg = self.create_test_zarrnii()

        # Apply Otsu segmentation with custom chunks
        custom_chunks = (1, 10, 15, 15)
        segmented = znimg.segment_otsu(chunk_size=custom_chunks)

        # Verify custom chunks were applied
        assert segmented.data.chunks == znimg.data.rechunk(custom_chunks).chunks


@pytest.mark.usefixtures("cleandir")
class TestSegmentationWorkflow:
    """Test complete segmentation workflows."""

    def test_segmentation_to_ome_zarr_workflow(self):
        """Test segmenting an image and saving as OME-Zarr."""
        # Create test data
        np.random.seed(42)
        image_data = np.random.rand(1, 10, 20, 20).astype(np.float32)
        image_data[0, 3:7, 8:12, 8:12] += 1.0  # Add a bright region

        darr = da.from_array(image_data, chunks=(1, 5, 10, 10))
        znimg = ZarrNii.from_darr(darr, axes_order="ZYX", orientation="RAS")

        # Apply segmentation
        segmented = znimg.segment_otsu()

        # Save as OME-Zarr
        output_path = "test_segmented.ome.zarr"
        segmented.to_ome_zarr(output_path)

        # Load back and verify
        loaded = ZarrNii.from_ome_zarr(output_path)

        assert loaded.shape == segmented.shape
        assert loaded.data.dtype == segmented.data.dtype
        assert_array_equal(loaded.data.compute(), segmented.data.compute())

    def test_plugin_repr(self):
        """Test plugin string representation."""
        plugin = OtsuSegmentation(nbins=128, extra_param="test")
        repr_str = repr(plugin)

        assert "OtsuSegmentation" in repr_str
        assert "nbins=128" in repr_str
        assert "extra_param=test" in repr_str
