"""
Tests for segmentation plugins functionality.

This module tests the plugin architecture
to ensure they work correctly with ZarrNii images.
"""

import dask.array as da
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from zarrnii import ZarrNii
from zarrnii.plugins import SegmentationPlugin


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
