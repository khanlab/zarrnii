"""
Tests for segmentation plugins functionality.

This module tests the plugin architecture
to ensure they work correctly with ZarrNii images.
"""

import dask.array as da
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from zarrnii_plugin_api import hookimpl

from zarrnii import ZarrNii


class TestPluginApiImport:
    """Test that zarrnii_plugin_api can be imported independently."""

    def test_hookimpl_importable(self):
        """Test that hookimpl can be imported from zarrnii_plugin_api."""
        from zarrnii_plugin_api import hookimpl

        assert hookimpl is not None

    def test_hookspec_importable(self):
        """Test that hookspec can be imported from zarrnii_plugin_api."""
        from zarrnii_plugin_api import hookspec

        assert hookspec is not None

    def test_zarrnispec_importable(self):
        """Test that ZarrNiiSpec can be imported from zarrnii_plugin_api."""
        from zarrnii_plugin_api import ZarrNiiSpec

        assert ZarrNiiSpec is not None


class TestSegmentationPlugin:
    """Test the pluggy-based segmentation plugin interface."""

    def test_plugin_interface(self):
        """Test that plain pluggy-style plugins implement the required interface."""

        # Create a minimal implementation using @hookimpl only (no inheritance)
        class TestPlugin:
            @hookimpl
            def segment(self, image, metadata=None):
                return np.ones_like(image, dtype=np.uint8)

            @hookimpl
            def segmentation_plugin_name(self):
                return "Test Plugin"

            @hookimpl
            def segmentation_plugin_description(self):
                return "A test plugin"

        plugin = TestPlugin()

        # Test methods directly (no properties required)
        assert plugin.segmentation_plugin_name() == "Test Plugin"
        assert plugin.segmentation_plugin_description() == "A test plugin"

        # Test segmentation
        test_image = np.random.rand(10, 10).astype(np.float32)
        result = plugin.segment(test_image)
        assert result.shape == test_image.shape
        assert result.dtype == np.uint8
        assert np.all(result == 1)

    def test_plugin_can_be_registered_with_manager(self):
        """Test that a plain pluggy-style plugin can be registered with the manager."""
        from zarrnii.plugins import get_plugin_manager

        class TestPlugin:
            @hookimpl
            def segment(self, image, metadata=None):
                return np.zeros_like(image, dtype=np.uint8)

            @hookimpl
            def segmentation_plugin_name(self):
                return "Test Plugin"

            @hookimpl
            def segmentation_plugin_description(self):
                return "A test plugin"

        pm = get_plugin_manager()
        plugin = TestPlugin()
        pm.register(plugin)

        results = pm.hook.segment(image=np.ones((5, 5), dtype=np.float32))
        assert len(results) >= 1

        pm.unregister(plugin)
