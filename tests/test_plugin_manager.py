"""
Tests for pluggy-based plugin manager functionality.

This module tests that the plugin manager and hook system work correctly
with the pluggy framework.
"""

import numpy as np
import pytest

from zarrnii.plugins import (
    GaussianBiasFieldCorrection,
    LocalOtsuSegmentation,
    ThresholdSegmentation,
    get_plugin_manager,
)


class TestPluggyPluginManager:
    """Test pluggy plugin manager functionality."""

    def test_plugin_manager_creation(self):
        """Test that we can create a plugin manager."""
        pm = get_plugin_manager()
        assert pm is not None
        assert pm.project_name == "zarrnii"

    def test_register_segmentation_plugin(self):
        """Test that we can register a segmentation plugin with the manager."""
        pm = get_plugin_manager()
        plugin = LocalOtsuSegmentation()

        # Register the plugin
        pm.register(plugin)

        # Check that plugin is registered
        assert plugin in pm.get_plugins()

        # Unregister for cleanup
        pm.unregister(plugin)

    def test_register_scaled_processing_plugin(self):
        """Test that we can register a scaled processing plugin with the manager."""
        pm = get_plugin_manager()
        plugin = GaussianBiasFieldCorrection()

        # Register the plugin
        pm.register(plugin)

        # Check that plugin is registered
        assert plugin in pm.get_plugins()

        # Unregister for cleanup
        pm.unregister(plugin)

    def test_plugin_hook_calls(self):
        """Test that hooks can be called through the plugin manager."""
        pm = get_plugin_manager()
        plugin = ThresholdSegmentation(thresholds=0.5)

        # Register the plugin
        pm.register(plugin)

        # Call hook through plugin manager
        test_image = np.random.rand(10, 10).astype(np.float32)
        results = pm.hook.segment(image=test_image)

        # Should get a list of results from all registered plugins
        assert isinstance(results, list)
        assert len(results) == 1  # One plugin registered
        assert results[0].shape == test_image.shape

        # Unregister for cleanup
        pm.unregister(plugin)

    def test_multiple_plugins_registered(self):
        """Test that multiple plugins can be registered and called."""
        pm = get_plugin_manager()
        plugin1 = LocalOtsuSegmentation()
        plugin2 = ThresholdSegmentation(thresholds=0.5)

        # Register both plugins
        pm.register(plugin1)
        pm.register(plugin2)

        # Check both are registered
        plugins = pm.get_plugins()
        assert plugin1 in plugins
        assert plugin2 in plugins

        # Call hook - should get results from both
        test_image = np.random.rand(10, 10).astype(np.float32) * 255
        results = pm.hook.segment(image=test_image)

        # Should get results from both plugins
        assert len(results) == 2

        # Unregister for cleanup
        pm.unregister(plugin1)
        pm.unregister(plugin2)

    def test_plugin_name_and_description_hooks(self):
        """Test that name and description hooks work."""
        pm = get_plugin_manager()
        plugin = LocalOtsuSegmentation()

        # Register the plugin
        pm.register(plugin)

        # Call name hook
        names = pm.hook.segmentation_plugin_name()
        assert isinstance(names, list)
        assert "Local Otsu Thresholding" in names

        # Call description hook
        descriptions = pm.hook.segmentation_plugin_description()
        assert isinstance(descriptions, list)
        assert any("Local Otsu" in d for d in descriptions)

        # Unregister for cleanup
        pm.unregister(plugin)
