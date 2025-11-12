"""
Plugin manager for ZarrNii plugins.

This module provides the plugin manager that discovers and manages plugins
using the pluggy framework.
"""

from __future__ import annotations

import pluggy

from . import hookspecs


def get_plugin_manager() -> pluggy.PluginManager:
    """
    Get or create the ZarrNii plugin manager.

    Returns:
        PluginManager instance configured for ZarrNii plugins
    """
    pm = pluggy.PluginManager("zarrnii")
    pm.add_hookspecs(hookspecs)
    return pm


# Global plugin manager instance
_plugin_manager = None


def get_global_plugin_manager() -> pluggy.PluginManager:
    """
    Get the global plugin manager instance.

    Returns:
        Global PluginManager instance
    """
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = get_plugin_manager()
    return _plugin_manager
