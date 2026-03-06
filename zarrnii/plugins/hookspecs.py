"""
Plugin hook specifications for ZarrNii plugins.

This module re-exports the hook specifications from :mod:`zarrnii_plugin_api`.
"""

from zarrnii_plugin_api import ZarrNiiSpec, hookspec

__all__ = ["hookspec", "ZarrNiiSpec"]
