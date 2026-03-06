"""
ZarrNii Plugin API.

Minimal package for writing ZarrNii plugins.  Only ``pluggy`` is required —
no dependency on the core ``zarrnii`` package is needed.

Usage::

    from zarrnii_plugin_api import hookimpl

    class MySegmentationPlugin:

        @hookimpl
        def segment(self, image, metadata=None):
            ...

        @hookimpl
        def segmentation_plugin_name(self) -> str:
            return "My Plugin"

        @hookimpl
        def segmentation_plugin_description(self) -> str:
            return "A custom segmentation plugin"
"""

import pluggy

from .hookspecs import ZarrNiiSpec

hookspec = pluggy.HookspecMarker("zarrnii")
hookimpl = pluggy.HookimplMarker("zarrnii")

__all__ = ["hookspec", "hookimpl", "ZarrNiiSpec"]
