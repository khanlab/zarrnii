"""
Pluggy hook markers for ZarrNii plugins.

This module defines the hook implementation and specification markers
used by ZarrNii plugins.

Example::

    from zarrnii.plugins import hookimpl

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

#: Hook implementation marker for ZarrNii plugins.
hookimpl = pluggy.HookimplMarker("zarrnii")

#: Hook specification marker for ZarrNii hook specs.
hookspec = pluggy.HookspecMarker("zarrnii")

__all__ = ["hookimpl", "hookspec"]
