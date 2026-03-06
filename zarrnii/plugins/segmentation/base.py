"""
Segmentation plugin base module.

.. deprecated::
    The ``SegmentationPlugin`` base class has been removed.  Implement plugins
    as plain classes using ``@hookimpl`` from :mod:`zarrnii_plugin_api` instead.
    See :mod:`zarrnii.plugins.segmentation` for examples.
"""

# Re-export hookimpl from zarrnii_plugin_api for backward compatibility
from zarrnii_plugin_api import hookimpl  # noqa: F401
