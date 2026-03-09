"""
Segmentation plugin base module.

.. deprecated::
    The ``SegmentationPlugin`` base class has been removed.  Implement plugins
    as plain classes using ``@hookimpl`` from :mod:`zarrnii.plugins` instead.
    See :mod:`zarrnii.plugins.segmentation` for examples.
"""

# Re-export hookimpl from zarrnii.plugins.markers for backward compatibility
from zarrnii.plugins.markers import hookimpl  # noqa: F401
