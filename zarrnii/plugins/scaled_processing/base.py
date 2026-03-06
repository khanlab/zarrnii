"""
Scaled processing plugin base module.

.. deprecated::
    The ``ScaledProcessingPlugin`` base class has been removed.  Implement
    plugins as plain classes using ``@hookimpl`` from :mod:`zarrnii_plugin_api`
    instead.  See :mod:`zarrnii.plugins.scaled_processing` for examples.
"""

# Re-export hookimpl from zarrnii_plugin_api for backward compatibility
from zarrnii_plugin_api import hookimpl  # noqa: F401
