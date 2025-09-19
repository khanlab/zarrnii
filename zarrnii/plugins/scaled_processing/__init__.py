"""
Scaled processing plugins for ZarrNii.

This module provides a plugin architecture for multi-resolution operations where
algorithms are run at low resolution and applied to full resolution data.
"""

from .base import ScaledProcessingPlugin
from .bias_field import BiasFieldCorrection

__all__ = [
    "ScaledProcessingPlugin",
    "BiasFieldCorrection",
]
