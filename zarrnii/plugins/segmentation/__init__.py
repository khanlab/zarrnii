"""
Segmentation plugins for ZarrNii.

This module provides a plugin architecture for different segmentation algorithms
that can be applied to ZarrNii images.
"""

from .local_otsu import LocalOtsuSegmentation
from .threshold import ThresholdSegmentation

__all__ = [
    "LocalOtsuSegmentation",
    "ThresholdSegmentation",
]
