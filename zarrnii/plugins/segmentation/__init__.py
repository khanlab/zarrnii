"""
Segmentation plugins for ZarrNii.

This module provides a plugin architecture for different segmentation algorithms
that can be applied to ZarrNii images.
"""

from .base import SegmentationPlugin
from .local_otsu import (  # OtsuSegmentation for backward compatibility
    LocalOtsuSegmentation,
    OtsuSegmentation,
)
from .threshold import ThresholdSegmentation

__all__ = [
    "SegmentationPlugin",
    "OtsuSegmentation",  # Backward compatibility
    "LocalOtsuSegmentation",
    "ThresholdSegmentation",
]
