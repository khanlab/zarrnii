"""
Segmentation plugins for ZarrNii.

This module provides a plugin architecture for different segmentation algorithms
that can be applied to ZarrNii images.
"""

from .base import SegmentationPlugin
from .otsu import OtsuSegmentation

__all__ = [
    "SegmentationPlugin",
    "OtsuSegmentation",
]
