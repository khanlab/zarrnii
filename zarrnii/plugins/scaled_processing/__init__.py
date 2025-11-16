"""
Scaled processing plugins for ZarrNii.

This module provides a plugin architecture for multi-resolution operations where
algorithms are run at low resolution and applied to full resolution data.
"""

from .base import ScaledProcessingPlugin
from .gaussian_biasfield import GaussianBiasFieldCorrection
from .n4_biasfield import N4BiasFieldCorrection
from .segmentation_cleaner import SegmentationCleaner

__all__ = [
    "ScaledProcessingPlugin",
    "GaussianBiasFieldCorrection",
    "N4BiasFieldCorrection",
    "SegmentationCleaner",
]
