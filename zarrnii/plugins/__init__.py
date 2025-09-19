"""
ZarrNii plugins package.

This package provides extensible plugin architectures for various image processing
tasks such as segmentation, filtering, and analysis.
"""

from .scaled_processing import *
from .segmentation import *

__all__ = [
    "SegmentationPlugin",
    "OtsuSegmentation",
    "ScaledProcessingPlugin",
    "BiasFieldCorrection",
]
