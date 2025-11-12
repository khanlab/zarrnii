"""
ZarrNii plugins package.

This package provides extensible plugin architectures for various image processing
tasks such as segmentation, filtering, and analysis using the pluggy framework.
"""

from .hookspecs import hookspec
from .plugin_manager import get_global_plugin_manager, get_plugin_manager
from .scaled_processing import *
from .segmentation import *

__all__ = [
    "SegmentationPlugin",
    "OtsuSegmentation",  # Backward compatibility
    "LocalOtsuSegmentation",
    "ThresholdSegmentation",
    "ScaledProcessingPlugin",
    "GaussianBiasFieldCorrection",
    "N4BiasFieldCorrection",
    "hookspec",
    "get_plugin_manager",
    "get_global_plugin_manager",
]
