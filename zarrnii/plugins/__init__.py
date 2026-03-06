"""
ZarrNii plugins package.

This package provides extensible plugin architectures for various image processing
tasks such as segmentation, filtering, and analysis using the pluggy framework.

Plugin authors should use :mod:`zarrnii_plugin_api` to write plugins without
depending on the core ``zarrnii`` package.
"""

from zarrnii_plugin_api import hookimpl, hookspec

from .plugin_manager import get_global_plugin_manager, get_plugin_manager
from .scaled_processing import *
from .segmentation import *

__all__ = [
    "LocalOtsuSegmentation",
    "ThresholdSegmentation",
    "GaussianBiasFieldCorrection",
    "N4BiasFieldCorrection",
    "SegmentationCleaner",
    "hookspec",
    "hookimpl",
    "get_plugin_manager",
    "get_global_plugin_manager",
]
