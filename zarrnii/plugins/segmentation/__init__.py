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

# StarDist plugin - optional dependency
try:
    # Check if StarDist is actually available
    import stardist  # noqa: F401

    from .stardist import StarDistSegmentation  # noqa: F401

    __all__.append("StarDistSegmentation")
except ImportError:
    # StarDist dependencies not available
    pass
