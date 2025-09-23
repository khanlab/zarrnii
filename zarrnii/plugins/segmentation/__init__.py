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

# StarDist plugin - optional dependency
try:
    # Check if StarDist is actually available
    import stardist  # noqa: F401

    from .stardist import StarDistSegmentation  # noqa: F401

    __all__.append("StarDistSegmentation")
except ImportError:
    # StarDist dependencies not available
    pass
