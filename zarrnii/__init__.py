from .core import ZarrNii, affine_to_orientation
from .plugins import OtsuSegmentation, SegmentationPlugin
from .transform import AffineTransform, DisplacementTransform, Transform

# Visualization imports with optional dependency handling
try:
    from . import visualization
    _has_visualization = True
    # Make stop_servers available at package level for convenience
    stop_servers = visualization.stop_servers
except ImportError:
    _has_visualization = False
    visualization = None
    stop_servers = None

__all__ = [
    "ZarrNii",
    "Transform",
    "AffineTransform",
    "DisplacementTransform",
    "affine_to_orientation",
    "SegmentationPlugin",
    "OtsuSegmentation",
]

# Add visualization to __all__ if available
if _has_visualization:
    __all__.extend(["visualization", "stop_servers"])
