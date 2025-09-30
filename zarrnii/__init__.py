from .analysis import compute_histogram, compute_otsu_thresholds
from .atlas import (
    AmbiguousTemplateFlowQueryError,
    Template,
    ZarrNiiAtlas,
    get,
    get_template,
)
from .core import ZarrNii, affine_to_orientation
from .plugins import OtsuSegmentation  # Backward compatibility
from .plugins import (
    GaussianBiasFieldCorrection,
    LocalOtsuSegmentation,
    N4BiasFieldCorrection,
    ScaledProcessingPlugin,
    SegmentationPlugin,
    ThresholdSegmentation,
)
from .transform import AffineTransform, DisplacementTransform, Transform

__all__ = [
    "AmbiguousTemplateFlowQueryError",
    "Template",
    "ZarrNii",
    "ZarrNiiAtlas",
    "Transform",
    "AffineTransform",
    "DisplacementTransform",
    "affine_to_orientation",
    "SegmentationPlugin",
    "OtsuSegmentation",  # Backward compatibility
    "LocalOtsuSegmentation",
    "ThresholdSegmentation",
    "ScaledProcessingPlugin",
    "GaussianBiasFieldCorrection",
    "N4BiasFieldCorrection",
    "compute_histogram",
    "compute_otsu_thresholds",
    "get",
    "get_template",
]
