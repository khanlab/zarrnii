from .analysis import compute_histogram, compute_otsu_thresholds
from .atlas import (
    AmbiguousTemplateFlowQueryError,
    Template,
    ZarrNiiAtlas,
    get,
    get_template,
    import_lut_csv_as_tsv,
    import_lut_itksnap_as_tsv,
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
    "ZarrNii",
    "ZarrNiiAtlas",
    "Transform",
    "AffineTransform",
    "DisplacementTransform",
    "SegmentationPlugin",
    "OtsuSegmentation",  # Backward compatibility
    "LocalOtsuSegmentation",
    "ThresholdSegmentation",
    "ScaledProcessingPlugin",
    "GaussianBiasFieldCorrection",
    "N4BiasFieldCorrection",
]
