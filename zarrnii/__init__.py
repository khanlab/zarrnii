from .analysis import compute_histogram, compute_otsu_thresholds
from .atlas import (
    AmbiguousTemplateFlowQueryError,
    Atlas,
    Template,
    add_template_to_templateflow,
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
    "AmbiguousTemplateFlowQueryError",
    "Atlas",
    "Template",
    "ZarrNii",
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
    "import_lut_csv_as_tsv",
    "import_lut_itksnap_as_tsv",
    "add_template_to_templateflow",
    "get",
    "get_template",
]
