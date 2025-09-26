from .analysis import compute_histogram, compute_otsu_thresholds
from .atlas import (
    Atlas,
    Template,
    get_builtin_atlas,
    get_builtin_template,
    get_builtin_template_atlas,
    import_lut_csv_as_tsv,
    import_lut_itksnap_as_tsv,
    list_builtin_atlases,
    list_builtin_templates,
)
from .core import ZarrNii, affine_to_orientation
from .plugins import OtsuSegmentation  # Backward compatibility
from .plugins import (
    BiasFieldCorrection,
    LocalOtsuSegmentation,
    ScaledProcessingPlugin,
    SegmentationPlugin,
    ThresholdSegmentation,
)
from .transform import AffineTransform, DisplacementTransform, Transform

__all__ = [
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
    "BiasFieldCorrection",
    "import_lut_csv_as_tsv",
    "import_lut_itksnap_as_tsv",
    "get_builtin_atlas",
    "get_builtin_template",
    "get_builtin_template_atlas",
    "list_builtin_atlases",
    "list_builtin_templates",
    "compute_histogram",
    "compute_otsu_thresholds",
]
