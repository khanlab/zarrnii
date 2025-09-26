from .analysis import compute_histogram, compute_otsu_thresholds
from .atlas import (
    Atlas,
    Template,
    get_builtin_atlas,
    get_builtin_template,
    get_builtin_template_atlas,
    get_templateflow_template,
    import_lut_csv_as_tsv,
    import_lut_itksnap_as_tsv,
    install_zarrnii_templates,
    list_builtin_atlases,
    list_builtin_templates,
    list_templateflow_templates,
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
    "get_builtin_atlas",
    "get_builtin_template",
    "get_builtin_template_atlas",
    "get_templateflow_template",
    "install_zarrnii_templates",
    "list_builtin_atlases",
    "list_builtin_templates",
    "list_templateflow_templates",
]
