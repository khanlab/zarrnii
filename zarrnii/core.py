"""Core facade module for backward compatibility.

This module re-exports all classes and functions from the modular structure
to maintain backward compatibility with existing code that imports from
zarrnii.core.

For new development, prefer importing directly from specific modules:
- zarrnii.zarrnii for the main ZarrNii class
- zarrnii.io for I/O utilities
- zarrnii.processing for image processing functions
- zarrnii.utils for utility functions
- zarrnii.transform for transformation classes
- zarrnii.enums for enumerations
"""

from .enums import ImageType, TransformType
from .io import (
    _extract_channel_labels_from_omero,
    _select_channels_from_image,
    _select_dimensions_from_image,
    _select_dimensions_from_image_with_omero,
    get_multiscales,
    load_ngff_image,
    save_ngff_image,
)
from .processing import (
    apply_near_isotropic_downsampling,
    apply_transform_to_ngff_image,
    crop_ngff_image,
    downsample_ngff_image,
)
from .transform import AffineTransform, Transform
from .utils import (
    affine_to_orientation,
    align_affine_to_input_orientation,
    construct_affine_with_orientation,
    get_affine_matrix,
    get_affine_transform,
    orientation_to_affine,
)

# Re-export from modular structure for backward compatibility
from .zarrnii import ZarrNii

# Backward compatibility aliases
_apply_near_isotropic_downsampling = apply_near_isotropic_downsampling
