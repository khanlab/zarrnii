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

# Re-export from modular structure for backward compatibility
from .zarrnii import ZarrNii
from .io import (
    load_ngff_image,
    save_ngff_image,
    get_multiscales,
    _select_dimensions_from_image,
    _select_channels_from_image,
    _extract_channel_labels_from_omero,
    _select_dimensions_from_image_with_omero,
)
from .processing import (
    crop_ngff_image,
    downsample_ngff_image,
    apply_transform_to_ngff_image,
    apply_near_isotropic_downsampling,
)
from .utils import (
    get_affine_matrix,
    get_affine_transform,
    affine_to_orientation,
    orientation_to_affine,
    align_affine_to_input_orientation,
    construct_affine_with_orientation,
)
from .transform import AffineTransform, Transform
from .enums import ImageType, TransformType

# Backward compatibility aliases
_apply_near_isotropic_downsampling = apply_near_isotropic_downsampling