"""Enumeration classes for ZarrNii type definitions."""

from enum import Enum, auto


class TransformType(Enum):
    """Enumeration of supported transform types.

    Attributes:
        AFFINE_RAS: Affine transformation in RAS coordinate system
        DISPLACEMENT_RAS: Displacement field transformation in RAS coordinate system
    """

    AFFINE_RAS = auto()
    DISPLACEMENT_RAS = auto()


class ImageType(Enum):
    """Enumeration of supported image formats.

    Attributes:
        OME_ZARR: OME-Zarr format (multiscale with metadata)
        ZARR: Standard Zarr format
        NIFTI: NIfTI format (Neuroimaging Informatics Technology Initiative)
        UNKNOWN: Unknown or unsupported format
    """

    OME_ZARR = auto()
    ZARR = auto()
    NIFTI = auto()
    UNKNOWN = auto()
