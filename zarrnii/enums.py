from enum import Enum, auto


class TransformType(Enum):
    AFFINE_RAS = auto()
    DISPLACEMENT_RAS = auto()


class ImageType(Enum):
    OME_ZARR = auto()
    ZARR = auto()
    NIFTI = auto()
    UNKNOWN = auto()
