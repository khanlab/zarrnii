"""Tests for enumeration classes."""

import pytest

from zarrnii.enums import ImageType, TransformType


def test_transform_type_enum():
    """Test TransformType enum values."""
    assert TransformType.AFFINE_RAS
    assert TransformType.DISPLACEMENT_RAS

    # Test that enum values are unique
    assert TransformType.AFFINE_RAS != TransformType.DISPLACEMENT_RAS

    # Test enum has expected number of values
    assert len(TransformType) == 2


def test_image_type_enum():
    """Test ImageType enum values."""
    assert ImageType.OME_ZARR
    assert ImageType.ZARR
    assert ImageType.NIFTI
    assert ImageType.UNKNOWN

    # Test that enum values are unique
    assert ImageType.OME_ZARR != ImageType.ZARR
    assert ImageType.ZARR != ImageType.NIFTI
    assert ImageType.NIFTI != ImageType.UNKNOWN

    # Test enum has expected number of values
    assert len(ImageType) == 4


def test_enum_string_representation():
    """Test string representation of enum values."""
    assert str(TransformType.AFFINE_RAS) == "TransformType.AFFINE_RAS"
    assert str(ImageType.OME_ZARR) == "ImageType.OME_ZARR"


def test_enum_equality():
    """Test enum equality and inequality."""
    # Same enum values should be equal
    assert ImageType.NIFTI == ImageType.NIFTI
    assert TransformType.AFFINE_RAS == TransformType.AFFINE_RAS

    # Different enum values should not be equal
    assert ImageType.NIFTI != ImageType.ZARR
    assert TransformType.AFFINE_RAS != TransformType.DISPLACEMENT_RAS
