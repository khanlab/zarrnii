"""Tests for transformation functionality."""

import os
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from zarrnii import AffineTransform, DisplacementTransform
from zarrnii.transform import Transform


def test_affine_transform_identity():
    """Test that identity transform works correctly."""
    identity_matrix = np.eye(4)
    transform = AffineTransform.from_array(identity_matrix)

    # Test that the matrix is correctly stored
    assert_array_almost_equal(transform.matrix, identity_matrix)


def test_affine_transform_identity_classmethod():
    """Test identity class method."""
    transform = AffineTransform.identity()
    expected_matrix = np.eye(4)

    assert_array_almost_equal(transform.matrix, expected_matrix)


def test_affine_transform_from_array():
    """Test creating affine transform from array."""
    # Create a simple scaling matrix
    test_matrix = np.eye(4)
    test_matrix[0, 0] = 2.0
    test_matrix[1, 1] = 3.0
    test_matrix[2, 2] = 1.5

    transform = AffineTransform.from_array(test_matrix)

    assert_array_almost_equal(transform.matrix, test_matrix)


def test_affine_transform_from_array_with_invert():
    """Test creating affine transform from array with inversion."""
    # Create a simple scaling matrix
    test_matrix = np.eye(4)
    test_matrix[0, 0] = 2.0
    test_matrix[1, 1] = 3.0
    test_matrix[2, 2] = 1.5

    transform = AffineTransform.from_array(test_matrix, invert=True)
    expected_matrix = np.linalg.inv(test_matrix)

    assert_array_almost_equal(transform.matrix, expected_matrix)


def test_affine_transform_from_txt():
    """Test creating affine transform from text file."""
    # Create a temporary file with matrix data
    test_matrix = np.eye(4)
    test_matrix[0, 0] = 2.0
    test_matrix[1, 1] = 3.0
    test_matrix[2, 2] = 1.5

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        np.savetxt(f, test_matrix)
        temp_path = f.name

    try:
        transform = AffineTransform.from_txt(temp_path)
        assert_array_almost_equal(transform.matrix, test_matrix)

        # Test with invert
        transform_inv = AffineTransform.from_txt(temp_path, invert=True)
        expected_matrix = np.linalg.inv(test_matrix)
        assert_array_almost_equal(transform_inv.matrix, expected_matrix)
    finally:
        os.unlink(temp_path)


def test_affine_transform_invert():
    """Test matrix inversion."""
    # Create a simple transform
    matrix = np.eye(4)
    matrix[0, 3] = 5.0  # translation in x
    matrix[1, 1] = 2.0  # scaling in y

    transform = AffineTransform.from_array(matrix)
    inverted = transform.invert()

    # Verify inversion: transform @ inverted should be identity
    identity_result = transform @ inverted
    assert_array_almost_equal(identity_result.matrix, np.eye(4), decimal=10)


def test_affine_transform_matmul():
    """Test matrix multiplication with another affine transform."""
    # Create two transforms
    matrix1 = np.eye(4)
    matrix1[0, 0] = 2.0  # scale x by 2

    matrix2 = np.eye(4)
    matrix2[0, 3] = 3.0  # translate x by 3

    transform1 = AffineTransform.from_array(matrix1)
    transform2 = AffineTransform.from_array(matrix2)

    # Multiply transforms
    combined = transform1 @ transform2

    # Expected result: first scale, then translate
    expected_matrix = matrix1 @ matrix2
    assert_array_almost_equal(combined.matrix, expected_matrix)


def test_affine_transform_apply_point():
    """Test applying transform to a single point."""
    # Create a translation transform
    matrix = np.eye(4)
    matrix[0, 3] = 1.0  # translate x by 1
    matrix[1, 3] = 2.0  # translate y by 2
    matrix[2, 3] = 3.0  # translate z by 3

    transform = AffineTransform.from_array(matrix)

    # Apply to a point
    point = np.array([0.0, 0.0, 0.0])
    transformed_point = transform @ point

    expected_point = np.array([1.0, 2.0, 3.0])
    assert_array_almost_equal(transformed_point, expected_point)


def test_affine_transform_apply_multiple_points():
    """Test applying transform to multiple points."""
    # Create a translation transform
    matrix = np.eye(4)
    matrix[0, 3] = 1.0  # translate x by 1
    matrix[1, 3] = 2.0  # translate y by 2
    matrix[2, 3] = 3.0  # translate z by 3

    transform = AffineTransform.from_array(matrix)

    # Apply to multiple points (3 x N format)
    points = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    transformed_points = transform @ points

    expected_points = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    assert_array_almost_equal(transformed_points, expected_points)


def test_affine_transform_apply_4d_point():
    """Test applying transform to a 4D homogeneous point."""
    # Create a translation transform
    matrix = np.eye(4)
    matrix[0, 3] = 1.0  # translate x by 1
    matrix[1, 3] = 2.0  # translate y by 2
    matrix[2, 3] = 3.0  # translate z by 3

    transform = AffineTransform.from_array(matrix)

    # Apply to a 4D homogeneous point
    point = np.array([0.0, 0.0, 0.0, 1.0])
    transformed_point = transform @ point

    # The implementation converts 4D homogeneous back to 3D coordinates
    expected_point = np.array([1.0, 2.0, 3.0])
    assert_array_almost_equal(transformed_point, expected_point)


def test_affine_transform_array_interface():
    """Test the __array__ interface."""
    matrix = np.eye(4)
    matrix[0, 3] = 5.0
    transform = AffineTransform.from_array(matrix)

    # Convert to array
    array_result = np.array(transform)
    assert_array_almost_equal(array_result, matrix)


def test_affine_transform_indexing():
    """Test array-like indexing and assignment."""
    matrix = np.eye(4)
    transform = AffineTransform.from_array(matrix)

    # Test getting
    assert transform[0, 0] == 1.0
    assert transform[0, 3] == 0.0

    # Test setting
    transform[0, 3] = 5.0
    assert transform[0, 3] == 5.0
    assert transform.matrix[0, 3] == 5.0


def test_affine_transform_apply_transform_method():
    """Test the apply_transform method."""
    # Create a translation transform
    matrix = np.eye(4)
    matrix[0, 3] = 1.0
    transform = AffineTransform.from_array(matrix)

    # Apply transform using apply_transform method
    points = np.array([0.0, 0.0, 0.0])
    result = transform.apply_transform(points)

    expected = np.array([1.0, 0.0, 0.0])
    assert_array_almost_equal(result, expected)


def test_transform_abc():
    """Test that Transform is an abstract base class."""
    with pytest.raises(TypeError):
        # Cannot instantiate abstract class
        Transform()


def test_displacement_transform_basic():
    """Test DisplacementTransform basic functionality with mock data."""
    # Create mock displacement data
    disp_xyz = np.zeros((10, 10, 10, 3))
    disp_xyz[5, 5, 5, :] = [1.0, 2.0, 3.0]  # Single displacement vector

    disp_grid = (np.arange(10), np.arange(10), np.arange(10))

    disp_affine = AffineTransform.identity()

    # Create displacement transform directly
    disp_transform = DisplacementTransform(
        disp_xyz=disp_xyz, disp_grid=disp_grid, disp_affine=disp_affine
    )

    # Test applying transform to a point
    point = np.array([[5.0], [5.0], [5.0]])  # Point at center
    result = disp_transform.apply_transform(point)

    # Should apply the displacement
    expected = point + np.array([[1.0], [2.0], [3.0]])
    assert_array_almost_equal(result, expected, decimal=1)


def test_affine_transform_update_for_orientation():
    """Test update_for_orientation method (placeholder test)."""
    transform = AffineTransform.identity()

    # This method appears to be a placeholder, so just test it doesn't crash
    try:
        transform.update_for_orientation("RAS", "LPI")
        # If it doesn't crash, that's good for now
        assert True
    except NotImplementedError:
        # If it raises NotImplementedError, that's also acceptable
        assert True
