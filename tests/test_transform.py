"""Tests for transformation functionality."""
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from zarrnii import AffineTransform


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