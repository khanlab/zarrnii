"""
Tests for centroid computation functions.

This module tests the compute_centroids function to ensure it correctly
identifies and localizes objects in binary segmentation images.
"""

import dask.array as da
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from zarrnii import ZarrNii, compute_centroids
from zarrnii.transform import AffineTransform


class TestComputeCentroidsStandalone:
    """Test the standalone compute_centroids function."""

    def test_compute_centroids_single_object(self):
        """Test centroid computation with a single object."""
        # Create a binary image with a single blob at known location
        img = np.zeros((50, 50, 50), dtype=np.uint8)
        # Place a 10x10x10 cube centered at (25, 25, 25)
        img[20:30, 20:30, 20:30] = 1

        # Convert to dask array
        dask_img = da.from_array(img, chunks=(25, 25, 25))

        # Identity affine (voxel coords = physical coords)
        affine = np.eye(4)

        # Compute centroids
        centroids = compute_centroids(dask_img, affine, depth=5)

        # Should find one centroid at (24.5, 24.5, 24.5) - center of the cube
        assert centroids.shape == (1, 3)
        assert_array_almost_equal(centroids[0], [24.5, 24.5, 24.5], decimal=1)

    def test_compute_centroids_multiple_objects(self):
        """Test centroid computation with multiple separate objects."""
        # Create a binary image with multiple blobs
        img = np.zeros((60, 60, 60), dtype=np.uint8)

        # Object 1: centered at (10, 10, 10)
        img[8:13, 8:13, 8:13] = 1

        # Object 2: centered at (30, 30, 30)
        img[28:33, 28:33, 28:33] = 1

        # Object 3: centered at (50, 50, 50)
        img[48:53, 48:53, 48:53] = 1

        # Convert to dask array
        dask_img = da.from_array(img, chunks=(30, 30, 30))

        # Identity affine
        affine = np.eye(4)

        # Compute centroids
        centroids = compute_centroids(dask_img, affine, depth=5)

        # Should find three centroids
        assert centroids.shape == (3, 3)

        # Check that centroids are approximately at expected locations
        expected_centroids = np.array(
            [[10.0, 10.0, 10.0], [30.0, 30.0, 30.0], [50.0, 50.0, 50.0]]
        )

        # Sort both arrays by first coordinate for comparison
        centroids_sorted = centroids[np.argsort(centroids[:, 0])]
        expected_sorted = expected_centroids[np.argsort(expected_centroids[:, 0])]

        assert_array_almost_equal(centroids_sorted, expected_sorted, decimal=0)

    def test_compute_centroids_with_affine_transform(self):
        """Test that affine transform is correctly applied."""
        # Create a simple binary image
        img = np.zeros((40, 40, 40), dtype=np.uint8)
        img[18:23, 18:23, 18:23] = 1  # Centered at (20, 20, 20)

        dask_img = da.from_array(img, chunks=(20, 20, 20))

        # Affine with 2mm spacing and 10mm offset
        affine = np.array(
            [[2.0, 0, 0, 10], [0, 2.0, 0, 10], [0, 0, 2.0, 10], [0, 0, 0, 1]]
        )

        # Compute centroids
        centroids = compute_centroids(dask_img, affine, depth=3)

        # Voxel centroid is at (20, 20, 20)
        # Physical coord = 2.0 * 20 + 10 = 50
        expected = np.array([[50.0, 50.0, 50.0]])

        assert centroids.shape == (1, 3)
        assert_array_almost_equal(centroids, expected, decimal=1)

    def test_compute_centroids_empty_image(self):
        """Test with empty image (no objects)."""
        img = np.zeros((50, 50, 50), dtype=np.uint8)
        dask_img = da.from_array(img, chunks=(25, 25, 25))
        affine = np.eye(4)

        centroids = compute_centroids(dask_img, affine, depth=5)

        # Should return empty array with correct shape
        assert centroids.shape == (0, 3)

    def test_compute_centroids_overlap_filtering(self):
        """Test that objects in overlap regions are not duplicated."""
        # Create image with object near chunk boundary
        img = np.zeros((60, 60, 60), dtype=np.uint8)

        # Place object exactly at chunk boundary (30, 30, 30)
        img[28:33, 28:33, 28:33] = 1

        dask_img = da.from_array(img, chunks=(30, 30, 30))
        affine = np.eye(4)

        # Use overlap that should catch this object
        centroids = compute_centroids(dask_img, affine, depth=10)

        # Should find exactly one centroid (not duplicated)
        assert centroids.shape == (1, 3)
        assert_array_almost_equal(centroids[0], [30.0, 30.0, 30.0], decimal=0)

    def test_compute_centroids_with_rechunk(self):
        """Test rechunking parameter."""
        img = np.zeros((50, 50, 50), dtype=np.uint8)
        img[20:30, 20:30, 20:30] = 1

        # Start with one chunk size
        dask_img = da.from_array(img, chunks=(50, 50, 50))
        affine = np.eye(4)

        # Rechunk to smaller chunks
        centroids = compute_centroids(dask_img, affine, depth=5, rechunk=(25, 25, 25))

        assert centroids.shape == (1, 3)
        assert_array_almost_equal(centroids[0], [24.5, 24.5, 24.5], decimal=1)

    def test_compute_centroids_different_depth_per_axis(self):
        """Test with different overlap depths per axis."""
        img = np.zeros((50, 50, 50), dtype=np.uint8)
        img[20:30, 20:30, 20:30] = 1

        dask_img = da.from_array(img, chunks=(25, 25, 25))
        affine = np.eye(4)

        # Different depth per axis
        centroids = compute_centroids(dask_img, affine, depth=(5, 10, 15))

        assert centroids.shape == (1, 3)
        assert_array_almost_equal(centroids[0], [24.5, 24.5, 24.5], decimal=1)

    def test_compute_centroids_with_affine_transform_object(self):
        """Test with AffineTransform object instead of numpy array."""
        img = np.zeros((40, 40, 40), dtype=np.uint8)
        img[18:23, 18:23, 18:23] = 1

        dask_img = da.from_array(img, chunks=(20, 20, 20))

        # Create AffineTransform object
        affine_matrix = np.array(
            [[2.0, 0, 0, 10], [0, 2.0, 0, 10], [0, 0, 2.0, 10], [0, 0, 0, 1]]
        )
        affine = AffineTransform.from_array(affine_matrix)

        centroids = compute_centroids(dask_img, affine, depth=3)

        expected = np.array([[50.0, 50.0, 50.0]])
        assert centroids.shape == (1, 3)
        assert_array_almost_equal(centroids, expected, decimal=1)

    def test_compute_centroids_invalid_affine_shape(self):
        """Test error handling for invalid affine matrix shape."""
        img = np.zeros((50, 50, 50), dtype=np.uint8)
        dask_img = da.from_array(img, chunks=(25, 25, 25))

        # Invalid affine matrix
        affine = np.eye(3)

        with pytest.raises(ValueError, match="Affine matrix must be 4x4"):
            compute_centroids(dask_img, affine, depth=5)


class TestZarrNiiComputeCentroids:
    """Test the ZarrNii.compute_centroids method."""

    def test_compute_centroids_method_basic(self):
        """Test basic centroid computation through ZarrNii method."""
        # Create a binary segmentation
        img = np.zeros((50, 50, 50), dtype=np.uint8)
        img[20:30, 20:30, 20:30] = 1

        dask_img = da.from_array(img, chunks=(25, 25, 25))

        # Create ZarrNii object using from_darr
        znii = ZarrNii.from_darr(
            dask_img, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), axes_order="ZYX"
        )

        # Compute centroids
        centroids = znii.compute_centroids(depth=5)

        assert centroids.shape == (1, 3)
        # Note: centroids should be in physical coordinates
        # With identity affine, they match voxel coordinates

    def test_compute_centroids_method_after_segmentation(self):
        """Test computing centroids after threshold segmentation."""
        # Create test image with bimodal distribution
        img = np.zeros((60, 60, 60), dtype=np.float32)

        # Background
        img[:30, :30, :30] = 0.2

        # Foreground objects
        img[10:20, 10:20, 10:20] = 0.8
        img[40:50, 40:50, 40:50] = 0.8

        dask_img = da.from_array(img, chunks=(30, 30, 30))

        znii = ZarrNii.from_darr(
            dask_img, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), axes_order="ZYX"
        )

        # Apply threshold segmentation
        binary = znii.segment_threshold(0.5)

        # Compute centroids
        centroids = binary.compute_centroids(depth=8)

        # Should find two objects
        assert centroids.shape[0] == 2
        assert centroids.shape[1] == 3

    def test_compute_centroids_with_scaled_affine(self):
        """Test with non-identity affine transform."""
        img = np.zeros((40, 40, 40), dtype=np.uint8)
        img[18:23, 18:23, 18:23] = 1

        dask_img = da.from_array(img, chunks=(20, 20, 20))

        # Create with 2mm spacing
        znii = ZarrNii.from_darr(
            dask_img,
            spacing=(2.0, 2.0, 2.0),
            origin=(10.0, 10.0, 10.0),
            axes_order="ZYX",
        )

        centroids = znii.compute_centroids(depth=3)

        # Physical coord = 2.0 * 20 + 10 = 50
        expected = np.array([[50.0, 50.0, 50.0]])
        assert centroids.shape == (1, 3)
        assert_array_almost_equal(centroids, expected, decimal=1)


class TestComputeCentroidsEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_voxel_objects(self):
        """Test with single-voxel objects."""
        img = np.zeros((50, 50, 50), dtype=np.uint8)

        # Add several single-voxel objects
        img[10, 10, 10] = 1
        img[20, 20, 20] = 1
        img[30, 30, 30] = 1

        dask_img = da.from_array(img, chunks=(25, 25, 25))
        affine = np.eye(4)

        centroids = compute_centroids(dask_img, affine, depth=3)

        # Should find three objects
        assert centroids.shape == (3, 3)

    def test_very_small_chunks(self):
        """Test with very small chunks."""
        img = np.zeros((30, 30, 30), dtype=np.uint8)
        img[13:18, 13:18, 13:18] = 1

        # Very small chunks
        dask_img = da.from_array(img, chunks=(10, 10, 10))
        affine = np.eye(4)

        centroids = compute_centroids(dask_img, affine, depth=5)

        assert centroids.shape == (1, 3)
        assert_array_almost_equal(centroids[0], [15.0, 15.0, 15.0], decimal=0)

    def test_object_spanning_multiple_chunks(self):
        """Test with object that spans multiple chunks."""
        img = np.zeros((60, 60, 60), dtype=np.uint8)

        # Large object spanning multiple chunks
        img[25:35, 25:35, 25:35] = 1

        dask_img = da.from_array(img, chunks=(20, 20, 20))
        affine = np.eye(4)

        centroids = compute_centroids(dask_img, affine, depth=8)

        # Should find exactly one object (not fragmented)
        assert centroids.shape == (1, 3)
        assert_array_almost_equal(centroids[0], [30.0, 30.0, 30.0], decimal=0)
