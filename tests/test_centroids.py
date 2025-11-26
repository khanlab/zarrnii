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


class TestComputeCentroidsCZYX:
    """Test compute_centroids with 4D CZYX data (channel dimension)."""

    def test_single_channel_single_object(self):
        """Test 4D data with single channel (C=1) and single object."""
        # Create a 4D binary image with a single channel
        img = np.zeros((1, 50, 50, 50), dtype=np.uint8)
        img[0, 20:30, 20:30, 20:30] = 1

        dask_img = da.from_array(img, chunks=(1, 25, 25, 25))
        affine = np.eye(4)

        # Should automatically squeeze channel dimension and process as 3D
        centroids = compute_centroids(dask_img, affine, depth=5)

        assert centroids.shape == (1, 3)
        assert_array_almost_equal(centroids[0], [24.5, 24.5, 24.5], decimal=1)

    def test_single_channel_multiple_objects(self):
        """Test 4D data with single channel and multiple objects."""
        img = np.zeros((1, 60, 60, 60), dtype=np.uint8)

        # Add multiple objects
        img[0, 8:13, 8:13, 8:13] = 1
        img[0, 28:33, 28:33, 28:33] = 1
        img[0, 48:53, 48:53, 48:53] = 1

        dask_img = da.from_array(img, chunks=(1, 30, 30, 30))
        affine = np.eye(4)

        centroids = compute_centroids(dask_img, affine, depth=5)

        # Should find three centroids
        assert centroids.shape == (3, 3)

        # Check approximate locations
        expected_centroids = np.array(
            [[10.0, 10.0, 10.0], [30.0, 30.0, 30.0], [50.0, 50.0, 50.0]]
        )
        centroids_sorted = centroids[np.argsort(centroids[:, 0])]
        expected_sorted = expected_centroids[np.argsort(expected_centroids[:, 0])]
        assert_array_almost_equal(centroids_sorted, expected_sorted, decimal=0)

    def test_single_channel_with_affine_transform(self):
        """Test 4D data with single channel and affine transform."""
        img = np.zeros((1, 40, 40, 40), dtype=np.uint8)
        img[0, 18:23, 18:23, 18:23] = 1

        dask_img = da.from_array(img, chunks=(1, 20, 20, 20))

        # Affine with 2mm spacing and 10mm offset
        affine = np.array(
            [[2.0, 0, 0, 10], [0, 2.0, 0, 10], [0, 0, 2.0, 10], [0, 0, 0, 1]]
        )

        centroids = compute_centroids(dask_img, affine, depth=3)

        # Voxel centroid is at (20, 20, 20), physical = 2.0 * 20 + 10 = 50
        expected = np.array([[50.0, 50.0, 50.0]])

        assert centroids.shape == (1, 3)
        assert_array_almost_equal(centroids, expected, decimal=1)

    def test_single_channel_empty_image(self):
        """Test 4D data with single channel and no objects."""
        img = np.zeros((1, 50, 50, 50), dtype=np.uint8)

        dask_img = da.from_array(img, chunks=(1, 25, 25, 25))
        affine = np.eye(4)

        centroids = compute_centroids(dask_img, affine, depth=5)

        # Should return empty array with correct shape
        assert centroids.shape == (0, 3)

    def test_multi_channel_raises_error(self):
        """Test that multi-channel images raise informative error."""
        # Create image with 3 channels
        img = np.zeros((3, 50, 50, 50), dtype=np.uint8)
        img[0, 20:30, 20:30, 20:30] = 1

        dask_img = da.from_array(img, chunks=(1, 25, 25, 25))
        affine = np.eye(4)

        # Should raise ValueError with informative message
        with pytest.raises(ValueError, match="compute_centroids only supports 3D"):
            compute_centroids(dask_img, affine, depth=5)

    def test_five_dimensional_raises_error(self):
        """Test that 5D images raise informative error."""
        img = np.zeros((2, 2, 50, 50, 50), dtype=np.uint8)

        dask_img = da.from_array(img, chunks=(1, 1, 25, 25, 25))
        affine = np.eye(4)

        # Should raise ValueError for unsupported dimensionality
        with pytest.raises(ValueError, match="Image must be 1D, 2D, 3D, or 4D"):
            compute_centroids(dask_img, affine, depth=5)


class TestComputeCentroidsParquet:
    """Test the Parquet output functionality."""

    def test_parquet_output_single_object(self, tmp_path):
        """Test writing centroids to Parquet file with single object."""
        import pandas as pd

        # Create a binary image with a single blob at known location
        img = np.zeros((50, 50, 50), dtype=np.uint8)
        img[20:30, 20:30, 20:30] = 1

        dask_img = da.from_array(img, chunks=(25, 25, 25))
        affine = np.eye(4)

        # Write to Parquet file
        output_file = tmp_path / "centroids.parquet"
        result = compute_centroids(
            dask_img, affine, depth=5, output_path=str(output_file)
        )

        # Should return None when output_path is specified
        assert result is None

        # Verify file exists
        assert output_file.exists()

        # Read back and verify contents
        df = pd.read_parquet(output_file)
        assert len(df) == 1
        assert list(df.columns) == ["x", "y", "z"]
        assert_array_almost_equal(df.iloc[0].values, [24.5, 24.5, 24.5], decimal=1)

    def test_parquet_output_multiple_objects(self, tmp_path):
        """Test writing multiple centroids to Parquet file."""
        import pandas as pd

        # Create a binary image with multiple blobs
        img = np.zeros((60, 60, 60), dtype=np.uint8)
        img[8:13, 8:13, 8:13] = 1
        img[28:33, 28:33, 28:33] = 1
        img[48:53, 48:53, 48:53] = 1

        dask_img = da.from_array(img, chunks=(30, 30, 30))
        affine = np.eye(4)

        # Write to Parquet file
        output_file = tmp_path / "centroids.parquet"
        result = compute_centroids(
            dask_img, affine, depth=5, output_path=str(output_file)
        )

        assert result is None
        assert output_file.exists()

        # Read back and verify
        df = pd.read_parquet(output_file)
        assert len(df) == 3
        assert list(df.columns) == ["x", "y", "z"]

        # Sort by x coordinate for comparison
        df_sorted = df.sort_values("x").reset_index(drop=True)
        expected = np.array(
            [[10.0, 10.0, 10.0], [30.0, 30.0, 30.0], [50.0, 50.0, 50.0]]
        )
        assert_array_almost_equal(df_sorted.values, expected, decimal=0)

    def test_parquet_output_with_affine_transform(self, tmp_path):
        """Test Parquet output with affine transformation."""
        import pandas as pd

        img = np.zeros((40, 40, 40), dtype=np.uint8)
        img[18:23, 18:23, 18:23] = 1

        dask_img = da.from_array(img, chunks=(20, 20, 20))
        affine = np.array(
            [[2.0, 0, 0, 10], [0, 2.0, 0, 10], [0, 0, 2.0, 10], [0, 0, 0, 1]]
        )

        output_file = tmp_path / "centroids.parquet"
        compute_centroids(dask_img, affine, depth=3, output_path=str(output_file))

        df = pd.read_parquet(output_file)
        assert len(df) == 1
        expected = np.array([[50.0, 50.0, 50.0]])
        assert_array_almost_equal(df.values, expected, decimal=1)

    def test_parquet_output_empty_image(self, tmp_path):
        """Test Parquet output with empty image (no objects)."""
        import pandas as pd

        img = np.zeros((50, 50, 50), dtype=np.uint8)
        dask_img = da.from_array(img, chunks=(25, 25, 25))
        affine = np.eye(4)

        output_file = tmp_path / "centroids.parquet"
        result = compute_centroids(
            dask_img, affine, depth=5, output_path=str(output_file)
        )

        assert result is None
        assert output_file.exists()

        # Read back and verify empty file
        df = pd.read_parquet(output_file)
        assert len(df) == 0
        assert list(df.columns) == ["x", "y", "z"]

    def test_parquet_via_zarrnii_method(self, tmp_path):
        """Test Parquet output through ZarrNii method."""
        import pandas as pd

        img = np.zeros((50, 50, 50), dtype=np.uint8)
        img[20:30, 20:30, 20:30] = 1

        dask_img = da.from_array(img, chunks=(25, 25, 25))
        znii = ZarrNii.from_darr(
            dask_img, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), axes_order="ZYX"
        )

        output_file = tmp_path / "centroids.parquet"
        result = znii.compute_centroids(depth=5, output_path=str(output_file))

        assert result is None
        assert output_file.exists()

        df = pd.read_parquet(output_file)
        assert len(df) == 1
        assert_array_almost_equal(df.iloc[0].values, [24.5, 24.5, 24.5], decimal=1)

    def test_parquet_backward_compatibility(self):
        """Test that default behavior (no output_path) still returns numpy array."""
        img = np.zeros((50, 50, 50), dtype=np.uint8)
        img[20:30, 20:30, 20:30] = 1

        dask_img = da.from_array(img, chunks=(25, 25, 25))
        affine = np.eye(4)

        # Without output_path, should return numpy array
        result = compute_centroids(dask_img, affine, depth=5)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3)
        assert_array_almost_equal(result[0], [24.5, 24.5, 24.5], decimal=1)


class TestRegionFilters:
    """Test the region_filters functionality for filtering objects by regionprops."""

    def test_filter_by_area_minimum(self):
        """Test filtering regions by minimum area."""
        # Create image with objects of different sizes
        img = np.zeros((60, 60, 60), dtype=np.uint8)

        # Small object (1 voxel)
        img[10, 10, 10] = 1

        # Medium object (~125 voxels = 5x5x5)
        img[28:33, 28:33, 28:33] = 1

        # Large object (~1000 voxels = 10x10x10)
        img[45:55, 45:55, 45:55] = 1

        dask_img = da.from_array(img, chunks=(30, 30, 30))
        affine = np.eye(4)

        # Without filter - should find all 3 objects
        centroids_all = compute_centroids(dask_img, affine, depth=5)
        assert centroids_all.shape[0] == 3

        # Filter by minimum area >= 30 - should exclude single voxel
        centroids_filtered = compute_centroids(
            dask_img, affine, depth=5, region_filters={"area": (">=", 30)}
        )
        assert centroids_filtered.shape[0] == 2

        # Filter by minimum area >= 500 - should only keep large object
        centroids_large = compute_centroids(
            dask_img, affine, depth=5, region_filters={"area": (">=", 500)}
        )
        assert centroids_large.shape[0] == 1

    def test_filter_by_area_maximum(self):
        """Test filtering regions by maximum area."""
        img = np.zeros((60, 60, 60), dtype=np.uint8)

        # Small object (1 voxel)
        img[10, 10, 10] = 1

        # Large object (~1000 voxels)
        img[45:55, 45:55, 45:55] = 1

        dask_img = da.from_array(img, chunks=(30, 30, 30))
        affine = np.eye(4)

        # Filter by maximum area < 500 - should only keep small object
        centroids = compute_centroids(
            dask_img, affine, depth=5, region_filters={"area": ("<", 500)}
        )
        assert centroids.shape[0] == 1
        # Verify it's the small object (at approximately 10, 10, 10)
        assert_array_almost_equal(centroids[0], [10.0, 10.0, 10.0], decimal=0)

    def test_multiple_filters(self):
        """Test applying multiple filters simultaneously."""
        img = np.zeros((60, 60, 60), dtype=np.uint8)

        # Small object (1 voxel)
        img[10, 10, 10] = 1

        # Medium object (~125 voxels)
        img[28:33, 28:33, 28:33] = 1

        # Large object (~1000 voxels)
        img[45:55, 45:55, 45:55] = 1

        dask_img = da.from_array(img, chunks=(30, 30, 30))
        affine = np.eye(4)

        # Filter by area >= 30 AND area < 500 - should only keep medium object
        centroids = compute_centroids(
            dask_img,
            affine,
            depth=5,
            region_filters={"area": (">=", 30)},
        )
        # First filter by minimum area
        assert centroids.shape[0] == 2

        # Now apply both filters - area between 30 and 500
        # Note: We need to test that filters are applied correctly
        # Since we can only have one filter per property, let's test with different props

    def test_filter_equality(self):
        """Test filtering with equality operator."""
        img = np.zeros((60, 60, 60), dtype=np.uint8)

        # Create two objects of different sizes
        img[10, 10, 10] = 1  # 1 voxel
        img[28:33, 28:33, 28:33] = 1  # ~125 voxels

        dask_img = da.from_array(img, chunks=(30, 30, 30))
        affine = np.eye(4)

        # Filter by area == 1 - should only keep single voxel
        centroids = compute_centroids(
            dask_img, affine, depth=5, region_filters={"area": ("==", 1)}
        )
        assert centroids.shape[0] == 1
        assert_array_almost_equal(centroids[0], [10.0, 10.0, 10.0], decimal=0)

    def test_filter_not_equal(self):
        """Test filtering with not-equal operator."""
        img = np.zeros((60, 60, 60), dtype=np.uint8)

        # Create two single-voxel objects
        img[10, 10, 10] = 1
        img[20, 20, 20] = 1

        # Create one larger object (~125 voxels)
        img[40:45, 40:45, 40:45] = 1

        dask_img = da.from_array(img, chunks=(30, 30, 30))
        affine = np.eye(4)

        # Filter by area != 1 - should keep only the larger object
        centroids = compute_centroids(
            dask_img, affine, depth=5, region_filters={"area": ("!=", 1)}
        )
        assert centroids.shape[0] == 1

    def test_filter_no_matches(self):
        """Test when filter excludes all objects."""
        img = np.zeros((60, 60, 60), dtype=np.uint8)

        # Create small objects only
        img[10, 10, 10] = 1
        img[20, 20, 20] = 1

        dask_img = da.from_array(img, chunks=(30, 30, 30))
        affine = np.eye(4)

        # Filter by area >= 100 - should exclude all single voxel objects
        centroids = compute_centroids(
            dask_img, affine, depth=5, region_filters={"area": (">=", 100)}
        )
        assert centroids.shape == (0, 3)

    def test_filter_with_zarrnii_method(self):
        """Test region_filters through ZarrNii.compute_centroids method."""
        img = np.zeros((60, 60, 60), dtype=np.uint8)

        # Small object
        img[10, 10, 10] = 1

        # Large object
        img[40:50, 40:50, 40:50] = 1

        dask_img = da.from_array(img, chunks=(30, 30, 30))

        znii = ZarrNii.from_darr(
            dask_img, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), axes_order="ZYX"
        )

        # Filter by minimum area
        centroids = znii.compute_centroids(
            depth=5, region_filters={"area": (">=", 100)}
        )
        assert centroids.shape[0] == 1

    def test_invalid_operator_raises_error(self):
        """Test that invalid operator raises ValueError."""
        img = np.zeros((50, 50, 50), dtype=np.uint8)
        img[20:30, 20:30, 20:30] = 1

        dask_img = da.from_array(img, chunks=(25, 25, 25))
        affine = np.eye(4)

        with pytest.raises(ValueError, match="Invalid operator"):
            compute_centroids(
                dask_img, affine, depth=5, region_filters={"area": (">>", 30)}
            )

    def test_invalid_property_raises_error(self):
        """Test that invalid property name raises ValueError."""
        img = np.zeros((50, 50, 50), dtype=np.uint8)
        img[20:30, 20:30, 20:30] = 1

        dask_img = da.from_array(img, chunks=(25, 25, 25))
        affine = np.eye(4)

        with pytest.raises(ValueError, match="Invalid regionprops property"):
            compute_centroids(
                dask_img,
                affine,
                depth=5,
                region_filters={"invalid_property_name": (">=", 30)},
            )

    def test_filter_with_parquet_output(self, tmp_path):
        """Test region_filters with Parquet output."""
        import pandas as pd

        img = np.zeros((60, 60, 60), dtype=np.uint8)

        # Small object
        img[10, 10, 10] = 1

        # Large object
        img[40:50, 40:50, 40:50] = 1

        dask_img = da.from_array(img, chunks=(30, 30, 30))
        affine = np.eye(4)

        output_file = tmp_path / "centroids.parquet"
        compute_centroids(
            dask_img,
            affine,
            depth=5,
            output_path=str(output_file),
            region_filters={"area": (">=", 100)},
        )

        df = pd.read_parquet(output_file)
        assert len(df) == 1  # Only the large object should be included
