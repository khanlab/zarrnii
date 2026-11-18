"""
Tests for density_from_points function.

This module tests the density_from_points function to ensure it correctly
creates density maps from point coordinates with proper coordinate
transformations.
"""

import os
import tempfile

import dask.array as da
import numpy as np
import pytest

from zarrnii import ZarrNii, density_from_points


class TestDensityFromPointsBasic:
    """Test basic functionality of density_from_points."""

    def test_density_from_points_single_point(self):
        """Test density map creation with a single point."""
        # Create a reference image with known properties
        ref_data = da.zeros((1, 10, 10, 10), chunks=(1, 5, 5, 5))
        ref_img = ZarrNii.from_darr(
            darr=ref_data,
            axes_order="ZYX",
            orientation="RAS",
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )

        # Create a single point in physical space at (5.0, 5.0, 5.0)
        # which should map to voxel (5, 5, 5) in ZYX order
        points = np.array([[5.0, 5.0, 5.0]])  # [x, y, z]

        # Create density map
        density = density_from_points(points, ref_img, in_physical_space=True)

        # Check that density has correct shape
        assert density.shape == (1, 10, 10, 10)

        # Compute the density and check that point is at correct location
        density_computed = density.data.compute()

        # Point (5.0, 5.0, 5.0) in physical space maps to voxel (5, 5, 5)
        # In ZYX order with channel: (c=0, z=5, y=5, x=5)
        assert density_computed[0, 5, 5, 5] == 1.0

        # Check that all other voxels are zero
        assert density_computed.sum() == 1.0

    def test_density_from_points_multiple_points_same_voxel(self):
        """Test that multiple points in same voxel accumulate."""
        # Create reference image
        ref_data = da.zeros((1, 10, 10, 10), chunks=(1, 5, 5, 5))
        ref_img = ZarrNii.from_darr(
            darr=ref_data,
            axes_order="ZYX",
            orientation="RAS",
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )

        # Create three points that all fall in the same voxel
        # Physical coords (x, y, z) = (2.1, 3.2, 4.3) etc.
        # Maps to voxel coords (x, y, z) = (2, 3, 4) after flooring
        points = np.array(
            [
                [2.1, 3.2, 4.3],  # All should map to voxel (x=2, y=3, z=4)
                [2.5, 3.5, 4.5],
                [2.9, 3.9, 4.9],
            ]
        )

        # Create density map
        density = density_from_points(points, ref_img, in_physical_space=True)

        # Compute the density
        density_computed = density.data.compute()

        # All three points should accumulate in one voxel
        # Voxel (x=2, y=3, z=4) in ZYX storage order: (c=0, z=4, y=3, x=2)
        assert density_computed[0, 4, 3, 2] == 3.0
        assert density_computed.sum() == 3.0

    def test_density_from_points_multiple_scattered_points(self):
        """Test density map with points scattered across multiple voxels."""
        # Create reference image
        ref_data = da.zeros((1, 20, 20, 20), chunks=(1, 10, 10, 10))
        ref_img = ZarrNii.from_darr(
            darr=ref_data,
            axes_order="ZYX",
            orientation="RAS",
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )

        # Create points at distinct locations
        points = np.array(
            [
                [5.0, 5.0, 5.0],  # Voxel (z=5, y=5, x=5)
                [10.0, 10.0, 10.0],  # Voxel (z=10, y=10, x=10)
                [15.0, 15.0, 15.0],  # Voxel (z=15, y=15, x=15)
            ]
        )

        # Create density map
        density = density_from_points(points, ref_img, in_physical_space=True)

        # Compute the density
        density_computed = density.data.compute()

        # Check that each point is in correct location
        assert density_computed[0, 5, 5, 5] == 1.0
        assert density_computed[0, 10, 10, 10] == 1.0
        assert density_computed[0, 15, 15, 15] == 1.0
        assert density_computed.sum() == 3.0


class TestDensityFromPointsWithTransform:
    """Test density_from_points with affine transformations."""

    def test_density_with_scaled_affine(self):
        """Test density map with non-unit spacing."""
        # Create reference image with 2mm spacing
        ref_data = da.zeros((1, 10, 10, 10), chunks=(1, 5, 5, 5))
        ref_img = ZarrNii.from_darr(
            darr=ref_data,
            axes_order="ZYX",
            orientation="RAS",
            spacing=(2.0, 2.0, 2.0),  # 2mm spacing
            origin=(0.0, 0.0, 0.0),
        )

        # Create a point at physical coordinate (10.0, 10.0, 10.0)
        # With 2mm spacing, this should map to voxel (5, 5, 5)
        points = np.array([[10.0, 10.0, 10.0]])

        # Create density map
        density = density_from_points(points, ref_img, in_physical_space=True)

        # Compute the density
        density_computed = density.data.compute()

        # Point should be at voxel (z=5, y=5, x=5)
        assert density_computed[0, 5, 5, 5] == 1.0
        assert density_computed.sum() == 1.0

    def test_density_with_origin_offset(self):
        """Test density map with non-zero origin."""
        # Create reference image with origin at (10, 10, 10)
        ref_data = da.zeros((1, 10, 10, 10), chunks=(1, 5, 5, 5))
        ref_img = ZarrNii.from_darr(
            darr=ref_data,
            axes_order="ZYX",
            orientation="RAS",
            spacing=(1.0, 1.0, 1.0),
            origin=(10.0, 10.0, 10.0),  # Origin offset
        )

        # Create a point at physical coordinate (15.0, 15.0, 15.0)
        # With origin at (10, 10, 10) and unit spacing, this maps to voxel (5, 5, 5)
        points = np.array([[15.0, 15.0, 15.0]])

        # Create density map
        density = density_from_points(points, ref_img, in_physical_space=True)

        # Compute the density
        density_computed = density.data.compute()

        # Point should be at voxel (z=5, y=5, x=5)
        assert density_computed[0, 5, 5, 5] == 1.0
        assert density_computed.sum() == 1.0

    def test_density_with_scaling_and_offset(self):
        """Test density map with both scaling and origin offset."""
        # Create reference image with 2mm spacing and origin at (10, 10, 10)
        ref_data = da.zeros((1, 10, 10, 10), chunks=(1, 5, 5, 5))
        ref_img = ZarrNii.from_darr(
            darr=ref_data,
            axes_order="ZYX",
            orientation="RAS",
            spacing=(2.0, 2.0, 2.0),
            origin=(10.0, 10.0, 10.0),
        )

        # Physical coordinate (20.0, 20.0, 20.0) with origin (10, 10, 10) and
        # spacing 2mm maps to voxel (5, 5, 5)
        points = np.array([[20.0, 20.0, 20.0]])

        # Create density map
        density = density_from_points(points, ref_img, in_physical_space=True)

        # Compute the density
        density_computed = density.data.compute()

        # Point should be at voxel (z=5, y=5, x=5)
        assert density_computed[0, 5, 5, 5] == 1.0
        assert density_computed.sum() == 1.0


class TestDensityFromPointsVoxelSpace:
    """Test density_from_points with points already in voxel space."""

    def test_density_from_voxel_coords(self):
        """Test density map creation from voxel coordinates."""
        # Create reference image
        ref_data = da.zeros((1, 10, 10, 10), chunks=(1, 5, 5, 5))
        ref_img = ZarrNii.from_darr(
            darr=ref_data,
            axes_order="ZYX",
            orientation="RAS",
            spacing=(2.0, 2.0, 2.0),  # Non-unit spacing
            origin=(10.0, 10.0, 10.0),  # Non-zero origin
        )

        # Provide points directly in voxel coordinates
        # Point (5, 5, 5) in voxel space
        voxel_points = np.array([[5.0, 5.0, 5.0]])

        # Create density map with in_physical_space=False
        density = density_from_points(voxel_points, ref_img, in_physical_space=False)

        # Compute the density
        density_computed = density.data.compute()

        # Point should be at voxel (z=5, y=5, x=5)
        assert density_computed[0, 5, 5, 5] == 1.0
        assert density_computed.sum() == 1.0


class TestDensityFromPointsFileInput:
    """Test density_from_points with file inputs."""

    def test_density_from_npy_file(self):
        """Test loading points from .npy file."""
        # Create temporary npy file
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            npy_path = f.name
            points = np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
            np.save(f, points)

        try:
            # Create reference image
            ref_data = da.zeros((1, 10, 10, 10), chunks=(1, 5, 5, 5))
            ref_img = ZarrNii.from_darr(
                darr=ref_data,
                axes_order="ZYX",
                orientation="RAS",
                spacing=(1.0, 1.0, 1.0),
                origin=(0.0, 0.0, 0.0),
            )

            # Create density map from file
            density = density_from_points(npy_path, ref_img, in_physical_space=True)

            # Compute and check
            density_computed = density.data.compute()
            assert density_computed[0, 5, 5, 5] == 1.0
            assert density_computed[0, 6, 6, 6] == 1.0
            assert density_computed.sum() == 2.0
        finally:
            # Clean up
            os.unlink(npy_path)

    def test_density_from_parquet_file(self):
        """Test loading points from .parquet file."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")

        # Create temporary parquet file
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            # Create and save points as parquet
            df = pd.DataFrame({"x": [5.0, 6.0], "y": [5.0, 6.0], "z": [5.0, 6.0]})
            df.to_parquet(parquet_path)

            # Create reference image
            ref_data = da.zeros((1, 10, 10, 10), chunks=(1, 5, 5, 5))
            ref_img = ZarrNii.from_darr(
                darr=ref_data,
                axes_order="ZYX",
                orientation="RAS",
                spacing=(1.0, 1.0, 1.0),
                origin=(0.0, 0.0, 0.0),
            )

            # Create density map from file
            density = density_from_points(parquet_path, ref_img, in_physical_space=True)

            # Compute and check
            density_computed = density.data.compute()
            assert density_computed[0, 5, 5, 5] == 1.0
            assert density_computed[0, 6, 6, 6] == 1.0
            assert density_computed.sum() == 2.0
        finally:
            # Clean up
            os.unlink(parquet_path)


class TestDensityFromPointsEdgeCases:
    """Test edge cases and error handling."""

    def test_density_with_empty_points(self):
        """Test density map with no points."""
        # Create reference image
        ref_data = da.zeros((1, 10, 10, 10), chunks=(1, 5, 5, 5))
        ref_img = ZarrNii.from_darr(
            darr=ref_data,
            axes_order="ZYX",
            orientation="RAS",
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )

        # Empty points array
        points = np.empty((0, 3))

        # Create density map
        density = density_from_points(points, ref_img, in_physical_space=True)

        # Compute the density
        density_computed = density.data.compute()

        # All voxels should be zero
        assert density_computed.sum() == 0.0

    def test_density_with_points_outside_bounds(self):
        """Test that points outside image bounds are ignored."""
        # Create reference image
        ref_data = da.zeros((1, 10, 10, 10), chunks=(1, 5, 5, 5))
        ref_img = ZarrNii.from_darr(
            darr=ref_data,
            axes_order="ZYX",
            orientation="RAS",
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )

        # Create points, some inside and some outside bounds
        points = np.array(
            [
                [5.0, 5.0, 5.0],  # Inside
                [-1.0, -1.0, -1.0],  # Outside (negative)
                [20.0, 20.0, 20.0],  # Outside (too large)
            ]
        )

        # Create density map
        density = density_from_points(points, ref_img, in_physical_space=True)

        # Compute the density
        density_computed = density.data.compute()

        # Only the point at (5, 5, 5) should be counted
        assert density_computed[0, 5, 5, 5] == 1.0
        assert density_computed.sum() == 1.0

    def test_density_with_invalid_points_shape(self):
        """Test that invalid points shape raises error."""
        # Create reference image
        ref_data = da.zeros((1, 10, 10, 10), chunks=(1, 5, 5, 5))
        ref_img = ZarrNii.from_darr(
            darr=ref_data,
            axes_order="ZYX",
            orientation="RAS",
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )

        # Invalid shape: (N, 2) instead of (N, 3)
        invalid_points = np.array([[5.0, 5.0], [6.0, 6.0]])

        # Should raise ValueError
        with pytest.raises(ValueError, match="Points must have shape \\(N, 3\\)"):
            density_from_points(invalid_points, ref_img, in_physical_space=True)


class TestDensityFromPointsAxesOrder:
    """Test density_from_points with different axes orders."""

    def test_density_with_xyz_axes_order(self):
        """Test density map with XYZ axes order."""
        # Create reference image with XYZ order
        ref_data = da.zeros((1, 10, 10, 10), chunks=(1, 5, 5, 5))
        ref_img = ZarrNii.from_darr(
            darr=ref_data,
            axes_order="XYZ",
            orientation="RAS",
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )

        # Create a point at (5.0, 5.0, 5.0)
        points = np.array([[5.0, 5.0, 5.0]])

        # Create density map
        density = density_from_points(points, ref_img, in_physical_space=True)

        # Check axes order is preserved
        assert density.axes_order == "XYZ"

        # Compute the density
        density_computed = density.data.compute()

        # Point should be at correct location
        # With XYZ order: (c=0, x=5, y=5, z=5)
        assert density_computed[0, 5, 5, 5] == 1.0
        assert density_computed.sum() == 1.0


class TestDensityFromPointsMetadata:
    """Test that metadata is properly preserved."""

    def test_density_preserves_spacing(self):
        """Test that spacing is preserved in density map."""
        # Create reference image with non-unit spacing
        ref_data = da.zeros((1, 10, 10, 10), chunks=(1, 5, 5, 5))
        ref_img = ZarrNii.from_darr(
            darr=ref_data,
            axes_order="ZYX",
            orientation="RAS",
            spacing=(2.0, 1.5, 1.0),
            origin=(0.0, 0.0, 0.0),
        )

        # Create density map
        points = np.array([[5.0, 5.0, 5.0]])
        density = density_from_points(points, ref_img, in_physical_space=True)

        # Check that spacing is preserved
        assert density.scale["z"] == 2.0
        assert density.scale["y"] == 1.5
        assert density.scale["x"] == 1.0

    def test_density_preserves_origin(self):
        """Test that origin is preserved in density map."""
        # Create reference image with non-zero origin
        ref_data = da.zeros((1, 10, 10, 10), chunks=(1, 5, 5, 5))
        ref_img = ZarrNii.from_darr(
            darr=ref_data,
            axes_order="ZYX",
            orientation="RAS",
            spacing=(1.0, 1.0, 1.0),
            origin=(10.0, 20.0, 30.0),
        )

        # Create density map
        points = np.array([[15.0, 25.0, 35.0]])
        density = density_from_points(points, ref_img, in_physical_space=True)

        # Check that origin is preserved
        assert density.translation["z"] == 10.0
        assert density.translation["y"] == 20.0
        assert density.translation["x"] == 30.0

    def test_density_preserves_orientation(self):
        """Test that orientation is preserved in density map."""
        # Create reference image with specific orientation
        ref_data = da.zeros((1, 10, 10, 10), chunks=(1, 5, 5, 5))
        ref_img = ZarrNii.from_darr(
            darr=ref_data,
            axes_order="ZYX",
            orientation="LPI",
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )

        # Create density map
        points = np.array([[5.0, 5.0, 5.0]])
        density = density_from_points(points, ref_img, in_physical_space=True)

        # Check that orientation is preserved
        assert density.xyz_orientation == "LPI"


class TestDensityFromPointsWeights:
    """Test density_from_points with weights parameter."""

    def test_density_with_uniform_weights(self):
        """Test density map with uniform weights (should equal count * weight)."""
        # Create reference image
        ref_data = da.zeros((1, 10, 10, 10), chunks=(1, 5, 5, 5))
        ref_img = ZarrNii.from_darr(
            darr=ref_data,
            axes_order="ZYX",
            orientation="RAS",
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )

        # Create points at a single location
        points = np.array([[5.0, 5.0, 5.0]])
        weights = np.array([3.0])  # Weight of 3

        # Create density map with weights
        density = density_from_points(
            points, ref_img, in_physical_space=True, weights=weights
        )

        # Compute the density
        density_computed = density.data.compute()

        # Weighted density should be 3.0 at (5, 5, 5)
        assert density_computed[0, 5, 5, 5] == 3.0
        assert density_computed.sum() == 3.0

    def test_density_with_varying_weights(self):
        """Test density map with varying weights for different points."""
        # Create reference image
        ref_data = da.zeros((1, 10, 10, 10), chunks=(1, 5, 5, 5))
        ref_img = ZarrNii.from_darr(
            darr=ref_data,
            axes_order="ZYX",
            orientation="RAS",
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )

        # Create points at different locations with different weights
        points = np.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0], [7.0, 7.0, 7.0]])
        weights = np.array([1.0, 5.0, 10.0])  # Different weights

        # Create density map with weights
        density = density_from_points(
            points, ref_img, in_physical_space=True, weights=weights
        )

        # Compute the density
        density_computed = density.data.compute()

        # Check weighted values at each location
        assert density_computed[0, 2, 2, 2] == 1.0
        assert density_computed[0, 5, 5, 5] == 5.0
        assert density_computed[0, 7, 7, 7] == 10.0
        assert density_computed.sum() == 16.0  # 1 + 5 + 10

    def test_density_with_weights_same_voxel(self):
        """Test that weights accumulate for points in the same voxel."""
        # Create reference image
        ref_data = da.zeros((1, 10, 10, 10), chunks=(1, 5, 5, 5))
        ref_img = ZarrNii.from_darr(
            darr=ref_data,
            axes_order="ZYX",
            orientation="RAS",
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )

        # Create three points that fall in the same voxel with different weights
        points = np.array(
            [
                [5.1, 5.1, 5.1],  # All fall in voxel (5, 5, 5)
                [5.5, 5.5, 5.5],
                [5.9, 5.9, 5.9],
            ]
        )
        weights = np.array([10.0, 20.0, 30.0])  # Total should be 60

        # Create density map with weights
        density = density_from_points(
            points, ref_img, in_physical_space=True, weights=weights
        )

        # Compute the density
        density_computed = density.data.compute()

        # All weights should accumulate in voxel (5, 5, 5)
        assert density_computed[0, 5, 5, 5] == 60.0
        assert density_computed.sum() == 60.0

    def test_density_with_weights_voxel_space(self):
        """Test weighted density with points in voxel space."""
        # Create reference image
        ref_data = da.zeros((1, 10, 10, 10), chunks=(1, 5, 5, 5))
        ref_img = ZarrNii.from_darr(
            darr=ref_data,
            axes_order="ZYX",
            orientation="RAS",
            spacing=(2.0, 2.0, 2.0),  # Non-unit spacing
            origin=(10.0, 10.0, 10.0),  # Non-zero origin
        )

        # Points directly in voxel coordinates
        voxel_points = np.array([[3.0, 3.0, 3.0], [6.0, 6.0, 6.0]])
        weights = np.array([100.0, 250.0])

        # Create density map with in_physical_space=False
        density = density_from_points(
            voxel_points, ref_img, in_physical_space=False, weights=weights
        )

        # Compute the density
        density_computed = density.data.compute()

        # Check weighted values
        assert density_computed[0, 3, 3, 3] == 100.0
        assert density_computed[0, 6, 6, 6] == 250.0
        assert density_computed.sum() == 350.0

    def test_density_invalid_weights_shape(self):
        """Test that invalid weights shape raises error."""
        # Create reference image
        ref_data = da.zeros((1, 10, 10, 10), chunks=(1, 5, 5, 5))
        ref_img = ZarrNii.from_darr(
            darr=ref_data,
            axes_order="ZYX",
            orientation="RAS",
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )

        # Three points
        points = np.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0], [7.0, 7.0, 7.0]])

        # Wrong number of weights (2 instead of 3)
        invalid_weights = np.array([1.0, 2.0])

        # Should raise ValueError
        with pytest.raises(ValueError, match="Weights must have shape"):
            density_from_points(
                points, ref_img, in_physical_space=True, weights=invalid_weights
            )

    def test_density_invalid_weights_ndim(self):
        """Test that 2D weights array raises error."""
        # Create reference image
        ref_data = da.zeros((1, 10, 10, 10), chunks=(1, 5, 5, 5))
        ref_img = ZarrNii.from_darr(
            darr=ref_data,
            axes_order="ZYX",
            orientation="RAS",
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )

        # Three points
        points = np.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0], [7.0, 7.0, 7.0]])

        # 2D weights (invalid)
        invalid_weights = np.array([[1.0, 2.0, 3.0]])

        # Should raise ValueError
        with pytest.raises(ValueError, match="Weights must have shape"):
            density_from_points(
                points, ref_img, in_physical_space=True, weights=invalid_weights
            )

    def test_density_with_empty_points_and_weights(self):
        """Test density map with empty points and weights."""
        # Create reference image
        ref_data = da.zeros((1, 10, 10, 10), chunks=(1, 5, 5, 5))
        ref_img = ZarrNii.from_darr(
            darr=ref_data,
            axes_order="ZYX",
            orientation="RAS",
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )

        # Empty points and weights
        points = np.empty((0, 3))
        weights = np.empty((0,))

        # Create density map
        density = density_from_points(
            points, ref_img, in_physical_space=True, weights=weights
        )

        # Compute the density
        density_computed = density.data.compute()

        # All voxels should be zero
        assert density_computed.sum() == 0.0
