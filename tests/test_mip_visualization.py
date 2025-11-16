"""
Tests for MIP (Maximum Intensity Projection) visualization functions.

This module tests the MIP visualization functions to ensure they work
correctly with different planes, slab configurations, and channel colors.
"""

import dask.array as da
import ngff_zarr as nz
import numpy as np
import pytest

from zarrnii import ZarrNii
from zarrnii.analysis import create_mip_visualization


class TestStandaloneMIPFunction:
    """Test the standalone create_mip_visualization function."""

    def test_basic_mip_axial(self):
        """Test basic MIP creation in axial plane."""
        # Create test data with 1 channel
        np.random.seed(42)
        data = da.from_array(np.random.random((1, 50, 40, 30)), chunks=(1, 25, 20, 15))
        dims = ["c", "z", "y", "x"]
        scale = {"z": 2.0, "y": 1.0, "x": 1.0}

        # Create MIPs with 20 micron slabs
        mips = create_mip_visualization(
            data,
            dims,
            scale,
            plane="axial",
            slab_thickness_um=20.0,
            slab_spacing_um=20.0,
            channel_colors=["red"],
        )

        # Should create multiple slabs
        assert len(mips) > 0
        # Each MIP should be 2D RGB image
        assert mips[0].shape == (40, 30, 3)
        # Values should be in [0, 1] range
        assert 0.0 <= mips[0].min() <= mips[0].max() <= 1.0

    def test_mip_coronal_plane(self):
        """Test MIP creation in coronal plane."""
        data = da.random.random((1, 30, 40, 50), chunks=(1, 15, 20, 25))
        dims = ["c", "z", "y", "x"]
        scale = {"z": 1.0, "y": 1.0, "x": 1.0}

        mips = create_mip_visualization(
            data, dims, scale, plane="coronal", slab_thickness_um=10.0
        )

        assert len(mips) > 0
        # Coronal projects along y, shows x-z
        assert mips[0].shape == (30, 50, 3)

    def test_mip_sagittal_plane(self):
        """Test MIP creation in sagittal plane."""
        data = da.random.random((1, 30, 40, 50), chunks=(1, 15, 20, 25))
        dims = ["c", "z", "y", "x"]
        scale = {"z": 1.0, "y": 1.0, "x": 1.0}

        mips = create_mip_visualization(
            data, dims, scale, plane="sagittal", slab_thickness_um=10.0
        )

        assert len(mips) > 0
        # Sagittal projects along x, shows y-z
        assert mips[0].shape == (30, 40, 3)

    def test_mip_multi_channel(self):
        """Test MIP with multiple channels."""
        # Create 3-channel data
        data = da.random.random((3, 20, 30, 40), chunks=(1, 10, 15, 20))
        dims = ["c", "z", "y", "x"]
        scale = {"z": 1.0, "y": 1.0, "x": 1.0}

        mips = create_mip_visualization(
            data,
            dims,
            scale,
            plane="axial",
            slab_thickness_um=10.0,
            channel_colors=["red", "green", "blue"],
        )

        assert len(mips) > 0
        assert mips[0].shape == (30, 40, 3)

    def test_mip_single_channel_no_c_dim(self):
        """Test MIP with single channel data (no channel dimension)."""
        # Data without channel dimension
        data = da.random.random((20, 30, 40), chunks=(10, 15, 20))
        dims = ["z", "y", "x"]
        scale = {"z": 1.0, "y": 1.0, "x": 1.0}

        mips = create_mip_visualization(
            data, dims, scale, plane="axial", slab_thickness_um=10.0
        )

        assert len(mips) > 0
        assert mips[0].shape == (30, 40, 3)

    def test_mip_with_slab_info(self):
        """Test MIP with slab metadata return."""
        data = da.random.random((1, 50, 30, 40), chunks=(1, 25, 15, 20))
        dims = ["c", "z", "y", "x"]
        scale = {"z": 2.0, "y": 1.0, "x": 1.0}

        mips, slab_info = create_mip_visualization(
            data,
            dims,
            scale,
            plane="axial",
            slab_thickness_um=20.0,
            slab_spacing_um=20.0,
            return_slabs=True,
        )

        assert len(mips) == len(slab_info)
        # Check slab info structure
        assert "start_um" in slab_info[0]
        assert "end_um" in slab_info[0]
        assert "center_um" in slab_info[0]
        assert "start_idx" in slab_info[0]
        assert "end_idx" in slab_info[0]
        # Start should be less than end
        assert slab_info[0]["start_um"] < slab_info[0]["end_um"]

    def test_mip_custom_colors_rgb_tuple(self):
        """Test MIP with custom RGB tuple colors."""
        data = da.random.random((2, 20, 30, 40), chunks=(1, 10, 15, 20))
        dims = ["c", "z", "y", "x"]
        scale = {"z": 1.0, "y": 1.0, "x": 1.0}

        # Use RGB tuples
        mips = create_mip_visualization(
            data,
            dims,
            scale,
            plane="axial",
            slab_thickness_um=10.0,
            channel_colors=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        )

        assert len(mips) > 0
        assert mips[0].shape == (30, 40, 3)

    def test_mip_invalid_plane(self):
        """Test error handling for invalid plane."""
        data = da.random.random((1, 20, 30, 40), chunks=(1, 10, 15, 20))
        dims = ["c", "z", "y", "x"]
        scale = {"z": 1.0, "y": 1.0, "x": 1.0}

        with pytest.raises(ValueError, match="plane must be one of"):
            create_mip_visualization(data, dims, scale, plane="invalid")

    def test_mip_missing_projection_axis(self):
        """Test error when projection axis not in dims."""
        data = da.random.random((1, 20, 30), chunks=(1, 10, 15))
        dims = ["c", "y", "x"]  # Missing z
        scale = {"y": 1.0, "x": 1.0}

        with pytest.raises(ValueError, match="not found in dims"):
            create_mip_visualization(data, dims, scale, plane="axial")

    def test_mip_too_few_colors(self):
        """Test error when not enough colors provided."""
        data = da.random.random((3, 20, 30, 40), chunks=(1, 10, 15, 20))
        dims = ["c", "z", "y", "x"]
        scale = {"z": 1.0, "y": 1.0, "x": 1.0}

        with pytest.raises(ValueError, match="Provided .* colors but image has"):
            create_mip_visualization(
                data, dims, scale, plane="axial", channel_colors=["red", "green"]
            )

    def test_mip_large_slab_thickness(self):
        """Test MIP with slab thickness larger than volume."""
        data = da.random.random((1, 10, 30, 40), chunks=(1, 5, 15, 20))
        dims = ["c", "z", "y", "x"]
        scale = {"z": 1.0, "y": 1.0, "x": 1.0}

        # Slab thickness larger than volume should still work
        mips = create_mip_visualization(
            data,
            dims,
            scale,
            plane="axial",
            slab_thickness_um=100.0,
            slab_spacing_um=100.0,
        )

        # Should create at least one slab
        assert len(mips) >= 1

    def test_mip_non_uniform_spacing(self):
        """Test MIP with non-uniform voxel spacing."""
        data = da.random.random((1, 50, 30, 40), chunks=(1, 25, 15, 20))
        dims = ["c", "z", "y", "x"]
        scale = {"z": 5.0, "y": 1.0, "x": 1.5}  # Different spacings

        mips, slab_info = create_mip_visualization(
            data,
            dims,
            scale,
            plane="axial",
            slab_thickness_um=50.0,
            slab_spacing_um=50.0,
            return_slabs=True,
        )

        assert len(mips) > 0
        # With 5 micron z-spacing, 50 micron slab is 10 slices
        # Volume is 50 slices * 5 = 250 microns
        # Expect several slabs
        assert len(mips) >= 3


class TestZarrNiiMIPMethod:
    """Test the ZarrNii create_mip method."""

    def setup_method(self):
        """Set up test data."""
        # Create test data with 2 channels
        np.random.seed(123)
        data = da.random.random((2, 40, 50, 60), chunks=(1, 20, 25, 30))
        dims = ["c", "z", "y", "x"]
        scale = {"z": 2.0, "y": 1.0, "x": 1.0}
        translation = {"z": 0.0, "y": 0.0, "x": 0.0}

        ngff_image = nz.NgffImage(
            data=data, dims=dims, scale=scale, translation=translation, name="test"
        )

        self.znimg = ZarrNii(ngff_image=ngff_image, axes_order="ZYX", orientation="RAS")

    def test_create_mip_method(self):
        """Test ZarrNii create_mip method."""
        mips = self.znimg.create_mip(
            plane="axial",
            slab_thickness_um=40.0,
            slab_spacing_um=40.0,
            channel_colors=["red", "green"],
        )

        assert len(mips) > 0
        assert mips[0].shape == (50, 60, 3)
        assert 0.0 <= mips[0].min() <= mips[0].max() <= 1.0

    def test_create_mip_all_planes(self):
        """Test create_mip with all three planes."""
        for plane in ["axial", "coronal", "sagittal"]:
            mips = self.znimg.create_mip(
                plane=plane,
                slab_thickness_um=20.0,
                channel_colors=["red", "green"],
            )
            assert len(mips) > 0

    def test_create_mip_with_metadata(self):
        """Test create_mip with slab metadata."""
        mips, slab_info = self.znimg.create_mip(
            plane="axial",
            slab_thickness_um=40.0,
            slab_spacing_um=40.0,
            return_slabs=True,
        )

        assert len(mips) == len(slab_info)
        for info in slab_info:
            assert "start_um" in info
            assert "center_um" in info
            assert "end_um" in info

    def test_create_mip_default_colors(self):
        """Test create_mip with default colors."""
        # Should work without specifying colors
        mips = self.znimg.create_mip(plane="axial", slab_thickness_um=40.0)

        assert len(mips) > 0
        assert mips[0].shape[-1] == 3  # RGB


class TestMIPVisualization:
    """Test MIP visualization quality and correctness."""

    def test_mip_max_projection_correctness(self):
        """Test that MIP actually computes maximum intensity."""
        # Create data with known maximum values
        data_array = np.zeros((1, 10, 20, 20))
        # Put a bright spot in the middle slices
        data_array[0, 4:6, 10, 10] = 1.0
        data = da.from_array(data_array, chunks=(1, 5, 10, 10))
        dims = ["c", "z", "y", "x"]
        scale = {"z": 1.0, "y": 1.0, "x": 1.0}

        # Create MIP with slab that includes the bright spot
        mips = create_mip_visualization(
            data,
            dims,
            scale,
            plane="axial",
            slab_thickness_um=10.0,  # Covers all z
            slab_spacing_um=20.0,
            channel_colors=["white"],
        )

        # The bright spot should be visible in the MIP at position (10, 10)
        # Since it's white color and normalized, all RGB channels should be bright
        assert len(mips) == 1
        center_pixel = mips[0][10, 10, :]
        # Should be brighter than background
        assert center_pixel.mean() > 0.5

    def test_mip_color_blending(self):
        """Test that multi-channel MIPs blend colors correctly."""
        # Create 2-channel data with different intensities
        data_array = np.zeros((2, 10, 20, 20))
        # Channel 0: bright in top-left
        data_array[0, :, :10, :10] = 1.0
        # Channel 1: bright in bottom-right
        data_array[1, :, 10:, 10:] = 1.0
        data = da.from_array(data_array, chunks=(1, 5, 10, 10))
        dims = ["c", "z", "y", "x"]
        scale = {"z": 1.0, "y": 1.0, "x": 1.0}

        mips = create_mip_visualization(
            data,
            dims,
            scale,
            plane="axial",
            slab_thickness_um=10.0,
            channel_colors=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],  # Red, Green
        )

        assert len(mips) == 1
        mip = mips[0]

        # Top-left should be red (channel 0)
        top_left = mip[5, 5, :]
        assert top_left[0] > 0.5  # Red channel
        assert top_left[1] < 0.2  # Green channel minimal

        # Bottom-right should be green (channel 1)
        bottom_right = mip[15, 15, :]
        assert bottom_right[0] < 0.2  # Red channel minimal
        assert bottom_right[1] > 0.5  # Green channel

    def test_mip_slab_spacing(self):
        """Test that slab spacing creates expected number of slabs."""
        data = da.random.random((1, 100, 30, 40), chunks=(1, 50, 15, 20))
        dims = ["c", "z", "y", "x"]
        scale = {"z": 1.0, "y": 1.0, "x": 1.0}

        # With 100 micron volume, 20 micron spacing should give ~5 slabs
        mips, slab_info = create_mip_visualization(
            data,
            dims,
            scale,
            plane="axial",
            slab_thickness_um=10.0,
            slab_spacing_um=20.0,
            return_slabs=True,
        )

        # Should have 5-6 slabs
        assert 4 <= len(mips) <= 6

        # Slab centers should be roughly 20 microns apart
        if len(slab_info) >= 2:
            spacing = slab_info[1]["center_um"] - slab_info[0]["center_um"]
            assert 18.0 <= spacing <= 22.0  # Allow some tolerance
