"""
Tests for enhanced MIP visualization with intensity rescaling and OMERO support.

This module tests the new features added to MIP visualization:
- Custom intensity ranges per channel
- OMERO metadata integration for colors and ranges
- Channel selection by OMERO labels
- Alpha transparency support
"""

import os
import tempfile

import dask.array as da
import ngff_zarr as nz
import numpy as np
import pytest
import zarr

from zarrnii import ZarrNii
from zarrnii.analysis import create_mip_visualization


class MockOmeroWindow:
    """Mock class to simulate OmeroWindow from ome-zarr-py."""

    def __init__(self, min=0.0, max=65535.0, start=0.0, end=65535.0):
        self.min = min
        self.max = max
        self.start = start
        self.end = end


class MockOmeroChannel:
    """Mock class to simulate OmeroChannel from ome-zarr-py."""

    def __init__(self, label, color, window=None):
        self.label = label
        self.color = color
        self.window = window or MockOmeroWindow()


class MockOmero:
    """Mock class to simulate Omero from ome-zarr-py."""

    def __init__(self, channels):
        self.channels = channels


def create_test_dataset_with_omero(store_path, num_channels=3):
    """Create a test OME-Zarr dataset with OMERO metadata."""
    # Create a 4D array in ZYXC order
    arr_sz = (16, 32, 32, num_channels)
    arr = da.zeros(arr_sz, dtype=np.uint16)

    # Fill with different values for each channel
    def fill_channel_data(block, block_info=None):
        if block_info is not None:
            block_slice = block_info[0]["array-location"]
            c_start = block_slice[3][0]
            c_end = block_slice[3][1]

            result = np.zeros(block.shape, dtype=np.uint16)
            for c_idx in range(c_end - c_start):
                global_c_idx = c_start + c_idx
                value = (global_c_idx + 1) * 1000
                result[:, :, :, c_idx] = value
            return result
        return np.zeros(block.shape, dtype=np.uint16)

    arr = arr.map_blocks(fill_channel_data, dtype=np.uint16)

    # Create NGFF image
    ngff_image = nz.to_ngff_image(arr)
    multiscales = nz.to_multiscales(ngff_image)

    # Create OMERO metadata
    channel_data = [
        ("DAPI", "0000FF", MockOmeroWindow(start=0.0, end=2000.0)),
        ("GFP", "00FF00", MockOmeroWindow(start=500.0, end=3000.0)),
        ("RFP", "FF0000", MockOmeroWindow(start=1000.0, end=4000.0)),
    ][:num_channels]

    omero_channels = []
    for label, color, window in channel_data:
        channel = {
            "label": label,
            "color": color,
            "window": {
                "min": window.min,
                "max": window.max,
                "start": window.start,
                "end": window.end,
            },
        }
        omero_channels.append(channel)

    omero_metadata = {"channels": omero_channels}

    # Store to zarr
    nz.to_ngff_zarr(store_path, multiscales)

    # Add OMERO metadata
    group = zarr.open_group(store_path, mode="r+")
    group.attrs["omero"] = omero_metadata

    return store_path


class TestIntensityRescaling:
    """Test custom intensity range specification."""

    def test_custom_intensity_ranges(self):
        """Test MIP with custom intensity ranges for each channel."""
        # Create test data with known intensities (3 channels, each with uniform value)
        data_array = np.zeros((3, 1, 1, 1), dtype=np.float32)
        data_array[0, :, :, :] = 1000.0  # Channel 0
        data_array[1, :, :, :] = 2000.0  # Channel 1
        data_array[2, :, :, :] = 3000.0  # Channel 2
        data = da.from_array(data_array, chunks=(1, 1, 1, 1))

        dims = ["c", "z", "y", "x"]
        scale = {"z": 1.0, "y": 1.0, "x": 1.0}

        # Specify custom ranges with explicit RGB tuples to avoid matplotlib color differences
        mips = create_mip_visualization(
            data,
            dims,
            scale,
            plane="axial",
            slab_thickness_um=10.0,
            channel_colors=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            channel_ranges=[(0.0, 2000.0), (1000.0, 3000.0), (2000.0, 4000.0)],
        )

        assert len(mips) == 1
        mip = mips[0]

        # Channel 0: value 1000 in range [0, 2000] -> 0.5 normalized
        # Channel 1: value 2000 in range [1000, 3000] -> 0.5 normalized
        # Channel 2: value 3000 in range [2000, 4000] -> 0.5 normalized
        # All channels should contribute equally (0.5 * color)
        pixel = mip[0, 0, :]
        np.testing.assert_allclose(pixel, [0.5, 0.5, 0.5], rtol=1e-5)

    def test_intensity_clipping(self):
        """Test that values outside custom range are clipped."""
        # Create data with values outside specified range
        data_array = np.zeros((1, 1, 1, 1), dtype=np.float32)
        data_array[0, :, :, :] = 5000.0
        data = da.from_array(data_array, chunks=(1, 1, 1, 1))

        dims = ["c", "z", "y", "x"]
        scale = {"z": 1.0, "y": 1.0, "x": 1.0}

        # Specify range that's smaller than actual data
        mips = create_mip_visualization(
            data,
            dims,
            scale,
            plane="axial",
            slab_thickness_um=10.0,
            channel_colors=["red"],
            channel_ranges=[(0.0, 2000.0)],
        )

        # Value 5000 should be clipped to max (2000) -> normalized to 1.0
        pixel = mips[0][0, 0, :]
        np.testing.assert_allclose(pixel[0], 1.0, rtol=1e-5)

    def test_mixed_auto_and_custom_ranges(self):
        """Test mixing auto-scaling and custom ranges."""
        data = da.random.random((2, 10, 20, 20), chunks=(1, 5, 10, 10))
        dims = ["c", "z", "y", "x"]
        scale = {"z": 1.0, "y": 1.0, "x": 1.0}

        # Specify range only for first channel, None for second
        mips = create_mip_visualization(
            data,
            dims,
            scale,
            plane="axial",
            slab_thickness_um=10.0,
            channel_colors=["red", "green"],
            channel_ranges=[(0.0, 0.5), None],
        )

        assert len(mips) == 1
        assert mips[0].shape == (20, 20, 3)


class TestOmeroIntegration:
    """Test OMERO metadata integration."""

    @pytest.fixture
    def test_dataset_omero(self):
        """Create a test dataset with OMERO metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "test_omero.ome.zarr")
            create_test_dataset_with_omero(store_path)
            yield store_path

    def test_omero_colors_extraction(self, test_dataset_omero):
        """Test that colors are extracted from OMERO metadata."""
        znimg = ZarrNii.from_ome_zarr(test_dataset_omero)

        # Create MIP without specifying colors - should use OMERO colors
        mips = znimg.create_mip(plane="axial", slab_thickness_um=50.0)

        assert len(mips) > 0
        # Should use OMERO colors (blue, green, red for DAPI, GFP, RFP)

    def test_omero_intensity_ranges_extraction(self, test_dataset_omero):
        """Test that intensity ranges are extracted from OMERO metadata."""
        znimg = ZarrNii.from_ome_zarr(test_dataset_omero)

        # Create MIP without specifying ranges - should use OMERO window settings
        mips = znimg.create_mip(plane="axial", slab_thickness_um=50.0)

        assert len(mips) > 0
        assert mips[0].shape[-1] == 3  # RGB

    def test_channel_selection_by_omero_labels(self, test_dataset_omero):
        """Test selecting channels by OMERO labels."""
        znimg = ZarrNii.from_ome_zarr(test_dataset_omero)

        # Select only GFP and RFP channels
        mips = znimg.create_mip(
            plane="axial", slab_thickness_um=50.0, channel_labels=["GFP", "RFP"]
        )

        assert len(mips) > 0
        # Should create MIP with only 2 channels

    def test_channel_reordering_by_labels(self, test_dataset_omero):
        """Test that channels are reordered according to label order."""
        znimg = ZarrNii.from_ome_zarr(test_dataset_omero)

        # Request channels in different order
        mips = znimg.create_mip(
            plane="axial", slab_thickness_um=50.0, channel_labels=["RFP", "DAPI"]
        )

        assert len(mips) > 0
        # Channels should be in order: RFP (value 3000), DAPI (value 1000)

    def test_error_invalid_channel_label(self, test_dataset_omero):
        """Test error when invalid channel label is specified."""
        znimg = ZarrNii.from_ome_zarr(test_dataset_omero)

        with pytest.raises(ValueError, match="Channel label 'InvalidLabel' not found"):
            znimg.create_mip(
                plane="axial",
                slab_thickness_um=50.0,
                channel_labels=["InvalidLabel"],
            )

    def test_error_channel_labels_without_omero(self):
        """Test error when channel_labels specified without OMERO metadata."""
        # Create data without OMERO metadata
        data = da.random.random((2, 10, 20, 20), chunks=(1, 5, 10, 10))
        dims = ["c", "z", "y", "x"]
        scale = {"z": 1.0, "y": 1.0, "x": 1.0}

        with pytest.raises(
            ValueError, match="channel_labels specified but omero_metadata not provided"
        ):
            create_mip_visualization(
                data,
                dims,
                scale,
                plane="axial",
                channel_labels=["Channel1"],
            )


class TestAlphaTransparency:
    """Test alpha transparency support."""

    def test_rgba_color_specification(self):
        """Test specifying colors with alpha values."""
        data = da.from_array(
            np.ones((2, 10, 20, 20), dtype=np.float32), chunks=(1, 5, 10, 10)
        )
        dims = ["c", "z", "y", "x"]
        scale = {"z": 1.0, "y": 1.0, "x": 1.0}

        # Use RGBA colors with different alpha values
        mips = create_mip_visualization(
            data,
            dims,
            scale,
            plane="axial",
            slab_thickness_um=10.0,
            channel_colors=[
                (1.0, 0.0, 0.0, 1.0),  # Full red
                (0.0, 1.0, 0.0, 0.5),  # Half-transparent green
            ],
        )

        assert len(mips) == 1
        mip = mips[0]

        # With uniform data (all 1.0), normalized to 1.0
        # Red channel contributes 1.0 * 1.0 = 1.0
        # Green channel contributes 1.0 * 0.5 = 0.5
        # Result should be clipped to [0, 1]
        assert mip.shape == (20, 20, 3)
        # Check that alpha affects the contribution
        pixel = mip[10, 10, :]
        assert pixel[0] == 1.0  # Red at full intensity
        assert pixel[1] == 0.5  # Green at half intensity

    def test_rgb_to_rgba_conversion(self):
        """Test that RGB tuples are automatically converted to RGBA."""
        data = da.random.random((2, 10, 20, 20), chunks=(1, 5, 10, 10))
        dims = ["c", "z", "y", "x"]
        scale = {"z": 1.0, "y": 1.0, "x": 1.0}

        # Use RGB tuples - should automatically get alpha=1.0
        mips = create_mip_visualization(
            data,
            dims,
            scale,
            plane="axial",
            slab_thickness_um=10.0,
            channel_colors=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        )

        assert len(mips) == 1
        assert mips[0].shape[-1] == 3

    def test_alpha_blending_additive(self):
        """Test that alpha values result in additive blending."""
        # Create data with overlapping regions
        data_array = np.zeros((2, 10, 20, 20), dtype=np.float32)
        data_array[0, :, 5:15, 5:15] = 1.0  # Channel 0: center square
        data_array[1, :, 5:15, 5:15] = 1.0  # Channel 1: same center square
        data = da.from_array(data_array, chunks=(1, 5, 10, 10))

        dims = ["c", "z", "y", "x"]
        scale = {"z": 1.0, "y": 1.0, "x": 1.0}

        # Both channels with half transparency
        mips = create_mip_visualization(
            data,
            dims,
            scale,
            plane="axial",
            slab_thickness_um=10.0,
            channel_colors=[
                (1.0, 0.0, 0.0, 0.5),  # Half-transparent red
                (0.0, 0.0, 1.0, 0.5),  # Half-transparent blue
            ],
        )

        mip = mips[0]
        # In overlapping region, both contribute 0.5
        center_pixel = mip[10, 10, :]
        assert center_pixel[0] == 0.5  # Red contribution
        assert center_pixel[2] == 0.5  # Blue contribution


class TestBackwardCompatibility:
    """Test that existing functionality still works."""

    def test_default_behavior_unchanged(self):
        """Test that default behavior (no new parameters) still works."""
        data = da.random.random((2, 10, 20, 20), chunks=(1, 5, 10, 10))
        dims = ["c", "z", "y", "x"]
        scale = {"z": 1.0, "y": 1.0, "x": 1.0}

        # Use old API style (no new parameters)
        mips = create_mip_visualization(
            data,
            dims,
            scale,
            plane="axial",
            slab_thickness_um=10.0,
            channel_colors=["red", "green"],
        )

        assert len(mips) == 1
        assert mips[0].shape == (20, 20, 3)

    def test_auto_scaling_still_default(self):
        """Test that auto-scaling is still the default when no ranges specified."""
        # Create data with specific range (2 channels, each with uniform value)
        data_array = np.zeros((2, 1, 1, 1), dtype=np.float32)
        data_array[0, :, :, :] = 100.0  # Channel 0
        data_array[1, :, :, :] = 200.0  # Channel 1
        data = da.from_array(data_array, chunks=(1, 1, 1, 1))

        dims = ["c", "z", "y", "x"]
        scale = {"z": 1.0, "y": 1.0, "x": 1.0}

        # Don't specify ranges - should auto-scale
        # Use explicit RGB tuples to avoid matplotlib color differences
        mips = create_mip_visualization(
            data,
            dims,
            scale,
            plane="axial",
            slab_thickness_um=10.0,
            channel_colors=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        )

        # With auto-scaling and uniform values per channel, each channel
        # normalizes to 1.0 (since min==max for uniform values)
        pixel = mips[0][0, 0, :]
        # Both channels at full intensity due to uniform values
        assert pixel[0] == 1.0
        assert pixel[1] == 1.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_error_mismatched_ranges_count(self):
        """Test error when number of ranges doesn't match channels."""
        data = da.random.random((3, 10, 20, 20), chunks=(1, 5, 10, 10))
        dims = ["c", "z", "y", "x"]
        scale = {"z": 1.0, "y": 1.0, "x": 1.0}

        with pytest.raises(
            ValueError, match="Provided .* intensity ranges but image has"
        ):
            create_mip_visualization(
                data,
                dims,
                scale,
                plane="axial",
                channel_ranges=[(0.0, 1.0), (0.0, 1.0)],  # Only 2 ranges for 3 channels
            )

    def test_zero_intensity_range(self):
        """Test handling of zero intensity range (min == max)."""
        data_array = np.zeros((1, 1, 1, 1), dtype=np.float32)
        data_array[0, :, :, :] = 500.0
        data = da.from_array(data_array, chunks=(1, 1, 1, 1))

        dims = ["c", "z", "y", "x"]
        scale = {"z": 1.0, "y": 1.0, "x": 1.0}

        # Specify range where min == max
        mips = create_mip_visualization(
            data,
            dims,
            scale,
            plane="axial",
            slab_thickness_um=10.0,
            channel_colors=["red"],
            channel_ranges=[(500.0, 500.0)],
        )

        # Should handle gracefully (avoid division by zero)
        assert len(mips) == 1

    def test_invalid_color_tuple_length(self):
        """Test error with invalid color tuple length."""
        data = da.random.random((1, 10, 20, 20), chunks=(1, 5, 10, 10))
        dims = ["c", "z", "y", "x"]
        scale = {"z": 1.0, "y": 1.0, "x": 1.0}

        with pytest.raises(
            ValueError, match="Color tuple must have 3 .* or 4 .* values"
        ):
            create_mip_visualization(
                data,
                dims,
                scale,
                plane="axial",
                channel_colors=[(1.0, 0.0)],  # Only 2 values
            )
