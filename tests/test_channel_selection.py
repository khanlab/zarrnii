"""Tests for enhanced channel selection functionality in ZarrNii.from_ome_zarr."""

import os
import tempfile

import dask.array as da
import ngff_zarr as nz
import numpy as np
import pytest
import zarr

from zarrnii import ZarrNii


class MockOmeroWindow:
    """Mock class to simulate OmeroWindow from modern ome-zarr-py."""

    def __init__(self, min=0.0, max=65535.0, start=0.0, end=65535.0):
        self.min = min
        self.max = max
        self.start = start
        self.end = end


class MockOmeroChannel:
    """Mock class to simulate OmeroChannel from modern ome-zarr-py."""

    def __init__(self, label, color, window=None):
        self.label = label
        self.color = color
        self.window = window or MockOmeroWindow()


class MockOmero:
    """Mock class to simulate Omero from modern ome-zarr-py."""

    def __init__(self, channels):
        self.channels = channels


def create_test_dataset_with_omero_metadata(store_path, num_channels=3):
    """Create a test OME-Zarr dataset with proper omero metadata and channel labels."""

    # Create a 4D array in ZYXC order to match the axis labels
    arr_sz = (16, 32, 32, num_channels)  # (z, y, x, channels)
    arr = da.zeros(arr_sz, dtype=np.uint16)

    # Fill with different values for each channel for easy identification
    def fill_channel_data(block, block_info=None):
        if block_info is not None:
            # Get the channel slice info (last dimension)
            block_slice = block_info[0]["array-location"]
            c_start = block_slice[3][0]  # Channel is at index 3
            c_end = block_slice[3][1]

            # Create different values for each channel (100, 200, 300, etc.)
            result = np.zeros(block.shape, dtype=np.uint16)
            for c_idx in range(c_end - c_start):
                global_c_idx = c_start + c_idx
                value = (global_c_idx + 1) * 100
                result[:, :, :, c_idx] = value
            return result
        return np.zeros(block.shape, dtype=np.uint16)

    arr = arr.map_blocks(fill_channel_data, dtype=np.uint16)

    # Create NGFF image and multiscales
    ngff_image = nz.to_ngff_image(arr)
    multiscales = nz.to_multiscales(ngff_image)

    # Create omero metadata as dictionaries - this will be the format expected by ngff_zarr
    channel_labels_and_colors = [
        ("DAPI", "0000FF"),
        ("Abeta", "00FF00"),
        ("GFP", "FF0000"),
    ][:num_channels]  # Only include as many as we have channels

    # Create channel dictionaries with proper structure for ngff_zarr
    omero_channels = []
    for label, color in channel_labels_and_colors:
        channel = {
            "label": label,
            "color": color,
            "window": {
                "min": 0.0,
                "max": 65535.0,
                "start": 0.0,
                "end": 65535.0
            }
        }
        omero_channels.append(channel)

    omero_metadata = {
        "channels": omero_channels
    }

    # Store to zarr first
    nz.to_ngff_zarr(store_path, multiscales)

    # Now add omero metadata to the zarr group attributes
    # This follows the same pattern that ngff_zarr uses when reading
    group = zarr.open_group(store_path, mode="r+")
    group.attrs["omero"] = omero_metadata

    return store_path


class TestChannelSelection:
    """Test class for channel selection functionality."""

    @pytest.fixture
    def test_dataset(self):
        """Create a temporary test dataset with legacy omero metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "test_channels.ome.zarr")
            create_test_dataset_with_omero_metadata(store_path)
            yield store_path

    def test_load_all_channels_by_default(self, test_dataset):
        """Test that all channels are loaded by default when no channels specified."""
        znimg = ZarrNii.from_ome_zarr(test_dataset)

        # Should load all 3 channels, shape should be (1, z, y, x) after slicing for single channel or (3, z, y, x) for all
        # Since we load all channels, expect shape to be (z, y, x, num_channels) -> (16, 32, 32, 3)
        expected_shape = (16, 32, 32, 3)
        assert znimg.darr.shape == expected_shape

        # Check that omero metadata is preserved
        assert znimg.omero is not None
        assert hasattr(znimg.omero, 'channels')
        assert len(znimg.omero.channels) == 3

    def test_load_by_channel_labels_single(self, test_dataset):
        """Test loading a single channel by label."""
        znimg = ZarrNii.from_ome_zarr(test_dataset, channel_labels=["Abeta"])

        # Should load only 1 channel: shape (z, y, x, 1) = (16, 32, 32, 1)
        expected_shape = (16, 32, 32, 1)
        assert znimg.darr.shape == expected_shape

        # Check data value - Abeta is channel index 1, so should have value 200
        data_sum = znimg.darr.compute().sum()
        expected_sum = 200 * 16 * 32 * 32  # value * z * y * x
        assert data_sum == expected_sum

    def test_load_by_channel_labels_multiple(self, test_dataset):
        """Test loading multiple channels by labels."""
        znimg = ZarrNii.from_ome_zarr(test_dataset, channel_labels=["DAPI", "GFP"])

        # Should load 2 channels: shape (z, y, x, 2) = (16, 32, 32, 2)
        expected_shape = (16, 32, 32, 2)
        assert znimg.darr.shape == expected_shape

        # Check data values for each channel
        # DAPI=channel 0=100, GFP=channel 2=300
        channel_0_sum = znimg.darr[:, :, :, 0].compute().sum()
        channel_1_sum = znimg.darr[:, :, :, 1].compute().sum()
        expected_sums = [100 * 16 * 32 * 32, 300 * 16 * 32 * 32]  # DAPI=100, GFP=300
        assert [channel_0_sum, channel_1_sum] == expected_sums

    def test_load_by_channel_indices_backward_compatibility(self, test_dataset):
        """Test that loading by channel indices still works (backward compatibility)."""
        znimg = ZarrNii.from_ome_zarr(test_dataset, channels=[1, 2])

        # Should load 2 channels: shape (z, y, x, 2) = (16, 32, 32, 2)
        expected_shape = (16, 32, 32, 2)
        assert znimg.darr.shape == expected_shape

        # Check data values - indices 1 and 2 should have values 200 and 300
        channel_0_sum = znimg.darr[:, :, :, 0].compute().sum()
        channel_1_sum = znimg.darr[:, :, :, 1].compute().sum()
        expected_sums = [200 * 16 * 32 * 32, 300 * 16 * 32 * 32]
        assert [channel_0_sum, channel_1_sum] == expected_sums

    def test_error_both_channels_and_labels_specified(self, test_dataset):
        """Test that specifying both channels and channel_labels raises an error."""
        with pytest.raises(
            ValueError, match="Cannot specify both 'channels' and 'channel_labels'"
        ):
            ZarrNii.from_ome_zarr(test_dataset, channels=[0], channel_labels=["DAPI"])

    def test_error_invalid_channel_label(self, test_dataset):
        """Test that specifying a non-existent channel label raises an error."""
        with pytest.raises(ValueError, match="Channel label 'NonExistent' not found"):
            ZarrNii.from_ome_zarr(test_dataset, channel_labels=["NonExistent"])

    def test_error_channel_labels_without_omero_metadata(self):
        """Test that specifying channel_labels without omero metadata raises an error."""
        # Create dataset without omero metadata
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "test_no_omero.ome.zarr")

            # Create simple dataset without omero metadata
            arr = da.zeros((2, 16, 32, 32), dtype=np.uint16)
            ngff_image = nz.to_ngff_image(arr)
            multiscales = nz.to_multiscales(ngff_image)
            nz.to_ngff_zarr(store_path, multiscales)

            with pytest.raises(
                ValueError,
                match="Channel labels were specified but no omero metadata found",
            ):
                ZarrNii.from_ome_zarr(store_path, channel_labels=["DAPI"])

    def test_omero_metadata_preservation(self, test_dataset):
        """Test that omero metadata is properly preserved when loading by labels."""
        znimg = ZarrNii.from_ome_zarr(test_dataset, channel_labels=["Abeta"])

        # Check that omero metadata is preserved correctly
        assert znimg.omero is not None
        assert hasattr(znimg.omero, 'channels')
        assert znimg.omero.channels[1].label == "Abeta"  # Original full metadata

    def test_mixed_label_order(self, test_dataset):
        """Test that channel labels can be specified in any order."""
        znimg = ZarrNii.from_ome_zarr(test_dataset, channel_labels=["GFP", "DAPI"])

        # Should load 2 channels in the order specified by labels: (z, y, x, 2) = (16, 32, 32, 2)
        expected_shape = (16, 32, 32, 2)
        assert znimg.darr.shape == expected_shape

        # Check data values - GFP first (300), then DAPI (100)
        channel_0_sum = znimg.darr[:, :, :, 0].compute().sum()
        channel_1_sum = znimg.darr[:, :, :, 1].compute().sum()
        expected_sums = [300 * 16 * 32 * 32, 100 * 16 * 32 * 32]  # GFP=300, DAPI=100
        assert [channel_0_sum, channel_1_sum] == expected_sums

    # Tests for backward compatibility with modern OME-Zarr format
    def test_extract_channel_labels_helper_legacy_format(self):
        """Test the helper function with legacy dictionary format."""
        from zarrnii.core import _extract_channel_labels_from_omero

        # Legacy format: list of dictionaries
        legacy_channels = [
            {"label": "DAPI", "color": "0000FF"},
            {"label": "Abeta", "color": "00FF00"},
            {"label": "GFP", "color": "FF0000"},
        ]

        labels = _extract_channel_labels_from_omero(legacy_channels)
        assert labels == ["DAPI", "Abeta", "GFP"]

    def test_extract_channel_labels_helper_modern_format(self):
        """Test the helper function with modern object format."""
        from zarrnii.core import _extract_channel_labels_from_omero

        # Modern format: list of objects with .label attributes
        modern_channels = [
            MockOmeroChannel("DAPI", "0000FF"),
            MockOmeroChannel("Abeta", "00FF00"),
            MockOmeroChannel("GFP", "FF0000"),
        ]

        labels = _extract_channel_labels_from_omero(modern_channels)
        assert labels == ["DAPI", "Abeta", "GFP"]

    def test_extract_channel_labels_helper_mixed_format(self):
        """Test the helper function with mixed formats."""
        from zarrnii.core import _extract_channel_labels_from_omero

        # Mixed format: some dictionaries, some objects
        mixed_channels = [
            {"label": "DAPI", "color": "0000FF"},  # Legacy format
            MockOmeroChannel("Abeta", "00FF00"),  # Modern format
            {"label": "GFP", "color": "FF0000"},  # Legacy format
        ]

        labels = _extract_channel_labels_from_omero(mixed_channels)
        assert labels == ["DAPI", "Abeta", "GFP"]
