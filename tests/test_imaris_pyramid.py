"""Tests for Imaris pyramid generation functionality."""

import os
import tempfile

import dask.array as da
import numpy as np
import pytest

from zarrnii import ZarrNii

# Skip all tests if h5py is not available
h5py = pytest.importorskip("h5py", reason="h5py required for Imaris support")


class TestImarisPyramid:
    """Test multi-resolution pyramid generation in Imaris export."""

    @pytest.mark.usefixtures("cleandir")
    def test_pyramid_level_calculation(self):
        """Test that pyramid levels are calculated correctly."""
        # Test with a moderately sized volume
        data_size = (128, 256, 192)  # Z, Y, X
        pyramid_levels = ZarrNii._compute_imaris_pyramid_levels(data_size)

        # Should have at least 2 levels (original + at least one downsampled)
        assert len(pyramid_levels) >= 2

        # First level should be original size
        assert pyramid_levels[0] == data_size

        # Each subsequent level should be smaller or equal
        for i in range(1, len(pyramid_levels)):
            prev_z, prev_y, prev_x = pyramid_levels[i - 1]
            curr_z, curr_y, curr_x = pyramid_levels[i]
            assert curr_z <= prev_z
            assert curr_y <= prev_y
            assert curr_x <= prev_x
            # At least one dimension should be downsampled
            assert curr_z < prev_z or curr_y < prev_y or curr_x < prev_x

    @pytest.mark.usefixtures("cleandir")
    def test_pyramid_level_calculation_small_volume(self):
        """Test pyramid calculation for a small volume."""
        # Small volume should have fewer pyramid levels
        data_size = (16, 32, 32)  # Z, Y, X
        pyramid_levels = ZarrNii._compute_imaris_pyramid_levels(data_size)

        # First level should be original size
        assert pyramid_levels[0] == data_size

        # Should stop when volume is small enough
        last_level = pyramid_levels[-1]
        volume_mb = (last_level[0] * last_level[1] * last_level[2] * 4.0) / (
            1024.0 * 1024.0
        )
        assert volume_mb <= 1.0 or last_level == (1, 1, 1)

    @pytest.mark.usefixtures("cleandir")
    def test_to_imaris_creates_pyramid(self):
        """Test that to_imaris creates multiple resolution levels."""
        # Create a test dataset large enough to generate multiple pyramid levels
        shape = (128, 256, 192)  # Z, Y, X (24 MB, should generate multiple levels)
        chunks = (16, 256, 192)

        data = da.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        data = da.rechunk(data, chunks=chunks)
        data = data[np.newaxis, ...]  # Add channel dimension

        znimg = ZarrNii.from_darr(data, spacing=[1.0, 1.0, 1.0])

        # Save to Imaris
        output_path = "test_pyramid.ims"
        result_path = znimg.to_imaris(output_path)

        assert result_path == output_path
        assert os.path.exists(output_path)

        # Verify the file has multiple resolution levels
        with h5py.File(output_path, "r") as f:
            dataset_group = f["DataSet"]

            # Count resolution levels
            res_levels = [
                key for key in dataset_group.keys() if key.startswith("ResolutionLevel")
            ]
            assert (
                len(res_levels) >= 2
            ), f"Expected at least 2 resolution levels, found {len(res_levels)}"

            # Verify each level has correct structure
            for level_key in res_levels:
                res_group = dataset_group[level_key]
                assert "TimePoint 0" in res_group
                time_group = res_group["TimePoint 0"]
                assert "Channel 0" in time_group
                channel_group = time_group["Channel 0"]
                assert "Data" in channel_group
                assert "Histogram" in channel_group

                # Check that data exists
                data_dataset = channel_group["Data"]
                assert data_dataset.shape[0] > 0
                assert data_dataset.shape[1] > 0
                assert data_dataset.shape[2] > 0

    @pytest.mark.usefixtures("cleandir")
    def test_pyramid_sizes_decrease(self):
        """Test that pyramid levels have decreasing sizes."""
        # Create a test dataset
        shape = (128, 256, 192)  # Z, Y, X
        data = da.zeros(shape, dtype=np.float32, chunks=(16, 256, 192))
        data = data[np.newaxis, ...]

        znimg = ZarrNii.from_darr(data, spacing=[1.0, 1.0, 1.0])

        output_path = "test_pyramid_sizes.ims"
        znimg.to_imaris(output_path)

        with h5py.File(output_path, "r") as f:
            dataset_group = f["DataSet"]

            # Get all resolution levels in order
            res_levels = sorted(
                [
                    key
                    for key in dataset_group.keys()
                    if key.startswith("ResolutionLevel")
                ],
                key=lambda x: int(x.split()[-1]),
            )

            prev_volume = None
            for level_key in res_levels:
                channel_group = dataset_group[level_key]["TimePoint 0"]["Channel 0"]
                data_dataset = channel_group["Data"]

                # Get dimensions
                z, y, x = data_dataset.shape
                volume = z * y * x

                # Each level should be smaller or equal to previous
                if prev_volume is not None:
                    assert (
                        volume <= prev_volume
                    ), f"Level {level_key} has volume {volume} > previous {prev_volume}"

                prev_volume = volume

    @pytest.mark.usefixtures("cleandir")
    def test_pyramid_chunking(self):
        """Test that all pyramid levels use appropriate chunking."""
        shape = (128, 256, 192)  # Z, Y, X
        data = da.zeros(shape, dtype=np.float32, chunks=(16, 256, 192))
        data = data[np.newaxis, ...]

        znimg = ZarrNii.from_darr(data, spacing=[1.0, 1.0, 1.0])

        output_path = "test_pyramid_chunks.ims"
        znimg.to_imaris(output_path)

        with h5py.File(output_path, "r") as f:
            dataset_group = f["DataSet"]

            for level_key in dataset_group.keys():
                if not level_key.startswith("ResolutionLevel"):
                    continue

                channel_group = dataset_group[level_key]["TimePoint 0"]["Channel 0"]
                data_dataset = channel_group["Data"]

                # Check chunking
                chunks = data_dataset.chunks
                assert chunks is not None, f"Level {level_key} has no chunking"

                # Chunks should follow 16x256x256 pattern (adjusted for small dimensions)
                z_chunk, y_chunk, x_chunk = chunks
                z_dim, y_dim, x_dim = data_dataset.shape

                assert z_chunk <= 16, f"Z chunk {z_chunk} exceeds 16"
                assert y_chunk <= 256, f"Y chunk {y_chunk} exceeds 256"
                assert x_chunk <= 256, f"X chunk {x_chunk} exceeds 256"

                # Chunks should not exceed dimensions
                assert z_chunk <= z_dim
                assert y_chunk <= y_dim
                assert x_chunk <= x_dim

    @pytest.mark.usefixtures("cleandir")
    def test_multi_channel_pyramid(self):
        """Test pyramid generation with multi-channel data."""
        n_channels = 3
        shape = (n_channels, 128, 256, 192)  # C, Z, Y, X (large enough for pyramid)
        data = da.zeros(shape, dtype=np.float32, chunks=(1, 16, 256, 192))

        znimg = ZarrNii.from_darr(data, spacing=[1.0, 1.0, 1.0])

        output_path = "test_multichannel_pyramid.ims"
        znimg.to_imaris(output_path)

        with h5py.File(output_path, "r") as f:
            dataset_group = f["DataSet"]

            # Get resolution levels
            res_levels = [
                key for key in dataset_group.keys() if key.startswith("ResolutionLevel")
            ]
            assert len(res_levels) >= 2

            # Check that all channels exist in all levels
            for level_key in res_levels:
                time_group = dataset_group[level_key]["TimePoint 0"]
                for c in range(n_channels):
                    assert f"Channel {c}" in time_group
                    channel_group = time_group[f"Channel {c}"]
                    assert "Data" in channel_group
                    assert "Histogram" in channel_group
