"""Tests for memory-safe Imaris export functionality.

This test suite verifies that the to_imaris() method can handle large
datasets without loading the entire array into memory.
"""

import os
import tempfile

import dask.array as da
import numpy as np
import pytest

from zarrnii import ZarrNii

# Skip all tests if h5py is not available
h5py = pytest.importorskip("h5py", reason="h5py required for Imaris support")


class TestImarisMemorySafe:
    """Test memory-safe export to Imaris format."""

    @pytest.mark.usefixtures("cleandir")
    def test_to_imaris_with_dask_array(self):
        """Test that to_imaris works with Dask arrays without computing full array."""
        # Create a larger Dask array (simulate a larger dataset)
        # Use a pattern that we can verify later
        shape = (128, 256, 192)  # Z, Y, X
        chunks = (16, 256, 192)

        # Create test data with a known pattern
        data = da.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        data = da.rechunk(data, chunks=chunks)

        # Add channel dimension
        data = data[np.newaxis, ...]

        # Create ZarrNii instance
        znimg = ZarrNii.from_darr(data, spacing=[1.0, 2.0, 3.0])

        # Save to Imaris - this should NOT call compute() on the full array
        output_path = "test_dask_output.ims"
        result_path = znimg.to_imaris(output_path)

        assert result_path == output_path
        assert os.path.exists(output_path)

        # Verify the file structure is correct
        with h5py.File(output_path, "r") as f:
            # Check basic structure
            assert "DataSet" in f
            assert "DataSetInfo" in f
            assert "Thumbnail" in f

            # Check data was written correctly
            dataset = f["DataSet"]["ResolutionLevel 0"]["TimePoint 0"]["Channel 0"][
                "Data"
            ]
            assert dataset.shape == shape
            assert dataset.dtype == np.float32

            # Verify some data values (sample check, not full array)
            sample_slice = dataset[0, 0, :10]
            expected_slice = np.arange(10, dtype=np.float32)
            np.testing.assert_array_almost_equal(sample_slice, expected_slice)

            # Check histogram was created
            histogram = f["DataSet"]["ResolutionLevel 0"]["TimePoint 0"]["Channel 0"][
                "Histogram"
            ]
            assert histogram.shape == (256,)
            assert histogram.dtype == np.uint64

            # Check thumbnail was created
            thumbnail = f["Thumbnail"]["Data"]
            assert thumbnail.shape[0] == 256  # Height
            assert thumbnail.dtype == np.uint8

    @pytest.mark.usefixtures("cleandir")
    def test_to_imaris_multi_channel_dask(self):
        """Test multi-channel export with Dask arrays."""
        # Create multi-channel Dask array
        n_channels = 3
        shape = (n_channels, 64, 128, 96)  # C, Z, Y, X
        chunks = (1, 16, 128, 96)

        # Create test data with different values per channel
        data = da.zeros(shape, dtype=np.float32, chunks=chunks)
        for c in range(n_channels):
            data[c] = c * 100 + da.arange(np.prod(shape[1:]), dtype=np.float32).reshape(
                shape[1:]
            )

        # Create ZarrNii instance
        znimg = ZarrNii.from_darr(data, spacing=[1.0, 1.0, 1.0])

        # Save to Imaris
        output_path = "test_multichannel.ims"
        result_path = znimg.to_imaris(output_path)

        assert os.path.exists(result_path)

        # Verify multi-channel structure
        with h5py.File(output_path, "r") as f:
            # Check all channels exist
            for c in range(n_channels):
                channel_key = f"Channel {c}"
                assert channel_key in f["DataSet"]["ResolutionLevel 0"]["TimePoint 0"]

                # Verify channel data shape
                dataset = f["DataSet"]["ResolutionLevel 0"]["TimePoint 0"][channel_key][
                    "Data"
                ]
                assert dataset.shape == shape[1:]

                # Verify histogram exists for each channel
                histogram = f["DataSet"]["ResolutionLevel 0"]["TimePoint 0"][
                    channel_key
                ]["Histogram"]
                assert histogram.shape == (256,)

            # Check multi-channel thumbnail (should be 256 x (256 * n_channels))
            thumbnail = f["Thumbnail"]["Data"]
            assert thumbnail.shape == (256, 256 * n_channels)

    @pytest.mark.usefixtures("cleandir")
    def test_to_imaris_statistics_correctness(self):
        """Verify that streaming statistics match full-array computation."""
        # Create a small test array with known statistics
        shape = (32, 64, 48)  # Z, Y, X
        np.random.seed(42)
        test_data = np.random.rand(*shape).astype(np.float32) * 100

        # Create Dask array
        darr = da.from_array(test_data[np.newaxis, ...], chunks=(1, 16, 64, 48))
        znimg = ZarrNii.from_darr(darr, spacing=[1.0, 1.0, 1.0])

        # Save to Imaris
        output_path = "test_stats.ims"
        znimg.to_imaris(output_path)

        # Read back and verify statistics
        with h5py.File(output_path, "r") as f:
            channel_group = f["DataSet"]["ResolutionLevel 0"]["TimePoint 0"][
                "Channel 0"
            ]

            # Get histogram attributes
            hist_min_bytes = channel_group.attrs["HistogramMin"]
            hist_max_bytes = channel_group.attrs["HistogramMax"]

            # Convert byte arrays to strings
            hist_min = float("".join([b.decode() for b in hist_min_bytes]))
            hist_max = float("".join([b.decode() for b in hist_max_bytes]))

            # Verify against expected values
            expected_min = float(test_data.min())
            expected_max = float(test_data.max())

            assert abs(hist_min - expected_min) < 0.01
            assert abs(hist_max - expected_max) < 0.01

            # Verify histogram
            histogram = channel_group["Histogram"][:]

            # Compare with numpy histogram
            expected_hist, _ = np.histogram(
                test_data.flatten(), bins=256, range=(expected_min, expected_max)
            )

            np.testing.assert_array_equal(histogram, expected_hist.astype(np.uint64))

    @pytest.mark.usefixtures("cleandir")
    def test_to_imaris_thumbnail_correctness(self):
        """Verify that streaming thumbnail generation matches full-array MIP."""
        # Create a test array with a distinctive pattern
        shape = (48, 128, 96)  # Z, Y, X
        test_data = np.zeros(shape, dtype=np.float32)

        # Add some distinctive features at different Z levels
        test_data[10, 30:50, 40:60] = 255.0  # Bright spot in middle Z
        test_data[20, 60:80, 20:40] = 200.0  # Another spot
        test_data[35, 90:110, 70:90] = 180.0  # Third spot

        # Create Dask array
        darr = da.from_array(test_data[np.newaxis, ...], chunks=(1, 16, 128, 96))
        znimg = ZarrNii.from_darr(darr, spacing=[1.0, 1.0, 1.0])

        # Save to Imaris
        output_path = "test_thumbnail.ims"
        znimg.to_imaris(output_path)

        # Read back thumbnail
        with h5py.File(output_path, "r") as f:
            thumbnail = f["Thumbnail"]["Data"][:]

        # Compute expected thumbnail (MIP then downsample)
        expected_mip = np.max(test_data, axis=0)
        step_y = max(1, expected_mip.shape[0] // 256)
        step_x = max(1, expected_mip.shape[1] // 256)
        expected_thumb = expected_mip[::step_y, ::step_x]

        # Pad to 256x256 if needed
        if expected_thumb.shape[0] < 256 or expected_thumb.shape[1] < 256:
            padded = np.zeros((256, 256), dtype=expected_thumb.dtype)
            h, w = expected_thumb.shape
            padded[:h, :w] = expected_thumb
            expected_thumb = padded
        else:
            expected_thumb = expected_thumb[:256, :256]

        expected_thumb = expected_thumb.astype(np.uint8)

        # Compare thumbnails
        np.testing.assert_array_equal(thumbnail, expected_thumb)

    @pytest.mark.usefixtures("cleandir")
    def test_to_imaris_round_trip_with_dask(self):
        """Test round-trip with Dask array to verify data integrity."""
        # Create test data
        shape = (64, 128, 96)
        np.random.seed(123)
        test_data = np.random.rand(*shape).astype(np.float32) * 1000

        # Create Dask array with chunks
        darr = da.from_array(test_data[np.newaxis, ...], chunks=(1, 16, 128, 96))
        znimg = ZarrNii.from_darr(darr, spacing=[2.0, 1.5, 1.0])

        # Save to Imaris
        output_path = "test_roundtrip_dask.ims"
        znimg.to_imaris(output_path)

        # Load back
        znimg_loaded = ZarrNii.from_imaris(output_path)

        # Compare data
        original_data = znimg.darr.compute()
        loaded_data = znimg_loaded.darr.compute()

        np.testing.assert_array_almost_equal(original_data, loaded_data, decimal=5)

    @pytest.mark.usefixtures("cleandir")
    def test_to_imaris_dtype_preservation(self):
        """Test that different dtypes are handled correctly."""
        shape = (32, 64, 48)

        # Test float32
        data_float = da.from_array(
            np.random.rand(*shape).astype(np.float32)[np.newaxis, ...],
            chunks=(1, 16, 64, 48),
        )
        znimg = ZarrNii.from_darr(data_float, spacing=[1.0, 1.0, 1.0])
        znimg.to_imaris("test_float32.ims")

        with h5py.File("test_float32.ims", "r") as f:
            dataset = f["DataSet"]["ResolutionLevel 0"]["TimePoint 0"]["Channel 0"][
                "Data"
            ]
            assert dataset.dtype == np.float32

        # Test uint16
        data_uint16 = da.from_array(
            np.random.randint(0, 65535, size=shape, dtype=np.uint16)[np.newaxis, ...],
            chunks=(1, 16, 64, 48),
        )
        znimg = ZarrNii.from_darr(data_uint16, spacing=[1.0, 1.0, 1.0])
        znimg.to_imaris("test_uint16.ims")

        with h5py.File("test_uint16.ims", "r") as f:
            dataset = f["DataSet"]["ResolutionLevel 0"]["TimePoint 0"]["Channel 0"][
                "Data"
            ]
            assert dataset.dtype == np.uint16

    @pytest.mark.usefixtures("cleandir")
    def test_to_imaris_hdf5_chunking(self):
        """Test that HDF5 chunking is set to 16x256x256 (ZYX) for Imaris files."""
        # Test with large dimensions (larger than 16x256x256)
        shape_large = (64, 512, 384)  # Z, Y, X
        data_large = da.zeros(shape_large, dtype=np.float32, chunks=(16, 512, 384))
        data_large = data_large[np.newaxis, ...]

        znimg_large = ZarrNii.from_darr(data_large, spacing=[1.0, 1.0, 1.0])
        znimg_large.to_imaris("test_large_chunks.ims")

        with h5py.File("test_large_chunks.ims", "r") as f:
            dataset = f["DataSet"]["ResolutionLevel 0"]["TimePoint 0"]["Channel 0"][
                "Data"
            ]
            # Verify chunking is 16x256x256 (ZYX)
            assert dataset.chunks == (
                16,
                256,
                256,
            ), f"Expected (16, 256, 256), got {dataset.chunks}"

        # Test with small dimensions (smaller than 16x256x256)
        shape_small = (8, 128, 96)  # Z, Y, X
        data_small = da.zeros(shape_small, dtype=np.float32, chunks=(8, 128, 96))
        data_small = data_small[np.newaxis, ...]

        znimg_small = ZarrNii.from_darr(data_small, spacing=[1.0, 1.0, 1.0])
        znimg_small.to_imaris("test_small_chunks.ims")

        with h5py.File("test_small_chunks.ims", "r") as f:
            dataset = f["DataSet"]["ResolutionLevel 0"]["TimePoint 0"]["Channel 0"][
                "Data"
            ]
            # Verify chunking is adjusted for small dimensions (8x128x96)
            assert dataset.chunks == (
                8,
                128,
                96,
            ), f"Expected (8, 128, 96), got {dataset.chunks}"

        # Test with mixed dimensions (some smaller, some larger than default)
        shape_mixed = (32, 128, 512)  # Z, Y, X
        data_mixed = da.zeros(shape_mixed, dtype=np.float32, chunks=(16, 128, 512))
        data_mixed = data_mixed[np.newaxis, ...]

        znimg_mixed = ZarrNii.from_darr(data_mixed, spacing=[1.0, 1.0, 1.0])
        znimg_mixed.to_imaris("test_mixed_chunks.ims")

        with h5py.File("test_mixed_chunks.ims", "r") as f:
            dataset = f["DataSet"]["ResolutionLevel 0"]["TimePoint 0"]["Channel 0"][
                "Data"
            ]
            # Verify chunking is (16, 128, 256) - mixed
            assert dataset.chunks == (
                16,
                128,
                256,
            ), f"Expected (16, 128, 256), got {dataset.chunks}"
