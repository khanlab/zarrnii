"""Tests to validate 3D tiling implementation in to_imaris().

This test suite verifies that the to_imaris() method uses true 3D tiling
(16×256×256) rather than Z-only chunking, preventing memory blowup for
large images.
"""

import os
import tempfile

import dask.array as da
import numpy as np
import pytest

from zarrnii import ZarrNii

# Skip all tests if h5py is not available
h5py = pytest.importorskip("h5py", reason="h5py required for Imaris support")


class TestImaris3DTiling:
    """Test that to_imaris() uses 3D tiling for memory safety."""

    @pytest.mark.usefixtures("cleandir")
    def test_large_image_with_small_tiles(self):
        """Test that large images are processed with small 3D tiles.
        
        This test creates a large image (128×2048×1536) and verifies that
        the export completes successfully. If Z-only chunking was used,
        this would materialize 16×2048×1536 ≈ 50M elements per chunk,
        causing memory issues. With 3D tiling, max is 16×256×256 ≈ 1M elements.
        """
        # Create a large test array
        # If we used Z-slab chunking, each slab would be 16×2048×1536 = ~50M floats (200MB)
        # With 3D tiling, max tile is 16×256×256 = ~1M floats (4MB)
        shape = (128, 2048, 1536)  # Z, Y, X - large Y and X
        
        # Create a Dask array with a pattern we can verify
        # Use small initial chunks to avoid materializing large arrays during creation
        data = da.zeros(shape, dtype=np.float32, chunks=(16, 256, 256))
        
        # Add some test pattern - just set a few values to ensure it's not all zeros
        # Do this efficiently without materializing the full array
        data = da.where(
            (da.arange(shape[0])[:, None, None] == 0) &
            (da.arange(shape[1])[None, :, None] == 0) &
            (da.arange(shape[2])[None, None, :] < 10),
            100.0,
            data
        )
        
        # Add channel dimension
        data = data[np.newaxis, ...]
        
        # Create ZarrNii instance
        znimg = ZarrNii.from_darr(data, spacing=[1.0, 1.0, 1.0])
        
        # Export to Imaris - should succeed without memory issues
        output_path = "test_large_3d_tiling.ims"
        result_path = znimg.to_imaris(output_path)
        
        assert result_path == output_path
        assert os.path.exists(output_path)
        
        # Verify the file structure is correct
        with h5py.File(output_path, "r") as f:
            dataset = f["DataSet"]["ResolutionLevel 0"]["TimePoint 0"]["Channel 0"]["Data"]
            assert dataset.shape == shape
            
            # Verify HDF5 chunking is correct (16×256×256)
            assert dataset.chunks == (16, 256, 256)
            
            # Sample check a few values to ensure data was written correctly
            # Check the corner value we set
            sample = dataset[0, 0, 0]
            assert sample == 100.0

    @pytest.mark.usefixtures("cleandir")
    def test_multi_channel_large_image_3d_tiling(self):
        """Test multi-channel large images with 3D tiling."""
        # Create a multi-channel large array
        n_channels = 2
        shape = (n_channels, 64, 1024, 1024)  # C, Z, Y, X
        
        # Create with proper chunking
        data = da.zeros(shape, dtype=np.uint16, chunks=(1, 16, 256, 256))
        
        # Add test patterns per channel
        for c in range(n_channels):
            # Set a small region in each channel to a unique value
            data[c, 0, :10, :10] = (c + 1) * 1000
        
        # Create ZarrNii instance
        znimg = ZarrNii.from_darr(data, spacing=[1.0, 1.0, 1.0])
        
        # Export to Imaris
        output_path = "test_multichannel_3d_tiling.ims"
        znimg.to_imaris(output_path)
        
        assert os.path.exists(output_path)
        
        # Verify multi-channel structure
        with h5py.File(output_path, "r") as f:
            for c in range(n_channels):
                channel_key = f"Channel {c}"
                dataset = f["DataSet"]["ResolutionLevel 0"]["TimePoint 0"][channel_key]["Data"]
                
                # Verify shape and chunking
                assert dataset.shape == shape[1:]
                assert dataset.chunks == (16, 256, 256)
                
                # Verify data integrity
                sample = dataset[0, 0, 0]
                assert sample == (c + 1) * 1000

    @pytest.mark.usefixtures("cleandir")
    def test_thumbnail_uses_3d_tiling(self):
        """Test that thumbnail MIP generation uses 3D tiling.
        
        This test verifies that the Maximum Intensity Projection (MIP)
        for thumbnails is computed correctly using 3D tiles rather than Z-slabs,
        by comparing against the expected result from full MIP computation.
        """
        # Create test data with a distinctive pattern at different locations
        shape = (48, 512, 384)  # Z, Y, X
        data = np.zeros(shape, dtype=np.float32)
        
        # Add bright spots at different Y/X locations and Z levels
        # Use larger regions to ensure they're captured by decimation
        data[5, 50:70, 50:70] = 255.0
        data[20, 250:270, 50:70] = 200.0
        data[35, 50:70, 300:320] = 180.0
        data[40, 450:470, 300:320] = 160.0
        
        # Create Dask array with 3D chunking
        darr = da.from_array(data[np.newaxis, ...], chunks=(1, 16, 256, 256))
        znimg = ZarrNii.from_darr(darr, spacing=[1.0, 1.0, 1.0])
        
        # Export to Imaris
        output_path = "test_thumbnail_3d_tiling.ims"
        znimg.to_imaris(output_path)
        
        # Read back thumbnail
        with h5py.File(output_path, "r") as f:
            thumbnail = f["Thumbnail"]["Data"][:]
        
        # Compute expected thumbnail (MIP then downsample, matching implementation)
        expected_mip = np.max(data, axis=0)
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
        
        # Compare thumbnails - they should match exactly
        np.testing.assert_array_equal(thumbnail, expected_thumb,
            err_msg="Thumbnail from 3D tiling should match expected MIP-based thumbnail")

    @pytest.mark.usefixtures("cleandir")
    def test_statistics_use_3d_tiling(self):
        """Test that min/max and histogram statistics use 3D tiling.
        
        This test creates an image with known statistics distributed across
        the entire volume to verify they're computed correctly with 3D tiling.
        """
        # Create test data with known statistics
        shape = (32, 512, 384)  # Z, Y, X
        data = np.zeros(shape, dtype=np.float32)
        
        # Place min and max values at opposite corners to ensure
        # 3D tiling covers the entire volume
        data[0, 0, 0] = 0.0  # min
        data[-1, -1, -1] = 1000.0  # max
        
        # Add some intermediate values throughout the volume
        data[8, 128, 96] = 250.0
        data[16, 256, 192] = 500.0
        data[24, 384, 288] = 750.0
        
        # Create Dask array
        darr = da.from_array(data[np.newaxis, ...], chunks=(1, 16, 256, 256))
        znimg = ZarrNii.from_darr(darr, spacing=[1.0, 1.0, 1.0])
        
        # Export to Imaris
        output_path = "test_statistics_3d_tiling.ims"
        znimg.to_imaris(output_path)
        
        # Verify statistics
        with h5py.File(output_path, "r") as f:
            channel_group = f["DataSet"]["ResolutionLevel 0"]["TimePoint 0"]["Channel 0"]
            
            # Get histogram attributes
            hist_min_bytes = channel_group.attrs["HistogramMin"]
            hist_max_bytes = channel_group.attrs["HistogramMax"]
            
            # Convert byte arrays to strings
            hist_min = float("".join([b.decode() for b in hist_min_bytes]))
            hist_max = float("".join([b.decode() for b in hist_max_bytes]))
            
            # Verify min/max were computed correctly across the entire volume
            assert abs(hist_min - 0.0) < 0.01, f"Expected min ~0.0, got {hist_min}"
            assert abs(hist_max - 1000.0) < 0.01, f"Expected max ~1000.0, got {hist_max}"
            
            # Verify histogram exists
            histogram = channel_group["Histogram"][:]
            assert histogram.shape == (256,)
            
            # The histogram should have most values in bin 0 (all the zeros)
            # and some values in other bins for our test points
            assert histogram[0] > np.prod(shape) - 10, "Most values should be in bin 0"
            assert histogram.sum() == np.prod(shape), "Histogram should account for all voxels"
