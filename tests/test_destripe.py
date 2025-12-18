"""Tests for destriping functionality."""

import dask.array as da
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from zarrnii.destripe import (
    _has_allowed_chunking,
    _odd,
    destripe,
    destripe_block,
    downsample_grid,
    phasecong,
    upsample_grid,
)


class TestOddHelper:
    """Tests for the _odd helper function."""

    def test_odd_even_input(self):
        """Test that even numbers are converted to odd."""
        assert _odd(2) == 3
        assert _odd(4) == 5
        assert _odd(100) == 101

    def test_odd_odd_input(self):
        """Test that odd numbers remain unchanged."""
        assert _odd(1) == 1
        assert _odd(3) == 3
        assert _odd(99) == 99

    def test_odd_zero(self):
        """Test that zero is converted to 1 (odd)."""
        assert _odd(0) == 1

    def test_odd_negative_even(self):
        """Test negative even numbers."""
        assert _odd(-2) == -1
        assert _odd(-4) == -3

    def test_odd_negative_odd(self):
        """Test negative odd numbers."""
        assert _odd(-1) == -1
        assert _odd(-3) == -3


class TestDownsampleGrid:
    """Tests for the downsample_grid function."""

    def test_downsample_basic(self):
        """Test basic downsampling with factor=2."""
        img = np.arange(16).reshape(4, 4).astype(np.float32)
        result = downsample_grid(img, factor=2)

        # Should produce 2x2x4 stack (h_small=2, w_small=2, channels=4)
        assert result.shape == (2, 2, 4)

    def test_downsample_factor_3(self):
        """Test downsampling with factor=3."""
        img = np.arange(81).reshape(9, 9).astype(np.float32)
        result = downsample_grid(img, factor=3)

        # Should produce 3x3x9 stack (h_small=3, w_small=3, channels=9)
        assert result.shape == (3, 3, 9)

    def test_downsample_preserves_values(self):
        """Test that downsampling preserves the correct values."""
        img = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        result = downsample_grid(img, factor=2)

        # Check first channel (offset 0,0)
        expected_ch0 = np.array([[1, 3], [9, 11]])
        assert_array_equal(result[:, :, 0], expected_ch0)

        # Check second channel (offset 0,1)
        expected_ch1 = np.array([[2, 4], [10, 12]])
        assert_array_equal(result[:, :, 1], expected_ch1)

    def test_downsample_crops_excess(self):
        """Test that images not divisible by factor are cropped."""
        img = np.arange(25).reshape(5, 5).astype(np.float32)
        result = downsample_grid(img, factor=2)

        # Should crop to 4x4 and produce 2x2x4 stack
        assert result.shape == (2, 2, 4)


class TestUpsampleGrid:
    """Tests for the upsample_grid function."""

    def test_upsample_basic(self):
        """Test basic upsampling with factor=2."""
        # Create a simple 2x2x4 stack
        stack = np.ones((2, 2, 4), dtype=np.float32)
        result = upsample_grid(stack, factor=2)

        # Should produce 4x4 image
        assert result.shape == (4, 4)

    def test_upsample_factor_3(self):
        """Test upsampling with factor=3."""
        stack = np.ones((3, 3, 9), dtype=np.float32)
        result = upsample_grid(stack, factor=3)

        # Should produce 9x9 image
        assert result.shape == (9, 9)

    def test_upsample_reconstructs_values(self):
        """Test that upsampling correctly reconstructs values."""
        # Create a stack with distinct values in each channel
        stack = np.zeros((2, 2, 4), dtype=np.float32)
        stack[:, :, 0] = 1.0
        stack[:, :, 1] = 2.0
        stack[:, :, 2] = 3.0
        stack[:, :, 3] = 4.0

        result = upsample_grid(stack, factor=2)

        # Check that values are interleaved correctly
        assert result[0, 0] == 1.0  # channel 0 (offset 0,0)
        assert result[0, 1] == 2.0  # channel 1 (offset 0,1)
        assert result[1, 0] == 3.0  # channel 2 (offset 1,0)
        assert result[1, 1] == 4.0  # channel 3 (offset 1,1)

    def test_downsample_upsample_roundtrip(self):
        """Test that downsample followed by upsample recovers original."""
        img = np.arange(16).reshape(4, 4).astype(np.float32)
        stack = downsample_grid(img, factor=2)
        reconstructed = upsample_grid(stack, factor=2)

        assert_array_equal(reconstructed, img)

    def test_downsample_upsample_larger_image(self):
        """Test roundtrip with larger image."""
        img = np.random.rand(64, 64).astype(np.float32)
        factor = 4
        stack = downsample_grid(img, factor=factor)
        reconstructed = upsample_grid(stack, factor=factor)

        # Should recover original up to the cropped size
        h, w = img.shape
        h_crop = (h // factor) * factor
        w_crop = (w // factor) * factor
        assert_array_equal(reconstructed, img[:h_crop, :w_crop])


class TestHasAllowedChunking:
    """Tests for the _has_allowed_chunking validation function."""

    def test_valid_3d_chunking(self):
        """Test valid 3D chunking (Z,Y,X) with Z chunked as 1s."""
        arr = da.zeros((10, 512, 512), chunks=(1, 512, 512))
        assert _has_allowed_chunking(arr) is True

    def test_valid_4d_chunking(self):
        """Test valid 4D chunking (C,Z,Y,X)."""
        arr = da.zeros((2, 10, 512, 512), chunks=(1, 1, 512, 512))
        assert _has_allowed_chunking(arr) is True

    def test_valid_5d_chunking(self):
        """Test valid 5D chunking (T,C,Z,Y,X)."""
        arr = da.zeros((3, 2, 10, 512, 512), chunks=(1, 1, 1, 512, 512))
        assert _has_allowed_chunking(arr) is True

    def test_invalid_z_chunking(self):
        """Test invalid Z chunking (chunk size != 1)."""
        arr = da.zeros((10, 512, 512), chunks=(2, 512, 512))
        assert _has_allowed_chunking(arr) is False

    def test_invalid_y_chunking(self):
        """Test invalid Y chunking (multiple chunks)."""
        arr = da.zeros((10, 512, 512), chunks=(1, 256, 512))
        assert _has_allowed_chunking(arr) is False

    def test_invalid_x_chunking(self):
        """Test invalid X chunking (multiple chunks)."""
        arr = da.zeros((10, 512, 512), chunks=(1, 512, 256))
        assert _has_allowed_chunking(arr) is False

    def test_invalid_leading_dim_chunking(self):
        """Test invalid leading dimension chunking in 4D."""
        arr = da.zeros((4, 10, 512, 512), chunks=(2, 1, 512, 512))
        assert _has_allowed_chunking(arr) is False

    def test_invalid_too_few_dims(self):
        """Test rejection of 2D arrays."""
        arr = da.zeros((512, 512), chunks=(512, 512))
        assert _has_allowed_chunking(arr) is False

    def test_invalid_too_many_dims(self):
        """Test rejection of 6D arrays."""
        arr = da.zeros((2, 2, 2, 10, 512, 512), chunks=(1, 1, 1, 1, 512, 512))
        assert _has_allowed_chunking(arr) is False


class TestPhasecong:
    """Tests for the phasecong function."""

    def test_phasecong_basic_square_image(self):
        """Test phase congruency on a simple square image."""
        # Create a simple 64x64 image with a vertical stripe pattern
        img = np.zeros((64, 64), dtype=np.float32)
        img[:, 10:20] = 1.0  # Vertical stripe

        pc, ori = phasecong(img, nscale=2, norient=4, min_wave_length=3)

        # Check output shapes
        assert pc.shape == (64, 64)
        assert ori.shape == (64, 64)

        # Check data types
        assert pc.dtype == np.float32
        assert ori.dtype == np.float32

        # Phase congruency should be non-negative
        assert np.all(pc >= 0)

        # Orientation should be in range [0, 180)
        assert np.all(ori >= 0)
        assert np.all(ori < 180)

    def test_phasecong_uniform_image(self):
        """Test phase congruency on uniform image (no features)."""
        img = np.ones((64, 64), dtype=np.float32) * 0.5

        pc, ori = phasecong(img, nscale=2, norient=4)

        # Uniform image should have very low phase congruency
        assert np.mean(pc) < 0.1

    def test_phasecong_invalid_non_square(self):
        """Test that non-square images raise ValueError."""
        img = np.zeros((64, 128), dtype=np.float32)

        with pytest.raises(ValueError, match="must be square 2D"):
            phasecong(img)

    def test_phasecong_invalid_3d(self):
        """Test that 3D images raise ValueError."""
        img = np.zeros((64, 64, 64), dtype=np.float32)

        with pytest.raises(ValueError, match="must be square 2D"):
            phasecong(img)

    def test_phasecong_default_parameters(self):
        """Test phase congruency with default parameters."""
        img = np.random.rand(128, 128).astype(np.float32)

        # Should work with default parameters
        pc, ori = phasecong(img)

        assert pc.shape == (128, 128)
        assert ori.shape == (128, 128)


class TestDestripeBlock:
    """Tests for the destripe_block function."""

    def test_destripe_block_basic(self):
        """Test basic destripe_block functionality."""
        # Create a simple 2D block with some structure
        block = np.random.rand(128, 128).astype(np.float32)

        result = destripe_block(
            block,
            bg_thresh=0.004,
            factor=8,
            phase_size=64,
            med_size_min=3,
            med_size_max=5,
        )

        # Check shape preservation
        assert result.shape == block.shape

        # Check dtype preservation
        assert result.dtype == block.dtype

    def test_destripe_block_with_stripes(self):
        """Test destripe_block with artificial vertical stripes."""
        # Create an image with vertical stripes
        block = np.ones((256, 256), dtype=np.float32) * 0.5

        # Add vertical stripes
        for i in range(0, 256, 20):
            block[:, i : i + 5] = 0.8

        result = destripe_block(block, factor=16, phase_size=128)

        # Shape and dtype should be preserved
        assert result.shape == block.shape
        assert result.dtype == block.dtype

        # The function should run without errors and produce output in valid range
        assert np.all(np.isfinite(result))
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_destripe_block_integer_input(self):
        """Test destripe_block with integer input."""
        block = (np.random.rand(128, 128) * 255).astype(np.uint8)

        result = destripe_block(block, factor=8, phase_size=64)

        # Check shape and dtype preservation
        assert result.shape == block.shape
        assert result.dtype == block.dtype

    def test_destripe_block_3d_input_squeezed(self):
        """Test destripe_block with 3D input that can be squeezed to 2D."""
        # Create 3D array with singleton dimension
        block = np.random.rand(1, 128, 128).astype(np.float32)

        result = destripe_block(block, factor=8, phase_size=64)

        # Should handle squeezing and return same shape
        assert result.shape == block.shape

    def test_destripe_block_preserves_range(self):
        """Test that destripe_block approximately preserves intensity range."""
        block = np.random.rand(128, 128).astype(np.float32)
        original_min = block.min()
        original_max = block.max()

        result = destripe_block(block, factor=8, phase_size=64)

        # Result should be in a reasonable range relative to input
        # Tolerance of 0.2 accounts for:
        # - Median filtering and phase congruency corrections
        # - Background masking and morphological operations
        # - Normalization and rescaling steps in the algorithm
        tolerance = 0.2
        assert result.min() >= original_min - tolerance
        assert result.max() <= original_max + tolerance

    def test_destripe_block_nan_handling(self):
        """Test that NaN values are handled correctly."""
        block = np.random.rand(128, 128).astype(np.float32)
        block[10:20, 10:20] = np.nan

        result = destripe_block(block, factor=8, phase_size=64)

        # NaN values should be converted to 0
        assert not np.any(np.isnan(result))


class TestDestripe:
    """Tests for the main destripe function."""

    def test_destripe_3d_valid_chunking(self):
        """Test destripe with valid 3D array chunking."""
        # Create a 3D dask array with proper chunking
        arr = da.random.random((10, 256, 256), chunks=(1, 256, 256)).astype(np.float32)

        result = destripe(arr, factor=16, phase_size=128)

        # Check that result is a dask array
        assert isinstance(result, da.Array)

        # Check shape preservation
        assert result.shape == arr.shape

        # Check dtype preservation
        assert result.dtype == arr.dtype

        # Check chunking preservation
        assert result.chunks == arr.chunks

    def test_destripe_4d_valid_chunking(self):
        """Test destripe with valid 4D array chunking (C,Z,Y,X)."""
        arr = da.random.random((2, 10, 256, 256), chunks=(1, 1, 256, 256)).astype(
            np.float32
        )

        result = destripe(arr, factor=16, phase_size=128)

        assert isinstance(result, da.Array)
        assert result.shape == arr.shape
        assert result.dtype == arr.dtype

    def test_destripe_5d_valid_chunking(self):
        """Test destripe with valid 5D array chunking (T,C,Z,Y,X)."""
        arr = da.random.random((3, 2, 10, 256, 256), chunks=(1, 1, 1, 256, 256)).astype(
            np.float32
        )

        result = destripe(arr, factor=16, phase_size=128)

        assert isinstance(result, da.Array)
        assert result.shape == arr.shape
        assert result.dtype == arr.dtype

    def test_destripe_invalid_z_chunking(self):
        """Test that invalid Z chunking raises ValueError."""
        arr = da.random.random((10, 256, 256), chunks=(2, 256, 256)).astype(np.float32)

        with pytest.raises(ValueError, match="Incorrect shape or chunking"):
            destripe(arr)

    def test_destripe_invalid_y_chunking(self):
        """Test that invalid Y chunking raises ValueError."""
        arr = da.random.random((10, 256, 256), chunks=(1, 128, 256)).astype(np.float32)

        with pytest.raises(ValueError, match="Incorrect shape or chunking"):
            destripe(arr)

    def test_destripe_invalid_x_chunking(self):
        """Test that invalid X chunking raises ValueError."""
        arr = da.random.random((10, 256, 256), chunks=(1, 256, 128)).astype(np.float32)

        with pytest.raises(ValueError, match="Incorrect shape or chunking"):
            destripe(arr)

    def test_destripe_invalid_too_few_dims(self):
        """Test that 2D arrays are rejected."""
        arr = da.random.random((256, 256), chunks=(256, 256)).astype(np.float32)

        with pytest.raises(ValueError, match="Incorrect shape or chunking"):
            destripe(arr)

    def test_destripe_invalid_too_many_dims(self):
        """Test that 6D arrays are rejected."""
        arr = da.random.random(
            (2, 2, 2, 10, 256, 256), chunks=(1, 1, 1, 1, 256, 256)
        ).astype(np.float32)

        with pytest.raises(ValueError, match="Incorrect shape or chunking"):
            destripe(arr)

    def test_destripe_compute_small_array(self):
        """Test that destripe can be computed on a small array."""
        # Create a small 3D array
        arr_np = np.random.rand(5, 128, 128).astype(np.float32)
        arr = da.from_array(arr_np, chunks=(1, 128, 128))

        result = destripe(arr, factor=8, phase_size=64)
        result_computed = result.compute()

        # Check that computation succeeds and returns expected shape
        assert result_computed.shape == arr_np.shape
        assert result_computed.dtype == arr_np.dtype

    def test_destripe_custom_parameters(self):
        """Test destripe with custom parameters."""
        arr = da.random.random((5, 128, 128), chunks=(1, 128, 128)).astype(np.float32)

        result = destripe(
            arr,
            bg_thresh=0.01,
            factor=8,
            diff_thresh=0.01,
            med_size_min=5,
            med_size_max=11,
            phase_size=64,
            ori_target_deg=90.0,
            ori_tol_deg=10.0,
        )

        # Should succeed with custom parameters
        assert isinstance(result, da.Array)
        assert result.shape == arr.shape

    def test_destripe_preserves_metadata(self):
        """Test that destripe preserves dask array metadata."""
        arr = da.random.random((5, 256, 256), chunks=(1, 256, 256)).astype(np.float32)
        arr_name = arr.name

        result = destripe(arr, factor=16, phase_size=128)

        # Result should be a new dask array (different name)
        assert result.name != arr_name

        # But shape, dtype, chunks should be preserved
        assert result.shape == arr.shape
        assert result.dtype == arr.dtype
        assert result.chunks == arr.chunks
