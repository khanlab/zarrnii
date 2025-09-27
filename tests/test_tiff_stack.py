"""Test TIFF stack export functionality."""

import os
import tempfile

import dask.array as da
import ngff_zarr as nz
import numpy as np
import pytest

from zarrnii import ZarrNii


class TestTiffStackExport:
    """Test the to_tiff_stack method."""

    def test_basic_tiff_stack_export(self):
        """Test basic TIFF stack export with 3D data."""
        # Create 4D test data (C=1, Z=5, Y=32, X=32) that matches ZarrNii's expected structure
        data = np.random.rand(1, 5, 32, 32).astype(np.float32)
        dask_data = da.from_array(data, chunks=(1, 2, 16, 16))
        znii = ZarrNii.from_darr(dask_data, spacing=(1.0, 1.0, 1.0))

        with tempfile.TemporaryDirectory() as temp_dir:
            pattern = os.path.join(temp_dir, "test_z{z:04d}.tif")
            # Use float32 to preserve original values for comparison
            znii.to_tiff_stack(pattern, dtype="float32")

            # Check that files were created
            assert os.path.exists(temp_dir)
            files = sorted([f for f in os.listdir(temp_dir) if f.endswith(".tif")])
            assert len(files) == 5  # Should be 5 Z-slices
            assert files[0] == "test_z0000.tif"
            assert files[-1] == "test_z0004.tif"
            assert files[0] == "test_z0000.tif"
            assert files[-1] == "test_z0004.tif"

            # Verify file contents
            try:
                import tifffile

                for i, filename in enumerate(files):
                    filepath = os.path.join(temp_dir, filename)
                    loaded_slice = tifffile.imread(filepath)
                    # Expected slice should be data[0, i, :, :] since data has shape (C=1, Z=5, Y=32, X=32)
                    expected_slice = data[0, i, :, :]
                    np.testing.assert_array_almost_equal(loaded_slice, expected_slice)
            except ImportError:
                pytest.skip("tifffile not available for verification")

    def test_tiff_stack_default_uint16_rescaling(self):
        """Test that default behavior uses uint16 with rescaling."""
        # Create 4D test data
        data = np.random.rand(1, 3, 16, 16).astype(np.float32)
        dask_data = da.from_array(data, chunks=(1, 2, 8, 8))
        znii = ZarrNii.from_darr(dask_data, spacing=(1.0, 1.0, 1.0))

        with tempfile.TemporaryDirectory() as temp_dir:
            pattern = os.path.join(temp_dir, "default_z{z:04d}.tif")
            # Use default parameters (should be uint16 with rescaling)
            znii.to_tiff_stack(pattern)

            files = sorted([f for f in os.listdir(temp_dir) if f.endswith(".tif")])
            assert len(files) == 3

            # Verify default behavior produces uint16 files
            try:
                import tifffile

                loaded_slice = tifffile.imread(os.path.join(temp_dir, files[0]))
                assert loaded_slice.dtype == np.uint16
                # Should use full uint16 range due to rescaling
                assert np.max(loaded_slice) > 30000  # Should be well into uint16 range
            except ImportError:
                pytest.skip("tifffile not available for verification")

    def test_tiff_stack_with_multichannel_data(self):
        """Test TIFF stack export with multichannel 4D data."""
        # Create 4D test data (C=3, Z=4, Y=16, X=16)
        data = np.random.rand(3, 4, 16, 16).astype(np.float32)
        dask_data = da.from_array(data, chunks=(1, 2, 8, 8))
        znii = ZarrNii.from_darr(dask_data, spacing=(1.0, 1.0, 1.0))

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test saving all channels as multichannel TIFFs
            pattern = os.path.join(temp_dir, "multichannel_z{z:04d}.tif")
            znii.to_tiff_stack(pattern)

            files = sorted([f for f in os.listdir(temp_dir) if f.endswith(".tif")])
            assert len(files) == 4  # 4 Z-slices

            # Test saving specific channel
            pattern2 = os.path.join(temp_dir, "channel0_z{z:04d}.tif")
            znii.to_tiff_stack(pattern2, channel=0)

            channel_files = sorted(
                [f for f in os.listdir(temp_dir) if f.startswith("channel0")]
            )
            assert len(channel_files) == 4

    def test_tiff_stack_with_5d_data_single_timepoint(self):
        """Test TIFF stack export with 5D data selecting single timepoint."""
        # Create 5D test data (T, C, Z, Y, X)
        data = np.random.rand(2, 3, 4, 16, 16).astype(np.float32)
        dask_data = da.from_array(data, chunks=(1, 1, 2, 8, 8))
        znii = ZarrNii.from_darr(
            dask_data, spacing=(1.0, 1.0, 1.0), dims=["t", "c", "z", "y", "x"]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            pattern = os.path.join(temp_dir, "t0_z{z:04d}.tif")
            znii.to_tiff_stack(pattern, timepoint=0)

            files = sorted([f for f in os.listdir(temp_dir) if f.endswith(".tif")])
            assert len(files) == 4  # 4 Z-slices

    def test_tiff_stack_automatic_pattern(self):
        """Test TIFF stack export with automatic pattern generation."""
        # Create 4D test data (C=1, Z=3, Y=16, X=16)
        data = np.random.rand(1, 3, 16, 16).astype(np.float32)
        dask_data = da.from_array(data, chunks=(1, 2, 8, 8))
        znii = ZarrNii.from_darr(dask_data, spacing=(1.0, 1.0, 1.0))

        with tempfile.TemporaryDirectory() as temp_dir:
            # Pattern without z format specifier
            pattern = os.path.join(temp_dir, "output.tif")
            znii.to_tiff_stack(pattern)

            files = sorted([f for f in os.listdir(temp_dir) if f.endswith(".tif")])
            assert len(files) == 3
            assert files[0] == "output_0000.tif"
            assert files[-1] == "output_0002.tif"

    def test_tiff_stack_compression_options(self):
        """Test TIFF stack with compression options."""
        # Create 4D test data (C=1, Z=2, Y=16, X=16)
        data = np.random.rand(1, 2, 16, 16).astype(np.float32)
        dask_data = da.from_array(data, chunks=(1, 1, 8, 8))
        znii = ZarrNii.from_darr(dask_data, spacing=(1.0, 1.0, 1.0))

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with compression (default)
            pattern1 = os.path.join(temp_dir, "compressed_z{z:04d}.tif")
            znii.to_tiff_stack(pattern1, compress=True)

            # Test without compression
            pattern2 = os.path.join(temp_dir, "uncompressed_z{z:04d}.tif")
            znii.to_tiff_stack(pattern2, compress=False)

            compressed_files = [
                f for f in os.listdir(temp_dir) if f.startswith("compressed")
            ]
            uncompressed_files = [
                f for f in os.listdir(temp_dir) if f.startswith("uncompressed")
            ]

            assert len(compressed_files) == 2
            assert len(uncompressed_files) == 2

    def test_tiff_stack_datatype_conversion(self):
        """Test TIFF stack export with different data types."""
        # Create 4D test data with float values in range [0, 1000]
        data = np.random.rand(1, 3, 16, 16).astype(np.float32) * 1000
        dask_data = da.from_array(data, chunks=(1, 2, 8, 8))
        znii = ZarrNii.from_darr(dask_data, spacing=(1.0, 1.0, 1.0))

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test uint8 conversion with rescaling
            pattern_uint8 = os.path.join(temp_dir, "uint8_z{z:04d}.tif")
            znii.to_tiff_stack(pattern_uint8, dtype="uint8", rescale=True)

            # Test uint16 conversion with rescaling (default)
            pattern_uint16 = os.path.join(temp_dir, "uint16_z{z:04d}.tif")
            znii.to_tiff_stack(pattern_uint16, dtype="uint16", rescale=True)

            # Test int16 conversion
            pattern_int16 = os.path.join(temp_dir, "int16_z{z:04d}.tif")
            znii.to_tiff_stack(pattern_int16, dtype="int16", rescale=True)

            # Test float32 (no conversion)
            pattern_float32 = os.path.join(temp_dir, "float32_z{z:04d}.tif")
            znii.to_tiff_stack(pattern_float32, dtype="float32", rescale=False)

            # Verify all files were created
            uint8_files = [f for f in os.listdir(temp_dir) if f.startswith("uint8")]
            uint16_files = [f for f in os.listdir(temp_dir) if f.startswith("uint16")]
            int16_files = [f for f in os.listdir(temp_dir) if f.startswith("int16")]
            float32_files = [f for f in os.listdir(temp_dir) if f.startswith("float32")]

            assert len(uint8_files) == 3
            assert len(uint16_files) == 3
            assert len(int16_files) == 3
            assert len(float32_files) == 3

            # Verify data types if tifffile is available
            try:
                import tifffile

                uint8_data = tifffile.imread(os.path.join(temp_dir, uint8_files[0]))
                uint16_data = tifffile.imread(os.path.join(temp_dir, uint16_files[0]))
                int16_data = tifffile.imread(os.path.join(temp_dir, int16_files[0]))
                float32_data = tifffile.imread(os.path.join(temp_dir, float32_files[0]))

                assert uint8_data.dtype == np.uint8
                assert uint16_data.dtype == np.uint16
                assert int16_data.dtype == np.int16
                assert float32_data.dtype == np.float32

                # Verify rescaling worked (uint8 should use full 0-255 range)
                assert np.min(uint8_data) < 50  # Should be close to 0
                assert np.max(uint8_data) > 200  # Should be close to 255
            except ImportError:
                pytest.skip("tifffile not available for verification")

    def test_tiff_stack_rescaling_options(self):
        """Test TIFF stack export with different rescaling options."""
        # Create data with known range [100, 900]
        data = np.full((1, 2, 8, 8), 100.0, dtype=np.float32)
        data[0, 1, :, :] = 900.0  # Second slice has higher values
        dask_data = da.from_array(data, chunks=(1, 1, 4, 4))
        znii = ZarrNii.from_darr(dask_data, spacing=(1.0, 1.0, 1.0))

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with rescaling (default)
            pattern_rescale = os.path.join(temp_dir, "rescale_z{z:04d}.tif")
            znii.to_tiff_stack(pattern_rescale, dtype="uint16", rescale=True)

            # Test without rescaling (clipping)
            pattern_no_rescale = os.path.join(temp_dir, "no_rescale_z{z:04d}.tif")
            znii.to_tiff_stack(pattern_no_rescale, dtype="uint16", rescale=False)

            # Verify files were created
            rescale_files = [f for f in os.listdir(temp_dir) if f.startswith("rescale")]
            no_rescale_files = [
                f for f in os.listdir(temp_dir) if f.startswith("no_rescale")
            ]

            assert len(rescale_files) == 2
            assert len(no_rescale_files) == 2

            try:
                import tifffile

                # Check rescaled data - should use full uint16 range
                rescaled_slice0 = tifffile.imread(
                    os.path.join(temp_dir, "rescale_z0000.tif")
                )
                rescaled_slice1 = tifffile.imread(
                    os.path.join(temp_dir, "rescale_z0001.tif")
                )

                # With rescaling, slice 0 (value 100) should map to 0, slice 1 (value 900) to 65535
                assert rescaled_slice0[0, 0] == 0  # Min value maps to 0
                assert rescaled_slice1[0, 0] == 65535  # Max value maps to 65535

                # Check non-rescaled data - should be clipped to original values
                no_rescale_slice0 = tifffile.imread(
                    os.path.join(temp_dir, "no_rescale_z0000.tif")
                )
                no_rescale_slice1 = tifffile.imread(
                    os.path.join(temp_dir, "no_rescale_z0001.tif")
                )

                # Without rescaling, values should be clipped to dtype range
                assert no_rescale_slice0[0, 0] == 100  # Original value preserved
                assert no_rescale_slice1[0, 0] == 900  # Original value preserved

            except ImportError:
                pytest.skip("tifffile not available for verification")

    def test_tiff_stack_unsupported_dtype(self):
        """Test error handling for unsupported data types."""
        data = np.random.rand(1, 2, 8, 8).astype(np.float32)
        dask_data = da.from_array(data, chunks=(1, 1, 4, 4))
        znii = ZarrNii.from_darr(dask_data, spacing=(1.0, 1.0, 1.0))

        with tempfile.TemporaryDirectory() as temp_dir:
            pattern = os.path.join(temp_dir, "test_z{z:04d}.tif")

            # Should raise ValueError for unsupported dtype
            with pytest.raises(ValueError, match="Unsupported dtype"):
                znii.to_tiff_stack(pattern, dtype="int32")

    def test_tiff_stack_error_conditions(self):
        """Test error conditions for TIFF stack export."""
        # 5D data without timepoint selection
        data_5d = np.random.rand(2, 3, 4, 16, 16).astype(np.float32)
        dask_data_5d = da.from_array(data_5d, chunks=(1, 1, 2, 8, 8))
        znii_5d = ZarrNii.from_darr(
            dask_data_5d, spacing=(1.0, 1.0, 1.0), dims=["t", "c", "z", "y", "x"]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            pattern = os.path.join(temp_dir, "test_z{z:04d}.tif")

            # Should raise ValueError for multiple timepoints without selection
            with pytest.raises(ValueError, match="Must specify 'timepoint' parameter"):
                znii_5d.to_tiff_stack(pattern)

            # Should raise ValueError for out-of-range timepoint
            with pytest.raises(ValueError, match="Timepoint .* is out of range"):
                znii_5d.to_tiff_stack(pattern, timepoint=5)

            # Should raise ValueError for out-of-range channel
            with pytest.raises(ValueError, match="Channel .* is out of range"):
                znii_5d.to_tiff_stack(pattern, timepoint=0, channel=5)

    def test_tiff_stack_without_z_dimension(self):
        """Test error when data doesn't have Z dimension."""
        # Create 2D data by making a degenerate case with Z=1
        data = np.random.rand(1, 1, 32, 32).astype(np.float32)  # C=1, Z=1, Y=32, X=32
        dask_data = da.from_array(data, chunks=(1, 1, 16, 16))
        znii = ZarrNii.from_darr(dask_data, spacing=(1.0, 1.0, 1.0))

        # Now manually modify the znii to have no Z dimension for testing
        # This is a bit artificial but tests the error condition
        import ngff_zarr as nz

        # Create data without Z dimension by using only 2D spatial
        data_2d = np.random.rand(1, 32, 32).astype(np.float32)  # C, Y, X
        dask_data_2d = da.from_array(data_2d, chunks=(1, 16, 16))
        ngff_image_2d = nz.NgffImage(
            data=dask_data_2d,
            dims=["c", "y", "x"],  # No Z dimension
            scale={"y": 1.0, "x": 1.0},
            translation={"y": 0.0, "x": 0.0},
            name="test_2d",
        )
        znii_2d = ZarrNii.from_ngff_image(ngff_image_2d)

        with tempfile.TemporaryDirectory() as temp_dir:
            pattern = os.path.join(temp_dir, "test_z{z:04d}.tif")

            with pytest.raises(ValueError, match="Data must have a Z dimension"):
                znii_2d.to_tiff_stack(pattern)

    def test_tiff_stack_directory_creation(self):
        """Test that output directories are created automatically."""
        # Create 4D test data (C=1, Z=2, Y=16, X=16)
        data = np.random.rand(1, 2, 16, 16).astype(np.float32)
        dask_data = da.from_array(data, chunks=(1, 1, 8, 8))
        znii = ZarrNii.from_darr(dask_data, spacing=(1.0, 1.0, 1.0))

        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = os.path.join(temp_dir, "subdir", "nested")
            pattern = os.path.join(nested_dir, "test_z{z:04d}.tif")

            output_dir = znii.to_tiff_stack(pattern)

            assert os.path.exists(nested_dir)
            files = os.listdir(nested_dir)
            assert len(files) == 2
            assert output_dir == nested_dir

    def test_tiff_stack_import_error_handling(self):
        """Test handling when tifffile is not available."""
        # Create 4D test data (C=1, Z=2, Y=16, X=16)
        data = np.random.rand(1, 2, 16, 16).astype(np.float32)
        dask_data = da.from_array(data, chunks=(1, 1, 8, 8))
        znii = ZarrNii.from_darr(dask_data, spacing=(1.0, 1.0, 1.0))

        # Mock the import to fail
        import sys

        original_tifffile = sys.modules.get("tifffile")
        if "tifffile" in sys.modules:
            del sys.modules["tifffile"]

        # Temporarily remove tifffile from path
        import importlib.util

        original_find_spec = importlib.util.find_spec

        def mock_find_spec(name):
            if name == "tifffile":
                return None
            return original_find_spec(name)

        importlib.util.find_spec = mock_find_spec

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                pattern = os.path.join(temp_dir, "test_z{z:04d}.tif")

                with pytest.raises(ImportError, match="tifffile is required"):
                    znii.to_tiff_stack(pattern)
        finally:
            # Restore original state
            importlib.util.find_spec = original_find_spec
            if original_tifffile:
                sys.modules["tifffile"] = original_tifffile

    def test_tiff_stack_singleton_dimensions(self):
        """Test TIFF stack with singleton time/channel dimensions."""
        # Create 5D data with singleton dimensions (T=1, C=1, Z=3, Y=16, X=16)
        data = np.random.rand(1, 1, 3, 16, 16).astype(np.float32)
        dask_data = da.from_array(data, chunks=(1, 1, 2, 8, 8))
        znii = ZarrNii.from_darr(
            dask_data, spacing=(1.0, 1.0, 1.0), dims=["t", "c", "z", "y", "x"]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            pattern = os.path.join(temp_dir, "singleton_z{z:04d}.tif")
            znii.to_tiff_stack(pattern)

            files = sorted([f for f in os.listdir(temp_dir) if f.endswith(".tif")])
            assert len(files) == 3  # 3 Z-slices

            # Verify the data shape is correct (should be 2D per slice)
            try:
                import tifffile

                loaded_slice = tifffile.imread(os.path.join(temp_dir, files[0]))
                assert loaded_slice.shape == (16, 16)  # Should be 2D
            except ImportError:
                pytest.skip("tifffile not available for verification")
