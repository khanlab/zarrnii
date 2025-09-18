"""Tests for Imaris I/O functionality."""

import os
import tempfile

import dask.array as da
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from zarrnii import ZarrNii

# Skip all tests if h5py is not available
h5py = pytest.importorskip("h5py", reason="h5py required for Imaris support")


@pytest.fixture
def sample_3d_data():
    """Create sample 3D data for testing."""
    return np.random.rand(64, 128, 96).astype(np.float32)


@pytest.fixture
def sample_imaris_file(tmp_path, sample_3d_data):
    """Create a sample Imaris file for testing."""
    imaris_path = tmp_path / "test_sample.ims"

    # Create a basic Imaris file structure
    with h5py.File(str(imaris_path), "w") as f:
        # Set file attributes
        f.attrs["ImarisVersion"] = "9.0.0"
        f.attrs["ImarisDataSet"] = "ImarisDataSet"
        f.attrs["ImageSizeX"] = 96.0
        f.attrs["ImageSizeY"] = 128.0
        f.attrs["ImageSizeZ"] = 64.0

        # Create dataset structure
        dataset_group = f.create_group("DataSet")
        res_group = dataset_group.create_group("ResolutionLevel 0")
        time_group = res_group.create_group("TimePoint 0")
        channel_group = time_group.create_group("Channel 0")

        # Save the data
        channel_group.create_dataset("Data", data=sample_3d_data, compression="gzip")

        # Add basic metadata
        info_group = f.create_group("DataSetInfo")
        info_group.create_group("Image")
        channel_info_group = info_group.create_group("Channel 0")
        channel_info_group.attrs["Name"] = "Test Channel"

        time_info_group = f.create_group("DataSetTimes")
        time_info_group.create_dataset("Time", data=[0.0])

    return str(imaris_path)


class TestImarisIO:
    """Test Imaris I/O functionality."""

    def test_import_error_without_h5py(self, monkeypatch):
        """Test that appropriate error is raised when h5py is not available."""

        # Mock h5py import to fail
        def mock_import(name, *args, **kwargs):
            if name == "h5py":
                raise ImportError("No module named 'h5py'")
            return __import__(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)

        with pytest.raises(ImportError, match="h5py is required for Imaris support"):
            ZarrNii.from_imaris("dummy_path.ims")

    def test_from_imaris_basic(self, sample_imaris_file, sample_3d_data):
        """Test basic loading from Imaris file."""
        znimg = ZarrNii.from_imaris(sample_imaris_file)

        # Check basic properties
        assert znimg is not None
        assert hasattr(znimg, "darr")
        assert hasattr(znimg, "axes_order")
        assert znimg.axes_order == "ZYX"  # Default

        # Check data shape (should have channel dimension added)
        expected_shape = (1,) + sample_3d_data.shape
        assert znimg.darr.shape == expected_shape

        # Check data content
        loaded_data = znimg.darr.compute()
        assert_array_almost_equal(loaded_data[0], sample_3d_data)

    def test_from_imaris_invalid_file(self, tmp_path):
        """Test error handling for invalid Imaris file."""
        # Create a file that's not an Imaris file
        invalid_file = tmp_path / "invalid.ims"
        with h5py.File(str(invalid_file), "w") as f:
            f.create_dataset("dummy", data=[1, 2, 3])

        with pytest.raises(
            ValueError, match="does not appear to be a valid Imaris file"
        ):
            ZarrNii.from_imaris(str(invalid_file))

    def test_from_imaris_nonexistent_file(self):
        """Test error handling for nonexistent file."""
        with pytest.raises(OSError):
            ZarrNii.from_imaris("nonexistent_file.ims")

    def test_from_imaris_invalid_level(self, sample_imaris_file):
        """Test error handling for invalid resolution level."""
        with pytest.raises(ValueError, match="Level 5 not available"):
            ZarrNii.from_imaris(sample_imaris_file, level=5)

    def test_from_imaris_invalid_timepoint(self, sample_imaris_file):
        """Test error handling for invalid timepoint."""
        with pytest.raises(ValueError, match="Timepoint 5 not available"):
            ZarrNii.from_imaris(sample_imaris_file, timepoint=5)

    def test_from_imaris_invalid_channel(self, sample_imaris_file):
        """Test error handling for invalid channel."""
        with pytest.raises(ValueError, match="Channel 5 not available"):
            ZarrNii.from_imaris(sample_imaris_file, channel=5)

    def test_from_imaris_custom_parameters(self, sample_imaris_file):
        """Test loading with custom parameters."""
        znimg = ZarrNii.from_imaris(
            sample_imaris_file, axes_order="XYZ", orientation="LPI"
        )

        assert znimg.axes_order == "XYZ"
        assert znimg.orientation == "LPI"

    @pytest.mark.usefixtures("cleandir")
    def test_to_imaris_basic(self, sample_3d_data):
        """Test basic saving to Imaris file."""
        # Create ZarrNii instance
        darr = da.from_array(sample_3d_data[np.newaxis, ...], chunks="auto")
        znimg = ZarrNii.from_darr(darr, spacing=[1.0, 1.0, 1.0])

        # Save to Imaris
        output_path = "test_output.ims"
        result_path = znimg.to_imaris(output_path)

        assert result_path == output_path
        assert os.path.exists(output_path)

        # Verify the file can be read back
        znimg_reloaded = ZarrNii.from_imaris(output_path)
        assert znimg_reloaded.darr.shape == znimg.darr.shape

        # Check data content (allowing for small numerical differences due to HDF5 I/O)
        original_data = znimg.darr.compute()
        reloaded_data = znimg_reloaded.darr.compute()
        assert_array_almost_equal(original_data, reloaded_data, decimal=5)

    @pytest.mark.usefixtures("cleandir")
    def test_to_imaris_auto_extension(self, sample_3d_data):
        """Test that .ims extension is automatically added."""
        darr = da.from_array(sample_3d_data[np.newaxis, ...], chunks="auto")
        znimg = ZarrNii.from_darr(darr, spacing=[1.0, 1.0, 1.0])

        # Save without extension
        result_path = znimg.to_imaris("test_output")

        assert result_path == "test_output.ims"
        assert os.path.exists("test_output.ims")

    @pytest.mark.usefixtures("cleandir")
    def test_to_imaris_compression_options(self, sample_3d_data):
        """Test saving with different compression options."""
        darr = da.from_array(sample_3d_data[np.newaxis, ...], chunks="auto")
        znimg = ZarrNii.from_darr(darr, spacing=[1.0, 1.0, 1.0])

        # Save with different compression (using 'szip' which doesn't take options)
        result_path = znimg.to_imaris(
            "test_output.ims", compression="gzip", compression_opts=1
        )

        assert os.path.exists(result_path)

        # Verify the file structure
        with h5py.File(result_path, "r") as f:
            assert "DataSet" in f
            data_dataset = f["DataSet"]["ResolutionLevel 0"]["TimePoint 0"][
                "Channel 0"
            ]["Data"]
            assert data_dataset.compression == "gzip"
            assert data_dataset.compression_opts == 1

    def test_to_imaris_import_error(self, monkeypatch, sample_3d_data):
        """Test that appropriate error is raised when h5py is not available for saving."""
        darr = da.from_array(sample_3d_data[np.newaxis, ...], chunks="auto")
        znimg = ZarrNii.from_darr(darr, spacing=[1.0, 1.0, 1.0])

        # Mock h5py import to fail
        def mock_import(name, *args, **kwargs):
            if name == "h5py":
                raise ImportError("No module named 'h5py'")
            return __import__(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)

        with pytest.raises(ImportError, match="h5py is required for Imaris support"):
            znimg.to_imaris("dummy_path.ims")

    @pytest.mark.usefixtures("cleandir")
    def test_round_trip_imaris(self, sample_3d_data):
        """Test round-trip: create -> save -> load -> compare."""
        # Create original ZarrNii
        darr = da.from_array(sample_3d_data[np.newaxis, ...], chunks="auto")
        znimg_original = ZarrNii.from_darr(darr, spacing=[2.0, 1.5, 1.0])

        # Save to Imaris
        imaris_path = "test_roundtrip.ims"
        znimg_original.to_imaris(imaris_path)

        # Load back from Imaris
        znimg_loaded = ZarrNii.from_imaris(imaris_path)

        # Compare shapes
        assert znimg_loaded.darr.shape == znimg_original.darr.shape

        # Compare data (allowing for small numerical differences)
        original_data = znimg_original.darr.compute()
        loaded_data = znimg_loaded.darr.compute()
        assert_array_almost_equal(original_data, loaded_data, decimal=5)

    def test_imaris_metadata_extraction(self, tmp_path, sample_3d_data):
        """Test extraction of spatial metadata from Imaris file."""
        imaris_path = tmp_path / "test_metadata.ims"

        # Create Imaris file with specific metadata
        with h5py.File(str(imaris_path), "w") as f:
            f.attrs["ImarisVersion"] = "9.0.0"
            f.attrs["ImarisDataSet"] = "ImarisDataSet"
            f.attrs["ImageSizeX"] = 192.0  # 96 * 2.0 spacing
            f.attrs["ImageSizeY"] = 192.0  # 128 * 1.5 spacing
            f.attrs["ImageSizeZ"] = 128.0  # 64 * 2.0 spacing

            dataset_group = f.create_group("DataSet")
            res_group = dataset_group.create_group("ResolutionLevel 0")
            time_group = res_group.create_group("TimePoint 0")
            channel_group = time_group.create_group("Channel 0")
            channel_group.create_dataset("Data", data=sample_3d_data)

            # Add basic info groups
            info_group = f.create_group("DataSetInfo")
            info_group.create_group("Image")
            time_info_group = f.create_group("DataSetTimes")
            time_info_group.create_dataset("Time", data=[0.0])

        # Load and check spacing calculation
        znimg = ZarrNii.from_imaris(str(imaris_path))

        # The spacing should be calculated from ImageSize attributes
        zooms = znimg.get_zooms(axes_order="ZYX")
        expected_zooms = [128.0 / 64, 192.0 / 128, 192.0 / 96]  # [Z, Y, X]
        assert_array_almost_equal(zooms, expected_zooms, decimal=3)


class TestImarisIntegration:
    """Test integration with other ZarrNii functionality."""

    @pytest.mark.usefixtures("cleandir")
    def test_imaris_to_nifti_conversion(self, sample_imaris_file, sample_3d_data):
        """Test converting from Imaris to NIfTI."""
        # Load from Imaris
        znimg = ZarrNii.from_imaris(sample_imaris_file)

        # Convert to NIfTI
        nifti_path = "converted.nii"
        znimg.to_nifti(nifti_path)

        assert os.path.exists(nifti_path)

        # Load back from NIfTI and compare
        znimg_nifti = ZarrNii.from_nifti(nifti_path)

        # The shapes might differ due to axis reordering, but volume should be same
        assert np.prod(znimg_nifti.darr.shape) == np.prod(znimg.darr.shape)

    @pytest.mark.usefixtures("cleandir")
    def test_nifti_to_imaris_conversion(self, nifti_nib):
        """Test converting from NIfTI to Imaris."""
        # Save NIfTI file
        nifti_path = "test.nii"
        nifti_nib.to_filename(nifti_path)

        # Load from NIfTI
        znimg = ZarrNii.from_nifti(nifti_path)

        # Convert to Imaris
        imaris_path = "converted.ims"
        znimg.to_imaris(imaris_path)

        assert os.path.exists(imaris_path)

        # Load back from Imaris and compare
        znimg_imaris = ZarrNii.from_imaris(imaris_path)

        # The shapes might differ due to axis reordering, but volume should be same
        assert np.prod(znimg_imaris.darr.shape) == np.prod(znimg.darr.shape)

    @pytest.mark.usefixtures("cleandir")
    def test_imaris_with_transformations(self, sample_imaris_file):
        """Test applying transformations to Imaris-loaded data."""
        # Load from Imaris
        znimg = ZarrNii.from_imaris(sample_imaris_file)

        # Apply some transformations
        cropped = znimg.crop((5, 5, 5), (50, 100, 80))
        downsampled = cropped.downsample(level=1)

        # Save the transformed result back to Imaris
        output_path = "transformed.ims"
        downsampled.to_imaris(output_path)

        assert os.path.exists(output_path)

        # Verify we can load the transformed result
        znimg_transformed = ZarrNii.from_imaris(output_path)
        assert znimg_transformed is not None
        assert (
            znimg_transformed.darr.shape[1:] == downsampled.darr.shape[1:]
        )  # Compare spatial dimensions
