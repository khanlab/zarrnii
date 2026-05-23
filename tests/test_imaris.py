"""Tests for Imaris I/O functionality."""

import os
import sys
import tempfile

import dask.array as da
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from zarrnii import (
    ZarrNii,
    get_dask_client,
    get_imaris_scale_factors,
    get_scale_factors_from_file,
)

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
    darr = da.from_array(sample_3d_data[np.newaxis, ...], chunks="auto")
    ZarrNii.from_darr(darr, spacing=[1.0, 1.0, 1.0]).to_imaris(str(imaris_path))

    return str(imaris_path)


@pytest.fixture
def sample_multichannel_imaris_file(tmp_path):
    """Create a multi-channel Imaris file for channel selection tests."""
    imaris_path = tmp_path / "test_multichannel.ims"
    # Shape: (channels=3, z=16, y=32, x=24)
    sample_data = np.random.rand(3, 16, 32, 24).astype(np.float32)
    darr = da.from_array(sample_data, chunks="auto")
    ZarrNii.from_darr(darr, spacing=[1.0, 1.0, 1.0]).to_imaris(str(imaris_path))
    return str(imaris_path), sample_data


class TestImarisIO:
    """Test Imaris I/O functionality."""

    def test_import_error_without_imaris_reader(self, monkeypatch, tmp_path):
        """Test that appropriate error is raised when imaris reader is unavailable."""
        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "imaris_ims_zarr" or name.startswith("imaris_ims_zarr."):
                raise ImportError("No module named 'imaris_ims_zarr'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)
        monkeypatch.delitem(sys.modules, "imaris_ims_zarr", raising=False)
        dummy_path = tmp_path / "dummy.ims"
        dummy_path.write_bytes(b"")

        with pytest.raises(
            ImportError, match="imaris_ims_zarr is required for Imaris support"
        ):
            ZarrNii.from_imaris(str(dummy_path))

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

        with pytest.raises(ValueError, match="Unable to read Imaris file"):
            ZarrNii.from_imaris(str(invalid_file))

    def test_from_imaris_nonexistent_file(self):
        """Test error handling for nonexistent file."""
        with pytest.raises(FileNotFoundError, match="file does not exist"):
            ZarrNii.from_imaris("nonexistent_file.ims")

    def test_from_imaris_invalid_level(self, sample_imaris_file, sample_3d_data):
        """Test that exceeding available levels applies lazy downsampling."""
        # sample_imaris_file has only 1 level (level 0), so level=2 should
        # apply 2^2=4x lazy downsampling instead of raising ValueError
        znimg = ZarrNii.from_imaris(sample_imaris_file, level=2)
        orig_shape = sample_3d_data.shape  # (64, 128, 96)
        ds_shape = znimg.darr.shape[1:]  # spatial dims (z, y, x)
        assert ds_shape[0] == orig_shape[0] // 4
        assert ds_shape[1] == orig_shape[1] // 4
        assert ds_shape[2] == orig_shape[2] // 4

    def test_from_imaris_negative_level_raises(self, sample_imaris_file):
        """Test that a negative level raises ValueError."""
        with pytest.raises(ValueError, match="Level must be >= 0"):
            ZarrNii.from_imaris(sample_imaris_file, level=-1)

    def test_from_imaris_invalid_timepoint(self, sample_imaris_file):
        """Test error handling for invalid timepoint."""
        with pytest.raises(ValueError, match="Timepoint 5 not available"):
            ZarrNii.from_imaris(sample_imaris_file, timepoint=5)

    def test_from_imaris_invalid_channel(self, sample_imaris_file):
        """Test error handling for invalid channel."""
        with pytest.raises(ValueError, match="Channel index 5 not available"):
            ZarrNii.from_imaris(sample_imaris_file, channels=[5])

    def test_from_imaris_select_channels(self, sample_multichannel_imaris_file):
        """Test selecting multiple channels from Imaris data."""
        imaris_path, sample_data = sample_multichannel_imaris_file
        znimg = ZarrNii.from_imaris(imaris_path, channels=[1, 2])
        loaded_data = znimg.darr.compute()
        assert loaded_data.shape[0] == 2
        assert_array_almost_equal(loaded_data[0], sample_data[1], decimal=5)
        assert_array_almost_equal(loaded_data[1], sample_data[2], decimal=5)

    def test_from_imaris_channel_labels_require_set_channel_labels(
        self, sample_multichannel_imaris_file
    ):
        """Test that channel label selection requires set_channel_labels."""
        imaris_path, _ = sample_multichannel_imaris_file
        with pytest.raises(
            ValueError,
            match="'set_channel_labels' is required when 'channel_labels' is provided",
        ):
            ZarrNii.from_imaris(imaris_path, channel_labels=["GFP"])

    def test_from_imaris_select_channel_labels(self, sample_multichannel_imaris_file):
        """Test selecting channels by labels with explicit source labels."""
        imaris_path, sample_data = sample_multichannel_imaris_file
        znimg = ZarrNii.from_imaris(
            imaris_path,
            channel_labels=["GFP", "DAPI"],
            set_channel_labels=["DAPI", "GFP", "RFP"],
        )
        loaded_data = znimg.darr.compute()
        assert loaded_data.shape[0] == 2
        assert_array_almost_equal(loaded_data[0], sample_data[1], decimal=5)
        assert_array_almost_equal(loaded_data[1], sample_data[0], decimal=5)
        assert znimg.list_channels() == ["GFP", "DAPI"]

    def test_from_imaris_set_channel_labels_without_selection(
        self, sample_multichannel_imaris_file
    ):
        """Test setting source labels without channel filtering."""
        imaris_path, _ = sample_multichannel_imaris_file
        znimg = ZarrNii.from_imaris(
            imaris_path,
            set_channel_labels=["DAPI", "GFP", "RFP"],
        )
        assert znimg.list_channels() == ["DAPI", "GFP", "RFP"]

    def test_from_imaris_custom_parameters(self, sample_imaris_file):
        """Test loading with custom parameters."""
        znimg = ZarrNii.from_imaris(
            sample_imaris_file, axes_order="XYZ", orientation="LPI"
        )

        assert znimg.axes_order == "XYZ"
        assert znimg.orientation == "LPI"

    def test_from_imaris_downsample_near_isotropic_deprecated(self, sample_imaris_file):
        """Test that downsample_near_isotropic emits a DeprecationWarning."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ZarrNii.from_imaris(sample_imaris_file, downsample_near_isotropic=True)
        assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
        assert any("downsample_near_isotropic" in str(warning.message) for warning in w)

    def test_from_imaris_to_ome_zarr_with_distributed_scheduler(
        self, sample_imaris_file, tmp_path
    ):
        """Test Imaris-backed arrays can be written under distributed scheduler."""
        pytest.importorskip("dask.distributed", reason="dask.distributed not installed")

        output_path = tmp_path / "from_imaris_distributed.ome.zarr"
        znimg = ZarrNii.from_imaris(sample_imaris_file)
        with get_dask_client("distributed", threads=2, threads_per_worker=1) as client:
            assert client.cluster.processes is True
            znimg.to_ome_zarr(str(output_path), max_layer=0)

        reloaded = ZarrNii.from_ome_zarr(str(output_path))
        assert reloaded.darr.shape == znimg.darr.shape

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
        expected_zooms = [2.0, 1.5, 1.0]  # [Z, Y, X]
        darr = da.from_array(sample_3d_data[np.newaxis, ...], chunks="auto")
        ZarrNii.from_darr(darr, spacing=expected_zooms).to_imaris(str(imaris_path))

        # Load and check spacing calculation
        znimg = ZarrNii.from_imaris(str(imaris_path))

        # The spacing should be loaded from Imaris extent metadata
        zooms = znimg.get_zooms(axes_order="ZYX")
        assert_array_almost_equal(zooms, expected_zooms, decimal=3)

    def test_malformed_imaris_files(self, tmp_path):
        """Test error handling for various malformed Imaris files."""

        # Test file with no resolution levels
        imaris_path_1 = tmp_path / "no_reslevels.ims"
        with h5py.File(str(imaris_path_1), "w") as f:
            f.attrs["ImarisVersion"] = "9.0.0"
            dataset_group = f.create_group("DataSet")
            # Don't add any ResolutionLevel groups

        with pytest.raises(ValueError, match="Unable to read Imaris file"):
            ZarrNii.from_imaris(str(imaris_path_1))

        # Test file with no timepoints
        imaris_path_2 = tmp_path / "no_timepoints.ims"
        with h5py.File(str(imaris_path_2), "w") as f:
            f.attrs["ImarisVersion"] = "9.0.0"
            dataset_group = f.create_group("DataSet")
            res_group = dataset_group.create_group("ResolutionLevel 0")
            # Don't add any TimePoint groups

        with pytest.raises(ValueError, match="Unable to read Imaris file"):
            ZarrNii.from_imaris(str(imaris_path_2))

        # Test file with no channels
        imaris_path_3 = tmp_path / "no_channels.ims"
        with h5py.File(str(imaris_path_3), "w") as f:
            f.attrs["ImarisVersion"] = "9.0.0"
            dataset_group = f.create_group("DataSet")
            res_group = dataset_group.create_group("ResolutionLevel 0")
            time_group = res_group.create_group("TimePoint 0")
            # Don't add any Channel groups

        with pytest.raises(ValueError, match="Unable to read Imaris file"):
            ZarrNii.from_imaris(str(imaris_path_3))

        # Test file with missing Data dataset
        imaris_path_4 = tmp_path / "no_data.ims"
        with h5py.File(str(imaris_path_4), "w") as f:
            f.attrs["ImarisVersion"] = "9.0.0"
            dataset_group = f.create_group("DataSet")
            res_group = dataset_group.create_group("ResolutionLevel 0")
            time_group = res_group.create_group("TimePoint 0")
            channel_group = time_group.create_group("Channel 0")
            # Don't add Data dataset

        with pytest.raises(ValueError, match="Unable to read Imaris file"):
            ZarrNii.from_imaris(str(imaris_path_4))

        # Test inconsistent structure (ResolutionLevel exists but key doesn't match)
        imaris_path_5 = tmp_path / "inconsistent.ims"
        with h5py.File(str(imaris_path_5), "w") as f:
            f.attrs["ImarisVersion"] = "9.0.0"
            dataset_group = f.create_group("DataSet")
            # Create a group that would be detected by startswith but doesn't match exact key
            res_group = dataset_group.create_group(
                "ResolutionLevel 1"
            )  # But we'll try to access level 0
            time_group = res_group.create_group("TimePoint 0")
            channel_group = time_group.create_group("Channel 0")
            channel_group.create_dataset("Data", data=np.ones((10, 10, 10)))

        with pytest.raises(ValueError, match="Unable to read Imaris file"):
            ZarrNii.from_imaris(str(imaris_path_5), level=0)

        # Test inconsistent timepoint structure
        imaris_path_6 = tmp_path / "inconsistent_time.ims"
        with h5py.File(str(imaris_path_6), "w") as f:
            f.attrs["ImarisVersion"] = "9.0.0"
            dataset_group = f.create_group("DataSet")
            res_group = dataset_group.create_group("ResolutionLevel 0")
            time_group = res_group.create_group(
                "TimePoint 1"
            )  # But we'll try to access timepoint 0
            channel_group = time_group.create_group("Channel 0")
            channel_group.create_dataset("Data", data=np.ones((10, 10, 10)))

        with pytest.raises(ValueError, match="Unable to read Imaris file"):
            ZarrNii.from_imaris(str(imaris_path_6), timepoint=0)

        # Test inconsistent channel structure
        imaris_path_7 = tmp_path / "inconsistent_channel.ims"
        with h5py.File(str(imaris_path_7), "w") as f:
            f.attrs["ImarisVersion"] = "9.0.0"
            dataset_group = f.create_group("DataSet")
            res_group = dataset_group.create_group("ResolutionLevel 0")
            time_group = res_group.create_group("TimePoint 0")
            channel_group = time_group.create_group(
                "Channel 1"
            )  # But we'll try to access channel 0
            channel_group.create_dataset("Data", data=np.ones((10, 10, 10)))

        with pytest.raises(ValueError, match="Unable to read Imaris file"):
            ZarrNii.from_imaris(str(imaris_path_7), channels=[0])

    def test_imaris_without_metadata(self, tmp_path):
        """Test malformed Imaris file without required metadata."""
        imaris_path = tmp_path / "no_metadata.ims"
        sample_data = np.random.rand(32, 64, 48).astype(np.float32)

        # Create Imaris file without ImageSizeX/Y/Z attributes
        with h5py.File(str(imaris_path), "w") as f:
            f.attrs["ImarisVersion"] = "9.0.0"
            # Don't set ImageSizeX, ImageSizeY, ImageSizeZ attributes

            dataset_group = f.create_group("DataSet")
            res_group = dataset_group.create_group("ResolutionLevel 0")
            time_group = res_group.create_group("TimePoint 0")
            channel_group = time_group.create_group("Channel 0")
            channel_group.create_dataset("Data", data=sample_data)

            # Add basic info groups
            info_group = f.create_group("DataSetInfo")
            info_group.create_group("Image")
            time_info_group = f.create_group("DataSetTimes")
            time_info_group.create_dataset("Time", data=[0.0])

        with pytest.raises(ValueError, match="Unable to read Imaris file"):
            ZarrNii.from_imaris(str(imaris_path))


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

    @pytest.mark.usefixtures("cleandir")
    def test_xyz_axes_order_to_imaris(self):
        """Test saving to Imaris with XYZ axes_order (e.g., from NIfTI)."""
        # Create test data with a specific pattern to verify correct axis ordering
        # Data shape: (X=10, Y=20, Z=30)
        test_data = np.zeros((10, 20, 30), dtype=np.float32)
        # Set a marker at X=5, Y=10, Z=15
        test_data[5, 10, 15] = 100.0

        # Create ZarrNii with XYZ axes_order (like NIfTI loading)
        darr = da.from_array(test_data[np.newaxis, ...], chunks="auto")
        znimg_xyz = ZarrNii.from_darr(darr, spacing=[1.0, 2.0, 3.0], axes_order="XYZ")

        assert znimg_xyz.axes_order == "XYZ"
        assert znimg_xyz.darr.shape == (1, 10, 20, 30)

        # Save to Imaris
        imaris_path = "test_xyz_order.ims"
        znimg_xyz.to_imaris(imaris_path)

        assert os.path.exists(imaris_path)

        # Load back from Imaris (which uses ZYX order)
        znimg_loaded = ZarrNii.from_imaris(imaris_path)

        assert znimg_loaded.axes_order == "ZYX"
        # Shape should be transposed: (1, Z=30, Y=20, X=10)
        assert znimg_loaded.darr.shape == (1, 30, 20, 10)

        # Verify the marker is at the correct position after reordering
        loaded_data = znimg_loaded.darr.compute()[0]
        marker_pos = np.unravel_index(np.argmax(loaded_data), loaded_data.shape)

        # Marker should be at ZYX position: Z=15, Y=10, X=5
        assert marker_pos == (15, 10, 5), f"Expected (15, 10, 5), got {marker_pos}"
        assert loaded_data[15, 10, 5] == 100.0

        # Verify the entire data was correctly transposed
        original_data = znimg_xyz.darr.compute()[0]
        # XYZ[x, y, z] should equal ZYX[z, y, x]
        assert_array_equal(original_data[5, 10, 15], loaded_data[15, 10, 5])


def _string_to_byte_array(s: str) -> np.ndarray:
    """Convert string to byte array as required by Imaris."""
    return np.array([c.encode() for c in s])


def _create_minimal_ims(path, data_levels, spacing=(1.0, 1.0, 1.0)):
    """Create a minimal multi-resolution Imaris HDF5 file for testing.

    Args:
        path: Output .ims file path.
        data_levels: List of numpy arrays (CZYX), one per resolution level.
            Level 0 is the full-resolution base; subsequent levels are
            progressively downsampled.
        spacing: Voxel spacing in ZYX order for level 0.
    """
    sz, sy, sx = spacing
    n_levels = len(data_levels)
    base = data_levels[0]
    n_channels = base.shape[0]
    z0, y0, x0 = base.shape[1], base.shape[2], base.shape[3]

    with h5py.File(path, "w") as f:
        # Root attributes
        f.attrs["DataSetDirectoryName"] = _string_to_byte_array("DataSet")
        f.attrs["DataSetInfoDirectoryName"] = _string_to_byte_array("DataSetInfo")
        f.attrs["ImarisDataSet"] = _string_to_byte_array("ImarisDataSet")
        f.attrs["ImarisVersion"] = _string_to_byte_array("5.5.0")
        f.attrs["NumberOfDataSets"] = np.array([1], dtype=np.uint32)
        f.attrs["ThumbnailDirectoryName"] = _string_to_byte_array("Thumbnail")

        dataset_group = f.create_group("DataSet")

        for r, level_data in enumerate(data_levels):
            res_group = dataset_group.create_group(f"ResolutionLevel {r}")
            time_group = res_group.create_group("TimePoint 0")
            zr, yr, xr = level_data.shape[1], level_data.shape[2], level_data.shape[3]

            for c in range(n_channels):
                channel_group = time_group.create_group(f"Channel {c}")
                ch_data = level_data[c]
                d_min = float(ch_data.min())
                d_max = float(ch_data.max())
                channel_group.attrs["ImageSizeX"] = _string_to_byte_array(str(xr))
                channel_group.attrs["ImageSizeY"] = _string_to_byte_array(str(yr))
                channel_group.attrs["ImageSizeZ"] = _string_to_byte_array(str(zr))
                channel_group.attrs["ImageBlockSizeX"] = _string_to_byte_array(str(xr))
                channel_group.attrs["ImageBlockSizeY"] = _string_to_byte_array(str(yr))
                channel_group.attrs["ImageBlockSizeZ"] = _string_to_byte_array(
                    str(min(zr, 16))
                )
                channel_group.attrs["HistogramMin"] = _string_to_byte_array(
                    f"{d_min:.3f}"
                )
                channel_group.attrs["HistogramMax"] = _string_to_byte_array(
                    f"{d_max:.3f}"
                )
                channel_group.create_dataset(
                    "Data", data=ch_data.astype(np.float32), compression="gzip"
                )
                hist_data, _ = np.histogram(ch_data.flatten(), bins=256)
                channel_group.create_dataset(
                    "Histogram", data=hist_data.astype(np.uint64)
                )

        # DataSetInfo/Image — required for ims_reader initialization
        info_group = f.create_group("DataSetInfo")
        image_group = info_group.create_group("Image")
        image_group.attrs["X"] = _string_to_byte_array(str(x0))
        image_group.attrs["Y"] = _string_to_byte_array(str(y0))
        image_group.attrs["Z"] = _string_to_byte_array(str(z0))
        image_group.attrs["ExtMin0"] = _string_to_byte_array("0.000")
        image_group.attrs["ExtMin1"] = _string_to_byte_array("0.000")
        image_group.attrs["ExtMin2"] = _string_to_byte_array("0.000")
        image_group.attrs["ExtMax0"] = _string_to_byte_array(f"{sx * x0:.3f}")
        image_group.attrs["ExtMax1"] = _string_to_byte_array(f"{sy * y0:.3f}")
        image_group.attrs["ExtMax2"] = _string_to_byte_array(f"{sz * z0:.3f}")

        for c in range(n_channels):
            ch_info = info_group.create_group(f"Channel {c}")
            ch_info.attrs["Color"] = _string_to_byte_array("1.000 0.000 0.000")
            ch_info.attrs["Name"] = _string_to_byte_array(f"Channel {c}")
            ch_info.attrs["ColorMode"] = _string_to_byte_array("BaseColor")
            ch_info.attrs["ColorOpacity"] = _string_to_byte_array("1.000")
            ch_info.attrs["ColorRange"] = _string_to_byte_array("0 255")
            ch_info.attrs["GammaCorrection"] = _string_to_byte_array("1.000")
            ch_info.attrs["HistogramMin"] = _string_to_byte_array("0.000")
            ch_info.attrs["HistogramMax"] = _string_to_byte_array("255.000")


@pytest.fixture
def multi_level_imaris_file(tmp_path):
    """Create a multi-resolution Imaris file with 3 resolution levels."""
    imaris_path = tmp_path / "multi_level.ims"

    # Level 0: full resolution (C=1, Z=32, Y=64, X=48)
    data0 = np.random.rand(1, 32, 64, 48).astype(np.float32)
    # Level 1: 2x downsampled in all axes (C=1, Z=16, Y=32, X=24)
    data1 = data0[:, ::2, ::2, ::2]
    # Level 2: 4x downsampled in all axes (C=1, Z=8, Y=16, X=12)
    data2 = data0[:, ::4, ::4, ::4]

    _create_minimal_ims(str(imaris_path), [data0, data1, data2])

    # Expected cumulative scale factors: level1 → 2×, level2 → 4×
    expected_factors = [
        {"z": 2, "y": 2, "x": 2},
        {"z": 4, "y": 4, "x": 4},
    ]
    return str(imaris_path), expected_factors


class TestGetImarisScaleFactors:
    """Tests for get_imaris_scale_factors and get_scale_factors_from_file."""

    def test_single_level_returns_empty(self, sample_imaris_file):
        """Single-resolution IMS file returns an empty list."""
        factors = get_imaris_scale_factors(sample_imaris_file)
        assert factors == []

    def test_multi_level_returns_correct_factors(self, multi_level_imaris_file):
        """Multi-resolution IMS file returns correct cumulative scale factors."""
        ims_path, expected_factors = multi_level_imaris_file
        factors = get_imaris_scale_factors(ims_path)
        assert factors == expected_factors

    def test_nonexistent_file_raises(self):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="file does not exist"):
            get_imaris_scale_factors("nonexistent.ims")

    def test_get_scale_factors_from_file_dispatches_ims(self, multi_level_imaris_file):
        """get_scale_factors_from_file dispatches .ims to get_imaris_scale_factors."""
        ims_path, expected_factors = multi_level_imaris_file
        factors = get_scale_factors_from_file(ims_path)
        assert factors == expected_factors

    def test_get_scale_factors_from_file_dispatches_zarr(self, tmp_path):
        """get_scale_factors_from_file dispatches .zarr to get_ome_zarr_scale_factors."""
        from zarrnii import get_ome_zarr_scale_factors
        from zarrnii.core import save_ngff_image_with_ome_zarr

        zarr_path = str(tmp_path / "source.zarr")
        data = np.random.rand(1, 32, 64, 48).astype(np.float32)
        darr = da.from_array(data, chunks="auto")
        znii = ZarrNii.from_darr(darr, spacing=[1.0, 1.0, 1.0])
        znii.to_ome_zarr(zarr_path, max_layer=3)

        expected = get_ome_zarr_scale_factors(zarr_path)
        assert get_scale_factors_from_file(zarr_path) == expected

    def test_to_ome_zarr_match_scale_factors_from_single_level_ims(
        self, sample_imaris_file, tmp_path
    ):
        """match_scale_factors_from with a single-level IMS writes a flat zarr."""
        output_path = str(tmp_path / "output.zarr")
        znii = ZarrNii.from_imaris(sample_imaris_file)
        # Single-level IMS → scale_factors=[] → max_layer=1 (level 0 only)
        znii.to_ome_zarr(output_path, match_scale_factors_from=sample_imaris_file)

        from zarrnii import get_ome_zarr_scale_factors

        assert get_ome_zarr_scale_factors(output_path) == []

    def test_to_ome_zarr_match_scale_factors_from_multi_level_ims(
        self, multi_level_imaris_file, tmp_path
    ):
        """match_scale_factors_from with a multi-level IMS replicates the pyramid."""
        ims_path, expected_factors = multi_level_imaris_file
        output_path = str(tmp_path / "output.zarr")

        znii = ZarrNii.from_imaris(ims_path)
        znii.to_ome_zarr(output_path, match_scale_factors_from=ims_path)

        from zarrnii import get_ome_zarr_scale_factors

        assert get_ome_zarr_scale_factors(output_path) == expected_factors
