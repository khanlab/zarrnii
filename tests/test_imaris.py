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
        """Test extraction of spatial metadata from Imaris OME XML tags.

        Uses three *distinct* voxel sizes so that any Z<->X dimension swap is
        immediately visible (a common pitfall with ZYX vs. XYZ ordering).
        """
        imaris_path = tmp_path / "test_metadata.ims"

        # Use fully distinct values: px_x != px_y != px_z
        # DataAxis0 = X, DataAxis1 = Y, DataAxis2 = Z (Imaris convention)
        px_x, px_y, px_z = 1.0, 2.0, 3.0

        ome_xml = (
            '<root xmlns:ca="urn:info.openmicroscopy.org/linkedAnnotation/customAttributes">'
            "<ca:CustomAttributes>"
            f'<DataAxis0 PhysicalUnit="{px_x}"/>'
            f'<DataAxis1 PhysicalUnit="{px_y}"/>'
            f'<DataAxis2 PhysicalUnit="{px_z}"/>'
            '<AxesLabels FirstAxis-Unit="µm"/>'
            "</ca:CustomAttributes>"
            "</root>"
        )
        ome_bytes = np.frombuffer(ome_xml.encode("latin-1"), dtype="|S1")

        # Create Imaris file with OME XML metadata
        with h5py.File(str(imaris_path), "w") as f:
            f.attrs["ImarisVersion"] = "9.0.0"
            f.attrs["ImarisDataSet"] = "ImarisDataSet"

            dataset_group = f.create_group("DataSet")
            res_group = dataset_group.create_group("ResolutionLevel 0")
            time_group = res_group.create_group("TimePoint 0")
            channel_group = time_group.create_group("Channel 0")
            channel_group.create_dataset("Data", data=sample_3d_data)

            # Add DataSetInfo with OME XML tags
            info_group = f.create_group("DataSetInfo")
            info_group.create_group("Image")
            ome_tags = info_group.create_group("OME Image Tags")
            ome_tags.create_dataset("Image 0", data=ome_bytes)

            time_info_group = f.create_group("DataSetTimes")
            time_info_group.create_dataset("Time", data=[0.0])

        # Load and check spacing from OME XML
        znimg = ZarrNii.from_imaris(str(imaris_path))

        # Spacing must be [Z, Y, X] = [px_z, px_y, px_x]
        # Any Z<->X swap would give [px_x, px_y, px_z] = [1.0, 2.0, 3.0] instead
        # of the correct [3.0, 2.0, 1.0]
        zooms = znimg.get_zooms(axes_order="ZYX")
        expected_zooms = [px_z, px_y, px_x]
        assert_array_almost_equal(zooms, expected_zooms, decimal=6)

        # Unit should be micrometer (mapped from µm)
        assert znimg.ngff_image.axes_units.get("z") == "micrometer"
        assert znimg.ngff_image.axes_units.get("y") == "micrometer"
        assert znimg.ngff_image.axes_units.get("x") == "micrometer"

    def test_malformed_imaris_files(self, tmp_path):
        """Test error handling for various malformed Imaris files."""

        # Test file with no resolution levels
        imaris_path_1 = tmp_path / "no_reslevels.ims"
        with h5py.File(str(imaris_path_1), "w") as f:
            f.attrs["ImarisVersion"] = "9.0.0"
            dataset_group = f.create_group("DataSet")
            # Don't add any ResolutionLevel groups

        with pytest.raises(ValueError, match="No resolution levels found"):
            ZarrNii.from_imaris(str(imaris_path_1))

        # Test file with no timepoints
        imaris_path_2 = tmp_path / "no_timepoints.ims"
        with h5py.File(str(imaris_path_2), "w") as f:
            f.attrs["ImarisVersion"] = "9.0.0"
            dataset_group = f.create_group("DataSet")
            res_group = dataset_group.create_group("ResolutionLevel 0")
            # Don't add any TimePoint groups

        with pytest.raises(ValueError, match="No timepoints found"):
            ZarrNii.from_imaris(str(imaris_path_2))

        # Test file with no channels
        imaris_path_3 = tmp_path / "no_channels.ims"
        with h5py.File(str(imaris_path_3), "w") as f:
            f.attrs["ImarisVersion"] = "9.0.0"
            dataset_group = f.create_group("DataSet")
            res_group = dataset_group.create_group("ResolutionLevel 0")
            time_group = res_group.create_group("TimePoint 0")
            # Don't add any Channel groups

        with pytest.raises(ValueError, match="No channels found"):
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

        with pytest.raises(ValueError, match="No Data dataset found"):
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

        with pytest.raises(ValueError, match="Resolution level 0 not found"):
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

        with pytest.raises(ValueError, match="Timepoint 0 not found"):
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

        with pytest.raises(ValueError, match="Channel 0 not found"):
            ZarrNii.from_imaris(str(imaris_path_7), channel=0)

    def test_imaris_without_metadata(self, tmp_path):
        """Test loading Imaris file without spatial metadata attributes."""
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

        # This should work and use default spacing
        znimg = ZarrNii.from_imaris(str(imaris_path))

        # Should have default spacing of [1.0, 1.0, 1.0]
        zooms = znimg.get_zooms()
        expected_zooms = [1.0, 1.0, 1.0]
        assert_array_almost_equal(zooms, expected_zooms)


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

    @pytest.mark.usefixtures("cleandir")
    def test_imaris_to_nifti_spacing_roundtrip(self, tmp_path):
        """Test from_imaris -> to_nifti -> from_nifti preserves voxel sizes and
        data positions correctly with a non-trivial (RPI) orientation.

        Uses three fully distinct voxel sizes so that any Z<->X dimension swap is
        immediately detected.
        """
        import nibabel as nib

        # --- Create Imaris file with OME XML metadata ---
        # Unique voxel sizes per axis so a Z<->X swap is immediately visible
        sx, sy, sz = 1.0, 2.0, 3.0  # X, Y, Z physical voxel sizes (micrometers)
        nx, ny, nz = 30, 40, 50  # X, Y, Z voxel counts

        data_zyx = np.zeros((nz, ny, nx), dtype=np.float32)
        data_zyx[10, 20, 25] = 100.0  # marker at ZYX = (10, 20, 25)

        ome_xml = (
            '<root xmlns:ca="urn:info.openmicroscopy.org/linkedAnnotation/customAttributes">'
            "<ca:CustomAttributes>"
            f'<DataAxis0 PhysicalUnit="{sx}"/>'  # X
            f'<DataAxis1 PhysicalUnit="{sy}"/>'  # Y
            f'<DataAxis2 PhysicalUnit="{sz}"/>'  # Z
            '<AxesLabels FirstAxis-Unit="um"/>'
            "</ca:CustomAttributes>"
            "</root>"
        )
        ome_bytes = np.frombuffer(ome_xml.encode("latin-1"), dtype="|S1")

        imaris_path = str(tmp_path / "test_spacing.ims")
        with h5py.File(imaris_path, "w") as f:
            f.attrs["ImarisVersion"] = "9.0.0"
            ch = (
                f.create_group("DataSet")
                .create_group("ResolutionLevel 0")
                .create_group("TimePoint 0")
                .create_group("Channel 0")
            )
            ch.create_dataset("Data", data=data_zyx)
            info = f.create_group("DataSetInfo")
            info.create_group("Image")
            info.create_group("OME Image Tags").create_dataset(
                "Image 0", data=ome_bytes
            )

        # --- Load from Imaris with RPI orientation ---
        znii = ZarrNii.from_imaris(imaris_path, orientation="RPI")

        # Verify ZYX scale: [sz, sy, sx] = [3, 2, 1]
        assert znii.axes_order == "ZYX"
        assert_array_almost_equal(
            znii.get_zooms(axes_order="ZYX"), [sz, sy, sx], decimal=6
        )
        # Data shape must be (1, nz, ny, nx)
        assert znii.darr.shape == (1, nz, ny, nx)

        # --- Convert to NIfTI (preserve original units) ---
        nii_path = str(tmp_path / "test_spacing.nii")
        znii.to_nifti(nii_path, convert_units_to_mm=False)

        nii = nib.load(nii_path)

        # NIfTI is always XYZ: shape must be (nx, ny, nz) = (30, 40, 50)
        assert nii.shape == (
            nx,
            ny,
            nz,
        ), f"Expected NIfTI shape (nx={nx}, ny={ny}, nz={nz}), got {nii.shape}"

        # Voxel sizes in NIfTI must be (sx, sy, sz) = (1, 2, 3)
        nii_zooms = nii.header.get_zooms()[:3]
        assert_array_almost_equal(nii_zooms, [sx, sy, sz], decimal=5)

        # Affine diagonal signs must reflect RPI (R=+, P=-, I=-)
        aff_diag = np.diag(nii.affine)[:3]
        assert aff_diag[0] > 0, "X direction should be positive (R)"
        assert aff_diag[1] < 0, "Y direction should be negative (P = -A)"
        assert aff_diag[2] < 0, "Z direction should be negative (I = -S)"

        # Marker was at ZYX=(10,20,25); after ZYX->XYZ transpose it should
        # be at XYZ voxel position (X=25, Y=20, Z=10)
        nii_data = nii.get_fdata()
        marker_pos = np.unravel_index(np.argmax(nii_data), nii_data.shape)
        assert marker_pos == (
            25,
            20,
            10,
        ), f"Expected marker at (X=25, Y=20, Z=10), got {marker_pos}"

        # --- Round-trip: load NIfTI back ---
        znii2 = ZarrNii.from_nifti(nii_path)
        # XYZ scales must match original sx, sy, sz
        assert_array_almost_equal(
            znii2.get_zooms(axes_order="XYZ"), [sx, sy, sz], decimal=5
        )
