"""Tests for ZipStore support in zarrnii."""

import json
import os
import tempfile
import zipfile

import dask.array as da
import numpy as np
import pytest

from zarrnii import ZarrNii


class TestZipStoreSupport:
    """Test ZipStore functionality for OME-Zarr files."""

    @pytest.fixture
    def sample_zarrnii(self):
        """Create a sample ZarrNii object for testing."""
        # Create test data
        data = da.ones((1, 16, 32, 32), dtype=np.uint16, chunks=(1, 8, 16, 16))
        return ZarrNii.from_darr(data, axes_order="ZYX", orientation="RAS")

    def test_save_to_zip_file(self, sample_zarrnii):
        """Test saving OME-Zarr to a .zip file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "test.ome.zarr.zip")

            # Save to ZIP file
            result = sample_zarrnii.to_ome_zarr(zip_path, max_layer=2)

            # Should return self for method chaining
            assert result is sample_zarrnii

            # Check that ZIP file was created
            assert os.path.exists(zip_path)
            assert os.path.getsize(zip_path) > 0

            # ZIP file should be readable as zip
            with zipfile.ZipFile(zip_path, "r") as zf:
                files = zf.namelist()
                # zarr v3 uses zarr.json instead of .zgroup/.zattrs
                assert "zarr.json" in files
                # Should have multiscale levels
                assert any("0" in name for name in files)

    def test_save_to_ozx_file(self, sample_zarrnii):
        """Test saving OME-Zarr to the new .ozx extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ozx_path = os.path.join(tmpdir, "test.ozx")

            # Save to OZX file
            result = sample_zarrnii.to_ome_zarr(ozx_path, max_layer=2)

            # Should return self for method chaining
            assert result is sample_zarrnii

            # Check that OZX file was created
            assert os.path.exists(ozx_path)
            assert os.path.getsize(ozx_path) > 0

            # OZX file should be readable as zip
            with zipfile.ZipFile(ozx_path, "r") as zf:
                files = zf.namelist()
                # zarr v3 uses zarr.json
                assert "zarr.json" in files
                # Should have multiscale levels
                assert any("0" in name for name in files)

    def test_ome_zarr_zip_spec_compliance(self, sample_zarrnii):
        """Test that ZIP files comply with OME-Zarr single-file spec."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ozx_path = os.path.join(tmpdir, "test.ozx")
            sample_zarrnii.to_ome_zarr(ozx_path, max_layer=2)

            with zipfile.ZipFile(ozx_path, "r") as zf:
                files = zf.namelist()

                # 1. Root zarr.json MUST be first entry
                assert files[0] == "zarr.json", "Root zarr.json should be first entry"

                # 2. zarr.json files should be in breadth-first order at the start
                zarr_json_files = [f for f in files if f.endswith("zarr.json")]
                # Verify root is first, then level-specific ones
                assert zarr_json_files[0] == "zarr.json"
                # Other zarr.json should follow
                for i, _f in enumerate(zarr_json_files[1:], 1):
                    # Earlier zarr.json files should have shallower depth
                    depth_i = zarr_json_files[i].count("/")
                    for j in range(i):
                        depth_j = zarr_json_files[j].count("/")
                        assert (
                            depth_j <= depth_i
                        ), "zarr.json files should be in breadth-first order"

                # 3. ZIP archive comment should contain OME version JSON
                comment = zf.comment.decode("utf-8")
                assert comment, "ZIP archive should have a comment"
                comment_json = json.loads(comment)
                assert "ome" in comment_json, "Comment should have 'ome' key"
                assert "version" in comment_json["ome"], "Comment should have version"

                # 4. No compression should be used (STORED method)
                for info in zf.infolist():
                    if not info.is_dir():
                        assert (
                            info.compress_type == zipfile.ZIP_STORED
                        ), f"File {info.filename} should use STORED compression"

    def test_load_from_zip_file(self, sample_zarrnii):
        """Test loading OME-Zarr from a .zip file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "test.ome.zarr.zip")

            # Save to ZIP file first
            sample_zarrnii.to_ome_zarr(zip_path, max_layer=2)

            # Load from ZIP file
            loaded_zarrnii = ZarrNii.from_ome_zarr(zip_path, level=0)

            # Verify loaded data
            assert loaded_zarrnii.darr.shape == sample_zarrnii.darr.shape
            assert loaded_zarrnii.axes_order == sample_zarrnii.axes_order
            assert loaded_zarrnii.orientation == sample_zarrnii.orientation

            # Data should be the same
            np.testing.assert_array_equal(
                loaded_zarrnii.darr.compute(), sample_zarrnii.darr.compute()
            )

    def test_load_from_ozx_file(self, sample_zarrnii):
        """Test loading OME-Zarr from a .ozx file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ozx_path = os.path.join(tmpdir, "test.ozx")

            # Save to OZX file first
            sample_zarrnii.to_ome_zarr(ozx_path, max_layer=2)

            # Load from OZX file
            loaded_zarrnii = ZarrNii.from_ome_zarr(ozx_path, level=0)

            # Verify loaded data
            assert loaded_zarrnii.darr.shape == sample_zarrnii.darr.shape
            assert loaded_zarrnii.axes_order == sample_zarrnii.axes_order
            assert loaded_zarrnii.orientation == sample_zarrnii.orientation

            # Data should be the same
            np.testing.assert_array_equal(
                loaded_zarrnii.darr.compute(), sample_zarrnii.darr.compute()
            )

    def test_round_trip_zip_storage(self, sample_zarrnii):
        """Test round-trip save and load with ZIP storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "roundtrip.ome.zarr.zip")

            # Round trip
            sample_zarrnii.to_ome_zarr(zip_path, max_layer=3)
            loaded = ZarrNii.from_ome_zarr(zip_path, level=0)

            # Verify all properties are preserved
            assert loaded.darr.shape == sample_zarrnii.darr.shape
            assert loaded.axes_order == sample_zarrnii.axes_order
            assert loaded.orientation == sample_zarrnii.orientation

            # Verify data integrity
            original_data = sample_zarrnii.darr.compute()
            loaded_data = loaded.darr.compute()
            np.testing.assert_array_equal(loaded_data, original_data)

    def test_round_trip_ozx_storage(self, sample_zarrnii):
        """Test round-trip save and load with .ozx storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ozx_path = os.path.join(tmpdir, "roundtrip.ozx")

            # Round trip
            sample_zarrnii.to_ome_zarr(ozx_path, max_layer=3)
            loaded = ZarrNii.from_ome_zarr(ozx_path, level=0)

            # Verify all properties are preserved
            assert loaded.darr.shape == sample_zarrnii.darr.shape
            assert loaded.axes_order == sample_zarrnii.axes_order
            assert loaded.orientation == sample_zarrnii.orientation

            # Verify data integrity
            original_data = sample_zarrnii.darr.compute()
            loaded_data = loaded.darr.compute()
            np.testing.assert_array_equal(loaded_data, original_data)

    def test_zip_vs_regular_directory_equivalence(self, sample_zarrnii):
        """Test that ZIP and regular directory storage produce equivalent results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save to regular directory
            regular_path = os.path.join(tmpdir, "regular.ome.zarr")
            sample_zarrnii.to_ome_zarr(regular_path, max_layer=2)

            # Save to ZIP file
            zip_path = os.path.join(tmpdir, "zipped.ome.zarr.zip")
            sample_zarrnii.to_ome_zarr(zip_path, max_layer=2)

            # Load both and compare
            loaded_regular = ZarrNii.from_ome_zarr(regular_path, level=0)
            loaded_zip = ZarrNii.from_ome_zarr(zip_path, level=0)

            # Should have identical data
            np.testing.assert_array_equal(
                loaded_regular.darr.compute(), loaded_zip.darr.compute()
            )

            # Should have identical metadata
            assert loaded_regular.axes_order == loaded_zip.axes_order
            assert loaded_regular.orientation == loaded_zip.orientation
            assert loaded_regular.darr.shape == loaded_zip.darr.shape

    def test_multi_level_zip_access(self, sample_zarrnii):
        """Test accessing different pyramid levels in ZIP files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "multilevel.ome.zarr.zip")

            # Save with multiple levels
            sample_zarrnii.to_ome_zarr(zip_path, max_layer=3)

            # Load different levels
            level0 = ZarrNii.from_ome_zarr(zip_path, level=0)
            level1 = ZarrNii.from_ome_zarr(zip_path, level=1)
            level2 = ZarrNii.from_ome_zarr(zip_path, level=2)

            # Verify pyramid structure
            assert level0.darr.shape == sample_zarrnii.darr.shape
            # Each level should be roughly half the size in index 2 or 3 (y or x)
            assert level1.darr.shape[2] <= level0.darr.shape[2] // 2 + 1
            assert level2.darr.shape[2] <= level1.darr.shape[2] // 2 + 1

    def test_zip_file_extension_detection(self, sample_zarrnii):
        """Test that .zip and .ozx extensions are properly detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test various zip extensions including new .ozx
            extensions = [".zip", ".ome.zarr.zip", ".zarr.zip", ".ozx"]

            for ext in extensions:
                zip_path = os.path.join(tmpdir, f"test{ext}")

                # Should successfully save and load
                sample_zarrnii.to_ome_zarr(zip_path)
                loaded = ZarrNii.from_ome_zarr(zip_path)

                assert loaded.darr.shape == sample_zarrnii.darr.shape
                np.testing.assert_array_equal(
                    loaded.darr.compute(), sample_zarrnii.darr.compute()
                )

    def test_zip_error_handling(self):
        """Test error handling for invalid ZIP files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an empty file with .zip extension
            invalid_zip = os.path.join(tmpdir, "invalid.ome.zarr.zip")
            with open(invalid_zip, "w") as f:
                f.write("not a zip file")

            # Should raise appropriate error when trying to read
            with pytest.raises(Exception):
                ZarrNii.from_ome_zarr(invalid_zip)

    def test_ozx_error_handling(self):
        """Test error handling for invalid .ozx files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an empty file with .ozx extension
            invalid_ozx = os.path.join(tmpdir, "invalid.ozx")
            with open(invalid_ozx, "w") as f:
                f.write("not a zip file")

            # Should raise appropriate error when trying to read
            with pytest.raises(Exception):
                ZarrNii.from_ome_zarr(invalid_ozx)

    def test_from_file_with_ozx_extension(self, sample_zarrnii):
        """Test that from_file correctly handles .ozx extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ozx_path = os.path.join(tmpdir, "test.ozx")
            sample_zarrnii.to_ome_zarr(ozx_path, max_layer=2)

            # from_file should detect .ozx and use from_ome_zarr
            loaded = ZarrNii.from_file(ozx_path)
            assert loaded.darr.shape == sample_zarrnii.darr.shape
