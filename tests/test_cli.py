"""Tests for the CLI console scripts.

This module tests the z2n and n2z console scripts that provide command-line
interfaces for converting between OME-Zarr and NIfTI formats.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from zarrnii import ZarrNii


class TestCLIScripts:
    """Test class for CLI console scripts."""

    @pytest.fixture
    def sample_nifti(self, tmp_path):
        """Create a sample NIfTI file for testing."""
        # Create test data
        data = np.random.rand(64, 64, 32).astype(np.float32)
        affine = np.eye(4)
        affine[0, 0] = 2.0  # 2mm voxel spacing in x
        affine[1, 1] = 2.0  # 2mm voxel spacing in y
        affine[2, 2] = 3.0  # 3mm voxel spacing in z

        # Create NIfTI image
        img = nib.Nifti1Image(data, affine)
        nifti_path = tmp_path / "test.nii.gz"
        img.to_filename(str(nifti_path))

        return nifti_path

    @pytest.fixture
    def sample_zarr(self, tmp_path, sample_nifti):
        """Create a sample OME-Zarr file for testing."""
        zarr_path = tmp_path / "test.ome.zarr"

        # Load NIfTI and convert to Zarr
        znimg = ZarrNii.from_nifti(str(sample_nifti))
        znimg.to_ome_zarr(str(zarr_path))

        return zarr_path

    def run_cli_script(self, script_name, args, expect_success=True):
        """Helper to run CLI scripts and return result."""
        cmd = [sys.executable, "-m", "zarrnii.cli", script_name] + args
        result = subprocess.run(cmd, capture_output=True, text=True)

        if expect_success and result.returncode != 0:
            pytest.fail(
                f"CLI script {script_name} failed with return code {result.returncode}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        return result

    def test_n2z_basic_conversion(self, sample_nifti, tmp_path):
        """Test basic NIfTI to OME-Zarr conversion."""
        output_zarr = tmp_path / "output.ome.zarr"

        result = self.run_cli_script("n2z", [str(sample_nifti), str(output_zarr)])

        assert result.returncode == 0
        assert "Conversion completed successfully!" in result.stdout
        assert output_zarr.exists()

        # Verify the converted file can be loaded
        znimg = ZarrNii.from_ome_zarr(str(output_zarr))
        assert znimg.darr.shape == (1, 32, 64, 64)  # Expected ZYX format with C dim

    def test_z2n_basic_conversion(self, sample_zarr, tmp_path):
        """Test basic OME-Zarr to NIfTI conversion."""
        output_nifti = tmp_path / "output.nii.gz"

        result = self.run_cli_script("z2n", [str(sample_zarr), str(output_nifti)])

        assert result.returncode == 0
        assert "Conversion completed successfully!" in result.stdout
        assert output_nifti.exists()

        # Verify the converted file can be loaded
        img = nib.load(str(output_nifti))
        assert img.get_fdata().shape == (64, 64, 32)  # XYZ format

    def test_n2z_with_options(self, sample_nifti, tmp_path):
        """Test NIfTI to OME-Zarr conversion with various options."""
        output_zarr = tmp_path / "output.ome.zarr"

        result = self.run_cli_script(
            "n2z",
            [
                str(sample_nifti),
                str(output_zarr),
                "--axes-order",
                "ZYX",
                "--max-layer",
                "6",
                "--name",
                "test_image",
            ],
        )

        assert result.returncode == 0
        assert "Conversion completed successfully!" in result.stdout
        assert output_zarr.exists()

    def test_z2n_with_options(self, sample_zarr, tmp_path):
        """Test OME-Zarr to NIfTI conversion with various options."""
        output_nifti = tmp_path / "output.nii.gz"

        result = self.run_cli_script(
            "z2n",
            [
                str(sample_zarr),
                str(output_nifti),
                "--level",
                "0",
                "--axes-order",
                "XYZ",
                "--orientation",
                "LPI",
            ],
        )

        assert result.returncode == 0
        assert "Conversion completed successfully!" in result.stdout
        assert output_nifti.exists()

    def test_n2z_with_custom_zooms(self, sample_nifti, tmp_path):
        """Test NIfTI to OME-Zarr with custom zoom parameters."""
        output_zarr = tmp_path / "output.ome.zarr"

        result = self.run_cli_script(
            "n2z",
            [str(sample_nifti), str(output_zarr), "--as-ref", "--zooms", "1.5,1.5,2.5"],
        )

        assert result.returncode == 0
        assert "Conversion completed successfully!" in result.stdout
        assert output_zarr.exists()

    def test_z2n_with_channels(self, tmp_path):
        """Test OME-Zarr to NIfTI with channel selection."""
        # Create a multi-channel test zarr first
        nifti_path = tmp_path / "multichannel.nii.gz"
        zarr_path = tmp_path / "multichannel.ome.zarr"
        output_nifti = tmp_path / "output.nii.gz"

        # Create multi-channel data
        data = np.random.rand(32, 64, 64, 3).astype(np.float32)  # 3 channels
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)
        img.to_filename(str(nifti_path))

        # Convert to zarr
        znimg = ZarrNii.from_nifti(str(nifti_path))
        znimg.to_ome_zarr(str(zarr_path))

        # Convert back with channel selection (select single channel)
        result = self.run_cli_script(
            "z2n", [str(zarr_path), str(output_nifti), "--channels", "0"]
        )

        assert result.returncode == 0
        assert "Conversion completed successfully!" in result.stdout
        assert output_nifti.exists()

    def test_n2z_missing_input_file(self, tmp_path):
        """Test error handling for missing input file."""
        missing_file = tmp_path / "missing.nii.gz"
        output_zarr = tmp_path / "output.ome.zarr"

        result = self.run_cli_script(
            "n2z", [str(missing_file), str(output_zarr)], expect_success=False
        )

        assert result.returncode == 1
        assert "does not exist" in result.stderr

    def test_z2n_missing_input_file(self, tmp_path):
        """Test error handling for missing input file."""
        missing_file = tmp_path / "missing.ome.zarr"
        output_nifti = tmp_path / "output.nii.gz"

        result = self.run_cli_script(
            "z2n", [str(missing_file), str(output_nifti)], expect_success=False
        )

        assert result.returncode == 1
        assert "does not exist" in result.stderr

    def test_n2z_help(self):
        """Test that help option works for n2z."""
        result = subprocess.run(
            [sys.executable, "-m", "zarrnii.cli", "n2z", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Convert NIfTI to OME-Zarr format" in result.stdout
        assert "Examples:" in result.stdout

    def test_z2n_help(self):
        """Test that help option works for z2n."""
        result = subprocess.run(
            [sys.executable, "-m", "zarrnii.cli", "z2n", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Convert OME-Zarr to NIfTI format" in result.stdout
        assert "Examples:" in result.stdout

    def test_n2z_scale_factors(self, sample_nifti, tmp_path):
        """Test n2z with custom scale factors."""
        output_zarr = tmp_path / "output.ome.zarr"

        result = self.run_cli_script(
            "n2z", [str(sample_nifti), str(output_zarr), "--scale-factors", "2,4,8"]
        )

        assert result.returncode == 0
        assert "Conversion completed successfully!" in result.stdout
        assert output_zarr.exists()

    def test_round_trip_conversion(self, sample_nifti, tmp_path):
        """Test round-trip conversion NIfTI -> Zarr -> NIfTI."""
        zarr_path = tmp_path / "intermediate.ome.zarr"
        final_nifti = tmp_path / "final.nii.gz"

        # NIfTI to Zarr
        result1 = self.run_cli_script("n2z", [str(sample_nifti), str(zarr_path)])
        assert result1.returncode == 0

        # Zarr back to NIfTI
        result2 = self.run_cli_script("z2n", [str(zarr_path), str(final_nifti)])
        assert result2.returncode == 0

        # Compare original and final
        original_img = nib.load(str(sample_nifti))
        final_img = nib.load(str(final_nifti))

        # Check shapes match (accounting for potential axis reordering)
        assert final_img.get_fdata().shape == original_img.get_fdata().shape

    def test_zip_store_support(self, sample_nifti, tmp_path):
        """Test conversion with ZIP store support."""
        zarr_zip_path = tmp_path / "test.ome.zarr.zip"
        output_nifti = tmp_path / "output.nii.gz"

        # First create a ZIP store manually
        zarr_regular = tmp_path / "temp.ome.zarr"
        znimg = ZarrNii.from_nifti(str(sample_nifti))
        znimg.to_ome_zarr(str(zarr_zip_path))  # This should create ZIP format

        # Test conversion from ZIP store
        result = self.run_cli_script("z2n", [str(zarr_zip_path), str(output_nifti)])

        assert result.returncode == 0
        assert "Conversion completed successfully!" in result.stdout
        assert output_nifti.exists()

    def test_invalid_argument_parsing(self, sample_nifti, tmp_path):
        """Test error handling for invalid arguments."""
        output_zarr = tmp_path / "output.ome.zarr"

        # Test invalid zooms format
        result = self.run_cli_script(
            "n2z",
            [
                str(sample_nifti),
                str(output_zarr),
                "--zooms",
                "1.0,2.0",  # Should be 3 values
            ],
            expect_success=False,
        )

        assert result.returncode != 0

    def test_chunk_parsing(self, sample_nifti, tmp_path):
        """Test chunk argument parsing."""
        output_zarr = tmp_path / "output.ome.zarr"

        # Test with valid chunk specification
        result = self.run_cli_script(
            "n2z", [str(sample_nifti), str(output_zarr), "--chunks", "32,32,16"]
        )

        assert result.returncode == 0
        assert "Conversion completed successfully!" in result.stdout


class TestCLIArgumentParsing:
    """Test argument parsing functions independently."""

    def test_parse_int_list(self):
        """Test integer list parsing function."""
        from zarrnii.cli import parse_int_list

        assert parse_int_list("1,2,3") == [1, 2, 3]
        assert parse_int_list("0") == [0]
        assert parse_int_list("") == []
        assert parse_int_list(" 1 , 2 , 3 ") == [1, 2, 3]

        with pytest.raises(ValueError):
            parse_int_list("1,a,3")

    def test_parse_float_tuple(self):
        """Test float tuple parsing function."""
        from zarrnii.cli import parse_float_tuple

        assert parse_float_tuple("1.0,2.0,3.0") == (1.0, 2.0, 3.0)
        assert parse_float_tuple("1,2,3") == (1.0, 2.0, 3.0)

        with pytest.raises(Exception):  # ArgumentTypeError
            parse_float_tuple("1.0,2.0")  # Wrong number of values

        with pytest.raises(ValueError):
            parse_float_tuple("1.0,a,3.0")

    def test_parse_int_tuple(self):
        """Test integer tuple parsing function."""
        from zarrnii.cli import parse_int_tuple

        assert parse_int_tuple("1,2,3") == (1, 2, 3)
        assert parse_int_tuple("32") == (32,)

        with pytest.raises(ValueError):
            parse_int_tuple("1,a,3")
