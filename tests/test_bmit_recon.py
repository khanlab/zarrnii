"""Tests for ZarrNii.from_bmit_recon()."""

import os
import tempfile

import numpy as np
import pytest
import tifffile

from zarrnii import ZarrNii


def _make_scan_dir(tmpdir, pixel_size=5.2, unit="um", nth_row=20, n_slices=3):
    """Create a minimal BMIT scan directory with fake TIFF slices."""
    sli_dir = os.path.join(tmpdir, "sli")
    os.makedirs(sli_dir)

    for i in range(n_slices):
        path = os.path.join(sli_dir, f"slice_{i:04d}.tif")
        tifffile.imwrite(path, np.full((4, 5), i, dtype=np.uint16))

    params = f"""\
*** General ***
Input directory /beamlinedata/BMIT/projects/test
CT set .
Center of rotation 2199.0 (auto estimate)
Dimensions of projections 1548 x 4416 (height x width)
Number of projections 6000
*** Preprocessing ***
  None
*** Image filters ***
  Remove large spots disabled
 Phase retrieval enabled
  energy 30.0 keV
  pixel size {pixel_size} {unit}
  sample-detector distance 0.45 m
 delta/beta ratio 499.99999999999994
*** Ring removal ***
RR disabled
*** Region of interest ***
Vertical ROI defined
  first row 1
  height 1547
  reconstruct every {nth_row}th row
ROI in slice plane not defined
*** Reconstructed values ***
  32bit, histogram untouched
*** Optional reco parameters ***
"""
    with open(os.path.join(tmpdir, "reco_params_simple.txt"), "w") as fh:
        fh.write(params)

    return tmpdir


def test_from_bmit_recon_basic():
    """from_bmit_recon returns correct shape and spacing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_scan_dir(tmpdir, pixel_size=5.2, unit="um", nth_row=20, n_slices=3)
        znii = ZarrNii.from_bmit_recon(tmpdir)

        assert list(znii.dims) == ["c", "z", "y", "x"]
        assert znii.data.shape == (1, 3, 4, 5)
        # Spacing is stored internally in mm; 5.2 um = 0.0052 mm, 104 um = 0.104 mm
        assert znii.scale["z"] == pytest.approx(0.104)
        assert znii.scale["y"] == pytest.approx(0.0052)
        assert znii.scale["x"] == pytest.approx(0.0052)
        assert znii.ngff_image.axes_units == {
            "z": "millimeter",
            "y": "millimeter",
            "x": "millimeter",
        }


def test_from_bmit_recon_nth_row_1():
    """When every 1st row is reconstructed, z spacing equals pixel size."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_scan_dir(tmpdir, pixel_size=3.0, unit="um", nth_row=1, n_slices=2)
        znii = ZarrNii.from_bmit_recon(tmpdir)

        # 3.0 um = 0.003 mm; z and xy should be equal
        assert znii.scale["z"] == pytest.approx(0.003)
        assert znii.scale["y"] == pytest.approx(0.003)
        assert znii.scale["x"] == pytest.approx(0.003)


def test_from_bmit_recon_no_nth_row_line():
    """When 'reconstruct every Nth row' is absent, z spacing equals pixel size."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sli_dir = os.path.join(tmpdir, "sli")
        os.makedirs(sli_dir)
        for i in range(2):
            tifffile.imwrite(
                os.path.join(sli_dir, f"slice_{i:04d}.tif"),
                np.zeros((4, 5), dtype=np.uint16),
            )
        params = "pixel size 2.5 um\n"
        with open(os.path.join(tmpdir, "reco_params_simple.txt"), "w") as fh:
            fh.write(params)

        znii = ZarrNii.from_bmit_recon(tmpdir)
        # 2.5 um = 0.0025 mm
        assert znii.scale["z"] == pytest.approx(0.0025)
        assert znii.scale["y"] == pytest.approx(0.0025)


def test_from_bmit_recon_missing_scan_dir():
    """Raises FileNotFoundError when scan_dir does not exist."""
    with pytest.raises(FileNotFoundError, match="scan_dir does not exist"):
        ZarrNii.from_bmit_recon("/nonexistent/path/abc123")


def test_from_bmit_recon_missing_sli_dir():
    """Raises FileNotFoundError when 'sli' subdirectory is absent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "reco_params_simple.txt"), "w") as fh:
            fh.write("pixel size 5.2 um\n")
        with pytest.raises(FileNotFoundError, match="'sli' subdirectory"):
            ZarrNii.from_bmit_recon(tmpdir)


def test_from_bmit_recon_missing_params_file():
    """Raises FileNotFoundError when reco_params_simple.txt is absent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "sli"))
        with pytest.raises(FileNotFoundError, match="reco_params_simple.txt"):
            ZarrNii.from_bmit_recon(tmpdir)


def test_from_bmit_recon_no_tif_files():
    """Raises ValueError when sli/ contains no TIFF files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "sli"))
        with open(os.path.join(tmpdir, "reco_params_simple.txt"), "w") as fh:
            fh.write("pixel size 5.2 um\n")
        with pytest.raises(ValueError, match="No TIFF files found"):
            ZarrNii.from_bmit_recon(tmpdir)


def test_from_bmit_recon_missing_pixel_size():
    """Raises ValueError when pixel size cannot be parsed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sli_dir = os.path.join(tmpdir, "sli")
        os.makedirs(sli_dir)
        tifffile.imwrite(
            os.path.join(sli_dir, "slice_0000.tif"),
            np.zeros((4, 5), dtype=np.uint16),
        )
        with open(os.path.join(tmpdir, "reco_params_simple.txt"), "w") as fh:
            fh.write("no pixel size here\n")
        with pytest.raises(ValueError, match="pixel size"):
            ZarrNii.from_bmit_recon(tmpdir)


def test_from_bmit_recon_axes_units_override():
    """axes_units override is passed through correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_scan_dir(tmpdir, pixel_size=5.2, unit="um", nth_row=10, n_slices=2)
        znii = ZarrNii.from_bmit_recon(
            tmpdir,
            axes_units={"z": "millimeter", "y": "millimeter", "x": "millimeter"},
        )
        assert znii.ngff_image.axes_units == {
            "z": "millimeter",
            "y": "millimeter",
            "x": "millimeter",
        }
