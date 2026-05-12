"""Tests for ZarrNii.from_ome_tif()."""

import os
import tempfile

import dask.array as da
import numpy as np
import pytest
import tifffile

from zarrnii import ZarrNii


def _write_ome_tif(path, data, axes, spacing_xyz=(0.325, 0.325, 1.0), unit="um"):
    """Helper: write a small OME-TIFF with physical size metadata."""
    tifffile.imwrite(
        path,
        data,
        photometric="minisblack",
        ome=True,
        metadata={
            "axes": axes,
            "PhysicalSizeX": spacing_xyz[0],
            "PhysicalSizeY": spacing_xyz[1],
            "PhysicalSizeZ": spacing_xyz[2],
            "PhysicalSizeXUnit": unit,
            "PhysicalSizeYUnit": unit,
            "PhysicalSizeZUnit": unit,
        },
    )


class TestFromOmeTif:
    """Tests for ZarrNii.from_ome_tif class method."""

    def test_basic_zyx_zstack(self):
        """Load a single-channel ZYX z-stack."""
        data = np.random.randint(0, 1000, size=(10, 32, 32), dtype=np.uint16)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tmpf = f.name
        try:
            _write_ome_tif(tmpf, data, axes="ZYX")
            znii = ZarrNii.from_ome_tif(tmpf)

            # Data should be lazily loaded
            assert isinstance(znii.data, da.Array)
            # Shape: (C=1, Z=10, Y=32, X=32)
            assert znii.data.shape == (1, 10, 32, 32)
            # Dims: c, z, y, x
            assert list(znii.dims) == ["c", "z", "y", "x"]
            # Spacing
            assert znii.scale["z"] == pytest.approx(1.0)
            assert znii.scale["y"] == pytest.approx(0.325)
            assert znii.scale["x"] == pytest.approx(0.325)
        finally:
            os.unlink(tmpf)

    def test_multichannel_czyx_zstack(self):
        """Load a multi-channel CZYX z-stack."""
        data = np.random.randint(0, 1000, size=(3, 10, 32, 32), dtype=np.uint16)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tmpf = f.name
        try:
            _write_ome_tif(tmpf, data, axes="CZYX", spacing_xyz=(0.5, 0.5, 2.0))
            znii = ZarrNii.from_ome_tif(tmpf)

            assert znii.data.shape == (3, 10, 32, 32)
            assert list(znii.dims) == ["c", "z", "y", "x"]
            assert znii.scale["z"] == pytest.approx(2.0)
            assert znii.scale["y"] == pytest.approx(0.5)
            assert znii.scale["x"] == pytest.approx(0.5)
        finally:
            os.unlink(tmpf)

    def test_set_channel_labels_builds_omero(self):
        """set_channel_labels adds OMERO channel labels to loaded OME-TIFF."""
        data = np.random.randint(0, 1000, size=(3, 10, 32, 32), dtype=np.uint16)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tmpf = f.name
        try:
            _write_ome_tif(tmpf, data, axes="CZYX", spacing_xyz=(0.5, 0.5, 2.0))
            znii = ZarrNii.from_ome_tif(tmpf, set_channel_labels=["DAPI", "GFP", "RFP"])
            assert znii.omero is not None
            assert [ch.label for ch in znii.omero.channels] == ["DAPI", "GFP", "RFP"]
        finally:
            os.unlink(tmpf)

    def test_set_channel_labels_length_mismatch_raises(self):
        """set_channel_labels length must match source channel count."""
        data = np.random.randint(0, 1000, size=(3, 10, 32, 32), dtype=np.uint16)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tmpf = f.name
        try:
            _write_ome_tif(tmpf, data, axes="CZYX", spacing_xyz=(0.5, 0.5, 2.0))
            with pytest.raises(ValueError, match="set_channel_labels length"):
                ZarrNii.from_ome_tif(tmpf, set_channel_labels=["DAPI"])
        finally:
            os.unlink(tmpf)

    def test_axes_order_xyz(self):
        """Loading with axes_order='XYZ' transposes data correctly."""
        data = np.random.randint(0, 1000, size=(10, 32, 32), dtype=np.uint16)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tmpf = f.name
        try:
            _write_ome_tif(tmpf, data, axes="ZYX", spacing_xyz=(0.325, 0.325, 1.0))
            znii = ZarrNii.from_ome_tif(tmpf, axes_order="XYZ")

            # With XYZ order the spatial dims should be x, y, z
            assert list(znii.dims) == ["c", "x", "y", "z"]
            assert znii.scale["z"] == pytest.approx(1.0)
            assert znii.scale["x"] == pytest.approx(0.325)
            assert znii.scale["y"] == pytest.approx(0.325)
        finally:
            os.unlink(tmpf)

    def test_spacing_units_micrometer(self):
        """Physical spacing with 'um' unit maps to 'micrometer'."""
        data = np.zeros((5, 16, 16), dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tmpf = f.name
        try:
            _write_ome_tif(tmpf, data, axes="ZYX", unit="um")
            znii = ZarrNii.from_ome_tif(tmpf)
            axes_units = znii.ngff_image.axes_units
            assert axes_units["z"] == "micrometer"
            assert axes_units["y"] == "micrometer"
            assert axes_units["x"] == "micrometer"
        finally:
            os.unlink(tmpf)

    def test_data_is_lazy(self):
        """Data array should be a dask array (lazy)."""
        data = np.random.randint(0, 65535, size=(8, 64, 64), dtype=np.uint16)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tmpf = f.name
        try:
            _write_ome_tif(tmpf, data, axes="ZYX")
            znii = ZarrNii.from_ome_tif(tmpf)
            assert isinstance(znii.data, da.Array)
            # Confirm compute gives the expected values (C dim added)
            computed = znii.data.compute()
            np.testing.assert_array_equal(computed[0], data)
        finally:
            os.unlink(tmpf)

    def test_custom_name(self):
        """Custom name is forwarded to the NgffImage."""
        data = np.zeros((4, 16, 16), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tmpf = f.name
        try:
            _write_ome_tif(tmpf, data, axes="ZYX")
            znii = ZarrNii.from_ome_tif(tmpf, name="my_zstack")
            assert znii.ngff_image.name == "my_zstack"
        finally:
            os.unlink(tmpf)

    def test_default_name_is_basename(self):
        """When name is None, the basename of the file is used."""
        data = np.zeros((4, 16, 16), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tmpf = f.name
        try:
            _write_ome_tif(tmpf, data, axes="ZYX")
            znii = ZarrNii.from_ome_tif(tmpf)
            assert znii.ngff_image.name == os.path.basename(tmpf)
        finally:
            os.unlink(tmpf)

    def test_invalid_axes_order_raises(self):
        """Passing an unknown axes_order raises ValueError."""
        data = np.zeros((4, 16, 16), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tmpf = f.name
        try:
            _write_ome_tif(tmpf, data, axes="ZYX")
            with pytest.raises(ValueError, match="axes_order"):
                ZarrNii.from_ome_tif(tmpf, axes_order="INVALID")
        finally:
            os.unlink(tmpf)

    def test_invalid_level_applies_lazy_downsampling(self):
        """Requesting a level beyond available levels applies lazy downsampling."""
        data = np.zeros((4, 16, 16), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tmpf = f.name
        try:
            _write_ome_tif(tmpf, data, axes="ZYX")
            # Single-level TIFF; level=2 should apply 4x lazy downsampling
            znii = ZarrNii.from_ome_tif(tmpf, level=2)
            # Original shape (C=1, Z=4, Y=16, X=16) downsampled by 4 -> (1, 1, 4, 4)
            assert znii.data.shape == (1, 1, 4, 4)
        finally:
            os.unlink(tmpf)

    def test_invalid_level_negative_raises(self):
        """Requesting a negative level raises ValueError."""
        data = np.zeros((4, 16, 16), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tmpf = f.name
        try:
            _write_ome_tif(tmpf, data, axes="ZYX")
            with pytest.raises(ValueError, match="Level must be >= 0"):
                ZarrNii.from_ome_tif(tmpf, level=-1)
        finally:
            os.unlink(tmpf)

    def test_invalid_series_raises(self):
        """Requesting a non-existent series raises ValueError."""
        data = np.zeros((4, 16, 16), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tmpf = f.name
        try:
            _write_ome_tif(tmpf, data, axes="ZYX")
            with pytest.raises(ValueError, match="[Ss]eries"):
                ZarrNii.from_ome_tif(tmpf, series=5)
        finally:
            os.unlink(tmpf)

    def test_from_file_dispatches_tif(self):
        """from_file dispatches .tif extension to from_ome_tif."""
        data = np.zeros((4, 16, 16), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tmpf = f.name
        try:
            _write_ome_tif(tmpf, data, axes="ZYX")
            znii = ZarrNii.from_file(tmpf)
            assert znii.data.shape == (1, 4, 16, 16)
        finally:
            os.unlink(tmpf)

    def test_from_file_dispatches_tiff(self):
        """from_file dispatches .tiff extension to from_ome_tif."""
        data = np.zeros((4, 16, 16), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as f:
            tmpf = f.name
        try:
            _write_ome_tif(tmpf, data, axes="ZYX")
            znii = ZarrNii.from_file(tmpf)
            assert znii.data.shape == (1, 4, 16, 16)
        finally:
            os.unlink(tmpf)

    def test_orientation_parameter(self):
        """orientation is passed to the ZarrNii instance."""
        data = np.zeros((4, 16, 16), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tmpf = f.name
        try:
            _write_ome_tif(tmpf, data, axes="ZYX")
            znii = ZarrNii.from_ome_tif(tmpf, orientation="LPI")
            assert znii.xyz_orientation == "LPI"
        finally:
            os.unlink(tmpf)

    def test_fallback_spacing_no_metadata(self):
        """Falls back to 1.0 spacing when no physical size metadata present."""
        data = np.zeros((4, 16, 16), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tmpf = f.name
        try:
            # Write minimal TIFF without physical size
            tifffile.imwrite(tmpf, data, photometric="minisblack")
            znii = ZarrNii.from_ome_tif(tmpf)
            assert znii.scale["z"] == pytest.approx(1.0)
            assert znii.scale["y"] == pytest.approx(1.0)
            assert znii.scale["x"] == pytest.approx(1.0)
            # Dims should include c, z, y, x
            assert "z" in znii.dims
            assert "y" in znii.dims
            assert "x" in znii.dims
            assert "c" in znii.dims
        finally:
            os.unlink(tmpf)
