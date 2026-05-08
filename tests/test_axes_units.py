"""Tests for axes_units parameter in ZarrNii creation functions."""

import dask.array as da
import numpy as np
import pytest

from zarrnii import VALID_AXES_UNITS, ZarrNii


@pytest.fixture
def data_4d():
    """4-D dask array (C, Z, Y, X)."""
    return da.zeros((1, 8, 16, 16), dtype=np.uint16)


class TestValidAxesUnitsConstant:
    """Tests for the VALID_AXES_UNITS module-level constant."""

    def test_valid_axes_units_is_frozenset(self):
        assert isinstance(VALID_AXES_UNITS, frozenset)

    def test_common_units_present(self):
        for unit in ("micrometer", "millimeter", "nanometer", "meter", "centimeter"):
            assert unit in VALID_AXES_UNITS

    def test_does_not_contain_invalid(self):
        assert "um" not in VALID_AXES_UNITS
        assert "mm" not in VALID_AXES_UNITS
        assert "microns" not in VALID_AXES_UNITS


class TestFromDarrAxesUnits:
    """Tests for axes_units parameter in from_darr."""

    def test_axes_units_stored(self, data_4d):
        units = {"x": "micrometer", "y": "micrometer", "z": "micrometer"}
        znii = ZarrNii.from_darr(data_4d, axes_units=units)
        assert znii.ngff_image.axes_units == units

    def test_axes_units_default_none(self, data_4d):
        znii = ZarrNii.from_darr(data_4d)
        assert znii.ngff_image.axes_units is None

    def test_axes_units_millimeter(self, data_4d):
        units = {"x": "millimeter", "y": "millimeter", "z": "millimeter"}
        znii = ZarrNii.from_darr(data_4d, axes_units=units)
        assert znii.ngff_image.axes_units.get("x") == "millimeter"

    def test_axes_units_partial(self, data_4d):
        """Partial axes_units mapping (only some axes) is accepted."""
        units = {"z": "nanometer"}
        znii = ZarrNii.from_darr(data_4d, axes_units=units)
        assert znii.ngff_image.axes_units == units

    def test_axes_units_invalid_raises(self, data_4d):
        with pytest.raises(ValueError, match="Invalid axes_units value"):
            ZarrNii.from_darr(data_4d, axes_units={"x": "um"})

    def test_axes_units_invalid_full_message(self, data_4d):
        with pytest.raises(ValueError, match="OME-Zarr accepted space units"):
            ZarrNii.from_darr(data_4d, axes_units={"z": "microns"})

    def test_axes_units_mixed_valid_invalid_raises(self, data_4d):
        with pytest.raises(ValueError, match="Invalid axes_units value"):
            ZarrNii.from_darr(
                data_4d, axes_units={"x": "micrometer", "y": "mm", "z": "micrometer"}
            )


class TestFromTifStackAxesUnits:
    """Tests for axes_units parameter in from_tif_stack."""

    @pytest.fixture
    def tif_paths(self, tmp_path):
        """Create minimal 2-D TIFF slices for testing."""
        try:
            import tifffile
        except ImportError:
            pytest.skip("tifffile not available")
        paths = []
        for i in range(3):
            p = tmp_path / f"slice_{i:02d}.tif"
            tifffile.imwrite(str(p), np.zeros((8, 8), dtype=np.uint16))
            paths.append(str(p))
        return paths

    def test_axes_units_stored(self, tif_paths):
        units = {"x": "micrometer", "y": "micrometer", "z": "micrometer"}
        znii = ZarrNii.from_tif_stack(tif_paths, axes_units=units)
        assert znii.ngff_image.axes_units == units

    def test_axes_units_default_none(self, tif_paths):
        znii = ZarrNii.from_tif_stack(tif_paths)
        assert znii.ngff_image.axes_units is None

    def test_axes_units_invalid_raises(self, tif_paths):
        with pytest.raises(ValueError, match="Invalid axes_units value"):
            ZarrNii.from_tif_stack(tif_paths, axes_units={"x": "um"})


class TestFromImarisAxesUnits:
    """Tests for axes_units parameter in from_imaris."""

    @pytest.fixture
    def imaris_path(self, tmp_path):
        """Create a minimal valid Imaris (.ims) file."""
        path = str(tmp_path / "test.ims")
        data = da.from_array(np.zeros((1, 4, 4, 4), dtype=np.uint16), chunks="auto")
        ZarrNii.from_darr(data).to_imaris(path)
        return path

    def test_axes_units_stored(self, imaris_path):
        units = {"x": "micrometer", "y": "micrometer", "z": "micrometer"}
        znii = ZarrNii.from_imaris(imaris_path, axes_units=units)
        assert znii.ngff_image.axes_units == units

    def test_axes_units_default_none(self, imaris_path):
        znii = ZarrNii.from_imaris(imaris_path)
        assert znii.ngff_image.axes_units is None

    def test_axes_units_invalid_raises(self, imaris_path):
        with pytest.raises(ValueError, match="Invalid axes_units value"):
            ZarrNii.from_imaris(imaris_path, axes_units={"z": "mm"})


class TestFromOmeTifAxesUnitsOverride:
    """Tests for axes_units override parameter in from_ome_tif."""

    @pytest.fixture
    def ome_tif_path(self, tmp_path):
        """Create a minimal OME-TIFF file."""
        try:
            import tifffile
        except ImportError:
            pytest.skip("tifffile not available")
        path = str(tmp_path / "test.ome.tif")
        data = np.zeros((4, 8, 8), dtype=np.uint16)
        tifffile.imwrite(path, data, photometric="minisblack")
        return path

    def test_axes_units_override_stored(self, ome_tif_path):
        units = {"x": "nanometer", "y": "nanometer", "z": "nanometer"}
        znii = ZarrNii.from_ome_tif(ome_tif_path, axes_units=units)
        assert znii.ngff_image.axes_units == units

    def test_axes_units_no_override_uses_metadata(self, ome_tif_path):
        """Without override the unit from file metadata is used (default micrometer)."""
        znii = ZarrNii.from_ome_tif(ome_tif_path)
        # The fixture has no OME-XML, so it falls back to micrometer default
        assert znii.ngff_image.axes_units is not None
        assert znii.ngff_image.axes_units.get("x") == "micrometer"

    def test_axes_units_invalid_raises(self, ome_tif_path):
        with pytest.raises(ValueError, match="Invalid axes_units value"):
            ZarrNii.from_ome_tif(ome_tif_path, axes_units={"x": "um"})
