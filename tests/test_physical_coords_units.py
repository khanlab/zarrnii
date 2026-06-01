"""Tests for physical-coordinate unit conversion in crop(), crop_centered(), and sample_at_points()."""

import dask.array as da
import numpy as np
import pytest

from zarrnii import ZarrNii

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_MM_UNITS = {"x": "millimeter", "y": "millimeter", "z": "millimeter"}
_UM_UNITS = {"x": "micrometer", "y": "micrometer", "z": "micrometer"}


def _make_znii(
    shape=(1, 20, 20, 20),
    spacing=(1.0, 1.0, 1.0),
    origin=(0.0, 0.0, 0.0),
    axes_order="ZYX",
    axes_units=None,
):
    """Create a simple in-memory ZarrNii with known metadata."""
    data = da.zeros(shape, dtype="f4")
    return ZarrNii.from_darr(
        data,
        axes_order=axes_order,
        spacing=spacing,
        origin=origin,
        axes_units=axes_units,
    )


# ===========================================================================
# _convert_physical_coords_units helper (internal)
# ===========================================================================


class TestConvertPhysicalCoordsUnits:
    """Unit tests for the private _convert_physical_coords_units helper."""

    def _fn(self, xyz, from_units, to_units):
        from zarrnii.core import _convert_physical_coords_units

        return _convert_physical_coords_units(xyz, from_units, to_units)

    def test_no_conversion_same_units(self):
        result = self._fn((1.0, 2.0, 3.0), _MM_UNITS, _MM_UNITS)
        assert tuple(result) == (1.0, 2.0, 3.0)

    def test_no_conversion_both_none(self):
        result = self._fn((1.0, 2.0, 3.0), None, None)
        assert tuple(result) == (1.0, 2.0, 3.0)

    def test_mm_to_um_tuple(self):
        result = self._fn((1.0, 2.0, 3.0), _MM_UNITS, _UM_UNITS)
        assert tuple(result) == pytest.approx((1000.0, 2000.0, 3000.0))

    def test_um_to_mm_tuple(self):
        result = self._fn((1000.0, 2000.0, 3000.0), _UM_UNITS, _MM_UNITS)
        assert tuple(result) == pytest.approx((1.0, 2.0, 3.0))

    def test_mm_to_um_array_1d(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = self._fn(arr, _MM_UNITS, _UM_UNITS)
        np.testing.assert_allclose(result, [1000.0, 2000.0, 3000.0])

    def test_mm_to_um_array_n3(self):
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = self._fn(arr, _MM_UNITS, _UM_UNITS)
        np.testing.assert_allclose(
            result, [[1000.0, 2000.0, 3000.0], [4000.0, 5000.0, 6000.0]]
        )

    def test_from_none_defaults_to_mm(self):
        """None from_units should default to millimeter per axis."""
        result = self._fn((1.0, 2.0, 3.0), None, _UM_UNITS)
        assert tuple(result) == pytest.approx((1000.0, 2000.0, 3000.0))

    def test_to_none_defaults_to_mm(self):
        """None to_units should default to millimeter per axis."""
        result = self._fn((1000.0, 2000.0, 3000.0), _UM_UNITS, None)
        assert tuple(result) == pytest.approx((1.0, 2.0, 3.0))

    def test_partial_axes(self):
        """Per-axis units: only z differs."""
        from zarrnii.core import _convert_physical_coords_units

        result = _convert_physical_coords_units(
            (1.0, 2.0, 3.0),
            {"x": "millimeter", "y": "millimeter", "z": "micrometer"},
            {"x": "millimeter", "y": "millimeter", "z": "millimeter"},
        )
        # z: 3 um -> 0.003 mm; x,y unchanged
        assert tuple(result) == pytest.approx((1.0, 2.0, 0.003))

    def test_invalid_from_unit_raises(self):
        from zarrnii.core import _convert_physical_coords_units

        with pytest.raises(ValueError, match="Unsupported spatial unit"):
            _convert_physical_coords_units(
                (1.0, 2.0, 3.0), {"x": "parsec", "y": "mm", "z": "mm"}, None
            )


# ===========================================================================
# crop() with physical_coords=True and coords_units
# ===========================================================================


class TestCropCoordsUnits:
    """Test that crop() converts coordinates when coords_units != image units."""

    def _make_mm_image(self):
        """10-voxel cube with 1 mm/voxel, image in millimeter."""
        return _make_znii(
            shape=(1, 10, 10, 10),
            spacing=(1.0, 1.0, 1.0),
            axes_units=_MM_UNITS,
        )

    def _make_um_image(self):
        """10-voxel cube with 1000 µm/voxel (= 1 mm/voxel), image in micrometer."""
        return _make_znii(
            shape=(1, 10, 10, 10),
            spacing=(1000.0, 1000.0, 1000.0),
            axes_units=_UM_UNITS,
        )

    def test_crop_mm_image_with_mm_coords(self):
        """Baseline: mm coords on mm image, no conversion needed."""
        znii = self._make_mm_image()
        cropped = znii.crop(
            (2.0, 2.0, 2.0),
            (5.0, 5.0, 5.0),
            physical_coords=True,
            coords_units=_MM_UNITS,
        )
        assert cropped.shape[1:] == (3, 3, 3)

    def test_crop_um_image_with_mm_coords_explicit(self):
        """Crop micrometer image using millimeter coords (explicit units)."""
        znii = self._make_um_image()
        # 2 mm = 2000 um; 5 mm = 5000 um. spacing=1000 um/vox → voxels 2-5
        cropped = znii.crop(
            (2.0, 2.0, 2.0),
            (5.0, 5.0, 5.0),
            physical_coords=True,
            coords_units=_MM_UNITS,
        )
        assert cropped.shape[1:] == (3, 3, 3)

    def test_crop_um_image_with_mm_coords_default(self):
        """Default coords_units=None (mm) on a micrometer image."""
        znii = self._make_um_image()
        cropped = znii.crop(
            (2.0, 2.0, 2.0),
            (5.0, 5.0, 5.0),
            physical_coords=True,
        )
        assert cropped.shape[1:] == (3, 3, 3)

    def test_crop_mm_image_with_um_coords(self):
        """Crop millimeter image using micrometer coords."""
        znii = self._make_mm_image()
        # 2000 um = 2 mm; 5000 um = 5 mm → should give same result as mm crop above
        cropped = znii.crop(
            (2000.0, 2000.0, 2000.0),
            (5000.0, 5000.0, 5000.0),
            physical_coords=True,
            coords_units=_UM_UNITS,
        )
        assert cropped.shape[1:] == (3, 3, 3)

    def test_crop_mm_um_equivalence(self):
        """Crops with mm coords on mm image and µm coords on µm image are identical."""
        znii_mm = self._make_mm_image()
        znii_um = self._make_um_image()

        cropped_mm = znii_mm.crop(
            (1.0, 1.0, 1.0), (4.0, 4.0, 4.0), physical_coords=True
        )
        cropped_um = znii_um.crop(
            (1000.0, 1000.0, 1000.0),
            (4000.0, 4000.0, 4000.0),
            physical_coords=True,
            coords_units=_UM_UNITS,
        )
        assert cropped_mm.shape == cropped_um.shape

    def test_crop_invalid_coords_units_raises(self):
        znii = self._make_mm_image()
        with pytest.raises(ValueError, match="Invalid axes_units value"):
            znii.crop(
                (1.0, 1.0, 1.0),
                (4.0, 4.0, 4.0),
                physical_coords=True,
                coords_units={"x": "mm"},
            )

    def test_crop_coords_units_ignored_for_voxel_coords(self):
        """coords_units is silently ignored when physical_coords=False."""
        znii = self._make_mm_image()
        cropped = znii.crop(
            (2, 2, 2),
            (5, 5, 5),
            physical_coords=False,
            coords_units=_UM_UNITS,
        )
        assert cropped.shape[1:] == (3, 3, 3)

    def test_crop_batch_propagates_units(self):
        """Batch crop passes coords_units to each individual crop call."""
        znii = self._make_um_image()
        bboxes = [
            ((1.0, 1.0, 1.0), (3.0, 3.0, 3.0)),
            ((4.0, 4.0, 4.0), (7.0, 7.0, 7.0)),
        ]
        results = znii.crop(bboxes, physical_coords=True, coords_units=_MM_UNITS)
        assert len(results) == 2
        assert results[0].shape[1:] == (2, 2, 2)
        assert results[1].shape[1:] == (3, 3, 3)


# ===========================================================================
# crop_centered() with centers_units
# ===========================================================================


class TestCropCenteredUnitsConversion:
    """Test that crop_centered() converts centers when centers_units != image units."""

    def _make_mm_image(self):
        return _make_znii(
            shape=(1, 20, 20, 20),
            spacing=(1.0, 1.0, 1.0),
            axes_units=_MM_UNITS,
        )

    def _make_um_image(self):
        """20-voxel cube with 1000 µm/voxel spacing, image in micrometer."""
        return _make_znii(
            shape=(1, 20, 20, 20),
            spacing=(1000.0, 1000.0, 1000.0),
            axes_units=_UM_UNITS,
        )

    def test_mm_image_mm_units_baseline(self):
        """Baseline: mm center on mm image produces correct patch."""
        znii = self._make_mm_image()
        patch = znii.crop_centered(
            (10.0, 10.0, 10.0), patch_size=(4, 4, 4), centers_units=_MM_UNITS
        )
        assert patch.shape[1:] == (4, 4, 4)

    def test_mm_image_default_units(self):
        """Default centers_units=None (mm) on mm image."""
        znii = self._make_mm_image()
        patch = znii.crop_centered((10.0, 10.0, 10.0), patch_size=(4, 4, 4))
        assert patch.shape[1:] == (4, 4, 4)

    def test_um_image_mm_centers(self):
        """Centers in mm, image in µm → coords converted before patch extraction."""
        znii = self._make_um_image()
        # Center 10 mm = 10000 um; spacing 1000 um/vox → voxel center 10
        patch = znii.crop_centered(
            (10.0, 10.0, 10.0),
            patch_size=(4, 4, 4),
        )
        assert patch.shape[1:] == (4, 4, 4)

    def test_mm_image_um_centers(self):
        """Centers in µm, image in mm → coords converted."""
        znii = self._make_mm_image()
        # 10000 um = 10 mm on 1 mm/vox image → voxel center 10
        patch = znii.crop_centered(
            (10000.0, 10000.0, 10000.0),
            patch_size=(4, 4, 4),
            centers_units=_UM_UNITS,
        )
        assert patch.shape[1:] == (4, 4, 4)

    def test_mm_um_centers_equivalence(self):
        """mm center on mm image == µm center on µm image (same spatial location)."""
        znii_mm = self._make_mm_image()
        znii_um = self._make_um_image()

        patch_mm = znii_mm.crop_centered((10.0, 10.0, 10.0), patch_size=(4, 4, 4))
        patch_um = znii_um.crop_centered(
            (10000.0, 10000.0, 10000.0),
            patch_size=(4, 4, 4),
            centers_units=_UM_UNITS,
        )
        assert patch_mm.shape == patch_um.shape

    def test_batch_centers_propagates_units(self):
        """Batch mode passes centers_units to each individual crop_centered call."""
        znii = self._make_um_image()
        centers = [(5.0, 5.0, 5.0), (10.0, 10.0, 10.0)]
        patches = znii.crop_centered(
            centers, patch_size=(4, 4, 4), centers_units=_MM_UNITS
        )
        assert len(patches) == 2
        for p in patches:
            assert p.shape[1:] == (4, 4, 4)

    def test_invalid_centers_units_raises(self):
        znii = self._make_mm_image()
        with pytest.raises(ValueError, match="Invalid axes_units value"):
            znii.crop_centered(
                (5.0, 5.0, 5.0), patch_size=(4, 4, 4), centers_units={"x": "um"}
            )


# ===========================================================================
# sample_at_points() with points_units
# ===========================================================================


class TestSampleAtPointsUnitsConversion:
    """Test that sample_at_points() converts points when points_units != image units."""

    def _make_image(self, spacing, axes_units):
        """8-voxel cube with voxel value = z-index (channel 0)."""
        shape = (1, 8, 8, 8)
        arr = np.zeros(shape, dtype="f4")
        for z in range(8):
            arr[0, z, :, :] = float(z)
        return ZarrNii.from_darr(
            da.from_array(arr, chunks=(1, 4, 4, 4)),
            axes_order="ZYX",
            spacing=spacing,
            origin=(0.0, 0.0, 0.0),
            axes_units=axes_units,
        )

    def test_mm_image_mm_points_default(self):
        """Default points_units=None (mm) on mm image: no conversion needed."""
        znii = self._make_image(spacing=(1.0, 1.0, 1.0), axes_units=_MM_UNITS)
        # Physical z=2 → voxel z=2 → value 2.0
        result = znii.sample_at_points(np.array([[0.0, 0.0, 2.0]]), method="nearest")
        np.testing.assert_allclose(result[0, 0], 2.0, atol=1e-4)

    def test_mm_image_mm_points_explicit(self):
        """Explicit mm units on mm image."""
        znii = self._make_image(spacing=(1.0, 1.0, 1.0), axes_units=_MM_UNITS)
        result = znii.sample_at_points(
            np.array([[0.0, 0.0, 3.0]]), method="nearest", points_units=_MM_UNITS
        )
        np.testing.assert_allclose(result[0, 0], 3.0, atol=1e-4)

    def test_um_image_mm_points(self):
        """Points in mm, image in µm (spacing 1000 µm/vox = 1 mm/vox)."""
        znii = self._make_image(spacing=(1000.0, 1000.0, 1000.0), axes_units=_UM_UNITS)
        # Physical z=2 mm = 2000 µm; spacing 1000 µm/vox → voxel z=2 → value 2.0
        result = znii.sample_at_points(np.array([[0.0, 0.0, 2.0]]), method="nearest")
        np.testing.assert_allclose(result[0, 0], 2.0, atol=1e-4)

    def test_mm_image_um_points(self):
        """Points in µm, image in mm (spacing 1 mm/vox)."""
        znii = self._make_image(spacing=(1.0, 1.0, 1.0), axes_units=_MM_UNITS)
        # z=3000 µm = 3 mm → voxel z=3 → value 3.0
        result = znii.sample_at_points(
            np.array([[0.0, 0.0, 3000.0]]),
            method="nearest",
            points_units=_UM_UNITS,
        )
        np.testing.assert_allclose(result[0, 0], 3.0, atol=1e-4)

    def test_mm_um_equivalence(self):
        """Same physical location sampled via mm and µm gives identical result."""
        znii_mm = self._make_image(spacing=(1.0, 1.0, 1.0), axes_units=_MM_UNITS)
        znii_um = self._make_image(
            spacing=(1000.0, 1000.0, 1000.0), axes_units=_UM_UNITS
        )

        # Physical location z=4
        result_mm = znii_mm.sample_at_points(
            np.array([[0.0, 0.0, 4.0]]), method="nearest"
        )
        result_um = znii_um.sample_at_points(
            np.array([[0.0, 0.0, 4.0]]), method="nearest"
        )
        np.testing.assert_allclose(result_mm, result_um, atol=1e-4)

    def test_invalid_points_units_raises(self):
        znii = self._make_image(spacing=(1.0, 1.0, 1.0), axes_units=_MM_UNITS)
        with pytest.raises(ValueError, match="Invalid axes_units value"):
            znii.sample_at_points(np.array([[0.0, 0.0, 1.0]]), points_units={"x": "mm"})

    def test_image_no_axes_units_no_conversion(self):
        """When image has no axes_units (None), coordinates are assumed in mm."""
        znii = self._make_image(spacing=(1.0, 1.0, 1.0), axes_units=None)
        # No axes_units → treated as mm → no conversion for default points_units
        result = znii.sample_at_points(np.array([[0.0, 0.0, 2.0]]), method="nearest")
        np.testing.assert_allclose(result[0, 0], 2.0, atol=1e-4)
