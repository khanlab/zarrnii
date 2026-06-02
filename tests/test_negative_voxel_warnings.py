"""Tests for negative voxel coordinate warnings in spatial transform functions.

When physical coordinates are transformed to voxel space using the inverse
affine and the resulting voxel coordinates are negative, a UserWarning should
be emitted.  Negative voxel coordinates can indicate an orientation mismatch
between the supplied coordinates and the image's affine transform.
"""

import warnings

import dask.array as da
import numpy as np
import pytest

from zarrnii import ZarrNii

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_znii(
    shape=(1, 20, 20, 20),
    spacing=(1.0, 1.0, 1.0),
    origin=(0.0, 0.0, 0.0),
    axes_order="ZYX",
):
    """Create a simple in-memory ZarrNii with known spacing/origin."""
    data = da.from_array(
        np.arange(np.prod(shape), dtype="f4").reshape(shape),
        chunks=shape,
    )
    return ZarrNii.from_darr(
        data,
        axes_order=axes_order,
        spacing=spacing,
        origin=origin,
    )


# ---------------------------------------------------------------------------
# crop() warnings
# ---------------------------------------------------------------------------


def test_crop_physical_coords_no_warning_for_valid_coords():
    """crop() must NOT warn when all voxel coordinates are non-negative."""
    znii = _make_znii(origin=(0.0, 0.0, 0.0))
    # Physical coords inside the image – voxel coords will be positive

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        znii.crop(
            (1.0, 1.0, 1.0),
            (5.0, 5.0, 5.0),
            physical_coords=True,
        )
    # Should produce no negative-voxel warning
    neg_warns = [w for w in rec if "Negative voxel coordinates" in str(w.message)]
    assert len(neg_warns) == 0, "Unexpected negative-voxel warning for valid coords"


def test_crop_physical_coords_warning_for_negative_coords():
    """crop() must warn when physical→voxel conversion yields negative coords.

    We supply physical bbox_min coordinates that, after applying the inverse
    affine, land at negative voxel indices.  The image origin is at (0,0,0)
    with positive spacing so any negative physical coordinate maps to a
    negative voxel index.
    """
    znii = _make_znii(origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
    # Negative physical coordinates → negative voxel coordinates
    with pytest.warns(UserWarning, match="Negative voxel coordinates"):
        znii.crop(
            (-5.0, -5.0, -5.0),
            (5.0, 5.0, 5.0),
            physical_coords=True,
        )


def test_crop_voxel_coords_no_warning():
    """crop() must NOT warn when voxel coordinates are supplied directly."""
    znii = _make_znii()
    # Pass voxel coords directly – no affine inversion happens

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            znii.crop((0, 0, 0), (5, 5, 5))
        except Exception:
            pass
    neg_warns = [w for w in rec if "Negative voxel coordinates" in str(w.message)]
    assert len(neg_warns) == 0


# ---------------------------------------------------------------------------
# crop_centered() warnings
# ---------------------------------------------------------------------------


def test_crop_centered_no_warning_for_valid_center():
    """crop_centered() must NOT warn for a center inside the image domain."""
    znii = _make_znii(origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
    # Center well inside the image

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        znii.crop_centered(
            (10.0, 10.0, 10.0),
            patch_size=(4, 4, 4),
        )
    neg_warns = [w for w in rec if "Negative voxel coordinates" in str(w.message)]
    assert len(neg_warns) == 0


def test_crop_centered_warning_for_negative_voxel_center():
    """crop_centered() must warn when the center maps to negative voxel coords."""
    znii = _make_znii(origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
    # A strongly negative physical center maps to negative voxels
    with pytest.warns(UserWarning, match="Negative voxel coordinates"):
        znii.crop_centered(
            (-50.0, -50.0, -50.0),
            patch_size=(4, 4, 4),
        )


# ---------------------------------------------------------------------------
# sample_at_points() warnings
# ---------------------------------------------------------------------------


def test_sample_at_points_no_warning_for_valid_coords():
    """sample_at_points() must NOT warn when all points map to positive voxels."""
    znii = _make_znii(origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
    pts = np.array([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]])

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        znii.sample_at_points(pts)
    neg_warns = [w for w in rec if "Negative voxel coordinates" in str(w.message)]
    assert len(neg_warns) == 0


def test_sample_at_points_warning_for_negative_voxel_coords():
    """sample_at_points() must warn when any point maps to a negative voxel."""
    znii = _make_znii(origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
    # One valid point and one with negative physical coordinates
    pts = np.array([[5.0, 5.0, 5.0], [-10.0, -10.0, -10.0]])
    with pytest.warns(UserWarning, match="Negative voxel coordinates"):
        znii.sample_at_points(pts)


def test_sample_at_points_warning_reports_count():
    """sample_at_points() warning should mention the number of affected points."""
    znii = _make_znii(origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
    pts = np.array([[-1.0, -1.0, -1.0], [-2.0, -2.0, -2.0], [5.0, 5.0, 5.0]])
    with pytest.warns(UserWarning, match="2 of 3"):
        znii.sample_at_points(pts)
