"""Regression tests for issue #274.

Downsampling (in memory or when writing pyramid levels) shifts the translation
by half a block, so each coarse voxel centre lands on the centre of the fine
voxels it averages. On axes flipped in the affine (world coordinate decreasing
with voxel index) that shift must be negative.
"""

import dask.array as da
import numpy as np
import pytest

from zarrnii import ZarrNii

ORIENTATIONS = ["RAS", "LPI", "PIR", "RPS"]
AXES_ORDERS = ["ZYX", "XYZ"]


def _make(orientation, axes_order):
    return ZarrNii.from_darr(
        da.zeros((1, 8, 8, 8), dtype="float32"),
        spacing=(1.0, 2.0, 3.0),
        origin=(5.0, 6.0, 7.0),
        orientation=orientation,
        axes_order=axes_order,
    )


def _expected_downsampled_affine(A, factors):
    """Affine of a block-reduced image: scale columns by f, move to block 0 centre."""
    factors = np.asarray(factors, dtype=float)
    expected = A @ np.diag(np.r_[factors, 1.0])
    expected[:3, 3] = (A @ np.r_[(factors - 1) / 2, 1.0])[:3]
    return expected


@pytest.mark.parametrize("axes_order", AXES_ORDERS)
@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_downsample_affine_flipped_axes(orientation, axes_order):
    znimg = _make(orientation, axes_order)
    A = znimg.affine.matrix
    out = znimg.downsample(factors=4)
    np.testing.assert_allclose(
        out.affine.matrix, _expected_downsampled_affine(A, [4, 4, 4])
    )


@pytest.mark.parametrize("axes_order", AXES_ORDERS)
@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_downsample_affine_flipped_axes_anisotropic_factors(orientation, axes_order):
    znimg = _make(orientation, axes_order)
    A = znimg.affine.matrix
    # factors are given in the image's spatial axes order
    factors = [2, 4, 1]
    out = znimg.downsample(factors=factors)
    np.testing.assert_allclose(
        out.affine.matrix, _expected_downsampled_affine(A, factors)
    )


@pytest.mark.parametrize("backend", ["ome-zarr-py", "ngff-zarr"])
@pytest.mark.parametrize("axes_order", AXES_ORDERS)
@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_pyramid_level_affine_flipped_axes(tmp_path, orientation, axes_order, backend):
    znimg = _make(orientation, axes_order)
    path = str(tmp_path / "img.ome.zarr")
    znimg.to_ome_zarr(path, max_layer=2, backend=backend)

    A0 = ZarrNii.from_file(path, level=0).affine.matrix
    A1 = ZarrNii.from_file(path, level=1).affine.matrix
    # Level 1 is checked against level 0 as read back (not against znimg),
    # which is what the pyramid must keep consistent.
    assert np.all(np.sign(np.diag(np.linalg.inv(A0[:3, :3]) @ A1[:3, :3])) > 0)

    # Per-voxel-axis factors between the levels (2 for each downsampled axis)
    factors = np.abs(np.diag(np.linalg.inv(A0[:3, :3]) @ A1[:3, :3]))
    assert np.all(factors >= 1)
    np.testing.assert_allclose(A1, _expected_downsampled_affine(A0, factors))
