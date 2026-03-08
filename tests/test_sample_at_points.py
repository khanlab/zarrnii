"""Tests for ZarrNii.sample_at_points – block-aware physical-space interpolation."""

import os
import tempfile

import dask.array as da
import numpy as np
import pytest
import zarr

from zarrnii import ZarrNii

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_identity_znii(
    shape=(1, 20, 20, 20),
    chunks=(1, 10, 10, 10),
    spacing=(1.0, 1.0, 1.0),
    origin=(0.0, 0.0, 0.0),
    axes_order="ZYX",
    dtype="f4",
):
    """Create a simple in-memory ZarrNii with known spacing/origin."""
    data = da.from_array(
        np.arange(np.prod(shape), dtype=dtype).reshape(shape),
        chunks=chunks,
    )
    return ZarrNii.from_darr(
        data,
        axes_order=axes_order,
        spacing=spacing,
        origin=origin,
    )


def _save_and_reload(znii, tmpdir, name="test.zarr"):
    """Save ZarrNii to OME-Zarr and reload so zarr store info is available."""
    store_path = os.path.join(tmpdir, name)
    znii.to_ome_zarr(store_path, max_layer=0)
    return ZarrNii.from_ome_zarr(store_path, level=0)


# ---------------------------------------------------------------------------
# Basic functionality tests
# ---------------------------------------------------------------------------


def test_sample_at_points_returns_correct_shape():
    """Output shape must be (C, N) for N query points and C channels."""
    znii = _make_identity_znii(shape=(2, 10, 10, 10))
    pts = np.zeros((5, 3))  # 5 points at the origin
    result = znii.sample_at_points(pts)
    assert result.shape == (2, 5)


def test_sample_at_points_single_point():
    """A single point passed as a length-3 1-D array must work."""
    znii = _make_identity_znii()
    pt = np.array([0.0, 0.0, 0.0])
    result = znii.sample_at_points(pt)
    assert result.shape == (1, 1)


def test_sample_at_points_transposed_input():
    """(3, N) input must be treated the same as (N, 3)."""
    znii = _make_identity_znii()
    pts_nx3 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])  # (2, 3)
    pts_3xn = pts_nx3.T  # (3, 2)
    result_nx3 = znii.sample_at_points(pts_nx3)
    result_3xn = znii.sample_at_points(pts_3xn)
    np.testing.assert_array_equal(result_nx3, result_3xn)


def test_sample_at_points_empty_input():
    """An empty array of points must return an empty (C, 0) array."""
    znii = _make_identity_znii()
    pts = np.zeros((0, 3))
    result = znii.sample_at_points(pts)
    assert result.shape == (1, 0)


# ---------------------------------------------------------------------------
# Physical coordinate correctness
# ---------------------------------------------------------------------------


def test_sample_at_points_known_values_nearest():
    """Nearest-neighbour sampling must recover exact voxel values."""
    # Build a 4-D array where voxel [0, z, y, x] = z * 100 + y * 10 + x
    shape = (1, 8, 8, 8)
    arr = np.zeros(shape, dtype="f4")
    for z in range(8):
        for y in range(8):
            for x in range(8):
                arr[0, z, y, x] = z * 100 + y * 10 + x

    znii = ZarrNii.from_darr(
        da.from_array(arr, chunks=(1, 4, 4, 4)),
        axes_order="ZYX",
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )

    # With spacing=1 and origin=0 the affine is identity, so physical == voxel.
    # Physical (x, y, z) = voxel (x, y, z); in ZYX data that means index [z, y, x].
    query = np.array(
        [
            [0.0, 0.0, 0.0],  # voxel (z=0,y=0,x=0) → 0
            [2.0, 3.0, 1.0],  # physical x=2,y=3,z=1 → voxel z=1,y=3,x=2 → 132
            [5.0, 5.0, 5.0],  # z=5,y=5,x=5 → 555
        ]
    )
    result = znii.sample_at_points(query, method="nearest")
    expected = np.array([0.0, 132.0, 555.0])
    np.testing.assert_allclose(result[0], expected, atol=1e-4)


def test_sample_at_points_known_values_linear():
    """Linear interpolation must return the weighted average at sub-voxel points."""
    # Flat array: all voxels = 1.0, so any interpolation returns 1.0
    shape = (1, 10, 10, 10)
    arr = np.ones(shape, dtype="f4")
    znii = ZarrNii.from_darr(
        da.from_array(arr, chunks=(1, 5, 5, 5)),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )
    pts = np.array([[1.5, 2.5, 3.5], [4.0, 4.0, 4.0]])
    result = znii.sample_at_points(pts, method="linear")
    np.testing.assert_allclose(result[0], [1.0, 1.0], atol=1e-6)


def test_sample_at_points_non_unit_spacing():
    """Physical coords must be correctly converted when spacing != 1."""
    # spacing = (2, 2, 2): physical coord (x=4, y=4, z=4) → voxel (2, 2, 2)
    shape = (1, 6, 6, 6)
    arr = np.zeros(shape, dtype="f4")
    arr[0, 2, 2, 2] = 99.0  # voxel (z=2, y=2, x=2) → physical (x=4, y=4, z=4)

    znii = ZarrNii.from_darr(
        da.from_array(arr, chunks=(1, 3, 3, 3)),
        axes_order="ZYX",
        spacing=(2.0, 2.0, 2.0),
        origin=(0.0, 0.0, 0.0),
    )
    # Physical (x=4, y=4, z=4) should land on voxel (z=2, y=2, x=2) = 99
    result = znii.sample_at_points(np.array([[4.0, 4.0, 4.0]]), method="nearest")
    np.testing.assert_allclose(result[0, 0], 99.0, atol=1e-4)


def test_sample_at_points_non_zero_origin():
    """Physical coords must be correctly converted when origin != 0."""
    # origin = (10, 10, 10), spacing = (1, 1, 1)
    # physical (x=11, y=11, z=11) → voxel (z=1, y=1, x=1)
    shape = (1, 5, 5, 5)
    arr = np.zeros(shape, dtype="f4")
    arr[0, 1, 1, 1] = 7.0

    znii = ZarrNii.from_darr(
        da.from_array(arr, chunks=(1, 5, 5, 5)),
        axes_order="ZYX",
        spacing=(1.0, 1.0, 1.0),
        origin=(10.0, 10.0, 10.0),
    )
    result = znii.sample_at_points(np.array([[11.0, 11.0, 11.0]]), method="nearest")
    np.testing.assert_allclose(result[0, 0], 7.0, atol=1e-4)


def test_sample_at_points_out_of_bounds():
    """Points outside the image domain must receive fill_value."""
    znii = _make_identity_znii(shape=(1, 5, 5, 5))
    pts = np.array([[1000.0, 1000.0, 1000.0]])  # far outside
    result = znii.sample_at_points(pts, fill_value=-1.0)
    np.testing.assert_allclose(result[0, 0], -1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Multi-channel
# ---------------------------------------------------------------------------


def test_sample_at_points_multichannel():
    """Each channel must be interpolated independently."""
    shape = (3, 8, 8, 8)
    arr = np.zeros(shape, dtype="f4")
    arr[0] = 1.0
    arr[1] = 2.0
    arr[2] = 3.0

    znii = ZarrNii.from_darr(
        da.from_array(arr, chunks=(1, 4, 4, 4)),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )
    pts = np.array([[2.0, 2.0, 2.0]])
    result = znii.sample_at_points(pts, method="nearest")
    assert result.shape == (3, 1)
    np.testing.assert_allclose(result[:, 0], [1.0, 2.0, 3.0], atol=1e-4)


# ---------------------------------------------------------------------------
# Zarr-backed (block-aware) path
# ---------------------------------------------------------------------------


def test_sample_at_points_zarr_backed():
    """Block-aware path via zarr store must match the in-memory path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        shape = (1, 20, 20, 20)
        arr = np.random.RandomState(0).rand(*shape).astype("f4")
        znii_mem = ZarrNii.from_darr(
            da.from_array(arr, chunks=(1, 10, 10, 10)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )

        # Save and reload so the zarr store path is available
        znii_zarr = _save_and_reload(znii_mem, tmpdir)

        # Verify that zarr store info is available (block-aware path taken)
        assert znii_zarr.get_zarr_store_info() is not None

        # Query points spread across the volume
        pts = np.array(
            [[1.0, 1.0, 1.0], [5.0, 5.0, 5.0], [10.0, 10.0, 10.0], [15.0, 15.0, 15.0]]
        )

        result_zarr = znii_zarr.sample_at_points(pts, method="linear")
        result_mem = znii_mem.sample_at_points(pts, method="linear")

        np.testing.assert_allclose(result_zarr, result_mem, atol=1e-5)


def test_sample_at_points_zarr_backed_nearest():
    """Nearest-neighbour via zarr store must match in-memory nearest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        shape = (1, 16, 16, 16)
        arr = np.random.RandomState(1).rand(*shape).astype("f4")
        znii_mem = ZarrNii.from_darr(
            da.from_array(arr, chunks=(1, 8, 8, 8)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )

        znii_zarr = _save_and_reload(znii_mem, tmpdir)

        pts = np.random.RandomState(2).uniform(0, 14, (20, 3))
        result_zarr = znii_zarr.sample_at_points(pts, method="nearest")
        result_mem = znii_mem.sample_at_points(pts, method="nearest")

        np.testing.assert_allclose(result_zarr, result_mem, atol=1e-5)


def test_sample_at_points_zarr_backed_points_across_chunks():
    """Points that span multiple zarr chunks must all be resolved correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Array with values equal to x-voxel index so we can verify via physical x
        shape = (1, 20, 20, 20)
        arr = np.zeros(shape, dtype="f4")
        for x in range(20):
            arr[0, :, :, x] = float(x)  # value = x-voxel index

        znii_mem = ZarrNii.from_darr(
            da.from_array(arr, chunks=(1, 5, 5, 5)),  # 4 chunks along each axis
            axes_order="ZYX",
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )
        znii_zarr = _save_and_reload(znii_mem, tmpdir)

        # Query with physical x varying — physical x == voxel x (spacing=1, origin=0)
        pts = np.array(
            [
                [0.0, 0.0, 0.0],  # x-voxel 0 → value 0
                [5.0, 0.0, 0.0],  # x-voxel 5 → value 5
                [10.0, 0.0, 0.0],  # x-voxel 10 → value 10
                [15.0, 0.0, 0.0],  # x-voxel 15 → value 15
                [19.0, 0.0, 0.0],  # x-voxel 19 → value 19
            ]
        )
        result = znii_zarr.sample_at_points(pts, method="nearest")
        expected = np.array([0.0, 5.0, 10.0, 15.0, 19.0])
        np.testing.assert_allclose(result[0], expected, atol=0.5)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_sample_at_points_bad_1d_input():
    """1-D input with wrong length must raise ValueError."""
    znii = _make_identity_znii()
    with pytest.raises(ValueError, match="3 elements"):
        znii.sample_at_points(np.array([1.0, 2.0]))


def test_sample_at_points_bad_2d_shape():
    """2-D input that is neither (N,3) nor (3,N) must raise ValueError."""
    znii = _make_identity_znii()
    with pytest.raises(ValueError, match=r"\(N, 3\) or \(3, N\)"):
        znii.sample_at_points(np.zeros((4, 4)))


def test_sample_at_points_bad_ndim():
    """3-D or higher input must raise ValueError."""
    znii = _make_identity_znii()
    with pytest.raises(ValueError, match="1-D or 2-D"):
        znii.sample_at_points(np.zeros((2, 3, 3)))
