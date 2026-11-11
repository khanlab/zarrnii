"""Tests for block-wise resampling (ZarrNii.apply_transform / zarrnii.resample)."""

import dask
import dask.array as da
import nibabel as nib
import numpy as np
import pytest

from zarrnii import AffineTransform, DisplacementTransform, ZarrNii
from zarrnii.resample import (
    apply_chain,
    compose_chain,
    evaluate_lattice,
    lattice_error,
    lattice_positions,
)

MM = {"x": "millimeter", "y": "millimeter", "z": "millimeter"}


def _ramp(shape, coef, spacing, origin, orientation="RAS", chunks=16, axes_order="ZYX"):
    """Image whose value is a linear function of physical position."""
    znii = ZarrNii.from_darr(
        da.zeros((1, *shape), chunks=(1, chunks, chunks, chunks), dtype=np.float64),
        axes_order=axes_order,
        orientation=orientation,
        spacing=spacing,
        origin=origin,
        axes_units=MM,
    )
    aff = znii.get_affine_matrix()
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"))
    phys = np.tensordot(aff[:3, :3], idx, axes=1) + aff[:3, 3, None, None, None]
    vals = np.tensordot(np.asarray(coef[:3]), phys, axes=1) + coef[3]
    znii.data = da.from_array(vals[None], chunks=(1, chunks, chunks, chunks))
    return znii


def _ref_physical(ref):
    aff = ref.get_affine_matrix()
    shape = ref.shape[-3:]
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"))
    return np.tensordot(aff[:3, :3], idx, axes=1) + aff[:3, 3, None, None, None]


def _inside(flo, pts_phys):
    """Mask of physical points inside the floating image's voxel extent."""
    inv = np.linalg.inv(flo.get_affine_matrix())
    vox = np.tensordot(inv[:3, :3], pts_phys, axes=1) + inv[:3, 3, None, None, None]
    n = np.array(flo.shape[-3:])[:, None, None, None]
    # stay away from the outer half voxel where values are clamped
    return np.all((vox >= 0) & (vox <= n - 1), axis=0)


@pytest.mark.parametrize("io", ["graph", "direct"])
@pytest.mark.parametrize("orientation", ["RAS", "LPI", "PIR"])
def test_affine_ramp_exact(io, orientation):
    """Linear interpolation of a linear ramp is exact under any affine."""
    coef = (2.0, -3.0, 0.5, 7.0)
    flo = _ramp((40, 36, 30), coef, (0.5, 0.4, 0.6), (1.0, -2.0, 3.0), orientation)
    centre = _centre(flo)
    ref = _ramp((25, 30, 28), (0, 0, 0, 0), (0.7, 0.55, 0.6), (0, 0, 0))
    # shift the reference so its centre is near the floating image's centre
    ref_centre = _centre(ref)
    ref.ngff_image.translation.update(
        {
            d: ref.ngff_image.translation[d] + (centre - ref_centre + 1.0)[i]
            for i, d in enumerate("xyz")
        }
    )
    ref.data = da.zeros(ref.shape, chunks=(1, 10, 11, 12))

    theta = 0.3
    rot = np.eye(4)
    rot[:3, :3] = [
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ]
    rot[:3, 3] = centre - rot[:3, :3] @ centre + [0.5, -0.5, 1.0]
    out = flo.apply_transform(AffineTransform.from_array(rot), ref_znimg=ref, io=io)
    result = out.data.compute()[0]

    pts = _ref_physical(ref)
    moved = np.tensordot(rot[:3, :3], pts, axes=1) + rot[:3, 3, None, None, None]
    expected = np.tensordot(np.asarray(coef[:3]), moved, axes=1) + coef[3]
    inside = _inside(flo, moved)
    assert inside.sum() > 1000
    np.testing.assert_allclose(result[inside], expected[inside], rtol=1e-5, atol=1e-4)
    # fully outside the floating image -> fill
    assert np.all(result[~_inside_extent(flo, moved)] == 0)


def _centre(img):
    aff = img.get_affine_matrix(axes_order="XYZ")
    shape_xyz = np.array(
        img.shape[-3:][::-1] if img.axes_order == "ZYX" else img.shape[-3:]
    )
    return aff[:3, :3] @ ((shape_xyz - 1) / 2) + aff[:3, 3]


def _inside_extent(flo, pts_phys):
    inv = np.linalg.inv(flo.get_affine_matrix())
    vox = np.tensordot(inv[:3, :3], pts_phys, axes=1) + inv[:3, 3, None, None, None]
    n = np.array(flo.shape[-3:])[:, None, None, None]
    return np.all((vox >= -0.5) & (vox <= n - 0.5), axis=0)


def test_identity_reproduces_input_including_edges():
    flo = _ramp((20, 22, 24), (1.0, 2.0, 3.0, 0.0), (0.01, 0.01, 0.01), (0, 0, 0))
    ref = flo.copy()
    out = flo.apply_transform(AffineTransform.identity(), ref_znimg=ref)
    np.testing.assert_allclose(out.data.compute(), flo.data.compute(), atol=1e-6)


def test_xyz_axes_order_matches_zyx():
    coef = (1.0, 2.0, -1.0, 0.0)
    flo_zyx = _ramp((20, 24, 28), coef, (0.5, 0.5, 0.5), (0, 0, 0))
    ref = _ramp((15, 16, 17), (0, 0, 0, 0), (0.6, 0.6, 0.6), (1, 1, 1))
    flo_xyz = ZarrNii.from_darr(
        flo_zyx.data.transpose(0, 3, 2, 1),
        axes_order="XYZ",
        orientation="RAS",
        spacing=(0.5, 0.5, 0.5),
        origin=(0, 0, 0),
        axes_units=MM,
    )
    a = flo_zyx.apply_transform(AffineTransform.identity(), ref_znimg=ref)
    b = flo_xyz.apply_transform(AffineTransform.identity(), ref_znimg=ref)
    np.testing.assert_allclose(a.data.compute(), b.data.compute(), atol=1e-6)


def test_multichannel_and_lazy_input():
    flo = _ramp((24, 24, 24), (1.0, 0.0, 0.0, 0.0), (1, 1, 1), (0, 0, 0))
    two = flo.copy()
    two.data = da.concatenate([flo.data, 2 * flo.data + 1], axis=0)
    ref = _ramp((20, 20, 20), (0, 0, 0, 0), (1, 1, 1), (2, 2, 2))
    out = two.apply_transform(AffineTransform.identity(), ref_znimg=ref)
    assert out.shape == (2, 20, 20, 20)
    res = out.data.compute()
    np.testing.assert_allclose(res[1], 2 * res[0] + 1, atol=1e-5)


def test_ome_zarr_channel_selection(tmp_path):
    flo = _ramp((24, 24, 24), (1.0, 0.0, 0.0, 0.0), (1, 1, 1), (0, 0, 0))
    two = flo.copy()
    two.data = da.concatenate([flo.data, 10 * flo.data], axis=0).astype(np.float32)
    path = tmp_path / "two.ome.zarr"
    two.to_ome_zarr(str(path), max_layer=0)
    ch1 = ZarrNii.from_ome_zarr(str(path), channels=[1])
    ref = _ramp((20, 20, 20), (0, 0, 0, 0), (1, 1, 1), (2, 2, 2))
    for io in ["graph", "direct"]:
        out = ch1.apply_transform(AffineTransform.identity(), ref_znimg=ref, io=io)
        assert out.shape == (1, 20, 20, 20)
        np.testing.assert_allclose(
            out.data.compute()[0],
            np.broadcast_to(10.0 * (np.arange(20) + 2), (20, 20, 20)),
            rtol=1e-6,
        )


def test_empty_blocks_have_no_input_dependencies():
    flo = _ramp((32, 32, 32), (1.0, 0, 0, 0), (1, 1, 1), (0, 0, 0), chunks=16)
    # reference grid extends far beyond the floating image along x
    ref = ZarrNii.from_darr(
        da.zeros((1, 16, 16, 64), chunks=(1, 16, 16, 16)),
        spacing=(1, 1, 1),
        origin=(8, 8, 16),
        axes_units=MM,
    )
    out = flo.apply_transform(AffineTransform.identity(), ref_znimg=ref, fill_value=-1)
    graph = dict(out.data.__dask_graph__())
    deps = {
        k: dask.core.get_dependencies(graph, k)
        for k in graph
        if isinstance(k, tuple) and k[0] == out.data.name
    }
    # blocks 2 and 3 along x start at x >= 48 > 31.5
    assert not deps[(out.data.name, 0, 0, 0, 3)]
    assert deps[(out.data.name, 0, 0, 0, 0)]
    res = out.data.compute()
    assert np.all(res[..., 32:] == -1)
    np.testing.assert_allclose(res[0, 0, 0, :16], np.arange(16, 32))


@pytest.mark.parametrize("io", ["graph", "direct"])
def test_nearest_preserves_labels_and_dtype(io):
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 5, size=(1, 30, 30, 30)).astype(np.uint16)
    flo = ZarrNii.from_darr(
        da.from_array(labels, chunks=(1, 10, 10, 10)), spacing=(1, 1, 1), axes_units=MM
    )
    ref = ZarrNii.from_darr(
        da.zeros((1, 40, 40, 40), chunks=(1, 16, 16, 16)),
        spacing=(0.7, 0.7, 0.7),
        axes_units=MM,
    )
    rot = AffineTransform.from_array(
        np.array([[0.9, 0.2, 0, 1], [-0.2, 0.9, 0, 0], [0, 0, 1, 0.5], [0, 0, 0, 1.0]])
    )
    out = flo.apply_transform(rot, ref_znimg=ref, interpolation="nearest", io=io)
    res = out.data.compute()
    assert res.dtype == np.uint16
    assert set(np.unique(res)) <= set(range(5))


def test_integer_linear_output_is_rounded():
    data = np.full((1, 8, 8, 8), 100, dtype=np.uint8)
    data[..., 4:] = 201
    flo = ZarrNii.from_darr(da.from_array(data), spacing=(1, 1, 1), axes_units=MM)
    shift = AffineTransform.from_array(
        np.array([[1, 0, 0, 0.5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    )
    out = flo.apply_transform(shift, ref_znimg=flo).data.compute()
    assert out.dtype == np.uint8
    assert out[0, 2, 2, 3] == 150  # midway 100 -> 201, rounded half to even


def test_ref_grid_zoom_preserves_field_of_view():
    for orientation in ["RAS", "LPI"]:
        ref = _ramp(
            (10, 12, 14), (0, 0, 0, 0), (0.025, 0.025, 0.025), (1, 2, 3), orientation
        )
        z = ref.ref_grid(zoom=10, chunks=64)
        assert z.shape == (1, 100, 120, 140)
        assert z.data.chunks[1][0] == 64
        np.testing.assert_allclose(z.get_zooms(), [0.0025] * 3)

        def corners(img):
            aff = img.get_affine_matrix()
            n = np.array(img.shape[-3:])
            lo = aff[:3, :3] @ np.full(3, -0.5) + aff[:3, 3]
            hi = aff[:3, :3] @ (n - 0.5) + aff[:3, 3]
            return np.sort(np.stack([lo, hi]), axis=0)

        np.testing.assert_allclose(corners(z), corners(ref), atol=1e-9)


def test_ref_zoom_resamples_ramp():
    coef = (1.0, -2.0, 3.0, 0.0)
    flo = _ramp((60, 60, 60), coef, (0.1, 0.1, 0.1), (0, 0, 0))
    template = _ramp((5, 5, 5), (0, 0, 0, 0), (1.0, 1.0, 1.0), (1, 1, 1))
    out = flo.apply_transform(
        AffineTransform.identity(), ref_znimg=template, ref_zoom=4, chunks=8
    )
    assert out.shape == (1, 20, 20, 20)
    res = out.data.compute()[0]
    pts = _ref_physical(out)
    expected = np.tensordot(np.asarray(coef[:3]), pts, axes=1)
    np.testing.assert_allclose(res, expected, atol=1e-4)


def _smooth_warp(path, shape, spacing, origin, amp):
    """ITK-convention (LPS) displacement field NIfTI covering a grid."""
    aff = np.diag([*spacing, 1.0])
    aff[:3, 3] = origin
    g = np.stack(
        np.meshgrid(*[np.linspace(0, np.pi, n) for n in shape], indexing="ij"), -1
    )
    disp = amp * np.sin(g)[..., ::-1] * np.cos(g)
    img = nib.Nifti1Image(disp[:, :, :, None, :].astype(np.float32), aff)
    img.header.set_intent("vector")
    img.to_filename(path)


def test_lattice_matches_exact_for_smooth_warp(tmp_path):
    flo = _ramp((40, 40, 40), (1.0, 1.0, 1.0, 0.0), (0.05, 0.05, 0.05), (0, 0, 0))
    flo.data = flo.data**2  # non-linear, so coordinate errors show
    warp_path = tmp_path / "warp.nii.gz"
    _smooth_warp(warp_path, (20, 20, 20), (0.1, 0.1, 0.1), (0.0, 0.0, 0.0), 0.1)
    warp = DisplacementTransform.from_nifti(str(warp_path))
    ref = flo.ref_grid(zoom=1.5, chunks=16)

    exact = flo.apply_transform(warp, ref_znimg=ref, coord_step=1).data.compute()
    approx = flo.apply_transform(warp, ref_znimg=ref).data.compute()
    direct = flo.apply_transform(warp, ref_znimg=ref, io="direct").data.compute()
    np.testing.assert_array_equal(approx, direct)
    # gradient of the image is ~2 * 6 = 12 per mm, 0.1 voxel = 0.005 mm
    assert np.abs(exact - approx).max() < 12 * 0.005


def test_lattice_error_estimate_affine_is_zero():
    m = np.array([[0.9, 0.1, 0, 3], [0, 1.1, 0.2, -2], [0.05, 0, 1, 1], [0, 0, 0, 1.0]])
    steps = compose_chain([AffineTransform.from_array(m)])
    positions = [lattice_positions(n, 8) for n in (50, 40, 33)]
    assert lattice_error(steps, positions)[0] < 1e-9
    # last lattice point is the last voxel, not beyond it
    assert positions[2][-1] == 32
    values, _ = evaluate_lattice(steps, positions)
    np.testing.assert_allclose(
        values[:, -1, -1, -1],
        apply_chain(steps, np.array([[49.0], [39.0], [32.0]]))[:, 0],
    )


def test_task_size_independent_of_warp_size(tmp_path):
    """Tasks carry a cropped piece of the warp, not the whole field."""
    import pickle

    flo = _ramp((64, 64, 64), (1.0, 0, 0, 0), (0.05, 0.05, 0.05), (0, 0, 0))
    warp_path = tmp_path / "warp.nii.gz"
    _smooth_warp(warp_path, (64, 64, 64), (0.05,) * 3, (0.0, 0.0, 0.0), 0.05)
    warp = DisplacementTransform.from_nifti(str(warp_path))
    out = flo.apply_transform(warp, ref_znimg=flo.ref_grid(chunks=16))
    graph = dict(out.data.__dask_graph__())
    task_bytes = max(
        len(pickle.dumps(v))
        for k, v in graph.items()
        if isinstance(k, tuple) and k[0] == out.data.name
    )
    assert task_bytes < warp.disp_xyz.nbytes / 20


def test_custom_transform_shared_once():
    class Shift:
        def apply_transform(self, vecs):
            out = np.array(vecs, dtype=float)
            out[0] += 0.5
            return out

    flo = _ramp((20, 20, 20), (1.0, 0, 0, 0), (1, 1, 1), (0, 0, 0), chunks=8)
    out = flo.apply_transform(Shift(), ref_znimg=flo, coord_step=4)
    graph = dict(out.data.__dask_graph__())
    assert sum(isinstance(v, Shift) for v in graph.values()) == 1
    res = out.data.compute()[0]
    np.testing.assert_allclose(res[:, :, :19], flo.data.compute()[0, :, :, :19] + 0.5)


def test_compose_chain_merges_affines():
    a = AffineTransform.from_array(np.diag([2.0, 2.0, 2.0, 1.0]))
    b = AffineTransform.from_array(
        np.array([[1, 0, 0, 1.0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    )
    steps = compose_chain([a, b])
    assert len(steps) == 1
    pts = np.array([[1.0], [1.0], [1.0]])
    np.testing.assert_allclose(apply_chain(steps, pts), b @ (a @ pts))


def test_displacement_transform_edges(tmp_path):
    """Half-voxel clamp at the field's edge, zero displacement beyond (as ITK)."""
    path = tmp_path / "w.nii.gz"
    disp = np.zeros((4, 4, 4, 1, 3), dtype=np.float32)
    disp[..., 2] = 1.0  # +1 mm along z (z is not flipped LPS -> RAS)
    nib.Nifti1Image(disp, np.eye(4)).to_filename(path)
    warp = DisplacementTransform.from_nifti(str(path))
    pts = np.array([[1.0, 3.4, 3.6, -0.4, -0.6], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    out = warp.apply_transform(pts)
    np.testing.assert_allclose(out[2] - pts[2], [1, 1, 0, 1, 0])


def test_matches_ants_apply_transforms(tmp_path):
    ants = pytest.importorskip("ants")

    rng = np.random.default_rng(1)
    # smooth floating image so tiny coordinate differences don't matter
    from scipy.ndimage import gaussian_filter

    flo_np = gaussian_filter(rng.random((30, 34, 28)), 3).astype(np.float32)
    flo_aff = np.array(
        [[-0.5, 0, 0, 8.0], [0, 0.45, 0, -7.0], [0, 0, 0.55, -6.0], [0, 0, 0, 1]]
    )
    nib.Nifti1Image(flo_np, flo_aff).to_filename(tmp_path / "flo.nii.gz")
    ref_aff = np.array(
        [[0.6, 0, 0, -6.0], [0, 0.6, 0, -6.5], [0, 0, 0.6, -6.0], [0, 0, 0, 1]]
    )
    nib.Nifti1Image(np.zeros((20, 22, 20), np.float32), ref_aff).to_filename(
        tmp_path / "ref.nii.gz"
    )
    _smooth_warp(
        tmp_path / "warp.nii.gz", (20, 22, 20), (0.6,) * 3, ref_aff[:3, 3], 0.8
    )

    expected = ants.apply_transforms(
        fixed=ants.image_read(str(tmp_path / "ref.nii.gz")),
        moving=ants.image_read(str(tmp_path / "flo.nii.gz")),
        transformlist=[str(tmp_path / "warp.nii.gz")],
        interpolator="linear",
    ).numpy()

    flo = ZarrNii.from_nifti(str(tmp_path / "flo.nii.gz"), chunks=(10, 10, 10))
    ref = ZarrNii.from_nifti(str(tmp_path / "ref.nii.gz"), chunks=(8, 8, 8))
    warp = DisplacementTransform.from_nifti(str(tmp_path / "warp.nii.gz"))
    for coord_step in [1, "auto"]:
        out = flo.apply_transform(warp, ref_znimg=ref, coord_step=coord_step)
        res = out.data.compute()[0]  # XYZ, like the NIfTI
        assert res.shape == expected.shape
        np.testing.assert_allclose(res, expected, atol=2e-3 * np.ptp(flo_np))
