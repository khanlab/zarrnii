"""Tests for max_layer="auto" (issue #279)."""

import dask.array as da
import ngff_zarr as nz
import numpy as np
import pytest
import zarr

from zarrnii import ZarrNii
from zarrnii.core import (
    AUTO_PYRAMID_MIN_SIZE,
    _compute_isotropic_scale_factors,
    _resolve_max_layer,
)


def _ngff(shape_zyx, scale_zyx=(1.0, 1.0, 1.0)):
    return nz.to_ngff_image(
        da.zeros((1,) + tuple(shape_zyx), dtype="uint8", chunks=(1, 64, 64, 64)),
        dims=["c", "z", "y", "x"],
        scale=dict(zip("zyx", scale_zyx)),
        translation={d: 0.0 for d in "zyx"},
    )


def _auto_levels(img):
    return _resolve_max_layer(
        img, "auto", lambda n: _compute_isotropic_scale_factors(img, n)
    )


@pytest.mark.parametrize(
    "shape, expected",
    [
        ((3200, 5280, 4560), 7),  # example from the issue
        ((2208, 8576, 13568), 8),  # Imaris file from the issue
        ((100, 100, 100), 1),
        ((128, 128, 128), 1),
        ((129, 10, 10), 2),
        ((512, 512, 512), 3),
    ],
)
def test_auto_levels_isotropic(shape, expected):
    img = _ngff(shape)
    assert _auto_levels(img) == expected
    # matches 1 + ceil(log2(max / 128)) for isotropic data
    assert expected == 1 + max(0, int(np.ceil(np.log2(max(shape) / 128))))


@pytest.mark.parametrize(
    "shape, scale",
    [
        ((256, 2048, 2048), (4.0, 1.0, 1.0)),  # coarse z
        ((2048, 256, 256), (1.0, 4.0, 4.0)),  # coarse y, x
    ],
)
def test_auto_levels_anisotropic(shape, scale):
    img = _ngff(shape, scale)
    n = _auto_levels(img)
    coarsest = _compute_isotropic_scale_factors(img, n)[-1]
    assert all(s // coarsest[d] <= AUTO_PYRAMID_MIN_SIZE for d, s in zip("zyx", shape))
    # one level fewer would leave some dim above the target
    prev = _compute_isotropic_scale_factors(img, n - 1)
    prev_coarsest = prev[-1] if prev else {d: 1 for d in "zyx"}
    assert any(
        s // prev_coarsest[d] > AUTO_PYRAMID_MIN_SIZE for d, s in zip("zyx", shape)
    )


def test_explicit_int_is_unchanged():
    img = _ngff((1000, 1000, 1000))
    assert _resolve_max_layer(img, 4, lambda n: []) == 4


def test_invalid_string_raises():
    with pytest.raises(ValueError, match="auto"):
        _resolve_max_layer(_ngff((10, 10, 10)), "lots", lambda n: [])


def _n_levels(path):
    attrs = zarr.open_group(path, mode="r").attrs.asdict()
    ms = attrs["ome"]["multiscales"] if "ome" in attrs else attrs["multiscales"]
    return len(ms[0]["datasets"])


@pytest.mark.parametrize("backend", ["ome-zarr-py", "ngff-zarr"])
@pytest.mark.parametrize("shape, expected", [((64, 64, 64), 1), ((300, 64, 40), 3)])
def test_to_ome_zarr_default_is_auto(tmp_path, backend, shape, expected):
    znimg = ZarrNii.from_darr(
        da.zeros((1,) + shape, dtype="uint8", chunks=(1, 64, 64, 64))
    )
    path = str(tmp_path / "img.ome.zarr")
    znimg.to_ome_zarr(path, backend=backend)
    assert _n_levels(path) == expected
    coarsest = ZarrNii.from_file(path, level=expected - 1)
    assert max(coarsest.shape[1:]) <= AUTO_PYRAMID_MIN_SIZE


def test_to_ome_zarr_explicit_max_layer(tmp_path):
    znimg = ZarrNii.from_darr(da.zeros((1, 64, 64, 64), dtype="uint8"))
    path = str(tmp_path / "img.ome.zarr")
    znimg.to_ome_zarr(path, max_layer=3)
    assert _n_levels(path) == 3


def test_cli_max_layer_arg():
    import argparse

    from zarrnii.cli import _max_layer_arg

    assert _max_layer_arg("auto") == "auto"
    assert _max_layer_arg("AUTO") == "auto"
    assert _max_layer_arg("5") == 5
    for bad in ["0", "-1", "many"]:
        with pytest.raises(argparse.ArgumentTypeError):
            _max_layer_arg(bad)
