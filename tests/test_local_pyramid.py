"""Tests for to_ome_zarr(pyramid_source="local"): the pyramid without read-back."""

import os
import warnings

import dask.array as da
import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore, WrapperStore

from zarrnii import ZarrNii


def _volume(shape_czyx, chunks=(1, 32, 32, 32), seed=0):
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 4000, shape_czyx, dtype=np.uint16)
    return da.from_array(data, chunks=chunks)


def _write(tmp_path, name, znii, **kwargs):
    path = str(tmp_path / name)
    znii.to_ome_zarr(path, **kwargs)
    return zarr.open_group(path, mode="r")


def _levels(group):
    datasets = group.attrs["ome"]["multiscales"][0]["datasets"]
    return [group[d["path"]] for d in datasets]


def _assert_same_pyramid(a, b):
    levels_a, levels_b = _levels(a), _levels(b)
    assert len(levels_a) == len(levels_b) > 1
    for x, y in zip(levels_a, levels_b):
        assert x.shape == y.shape
        np.testing.assert_array_equal(x[...], y[...])
    assert dict(a.attrs) == dict(b.attrs)


@pytest.mark.parametrize(
    "shape, spacing, storage_options",
    [
        # isotropic, no shards
        ((1, 96, 80, 72), (1.0, 1.0, 1.0), None),
        # anisotropic: the first level halves z only, as for light-sheet data
        ((1, 120, 64, 64), (0.5, 1.0, 1.0), {"chunks": (1, 16, 16, 16)}),
        # sharded, several channels, sizes that are not multiples of the shard
        (
            (2, 75, 90, 66),
            (1.0, 1.0, 1.0),
            {"chunks": (1, 16, 16, 16), "shards": (1, 32, 32, 32)},
        ),
    ],
)
def test_local_pyramid_matches_store(tmp_path, shape, spacing, storage_options):
    znii = ZarrNii.from_darr(_volume(shape), spacing=spacing, axes_order="ZYX")
    kwargs = {"max_layer": 4}
    if storage_options:
        kwargs["storage_options"] = storage_options
    store = _write(tmp_path, "store.ome.zarr", znii, **kwargs)
    local = _write(
        tmp_path,
        "local.ome.zarr",
        znii,
        pyramid_source="local",
        pyramid_tmp_dir=str(tmp_path),
        **kwargs,
    )
    _assert_same_pyramid(store, local)


@pytest.mark.parametrize("block_bytes", [1, 600 * 1024, 256 * 2**20])
def test_local_pyramid_any_block_budget(tmp_path, block_bytes):
    """None, one or all lower levels written from level 0's blocks: the same result."""
    znii = ZarrNii.from_darr(
        _volume((1, 128, 96, 80)), spacing=(1.0, 1.0, 1.0), axes_order="ZYX"
    )
    opts = {"chunks": (1, 16, 16, 16), "shards": (1, 32, 32, 32)}
    store = _write(tmp_path, "store.ome.zarr", znii, max_layer=5, storage_options=opts)
    local = _write(
        tmp_path,
        "local.ome.zarr",
        znii,
        max_layer=5,
        storage_options=opts,
        pyramid_source="local",
        pyramid_block_bytes=block_bytes,
    )
    _assert_same_pyramid(store, local)


class _ReadCounter(WrapperStore):
    """Counts reads of data (not metadata) from the wrapped store."""

    def __init__(self, store, counts=None):
        super().__init__(store)
        self.counts = counts if counts is not None else {"reads": 0}

    def _with_store(self, store):
        return type(self)(store, self.counts)

    async def get(self, key, prototype, byte_range=None):
        if not key.endswith("zarr.json"):
            self.counts["reads"] += 1
        return await super().get(key, prototype, byte_range)

    async def get_partial_values(self, prototype, key_ranges):
        key_ranges = list(key_ranges)
        self.counts["reads"] += sum(not k.endswith("zarr.json") for k, _ in key_ranges)
        return await super().get_partial_values(prototype, key_ranges)


@pytest.mark.parametrize("source, reads_back", [("store", True), ("local", False)])
def test_local_pyramid_reads_nothing_back(tmp_path, source, reads_back):
    store = _ReadCounter(LocalStore(str(tmp_path / "out.ome.zarr")))
    znii = ZarrNii.from_darr(
        _volume((1, 64, 64, 64)), spacing=(1.0, 1.0, 1.0), axes_order="ZYX"
    )
    znii.to_ome_zarr(
        zarr.open_group(store, mode="w"), max_layer=3, pyramid_source=source
    )
    assert (store.counts["reads"] > 0) == reads_back


def test_local_pyramid_removes_its_temporary_store(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    znii = ZarrNii.from_darr(
        _volume((1, 64, 64, 64)), spacing=(1.0, 1.0, 1.0), axes_order="ZYX"
    )
    znii.to_ome_zarr(
        str(tmp_path / "out.ome.zarr"),
        max_layer=4,
        pyramid_source="local",
        pyramid_tmp_dir=str(scratch),
    )
    assert os.listdir(scratch) == []


def test_local_pyramid_other_methods_fall_back_to_store(tmp_path):
    znii = ZarrNii.from_darr(
        _volume((1, 64, 64, 64)), spacing=(1.0, 1.0, 1.0), axes_order="ZYX"
    )
    store = _write(
        tmp_path, "store.ome.zarr", znii, max_layer=3, scaling_method="nearest"
    )
    with pytest.warns(UserWarning, match="local_mean"):
        local = _write(
            tmp_path,
            "local.ome.zarr",
            znii,
            max_layer=3,
            scaling_method="nearest",
            pyramid_source="local",
        )
    _assert_same_pyramid(store, local)


def test_pyramid_source_checked(tmp_path):
    znii = ZarrNii.from_darr(
        _volume((1, 32, 32, 32)), spacing=(1.0, 1.0, 1.0), axes_order="ZYX"
    )
    with pytest.raises(ValueError, match="pyramid_source"):
        znii.to_ome_zarr(str(tmp_path / "x.ome.zarr"), pyramid_source="remote")


def test_store_is_the_default(tmp_path):
    """Without pyramid_source, nothing changes: levels are read back as before."""
    store = _ReadCounter(LocalStore(str(tmp_path / "out.ome.zarr")))
    znii = ZarrNii.from_darr(
        _volume((1, 64, 64, 64)), spacing=(1.0, 1.0, 1.0), axes_order="ZYX"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        znii.to_ome_zarr(zarr.open_group(store, mode="w"), max_layer=3)
    assert store.counts["reads"] > 0
