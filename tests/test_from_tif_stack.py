"""Tests for ZarrNii.from_tif_stack()."""

import os
import tempfile

import numpy as np
import pytest
import tifffile

from zarrnii import ZarrNii


def _write_tif(path, data):
    if data.ndim == 3:
        tifffile.imwrite(path, data, photometric="minisblack", metadata={"axes": "ZYX"})
    else:
        tifffile.imwrite(path, data, photometric="minisblack")


def test_from_tif_stack_flat_2d_slices_stacked_as_z():
    """Flat 2D TIFF list can be stacked as a single-channel Z stack."""
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        for z in range(3):
            path = os.path.join(tmpdir, f"z{z}.tif")
            _write_tif(path, np.full((4, 5), z, dtype=np.uint16))
            paths.append(path)

        znii = ZarrNii.from_tif_stack(
            paths,
            stack_mode="z",
            spacing=(2.0, 1.5, 1.0),
            origin=(10.0, 20.0, 30.0),
        )

        assert list(znii.dims) == ["c", "z", "y", "x"]
        assert znii.data.shape == (1, 3, 4, 5)
        assert znii.scale == {"z": 2.0, "y": 1.5, "x": 1.0}
        assert znii.translation == {"z": 10.0, "y": 20.0, "x": 30.0}
        np.testing.assert_array_equal(
            znii.data.compute()[0, :, 0, 0], np.array([0, 1, 2])
        )


def test_from_tif_stack_nested_channel_stacks_with_omero_metadata():
    """Nested per-channel 2D stacks become CZYX with OMERO convenience metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        for c in range(2):
            channel_paths = []
            for z in range(2):
                path = os.path.join(tmpdir, f"c{c}_z{z}.tif")
                _write_tif(path, np.full((3, 4), c * 10 + z, dtype=np.uint16))
                channel_paths.append(path)
            paths.append(channel_paths)

        znii = ZarrNii.from_tif_stack(
            paths,
            stack_mode="channel_z",
            channel_labels=["DAPI", "GFP"],
            channel_colors=["0000FF", "00FF00"],
        )

        assert list(znii.dims) == ["c", "z", "y", "x"]
        assert znii.data.shape == (2, 2, 3, 4)
        assert znii.omero is not None
        assert [ch.label for ch in znii.omero.channels] == ["DAPI", "GFP"]
        np.testing.assert_array_equal(
            znii.data.compute()[:, :, 0, 0], np.array([[0, 1], [10, 11]])
        )


def test_from_tif_stack_flat_3d_volumes_stacked_as_channels_auto():
    """Flat 3D TIFF list is inferred as channel-stacked volumes in auto mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        for c in range(2):
            path = os.path.join(tmpdir, f"ch{c}.tif")
            _write_tif(path, np.full((4, 5, 6), c + 1, dtype=np.uint16))
            paths.append(path)

        znii = ZarrNii.from_tif_stack(paths, stack_mode="auto")
        assert znii.data.shape == (2, 4, 5, 6)
        np.testing.assert_array_equal(
            znii.data.compute()[:, 0, 0, 0], np.array([1, 2], dtype=np.uint16)
        )


def test_from_tif_stack_flat_3d_volumes_concatenate_along_z():
    """stack_mode='z' concatenates flat 3D TIFF volumes along the z dimension."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p0 = os.path.join(tmpdir, "vol0.tif")
        p1 = os.path.join(tmpdir, "vol1.tif")
        _write_tif(p0, np.full((2, 5, 6), 7, dtype=np.uint16))
        _write_tif(p1, np.full((3, 5, 6), 9, dtype=np.uint16))

        znii = ZarrNii.from_tif_stack([p0, p1], stack_mode="z")
        assert znii.data.shape == (1, 5, 5, 6)

        computed = znii.data.compute()
        np.testing.assert_array_equal(computed[0, :2, 0, 0], np.array([7, 7]))
        np.testing.assert_array_equal(computed[0, 2:, 0, 0], np.array([9, 9, 9]))


def test_from_tif_stack_validation_errors():
    """Method raises clear errors for misused or ambiguous inputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p2d = os.path.join(tmpdir, "s0.tif")
        p3d = os.path.join(tmpdir, "v0.tif")
        _write_tif(p2d, np.zeros((4, 5), dtype=np.uint16))
        _write_tif(p3d, np.zeros((2, 4, 5), dtype=np.uint16))

        with pytest.raises(ValueError, match="Cannot infer stack_mode"):
            ZarrNii.from_tif_stack([p2d, p3d], stack_mode="auto")

        with pytest.raises(ValueError, match="requires nested path input"):
            ZarrNii.from_tif_stack([p2d], stack_mode="channel_z")

        with pytest.raises(ValueError, match="Provide either 'omero'"):
            ZarrNii.from_tif_stack(
                [p2d], stack_mode="z", omero=object(), channel_labels=["A"]
            )

        with pytest.raises(ValueError, match="Provide either 'omero'"):
            ZarrNii.from_tif_stack(
                [p2d], stack_mode="z", omero=object(), channel_windows=[None]
            )


def test_from_tif_stack_with_channel_windows():
    """channel_windows are forwarded to OMERO metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        for c in range(2):
            channel_paths = []
            for z in range(2):
                path = os.path.join(tmpdir, f"c{c}_z{z}.tif")
                _write_tif(path, np.full((3, 4), c * 10 + z, dtype=np.uint16))
                channel_paths.append(path)
            paths.append(channel_paths)

        znii = ZarrNii.from_tif_stack(
            paths,
            stack_mode="channel_z",
            channel_labels=["DAPI", "GFP"],
            channel_colors=["0000FF", "00FF00"],
            channel_windows=[
                {"min": 0, "max": 65535, "start": 100, "end": 2000},
                {"min": 0, "max": 65535, "start": 200, "end": 4000},
            ],
        )

        assert znii.omero is not None
        assert znii.omero.channels[0].window.end == 2000.0
        assert znii.omero.channels[1].window.end == 4000.0


def test_from_darr_with_channel_labels():
    """from_darr accepts channel_labels and builds OMERO metadata."""
    import dask.array as da

    data = da.zeros((2, 4, 8, 8), dtype=np.uint16)
    znii = ZarrNii.from_darr(
        data,
        channel_labels=["DAPI", "GFP"],
        channel_colors=["0000FF", "00FF00"],
    )
    assert znii.omero is not None
    assert [ch.label for ch in znii.omero.channels] == ["DAPI", "GFP"]
    assert [ch.color for ch in znii.omero.channels] == ["0000FF", "00FF00"]


def test_from_darr_with_channel_windows():
    """from_darr forwards channel_windows into OMERO metadata."""
    import dask.array as da

    data = da.zeros((2, 4, 8, 8), dtype=np.uint16)
    znii = ZarrNii.from_darr(
        data,
        channel_labels=["A", "B"],
        channel_windows=[
            {"min": 0, "max": 4095, "start": 10, "end": 1000},
            (0, 65535, 100, 2000),
        ],
    )
    assert znii.omero is not None
    assert znii.omero.channels[0].window.max == 4095.0
    assert znii.omero.channels[1].window.end == 2000.0


def test_from_darr_omero_conflict_raises():
    """from_darr raises when omero and channel convenience args are both given."""
    import dask.array as da

    data = da.zeros((2, 4, 8, 8), dtype=np.uint16)
    with pytest.raises(ValueError, match="Provide either 'omero'"):
        ZarrNii.from_darr(data, omero=object(), channel_labels=["A", "B"])

    with pytest.raises(ValueError, match="Provide either 'omero'"):
        ZarrNii.from_darr(data, omero=object(), channel_colors=["0000FF", "00FF00"])

    with pytest.raises(ValueError, match="Provide either 'omero'"):
        ZarrNii.from_darr(data, omero=object(), channel_windows=[None, None])


def test_from_darr_channel_label_length_mismatch_raises():
    """from_darr raises when channel_labels length != number of channels."""
    import dask.array as da

    data = da.zeros((3, 4, 8, 8), dtype=np.uint16)
    with pytest.raises(ValueError, match="channel_labels length"):
        ZarrNii.from_darr(data, channel_labels=["A", "B"])  # 2 != 3
