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
