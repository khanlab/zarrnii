"""
Tests for lazy evaluation of create_mip_visualization function.

This module tests that create_mip_visualization returns dask arrays
that are lazy and only computed when explicitly requested.
"""

import dask.array as da
import numpy as np
import pytest

from zarrnii.analysis import create_mip_visualization


def test_mip_returns_dask_arrays():
    """Test that create_mip_visualization returns dask arrays."""
    # Create test data with 2 channels
    data = da.random.random((2, 20, 20, 20), chunks=(1, 10, 10, 10))
    dims = ["c", "z", "y", "x"]
    scale = {"z": 2.0, "y": 1.0, "x": 1.0}  # in microns

    # Create axial MIPs with custom intensity ranges (scale in microns)
    mips = create_mip_visualization(
        data,
        dims,
        scale,
        plane="axial",
        slab_thickness_um=10.0,
        slab_spacing_um=10.0,
        channel_colors=["red", "green"],
        channel_ranges=[(0.0, 0.8), (0.2, 1.0)],
        scale_units="um",
    )

    # Verify that mips is a list
    assert isinstance(mips, list), "mips should be a list"
    assert len(mips) > 0, "mips should contain at least one element"

    # Verify each element is a dask array
    for i, mip in enumerate(mips):
        assert isinstance(mip, da.Array), f"mip[{i}] should be a dask array"
        # Verify shape has RGB dimension
        assert (
            len(mip.shape) == 3 and mip.shape[2] == 3
        ), f"mip[{i}] should have shape (height, width, 3)"


def test_mip_lazy_evaluation():
    """Test that create_mip_visualization does not eagerly compute."""
    # Create test data with 1 channel
    data = da.random.random((1, 10, 10, 10), chunks=(1, 5, 5, 5))
    dims = ["c", "z", "y", "x"]
    scale = {"z": 1.0, "y": 1.0, "x": 1.0}

    # Create MIPs
    mips = create_mip_visualization(
        data,
        dims,
        scale,
        plane="axial",
        slab_thickness_um=5.0,
        slab_spacing_um=5.0,
        channel_colors=["blue"],
        channel_ranges=[(0.0, 1.0)],  # Explicit range to avoid min/max computation
        scale_units="um",
    )

    # Verify that the result is lazy (dask array)
    for mip in mips:
        assert isinstance(mip, da.Array), "mip should be a dask array"

        # Verify we can compute it
        computed = mip.compute()
        assert isinstance(computed, np.ndarray), "computed mip should be numpy array"
        assert computed.shape[2] == 3, "computed mip should have 3 RGB channels"
        assert computed.dtype == np.float32, "computed mip should be float32"
        assert computed.min() >= 0.0, "computed mip values should be >= 0"
        assert computed.max() <= 1.0, "computed mip values should be <= 1"


def test_mip_with_return_slabs_lazy():
    """Test that create_mip_visualization with return_slabs returns lazy arrays."""
    # Create test data
    data = da.random.random((2, 15, 15, 15), chunks=(1, 8, 8, 8))
    dims = ["c", "z", "y", "x"]
    scale = {"z": 2.0, "y": 1.0, "x": 1.0}

    # Create MIPs with slab info
    mips, slab_info = create_mip_visualization(
        data,
        dims,
        scale,
        plane="axial",
        slab_thickness_um=10.0,
        slab_spacing_um=10.0,
        channel_colors=["red", "green"],
        channel_ranges=[(0.0, 1.0), (0.0, 1.0)],  # Explicit ranges
        return_slabs=True,
        scale_units="um",
    )

    # Verify mips are dask arrays
    assert isinstance(mips, list), "mips should be a list"
    for mip in mips:
        assert isinstance(mip, da.Array), "mip should be a dask array"

    # Verify slab_info is a list of dicts
    assert isinstance(slab_info, list), "slab_info should be a list"
    for info in slab_info:
        assert isinstance(info, dict), "slab info should be a dict"
        assert "start_um" in info
        assert "end_um" in info
        assert "center_um" in info
        assert "start_idx" in info
        assert "end_idx" in info


def test_mip_single_channel_lazy():
    """Test that create_mip_visualization works with single channel and is lazy."""
    # Create test data without channel dimension
    data = da.random.random((15, 15, 15), chunks=(8, 8, 8))
    dims = ["z", "y", "x"]
    scale = {"z": 1.0, "y": 1.0, "x": 1.0}

    # Create MIPs
    mips = create_mip_visualization(
        data,
        dims,
        scale,
        plane="axial",
        slab_thickness_um=5.0,
        slab_spacing_um=5.0,
        channel_colors=["cyan"],
        channel_ranges=[(0.0, 1.0)],  # Explicit range
        scale_units="um",
    )

    # Verify result is lazy
    assert isinstance(mips, list), "mips should be a list"
    for mip in mips:
        assert isinstance(mip, da.Array), "mip should be a dask array"

        # Compute and verify
        computed = mip.compute()
        assert isinstance(computed, np.ndarray), "computed mip should be numpy array"
        assert computed.shape[2] == 3, "computed mip should have 3 RGB channels"


def test_mip_different_planes_lazy():
    """Test that create_mip_visualization works with different planes and is lazy."""
    # Create test data
    data = da.random.random((2, 12, 12, 12), chunks=(1, 6, 6, 6))
    dims = ["c", "z", "y", "x"]
    scale = {"z": 1.0, "y": 1.0, "x": 1.0}

    for plane in ["axial", "coronal", "sagittal"]:
        mips = create_mip_visualization(
            data,
            dims,
            scale,
            plane=plane,
            slab_thickness_um=5.0,
            slab_spacing_um=5.0,
            channel_colors=["red", "green"],
            channel_ranges=[(0.0, 1.0), (0.0, 1.0)],
            scale_units="um",
        )

        # Verify lazy
        assert isinstance(mips, list), f"mips for plane {plane} should be a list"
        for mip in mips:
            assert isinstance(
                mip, da.Array
            ), f"mip for plane {plane} should be a dask array"
