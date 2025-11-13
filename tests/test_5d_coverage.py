"""Additional tests to improve coverage for 5D support functionality."""

import os
import tempfile

import dask.array as da
import ngff_zarr as nz
import numpy as np
import pytest
import zarr

from zarrnii import ZarrNii
from zarrnii.core import (
    _select_dimensions_from_image,
    _select_dimensions_from_image_with_omero,
    load_ngff_image,
)


def create_test_dataset_without_time(store_path, num_channels=2):
    """Create a test dataset without time dimension."""
    # Create 4D array (z, y, x, c)
    arr_sz = (8, 16, 16, num_channels)
    arr = da.ones(arr_sz, dtype=np.uint16)

    # Create NGFF image with 4D dimensions (no time)
    ngff_image = nz.to_ngff_image(arr, dims=["z", "y", "x", "c"])
    multiscales = nz.to_multiscales(ngff_image)

    # Save to zarr
    nz.to_ngff_zarr(store_path, multiscales)
    return store_path


def create_test_dataset_without_channels(store_path, num_timepoints=2):
    """Create a test dataset without channel dimension."""
    # Create 4D array (t, z, y, x)
    arr_sz = (num_timepoints, 8, 16, 16)
    arr = da.ones(arr_sz, dtype=np.uint16)

    # Create NGFF image with 4D dimensions (no channels)
    ngff_image = nz.to_ngff_image(arr, dims=["t", "z", "y", "x"])
    multiscales = nz.to_multiscales(ngff_image)

    # Save to zarr
    nz.to_ngff_zarr(store_path, multiscales)
    return store_path


def create_test_dataset_with_omero_metadata(
    store_path, num_timepoints=2, num_channels=2
):
    """Create a test dataset with omero metadata."""
    # Create 5D array (t, z, y, x, c)
    arr_sz = (num_timepoints, 8, 16, 16, num_channels)
    data = np.ones(arr_sz, dtype=np.uint16)

    # Fill with different values for each channel
    for c in range(num_channels):
        data[:, :, :, :, c] = (c + 1) * 100

    dask_data = da.from_array(data, chunks=(1, 4, 8, 8, 1))

    # Create NGFF image with 5D dimensions
    ngff_image = nz.to_ngff_image(dask_data, dims=["t", "z", "y", "x", "c"])
    multiscales = nz.to_multiscales(ngff_image)

    # Create omero metadata
    omero_channels = []
    channel_labels = ["DAPI", "GFP"][:num_channels]
    for i, label in enumerate(channel_labels):
        channel = {
            "label": label,
            "color": "FF0000" if i == 0 else "00FF00",
            "window": {"min": 0.0, "max": 65535.0, "start": 0.0, "end": 65535.0},
        }
        omero_channels.append(channel)

    omero_metadata = {"channels": omero_channels}

    # Save to zarr
    nz.to_ngff_zarr(store_path, multiscales)

    # Add omero metadata
    group = zarr.open_group(store_path, mode="a")
    group.attrs["omero"] = omero_metadata

    return store_path


class TestEdgeCases:
    """Test edge cases and error conditions for 5D support."""

    def test_load_ngff_image_with_timepoints_only(self):
        """Test loading with timepoints parameter in load_ngff_image function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "test.zarr")
            create_test_dataset_with_omero_metadata(
                store_path, num_timepoints=3, num_channels=2
            )

            # Test using from_ome_zarr which handles the sequential selection properly
            znimg = ZarrNii.from_ome_zarr(store_path, timepoints=[0, 2])
            assert znimg.darr.shape == (2, 8, 16, 16, 2)

    def test_load_ngff_image_with_channels_and_timepoints(self):
        """Test loading with both channels and timepoints in load_ngff_image function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "test.zarr")
            create_test_dataset_with_omero_metadata(
                store_path, num_timepoints=3, num_channels=2
            )

            # Test using from_ome_zarr which handles the sequential selection properly
            znimg = ZarrNii.from_ome_zarr(store_path, channels=[1], timepoints=[0, 2])
            assert znimg.darr.shape == (2, 8, 16, 16, 1)

    def test_select_dimensions_from_image_no_time_axis(self):
        """Test _select_dimensions_from_image when no time axis exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "test.zarr")
            create_test_dataset_without_time(store_path)

            multiscales = nz.from_ngff_zarr(store_path)
            ngff_image = multiscales.images[0]

            # This should work fine even without time axis
            result = _select_dimensions_from_image(
                ngff_image, multiscales, channels=[0], timepoints=None
            )
            assert result.data.shape == (8, 16, 16, 1)

    def test_select_dimensions_from_image_no_channel_axis(self):
        """Test _select_dimensions_from_image when no channel axis exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "test.zarr")
            create_test_dataset_without_channels(store_path)

            multiscales = nz.from_ngff_zarr(store_path)
            ngff_image = multiscales.images[0]

            # This should work fine even without channel axis
            result = _select_dimensions_from_image(
                ngff_image, multiscales, channels=None, timepoints=[0]
            )
            assert result.data.shape == (1, 8, 16, 16)

    def test_select_dimensions_with_omero_no_timepoints_no_channels(self):
        """Test _select_dimensions_from_image_with_omero with no selection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "test.zarr")
            create_test_dataset_with_omero_metadata(store_path)

            multiscales = nz.from_ngff_zarr(store_path)
            ngff_image = multiscales.images[0]

            # Test with no selection - should return original
            result_img, result_omero = _select_dimensions_from_image_with_omero(
                ngff_image, multiscales, None, None, None, None
            )
            assert result_img.data.shape == ngff_image.data.shape
            assert result_omero is None

    def test_select_dimensions_with_omero_channel_labels_error(self):
        """Test error when channel labels provided but no omero metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "test.zarr")
            create_test_dataset_without_time(store_path)

            multiscales = nz.from_ngff_zarr(store_path)
            ngff_image = multiscales.images[0]

            # This should raise an error
            with pytest.raises(
                ValueError,
                match="Channel labels were specified but no omero metadata found",
            ):
                _select_dimensions_from_image_with_omero(
                    ngff_image, multiscales, None, ["DAPI"], None, None
                )

    def test_select_timepoints_no_time_dimension(self):
        """Test select_timepoints when no time dimension exists."""
        # Create 4D data without time dimension
        data_4d = da.ones((1, 8, 16, 16), chunks=(1, 4, 8, 8))
        znimg = ZarrNii.from_darr(data_4d)

        # Should raise error when trying to select timepoints
        with pytest.raises(ValueError, match="No time dimension found in the data"):
            znimg.select_timepoints([0])

    def test_select_timepoints_none_parameter(self):
        """Test select_timepoints with None parameter returns copy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "test.zarr")
            create_test_dataset_with_omero_metadata(store_path, num_timepoints=3)

            znimg = ZarrNii.from_ome_zarr(store_path)
            result = znimg.select_timepoints(None)

            # Should return a copy with same shape
            assert result.darr.shape == znimg.darr.shape
            assert result is not znimg  # Different instance

    def test_from_nifti_5d_different_dimensions(self):
        """Test from_nifti with 5D data having different dimension configurations."""
        import tempfile

        import nibabel as nib

        # Test different 5D configurations
        test_cases = [
            (2, 1, 8, 16, 16),  # (t, c, z, y, x)
            (1, 3, 8, 16, 16),  # (c, t, z, y, x) - unusual but possible
        ]

        for shape in test_cases:
            data_5d = np.random.rand(*shape).astype(np.float32)
            affine = np.eye(4)

            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
                nifti_img = nib.Nifti1Image(data_5d, affine)
                nib.save(nifti_img, tmp.name)

                try:
                    znimg = ZarrNii.from_nifti(tmp.name)
                    assert znimg.darr.shape == shape
                    assert len(znimg.ngff_image.dims) == 5
                finally:
                    os.unlink(tmp.name)

    def test_complex_timepoint_channel_combinations(self):
        """Test various combinations of timepoint and channel selections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "test.zarr")
            create_test_dataset_with_omero_metadata(
                store_path, num_timepoints=4, num_channels=2
            )  # Only 2 channels

            # Test various combinations
            test_cases = [
                ({"timepoints": [0, 2, 3]}, (3, 8, 16, 16, 2)),
                ({"channels": [0, 1]}, (4, 8, 16, 16, 2)),  # Use valid channel indices
                (
                    {"timepoints": [1], "channels": [1]},
                    (1, 8, 16, 16, 1),
                ),  # Use valid channel index
                ({"timepoints": [0, 1, 2], "channels": [0]}, (3, 8, 16, 16, 1)),
            ]

            for kwargs, expected_shape in test_cases:
                znimg = ZarrNii.from_ome_zarr(store_path, **kwargs)
                assert (
                    znimg.darr.shape == expected_shape
                ), f"Failed for {kwargs}: got {znimg.darr.shape}, expected {expected_shape}"

    def test_channel_labels_with_5d_data(self):
        """Test channel label selection with 5D data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "test.zarr")
            create_test_dataset_with_omero_metadata(
                store_path, num_timepoints=2, num_channels=2
            )

            # Test selecting by channel labels
            znimg = ZarrNii.from_ome_zarr(store_path, channel_labels=["GFP"])
            assert znimg.darr.shape == (2, 8, 16, 16, 1)

            # Test combining with timepoints
            znimg = ZarrNii.from_ome_zarr(
                store_path, timepoints=[1], channel_labels=["DAPI"]
            )
            assert znimg.darr.shape == (1, 8, 16, 16, 1)

    def test_error_conditions(self):
        """Test various error conditions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "test.zarr")
            create_test_dataset_with_omero_metadata(
                store_path, num_timepoints=2, num_channels=2
            )

            # Test invalid timepoint indices
            with pytest.raises((IndexError, ValueError)):
                ZarrNii.from_ome_zarr(
                    store_path, timepoints=[5]
                )  # Only 2 timepoints available

            # Test invalid channel labels
            with pytest.raises(
                ValueError, match="Channel label 'InvalidChannel' not found"
            ):
                ZarrNii.from_ome_zarr(store_path, channel_labels=["InvalidChannel"])

    def test_dimension_handling_edge_cases(self):
        """Test edge cases in dimension handling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "test.zarr")
            create_test_dataset_with_omero_metadata(
                store_path, num_timepoints=1, num_channels=1
            )

            # Test with single timepoint and channel
            znimg = ZarrNii.from_ome_zarr(store_path, timepoints=[0], channels=[0])
            assert znimg.darr.shape == (1, 8, 16, 16, 1)

            # Test select_timepoints with single timepoint
            selected = znimg.select_timepoints([0])
            assert selected.darr.shape == (1, 8, 16, 16, 1)
