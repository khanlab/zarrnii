"""Tests for 5D image support with time dimension (T,C,Z,Y,X)."""

import os
import tempfile

import dask.array as da
import ngff_zarr as nz
import numpy as np
import pytest
import zarr

from zarrnii import ZarrNii


def create_5d_test_dataset(store_path, num_timepoints=3, num_channels=2):
    """Create a test OME-Zarr dataset with time and channel dimensions."""

    # Create a 5D array in TZYXC order to match expected axis labels
    arr_sz = (num_timepoints, 16, 32, 32, num_channels)  # (t, z, y, x, c)
    arr = da.zeros(arr_sz, dtype=np.uint16)

    # Fill with different values for each timepoint and channel for easy identification
    def fill_data(block, block_info=None):
        if block_info is not None:
            block_slice = block_info[0]["array-location"]
            t_start = block_slice[0][0]  # Time is at index 0
            t_end = block_slice[0][1]
            c_start = block_slice[4][0]  # Channel is at index 4
            c_end = block_slice[4][1]

            result = np.zeros(block.shape, dtype=np.uint16)
            for t_idx in range(t_end - t_start):
                global_t_idx = t_start + t_idx
                for c_idx in range(c_end - c_start):
                    global_c_idx = c_start + c_idx
                    # Value = (timepoint + 1) * 1000 + (channel + 1) * 100
                    value = (global_t_idx + 1) * 1000 + (global_c_idx + 1) * 100
                    result[t_idx, :, :, :, c_idx] = value
            return result
        return np.zeros(block.shape, dtype=np.uint16)

    arr = arr.map_blocks(fill_data, dtype=np.uint16)

    # Create NGFF image with 5D dimensions
    ngff_image = nz.to_ngff_image(arr, dims=["t", "z", "y", "x", "c"])
    multiscales = nz.to_multiscales(ngff_image)

    # Create omero metadata for channels
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

    # Save the dataset
    nz.to_ngff_zarr(store_path, multiscales)

    # Add omero metadata
    group = zarr.open_group(store_path, mode="a")
    group.attrs["omero"] = omero_metadata

    return store_path


@pytest.fixture
def test_5d_dataset():
    """Create a temporary 5D test dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = os.path.join(tmpdir, "test_5d.zarr")
        yield create_5d_test_dataset(store_path)


class Test5DSupport:
    """Test class for 5D image functionality."""

    def test_load_5d_all_data_by_default(self, test_5d_dataset):
        """Test that loading 5D data without selection loads all timepoints and channels."""
        znimg = ZarrNii.from_ome_zarr(test_5d_dataset)

        # Should load all data: shape (t, z, y, x, c) = (3, 16, 32, 32, 2)
        expected_shape = (3, 16, 32, 32, 2)
        assert znimg.darr.shape == expected_shape

        # Verify dimensions include time
        assert "t" in znimg.ngff_image.dims
        assert "c" in znimg.ngff_image.dims

    def test_load_by_timepoint_indices(self, test_5d_dataset):
        """Test loading specific timepoints by index."""
        znimg = ZarrNii.from_ome_zarr(test_5d_dataset, timepoints=[0, 2])

        # Should load 2 timepoints: shape (t, z, y, x, c) = (2, 16, 32, 32, 2)
        expected_shape = (2, 16, 32, 32, 2)
        assert znimg.darr.shape == expected_shape

        # Check data values - timepoint 0 should have values starting with 1000
        # timepoint 2 should have values starting with 3000
        t0_data = znimg.darr[0, :, :, :, 0].compute()
        t1_data = znimg.darr[1, :, :, :, 0].compute()  # This is actually timepoint 2

        # Timepoint 0, channel 0 should have value 1100 (1*1000 + 1*100)
        assert np.all(t0_data == 1100)
        # Timepoint 2, channel 0 should have value 3100 (3*1000 + 1*100)
        assert np.all(t1_data == 3100)

    def test_combine_timepoint_and_channel_selection(self, test_5d_dataset):
        """Test selecting both specific timepoints and channels."""
        znimg = ZarrNii.from_ome_zarr(test_5d_dataset, timepoints=[1], channels=[1])

        # Should load 1 timepoint, 1 channel: shape (t, z, y, x, c) = (1, 16, 32, 32, 1)
        expected_shape = (1, 16, 32, 32, 1)
        assert znimg.darr.shape == expected_shape

        # Check data value - timepoint 1, channel 1 should have value 2200 (2*1000 + 2*100)
        data_sum = znimg.darr.compute().sum()
        expected_sum = 2200 * 16 * 32 * 32  # value * z * y * x
        assert data_sum == expected_sum

    def test_spatial_transforms_preserve_time_channel_dims(self, test_5d_dataset):
        """Test that spatial transformations work correctly with 5D data."""
        znimg = ZarrNii.from_ome_zarr(test_5d_dataset)

        # Test cropping - should preserve T and C dimensions
        cropped = znimg.crop([0, 0, 0], [8, 16, 16])  # Crop spatial dimensions only
        # since crop dims are always defined x y z, x is the one cropped more..

        # Should maintain T and C dimensions but crop spatial ones
        expected_shape = (3, 16, 16, 8, 2)  # (t, z_cropped, y_cropped, x_cropped, c)
        assert cropped.darr.shape == expected_shape

    def test_downsample_5d_data(self, test_5d_dataset):
        """Test downsampling works with 5D data."""
        znimg = ZarrNii.from_ome_zarr(test_5d_dataset)

        # Test isotropic downsampling
        downsampled = znimg.downsample(factors=2)

        # Should downsample spatial dimensions but preserve T and C
        expected_shape = (3, 8, 16, 16, 2)  # (t, z/2, y/2, x/2, c)
        assert downsampled.darr.shape == expected_shape

    def test_select_timepoints_method(self, test_5d_dataset):
        """Test the select_timepoints method works like select_channels."""
        znimg = ZarrNii.from_ome_zarr(test_5d_dataset)

        # Select specific timepoints after loading
        selected = znimg.select_timepoints(timepoints=[0, 2])

        # Should have selected timepoints
        expected_shape = (2, 16, 32, 32, 2)
        assert selected.darr.shape == expected_shape

    def test_error_invalid_timepoint_index(self, test_5d_dataset):
        """Test error when invalid timepoint index is specified."""
        with pytest.raises((IndexError, ValueError)):
            ZarrNii.from_ome_zarr(
                test_5d_dataset, timepoints=[5]
            )  # Only 3 timepoints available

    def test_from_nifti_5d_data(self):
        """Test loading 5D data from NIfTI-like array."""
        import tempfile

        import nibabel as nib

        # Create a 5D array (t, z, y, x, c) - but NIfTI will see it as 5D
        data_5d = np.random.rand(2, 8, 16, 16, 2).astype(np.float32)

        # Create a dummy affine matrix
        affine = np.eye(4)
        affine[0, 0] = 1.5  # x spacing
        affine[1, 1] = 1.0  # y spacing
        affine[2, 2] = 2.0  # z spacing

        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            # Create NIfTI image
            nifti_img = nib.Nifti1Image(data_5d, affine)
            nib.save(nifti_img, tmp.name)

            try:
                # Load with ZarrNii
                znimg = ZarrNii.from_nifti(tmp.name)

                # Should handle 5D data appropriately
                assert znimg.darr.shape == (2, 8, 16, 16, 2)
                assert znimg.ngff_image.dims == [
                    "t",
                    "c",
                    "x",
                    "y",
                    "z",
                ]  # XYZ order by default for from_nifti

            finally:
                import os

                os.unlink(tmp.name)

    def test_backward_compatibility_4d_data(self):
        """Test that existing 4D functionality still works."""
        # Create a simple 4D array
        data = da.ones((1, 16, 32, 32), chunks=(1, 8, 16, 16))  # (c, z, y, x)
        znimg = ZarrNii.from_darr(data)

        # Should work as before
        assert znimg.darr.shape == (1, 16, 32, 32)
        assert znimg.ngff_image.dims == ["c", "z", "y", "x"]

        # Spatial operations should still work
        cropped = znimg.crop([0, 0, 0], [8, 16, 16])
        assert cropped.darr.shape == (1, 16, 16, 8)
