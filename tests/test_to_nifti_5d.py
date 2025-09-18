"""Tests for to_nifti method with 5D data handling."""

import os
import tempfile

import dask.array as da
import ngff_zarr as nz
import numpy as np
import pytest

from zarrnii import ZarrNii


def create_5d_test_dataset(store_path, num_timepoints=2, num_channels=1):
    """Create a test OME-Zarr dataset with time and channel dimensions."""

    # Create a 5D array in TZYXC order
    arr_sz = (num_timepoints, 8, 16, 16, num_channels)
    arr = da.ones(arr_sz, dtype=np.float32)

    # Create NGFF image with 5D dimensions
    ngff_image = nz.to_ngff_image(arr, dims=["t", "z", "y", "x", "c"])
    multiscales = nz.to_multiscales(ngff_image)

    # Save to zarr
    nz.to_ngff_zarr(store_path, multiscales)
    return store_path


@pytest.fixture
def test_5d_dataset():
    """Create a temporary 5D test dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = os.path.join(tmpdir, "test_5d.zarr")
        yield create_5d_test_dataset(store_path)


class TestToNifti5D:
    """Test to_nifti method with 5D data."""

    def test_to_nifti_multiple_timepoints_error(self, test_5d_dataset):
        """Test that to_nifti raises error with multiple timepoints."""
        znimg = ZarrNii.from_ome_zarr(test_5d_dataset)

        # Should raise error because we have 2 timepoints
        with pytest.raises(
            ValueError, match="NIfTI format doesn't support non-singleton t dimension"
        ):
            with tempfile.NamedTemporaryFile(suffix=".nii") as tmp:
                znimg.to_nifti(tmp.name)

    def test_to_nifti_single_timepoint_success(self, test_5d_dataset):
        """Test that to_nifti works with single timepoint selection."""
        znimg = ZarrNii.from_ome_zarr(test_5d_dataset)

        # Select a single timepoint
        znimg_single = znimg.select_timepoints([0])

        # Should work fine
        with tempfile.NamedTemporaryFile(suffix=".nii") as tmp:
            result = znimg_single.to_nifti(tmp.name)
            assert result == tmp.name
            assert os.path.exists(tmp.name)

    def test_to_nifti_singleton_dimensions(self):
        """Test to_nifti with singleton time and channel dimensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "test_singleton.zarr")
            # Create dataset with singleton dimensions
            create_5d_test_dataset(store_path, num_timepoints=1, num_channels=1)

            znimg = ZarrNii.from_ome_zarr(store_path)

            # Should work fine - singleton dimensions will be squeezed
            with tempfile.NamedTemporaryFile(suffix=".nii") as tmp:
                result = znimg.to_nifti(tmp.name)
                assert result == tmp.name
                assert os.path.exists(tmp.name)

    def test_to_nifti_multiple_channels_error(self):
        """Test that to_nifti raises error with multiple channels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "test_channels.zarr")
            # Create dataset with multiple channels
            create_5d_test_dataset(store_path, num_timepoints=1, num_channels=3)

            znimg = ZarrNii.from_ome_zarr(store_path)

            # Should raise error because we have 3 channels
            with pytest.raises(
                ValueError,
                match="NIfTI format doesn't support non-singleton c dimension",
            ):
                with tempfile.NamedTemporaryFile(suffix=".nii") as tmp:
                    znimg.to_nifti(tmp.name)

    def test_to_nifti_single_channel_selection(self):
        """Test to_nifti with single channel selection from multichannel data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "test_channels.zarr")
            # Create dataset with multiple channels
            create_5d_test_dataset(store_path, num_timepoints=1, num_channels=3)

            znimg = ZarrNii.from_ome_zarr(store_path)

            # Select single channel
            znimg_single = znimg.select_channels([1])

            # Should work fine
            with tempfile.NamedTemporaryFile(suffix=".nii") as tmp:
                result = znimg_single.to_nifti(tmp.name)
                assert result == tmp.name
                assert os.path.exists(tmp.name)

    def test_to_nifti_returns_nibabel_image(self, test_5d_dataset):
        """Test that to_nifti returns nibabel image when no filename provided."""
        znimg = ZarrNii.from_ome_zarr(test_5d_dataset)
        znimg_single = znimg.select_timepoints([0])

        # Should return nibabel image
        nifti_img = znimg_single.to_nifti()

        import nibabel as nib

        assert isinstance(nifti_img, nib.Nifti1Image)

        # Check data shape - should be 3D after squeezing singleton dimensions
        assert nifti_img.get_fdata().ndim == 3

    def test_to_nifti_with_4d_data_backward_compatibility(self):
        """Test that existing 4D data still works (backward compatibility)."""
        # Create 4D data
        data = da.ones((1, 8, 16, 16), chunks=(1, 4, 8, 8))
        znimg = ZarrNii.from_darr(data)

        # Should work as before
        with tempfile.NamedTemporaryFile(suffix=".nii") as tmp:
            result = znimg.to_nifti(tmp.name)
            assert result == tmp.name
            assert os.path.exists(tmp.name)

    def test_to_nifti_with_real_dataset(self):
        """Test to_nifti with real dataset 'sub-AS36F2_sample-brain_acq-downsampled_SPIM.ome.zarr'."""
        dataset_path = (
            "tests/data/sub-AS36F2_sample-brain_acq-downsampled_SPIM.ome.zarr"
        )

        # Skip test if dataset doesn't exist
        if not os.path.exists(dataset_path):
            pytest.skip("Real dataset not available for testing")

        # Replicate the exact code from the comment
        from zarrnii import ZarrNii

        znimg = ZarrNii.from_ome_zarr(dataset_path)

        # Since the original dataset has multiple channels, we need to select one
        # for NIfTI export (as NIfTI doesn't support multiple channels)
        if "c" in znimg.dims and znimg.data.shape[znimg.dims.index("c")] > 1:
            # Select first channel by label to work around channel selection issue
            available_channels = znimg.list_channels()
            if available_channels:
                znimg = ZarrNii.from_ome_zarr(
                    dataset_path, channel_labels=[available_channels[0]]
                )
            else:
                znimg = ZarrNii.from_ome_zarr(dataset_path, channels=[0])

        # Now test the to_nifti call - this should work without errors
        with tempfile.NamedTemporaryFile(suffix=".nii") as tmp:
            znimg.to_nifti(tmp.name)  # Assert no errors

            # Verify file was created and is valid
            assert os.path.exists(tmp.name)

            # Verify we can read it back with nibabel
            import nibabel as nib

            nifti_img = nib.load(tmp.name)
            assert nifti_img is not None
