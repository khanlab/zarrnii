"""Tests for NIfTI channel dimension support."""

import os
import tempfile

import nibabel as nib
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from zarrnii import ZarrNii


class TestNiftiChannelSupport:
    """Test channel dimension support in NIfTI I/O."""

    def test_write_multichannel_nifti(self):
        """Test writing multi-channel data to NIfTI."""
        # Create 4D data (X, Y, Z, C)
        data = np.random.rand(10, 20, 30, 3).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create NIfTI and save
            nifti_path = os.path.join(tmpdir, "multichannel.nii.gz")
            nifti_img = nib.Nifti1Image(data, np.eye(4))
            nib.save(nifti_img, nifti_path)
            
            # Load with ZarrNii
            znimg = ZarrNii.from_nifti(nifti_path)
            
            # Should have shape (C, X, Y, Z) with default XYZ axes_order
            assert znimg.shape == (3, 10, 20, 30)
            assert znimg.dims == ["c", "x", "y", "z"]
            
            # Save back to NIfTI
            output_path = os.path.join(tmpdir, "output.nii.gz")
            znimg.to_nifti(output_path)
            
            # Load and verify
            reloaded = nib.load(output_path)
            assert reloaded.shape == data.shape
            assert_array_equal(reloaded.get_fdata(), data)

    def test_write_multichannel_with_labels(self):
        """Test writing multi-channel data with channel labels."""
        # Create test data with 3 channels
        data = np.random.rand(10, 20, 30, 3).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create NIfTI with channel labels
            nifti_path = os.path.join(tmpdir, "labeled.nii.gz")
            nifti_img = nib.Nifti1Image(data, np.eye(4))
            
            # Add channel labels as extension
            import json
            channel_metadata = {"channel_labels": ["DAPI", "GFP", "RFP"]}
            ext = nib.nifti1.Nifti1Extension(
                0, json.dumps(channel_metadata).encode("utf-8")
            )
            nifti_img.header.extensions.append(ext)
            nib.save(nifti_img, nifti_path)
            
            # Load with ZarrNii
            znimg = ZarrNii.from_nifti(nifti_path)
            
            # Check shape
            assert znimg.shape == (3, 10, 20, 30)
            
            # Check channel labels were loaded
            labels = znimg.list_channels()
            assert labels == ["DAPI", "GFP", "RFP"]

    def test_roundtrip_multichannel_with_labels(self):
        """Test round-trip of multi-channel data with channel labels."""
        # Create synthetic multi-channel OME-Zarr
        import dask.array as da
        import ngff_zarr as nz
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 4D data (C, Z, Y, X)
            arr = da.from_array(
                np.random.random((3, 8, 16, 16)).astype(np.float32), 
                chunks=(1, 8, 16, 16)
            )
            
            # Create NGFF image with channel labels
            ngff_image = nz.to_ngff_image(arr, dims=["c", "z", "y", "x"])
            multiscales = nz.to_multiscales(ngff_image)
            
            # Add OMERO metadata with channel labels
            from ngff_zarr import Omero, OmeroChannel, OmeroWindow
            window = OmeroWindow(min=0.0, max=1.0, start=0.0, end=1.0)
            omero = Omero(
                channels=[
                    OmeroChannel(color="FF0000", window=window, label="Red"),
                    OmeroChannel(color="00FF00", window=window, label="Green"),
                    OmeroChannel(color="0000FF", window=window, label="Blue"),
                ]
            )
            multiscales.metadata.omero = omero
            
            # Save as OME-Zarr
            zarr_path = os.path.join(tmpdir, "test.zarr")
            nz.to_ngff_zarr(zarr_path, multiscales)
            
            # Load with ZarrNii
            znimg = ZarrNii.from_ome_zarr(zarr_path)
            
            # Verify channel labels
            labels = znimg.list_channels()
            assert labels == ["Red", "Green", "Blue"]
            
            # Save to NIfTI
            nifti_path = os.path.join(tmpdir, "output.nii.gz")
            znimg.to_nifti(nifti_path)
            
            # Load back from NIfTI (use same axes_order as original)
            znimg2 = ZarrNii.from_nifti(nifti_path, axes_order="ZYX")

            # Verify channel labels survived round-trip
            labels2 = znimg2.list_channels()
            assert labels2 == ["Red", "Green", "Blue"]

            # Verify data shape (should match original)
            assert znimg2.shape == znimg.shape

    def test_read_4d_nifti_as_channels(self):
        """Test that 4D NIfTI files are interpreted as channels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 4D NIfTI (X, Y, Z, C)
            data = np.random.rand(10, 20, 30, 4).astype(np.float32)
            nifti_path = os.path.join(tmpdir, "4d.nii.gz")
            nifti_img = nib.Nifti1Image(data, np.eye(4))
            nib.save(nifti_img, nifti_path)
            
            # Load with ZarrNii
            znimg = ZarrNii.from_nifti(nifti_path)
            
            # Should interpret 4th dimension as channels
            assert "c" in znimg.dims
            assert znimg.shape[0] == 4  # 4 channels
            
            # Verify data integrity
            computed_data = znimg.data.compute()
            # ZarrNii stores as (C, X, Y, Z), NIfTI stores as (X, Y, Z, C)
            # So we need to transpose back to compare
            if znimg.axes_order == "XYZ":
                # CXYZ -> XYZC for comparison
                transposed = computed_data.transpose(1, 2, 3, 0)
            else:
                # CZYX -> XYZC for comparison
                transposed = computed_data.transpose(3, 2, 1, 0)
            
            assert_array_equal(transposed, data)

    def test_3d_nifti_adds_channel_dimension(self):
        """Test that 3D NIfTI files get a channel dimension added."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 3D NIfTI
            data = np.random.rand(10, 20, 30).astype(np.float32)
            nifti_path = os.path.join(tmpdir, "3d.nii.gz")
            nifti_img = nib.Nifti1Image(data, np.eye(4))
            nib.save(nifti_img, nifti_path)
            
            # Load with ZarrNii
            znimg = ZarrNii.from_nifti(nifti_path)
            
            # Should have added channel dimension
            assert znimg.shape == (1, 10, 20, 30)
            assert znimg.dims[0] == "c"

    def test_channel_selection_to_nifti(self):
        """Test selecting channels before writing to NIfTI."""
        import dask.array as da
        import ngff_zarr as nz
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multi-channel data
            arr = da.from_array(
                np.random.random((5, 8, 16, 16)).astype(np.float32), 
                chunks=(1, 8, 16, 16)
            )
            
            ngff_image = nz.to_ngff_image(arr, dims=["c", "z", "y", "x"])
            multiscales = nz.to_multiscales(ngff_image)
            
            # Add channel labels
            from ngff_zarr import Omero, OmeroChannel, OmeroWindow
            window = OmeroWindow(min=0.0, max=1.0, start=0.0, end=1.0)
            omero = Omero(
                channels=[
                    OmeroChannel(color="FFFFFF", window=window, label=f"Ch{i}") for i in range(5)
                ]
            )
            multiscales.metadata.omero = omero
            
            zarr_path = os.path.join(tmpdir, "test.zarr")
            nz.to_ngff_zarr(zarr_path, multiscales)
            
            # Load and select specific channels
            znimg = ZarrNii.from_ome_zarr(zarr_path)
            znimg_selected = znimg.select_channels([1, 3])
            
            # Save to NIfTI
            nifti_path = os.path.join(tmpdir, "selected.nii.gz")
            znimg_selected.to_nifti(nifti_path)
            
            # Load back and verify
            reloaded = nib.load(nifti_path)
            # Should have 2 channels in 4th dimension
            assert reloaded.shape[3] == 2
            
            # Load with ZarrNii and check labels
            znimg2 = ZarrNii.from_nifti(nifti_path)
            labels = znimg2.list_channels()
            assert labels == ["Ch1", "Ch3"]

    def test_channel_labels_without_omero(self):
        """Test that files without OMERO metadata still work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 4D NIfTI without channel labels
            data = np.random.rand(10, 20, 30, 3).astype(np.float32)
            nifti_path = os.path.join(tmpdir, "no_labels.nii.gz")
            nifti_img = nib.Nifti1Image(data, np.eye(4))
            nib.save(nifti_img, nifti_path)
            
            # Load with ZarrNii
            znimg = ZarrNii.from_nifti(nifti_path)
            
            # Should work but have no channel labels
            assert znimg.shape == (3, 10, 20, 30)
            labels = znimg.list_channels()
            assert labels == []
            
            # Should still be able to save back
            output_path = os.path.join(tmpdir, "output.nii.gz")
            znimg.to_nifti(output_path)
            assert os.path.exists(output_path)

    def test_backward_compatibility_3d(self):
        """Test that 3D NIfTI I/O still works as before."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 3D NIfTI
            data = np.random.rand(10, 20, 30).astype(np.float32)
            nifti_path = os.path.join(tmpdir, "3d.nii.gz")
            nifti_img = nib.Nifti1Image(data, np.eye(4))
            nib.save(nifti_img, nifti_path)
            
            # Load and save back
            znimg = ZarrNii.from_nifti(nifti_path)
            output_path = os.path.join(tmpdir, "output.nii.gz")
            znimg.to_nifti(output_path)
            
            # Verify data matches
            reloaded = nib.load(output_path)
            assert reloaded.shape == data.shape
            assert_array_equal(reloaded.get_fdata(), data)
