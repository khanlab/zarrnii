import numpy as np
import pytest
from numpy.testing import assert_array_equal

from zarrnii import ZarrNii


@pytest.mark.usefixtures("cleandir")
def test_downsample(nifti_nib):
    nifti_nib.to_filename("test.nii")
    znimg = ZarrNii.from_nifti("test.nii")

    ds = np.array([2, 5, 8])
    znimg_downsampled = znimg.downsample(along_x=ds[0], along_y=ds[1], along_z=ds[2])

    # check size is same as expected size
    assert_array_equal(znimg.darr.shape[1:], znimg_downsampled.darr.shape[1:] * ds)




@pytest.mark.usefixtures("cleandir")
def test_downsample_on_read(nifti_nib):
    nifti_nib.to_filename("test.nii")
    znimg = ZarrNii.from_nifti("test.nii")

    # now we have znimg with axes_order == 'XYZ'
    znimg.to_ome_zarr("test_fromznimg.ome.zarr",max_layer=0)
    
    level=2
    znimg2 = ZarrNii.from_ome_zarr("test_fromznimg.ome.zarr",level=level)
    znimg2


    xyz_orig=znimg.darr.shape[1:]
    xyz_ds=znimg2.darr.shape[:0:-1]



    #x y and z are modified by 2^level
    assert_array_equal(int(xyz_orig[0]/(2**level)),xyz_ds[0])
    assert_array_equal(int(xyz_orig[1]/(2**level)),xyz_ds[1])
    assert_array_equal(int(xyz_orig[2]/(2**level)),xyz_ds[2])


@pytest.mark.usefixtures("cleandir")
def test_read_from_downsampled_level(nifti_nib):
    nifti_nib.to_filename("test.nii")
    znimg = ZarrNii.from_nifti("test.nii")

    # now we have znimg with axes_order == 'XYZ'
    znimg.to_ome_zarr("test_fromznimg.ome.zarr",max_layer=4)
    
    level=2
    znimg2 = ZarrNii.from_ome_zarr("test_fromznimg.ome.zarr",level=level)
    znimg2


    xyz_orig=znimg.darr.shape[1:]
    xyz_ds=znimg2.darr.shape[:0:-1]



    #x y and z are modified by 2^level
    assert_array_equal(int(xyz_orig[0]/(2**level)),xyz_ds[0])
    assert_array_equal(int(xyz_orig[1]/(2**level)),xyz_ds[1])
    assert_array_equal(int(xyz_orig[2]/(2**level)),xyz_ds[2])
