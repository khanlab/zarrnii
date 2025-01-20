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
