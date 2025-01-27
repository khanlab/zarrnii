import os
import tempfile

import nibabel as nib
import numpy as np
import pytest


@pytest.fixture
def nifti_nib():
    img_size = (100, 50, 200)
    pix_dims = (0.3, 0.2, 1.5, 1)

    nifti_nib = nib.Nifti1Image(np.random.rand(*img_size), affine=np.diag(pix_dims))

    return nifti_nib


@pytest.fixture
def cleandir():
    with tempfile.TemporaryDirectory() as newpath:
        old_cwd = os.getcwd()
        os.chdir(newpath)
        yield
        os.chdir(old_cwd)
