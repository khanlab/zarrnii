import os
import tempfile

import nibabel as nib
import numpy as np
import ngff_zarr as nz
import pytest
from zarrnii import ZarrNii

@pytest.fixture
def nifti_nib():
    img_size = (100, 50, 200)
    pix_dims = (0.3, 0.2, 1.5, 1)

    nifti_nib = nib.Nifti1Image(np.random.rand(*img_size), affine=np.diag(pix_dims))

    return nifti_nib

@pytest.fixture
def znimg_from_multiscales():
    img_size = (1,64, 128, 256)
    pix_dims = (1, 0.3, 0.2, 1.5, 1)

    ngff_image = nz.to_ngff_image(np.ones(img_size),
                             dims=['c','z','y','x'])
    multiscales = nz.to_multiscales(ngff_image,scale_factors=[2,4],
                                    chunks=(1,32,32,32))

    nz.to_ngff_zarr('test_znimg.ome.zarr',multiscales,
                    chunks_per_shard=16,
                    version='0.5')
    znimg = ZarrNii.from_ome_zarr('test_znimg.ome.zarr')

    return znimg



@pytest.fixture
def cleandir_fake():
    print('fake clean')

@pytest.fixture
def cleandir():
    with tempfile.TemporaryDirectory() as newpath:
        old_cwd = os.getcwd()
        os.chdir(newpath)
        yield
        os.chdir(old_cwd)
