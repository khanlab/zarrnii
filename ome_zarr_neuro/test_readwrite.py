#tests
import nibabel as nib
from ome_zarr_neuro.transform import DaskImage, TransformSpec
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal


#TODO: use tempfiles for test images

def test_read_write_nii():
    """create a nifti with nibabel, read it with DaskImage, then write it back as a nifti.
        ensure data, affine and header do not change."""

    img_size=(100,50,200)
    pix_dims=(0.3,0.2,1.5,1)
    
    nib.Nifti1Image(np.random.rand(*img_size),affine=np.diag(pix_dims)).to_filename('test.nii')
    nib_orig = nib.load('test.nii')
    
    dimg = DaskImage.from_path('./test.nii')
    dimg.to_nifti('test_fromdimg.nii')
    nib_dimg = nib.load('test_fromdimg.nii')

    #now compare nib_orig and nib_dimg
    assert_array_equal(nib_orig.affine,nib_dimg.affine)
    assert_array_equal(nib_orig.get_fdata(),nib_dimg.get_fdata())



def test_compare_daskimage_nii_to_zarr():
    """create a nifti with nibabel, read it with DaskImage, write it back as zarr,
        then read it again with DaskImage, 
        ensure data, affine and header do not change."""

    img_size=(100,50,200)
    pix_dims=(0.3,0.2,1.5,1)
    
    nib.Nifti1Image(np.random.rand(*img_size),affine=np.diag(pix_dims)).to_filename('test.nii')
    nib_orig = nib.load('test.nii')
    
    dimg = DaskImage.from_path('./test.nii')

    #now we have dimg with axes_nifti == True 
    #  this means, the affine is XYZ vox dims

    
    
    
    dimg.to_ome_zarr('test_fromdimg.ome.zarr')
    dimg2 = DaskImage.from_path('test_fromdimg.ome.zarr')
    dimg2

    #now we have dimg2 with axes_nifti == False
    #  this means, the affine is ZYX vox dims negated
    
    
    dimg2.to_nifti('test_fromdimg_tonii.nii')
    dimg3 = DaskImage.from_path('test_fromdimg_tonii.nii')
    
    assert_array_equal(dimg.vox2ras.affine,dimg3.vox2ras.affine)
    assert_array_equal(dimg.darr.compute(),dimg3.darr.compute())



def test_compare_daskimage_zarr_to_zarr():
    """create a nifti with nibabel, read it with DaskImage, write it back as zarr,
        then read it again with DaskImage, 
        ensure data, affine and header do not change."""

    img_size=(100,50,200)
    pix_dims=(0.3,0.2,1.5,1)
    
    nib.Nifti1Image(np.random.rand(*img_size),affine=np.diag(pix_dims)).to_filename('test.nii')
    nib_orig = nib.load('test.nii')
    
    dimg = DaskImage.from_path('./test.nii')

    #now we have dimg with axes_nifti == True 
    #  this means, the affine is XYZ vox dims

    dimg.to_ome_zarr('test_fromdimg.ome.zarr')
    dimg2 = DaskImage.from_path('test_fromdimg.ome.zarr')
    dimg2

    #now we have dimg2 with axes_nifti == False
    #  this means, the affine is ZYX vox dims negated
    
    
    dimg2.to_ome_zarr('test_fromdimg_tozarr.ome.zarr')
    dimg3 = DaskImage.from_path('test_fromdimg_tozarr.ome.zarr')
    
    assert_array_equal(dimg2.vox2ras.affine,dimg3.vox2ras.affine)
    assert_array_equal(dimg2.darr.compute(),dimg3.darr.compute())



def test_identity_transform_nii_refnifti():
    """ tests affine transform from nifti to zarr using identity transformation """
    img_size=(100,50,200)
    pix_dims=(0.3,0.2,1.5,1)
    
    nib.Nifti1Image(np.random.rand(*img_size),affine=np.diag(pix_dims)).to_filename('test.nii')
    
    flo_dimg = DaskImage.from_path('./test.nii')

    ref_dimg = DaskImage.from_path('./test.nii')

    interp_dimg = flo_dimg.apply_transform(TransformSpec.affine_ras_from_array(np.eye(4)),ref_dimg=ref_dimg)
    
    
    assert_array_almost_equal(flo_dimg.darr.compute(), interp_dimg.darr.compute())

    

    


    

