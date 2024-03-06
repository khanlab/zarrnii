import zarr
import dask.array as da
import nibabel as nib
import numpy as np
from pathlib import Path
from scipy.interpolate import interpn
from dask.diagnostics import ProgressBar






#this global variable is set by apply_transform, then used in map_blocks with interp_by_blocks
global _flo_darr


#vox2ras now works for both ome_zarr and nifti.. next, set-up image volume reading for each..


def get_dask_array_from_path(in_path,level,channel,chunks):
    """ returns a dask array whether a nifti or ome_zarr is provided """
    img_type = check_img_type(in_path)
    if img_type == 'ome_zarr':
        return da.from_zarr(in_path,component=f'/{level}')[channel,:,:,:].squeeze()
    elif img_type == 'nifti':
        return da.from_array(nib.load(in_path).get_fdata(),chunks=chunks)
    else:
        return None



def check_img_type(in_path):
    suffixes=Path(in_path).suffixes

    if len(suffixes) >1:
        if suffixes[-2] == '.ome' and suffixes[-1] == '.zarr':
            return 'ome_zarr'
        if suffixes[-2] == '.nii' and suffixes[-1] == '.gz':
            return 'nifti'
    if suffixes[-1] == '.zarr':
        return 'zarr'
    elif suffixes[-1] == 'nii':
        return 'nifti'
    else:
        return 'unknown'
   

def get_vox2ras(in_path:str,level=0) -> np.array:
    img_type = check_img_type(in_path)

    if img_type == 'ome_zarr':
        return get_vox2ras_zarr(in_path,level)
    elif img_type == 'nifti':
        return get_vox2ras_nii(in_path)
    else:
        print('unknown image type for vox2ras')
        return None


def get_ras2vox(in_path:str,level=0) -> np.array:
    img_type = check_img_type(in_path)
    if img_type == 'ome_zarr':
        return get_ras2vox_zarr(in_path,level)
    elif img_type == 'nifti':
        return get_ras2vox_nii(in_path)
    else:
        print('unknown image type for ras2vox')
        return None

def get_vox2ras_nii(in_nii_path:str) -> np.array:

    return nib.load(in_nii_path).affine

def get_ras2vox_nii(in_nii_path:str) -> np.array:

    return np.linalg.inv(get_vox2ras_nii(in_nii_path))



def get_vox2ras_zarr(in_zarr_path:str,level=0) -> np.array:

    #read coordinate transform from ome-zarr
    zi = zarr.open(in_zarr_path)
    attrs=zi['/'].attrs.asdict()
    multiscale=0 #first multiscale image
    transforms = attrs['multiscales'][multiscale]['datasets'][level]['coordinateTransformations']

    # 0. reorder_xfm -- changes from z,y,x to x,y,z ordering
    reorder_xfm = np.eye(4)
    reorder_xfm[:3,:3] = np.flip(reorder_xfm[:3,:3],axis=0) #reorders z-y-x to x-y-z and vice versa

    # 1. scaling_xfm (vox2ras in spim space)
    # this matches what the ome_zarr_to_nii affine has

    scaling_xfm = np.eye(4)
    scaling_xfm[0,0]=-transforms[0]['scale'][-1] #x  # 0-index in transforms is the first (and only) transform 
    scaling_xfm[1,1]=-transforms[0]['scale'][-2] #y
    scaling_xfm[2,2]=-transforms[0]['scale'][-3] #z


    return scaling_xfm @ reorder_xfm

def get_ras2vox_zarr(in_zarr_path:str,level=0) -> np.array:
    return np.linalg.inv(get_vox2ras_zarr(in_zarr_path,level=level))




def get_bounded_subregion(points: np.array ):

    """ 
    Uses the extent of points, along with the shape of the dask array,
    to return a subregion of the dask array that contains the points,
    along with the grid points corresponding to the indices from the 
    uncropped dask array (to be used for interpolation).

    if the points go outside the domain of the dask array, then it is
    capped off with floor or ceil.

    points are Nx3 or Nx4 coordinates (with/without homog coord)
    darr is the floating image dask array

    We use compute() on the floating dask array to immediately get it
    since dask doesn't support nd fancy indexing yet that interpn seems to use 
    """ 
    pad=1
    min_extent=np.floor(points.min(axis=1)[:3]-pad).astype('int') 
    max_extent=np.ceil(points.max(axis=1)[:3]+pad).astype('int')

    min_extent=np.maximum(min_extent,np.zeros(min_extent.shape))
    max_extent=np.minimum(max_extent,(_flo_darr.shape[-3],_flo_darr.shape[-2],_flo_darr.shape[-1]))

    subvol = _flo_darr[min_extent[0]:max_extent[0],
                          min_extent[1]:max_extent[1],
                          min_extent[2]:max_extent[2]].compute()

     # along with grid points for interpolation
    grid_points = (np.arange(min_extent[0],max_extent[0]),
                   np.arange(min_extent[1],max_extent[1]),
                   np.arange(min_extent[2],max_extent[2]))

    return (grid_points,subvol)
    

def interp_by_block(x,
                    transform_stack,
                    block_info=None,
                    interp_method='linear'
                    ):
    """
    main idea here is we take coordinates from the current block (ref image)
    transform them, then in that transformed space, then interpolate the 
    floating image intensities

    since the floating image is a dask array, we need to just load a 
    subset of it -- we use the bounds of the points in the transformed 
     space to define what range of the floating image we load in
    """
    arr_location = block_info[0]['array-location']
    
    xv,yv,zv=np.meshgrid(np.arange(arr_location[0][0],arr_location[0][1]),
            np.arange(arr_location[1][0],arr_location[1][1]),
            np.arange(arr_location[2][0],arr_location[2][1]),indexing='ij')

    #reshape them into a vectors (x,y,z,1) for each point, so we can matrix multiply
    xvf=xv.reshape((1,np.product(xv.shape)))
    yvf=yv.reshape((1,np.product(yv.shape)))
    zvf=zv.reshape((1,np.product(zv.shape)))
    homog=np.ones(xvf.shape)
    
    xfm_vecs=np.vstack((xvf,yvf,zvf,homog))
    

    #apply transforms one at a time (will need to edit this for warps)
    for transform in transform_stack:
        xfm_vecs = transform @ xfm_vecs
    

    #find bounding box required for flo vol
    (grid_points,flo_vol) = get_bounded_subregion(xfm_vecs)
                  
    #then finally interpolate those points on the template dseg volume
    interpolated = interpn(grid_points,flo_vol,
                        xfm_vecs[:3,:].T, #
                        method=interp_method,
                        bounds_error=False,
                        fill_value=0)
    
    return interpolated.reshape(x.shape).astype(block_info[None]['dtype'])


def apply_transform(flo_img_path:str, #can be ome_zarr or nifti
                    ref_img_path:str, #can be ome_zarr or nifti
                    transform_specs:list[str],
                    ref_channel=0,#channel to use if ref is ome_zarr
                    flo_channel=0,#channel to use if flo is ome_zarr
                    ref_level=0,#downsampling level to use if ref is ome_zarr
                    flo_level=0,#downsampling level to use if ref is ome_zarr
                    ref_chunks='auto', 
                    flo_chunks='auto') -> da.Array:
    """ return dask array applying transform to floating image.
        this is a lazy function, doesn't do any work until you compute() 
        on the returned dask array.

        note: global _flo_darr variable is needed since I can't pass
        a dask array as a kwarg to map_blocks -- could get around this
        by using a class, can do that later...
    """

    #load dask array for floating image
    global _flo_darr
    _flo_darr = get_dask_array_from_path(flo_img_path,level=flo_level,channel=flo_channel,chunks=flo_chunks)

    #load reference  (this will be ref space for now)
    ref_darr = get_dask_array_from_path(ref_img_path,level=ref_level,channel=ref_channel,chunks=ref_chunks)


    #the transform can be put together from flo to ref, then inverted
    # or equivalent to going from ref to flo
    
    # e.g., from flo to ref:
    # ref_ras2vox @ affine_flo_to_ref @ flo_vox2ras

    # or, from ref to flo:
    #  flo_ras2vox @ affine_ref_to_flo @ ref_vox2ras 

    # both options are below for educational sake:

    #the inverse of the affine is used for warping images
   # affine_ref_to_flo = np.loadtxt(flo_to_ref_affine_xfm)
   # affine_flo_to_ref= np.linalg.inv(affine_ref_to_flo)

    transform_stack=[]
    
    transform_stack.append(get_vox2ras(ref_img_path))

    for tfm_spec in transform_specs:
        if tfm_spec['type'] == 'affine_ras':
            aff = np.loadtxt(tfm_spec['path'])
            if tfm_spec.get('invert',False):
                aff = np.linalg.inv(aff)
            transform_stack.append(aff)

    transform_stack.append(get_ras2vox(flo_img_path))

#    inv_matrix_transform = ref_ras2vox @ affine_flo_to_ref @ flo_vox2ras
#    matrix_transform_alt = np.linalg.inv(inv_matrix_transform)

#    matrix_transform = flo_ras2vox @ affine_ref_to_flo @ ref_vox2ras

    #perform interpolation on each block in parallel
    darr_interp=da.map_blocks(interp_by_block,
                        ref_darr, dtype=np.uint16,
                        transform_stack=transform_stack)

    return darr_interp
                    
    

