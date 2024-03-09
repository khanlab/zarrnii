import zarr
import dask.array as da
import nibabel as nib
import numpy as np
from pathlib import Path
from scipy.interpolate import interpn
from dask.diagnostics import ProgressBar
from attrs import define
from enum import Enum, auto

class TransformType(Enum):
    AFFINE_RAS = auto()
    DISPLACEMENT_RAS = auto()

class ImageType(Enum):
    OME_ZARR = auto()
    ZARR = auto()
    NIFTI = auto()
    UNKNOWN = auto()

@define
class TransformSpec:
    tfm_type: TransformType
    affine: np.array = None
    disp_xyz: np.array = None
    disp_grid: np.array = None
    disp_ras2vox: np.array = None

    @classmethod
    def affine_ras_from_txt(cls,path,invert=False):
        affine = np.loadtxt(path)
        if invert:
            affine=np.linalg.inv(self.affine)

        return cls(TransformType.AFFINE_RAS,affine=affine)

    @classmethod
    def affine_ras_from_array(cls,affine,invert=False):
        if invert:
            affine=np.linalg.inv(affine)

        return cls(TransformType.AFFINE_RAS,affine=affine)

    @classmethod
    def displacement_from_nifti(cls,path):
        disp_nib = nib.load(path)
        disp_xyz = disp_nib.get_fdata().squeeze()
        disp_ras2vox = np.linalg.inv(disp_nib.affine)
   
        #convert from itk transform
        disp_xyz[:,:,:,0] = -disp_xyz[:,:,:,0]
        disp_xyz[:,:,:,1] = -disp_xyz[:,:,:,1]

        disp_grid = (np.arange(disp_xyz.shape[0]),
                np.arange(disp_xyz.shape[1]),
                np.arange(disp_xyz.shape[2]))


        return cls(TransformType.DISPLACEMENT_RAS,disp_xyz=disp_xyz,disp_grid=disp_grid,disp_ras2vox=disp_ras2vox)

    @classmethod
    def vox2ras_from_image(cls, path:str,level=0):
        img_type = ImageSpec.check_img_type(path)

        if img_type is ImageType.OME_ZARR:
            return cls(TransformType.AFFINE_RAS,affine=cls.get_vox2ras_zarr(path,level))
        elif img_type is ImageType.NIFTI:
            return cls(TransformType.AFFINE_RAS,affine=cls.get_vox2ras_nii(path))
        else:
            print('unknown image type for vox2ras')
            return None

    @classmethod
    def ras2vox_from_image(cls, path:str,level=0):
        img_type = ImageSpec.check_img_type(path)
        if img_type is ImageType.OME_ZARR:
            return cls(TransformType.AFFINE_RAS,affine=cls.get_ras2vox_zarr(path,level))
        elif img_type is ImageType.NIFTI:
            return cls(TransformType.AFFINE_RAS,affine=cls.get_ras2vox_nii(path))
        else:
            print('unknown image type for ras2vox')
            return None

    @staticmethod
    def get_vox2ras_nii(in_nii_path:str) -> np.array:

        return nib.load(in_nii_path).affine

    @staticmethod
    def get_ras2vox_nii(in_nii_path:str) -> np.array:

        return np.linalg.inv(TransformSpec.get_vox2ras_nii(in_nii_path))

    @staticmethod
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

    @staticmethod
    def get_ras2vox_zarr(in_zarr_path:str,level=0) -> np.array:
        return np.linalg.inv(TransformSpec.get_vox2ras_zarr(in_zarr_path,level=level))




    def apply_transform(self, vecs:np.array) -> np.array:
        
        if self.tfm_type == TransformType.AFFINE_RAS:
            return  self.affine @ vecs

        elif self.tfm_type == TransformType.DISPLACEMENT_RAS:
            #we have the grid points, the volumes to interpolate displacements
            
            #first we need to transform points to vox space of the warp
            vox_vecs = self.disp_ras2vox @ vecs

            #then interpolate the displacement in x, y, z:
            disp_vecs = np.zeros(vox_vecs.shape)

            for ax in range(3):
                disp_vecs[ax,:] = interpn(self.disp_grid,
                                self.disp_xyz[:,:,:,ax].squeeze(),
                                vox_vecs[:3,:].T,
                                method='linear',
                                bounds_error=True,
                                fill_value=0)
           
                    
            return vecs + disp_vecs

@define
class ImageSpec:
    darr: da.Array
    ras2vox: TransformSpec = None
    vox2ras: TransformSpec = None

    @classmethod
    def ref_from_path(cls,path,level,channels,chunks):
        """ ref image we dont need actual intensities, just the shape and affine"""
        img_type = cls.check_img_type(path)
        if img_type is ImageType.OME_ZARR:
            darr_base = da.from_zarr(path,component=f'/{level}')[channels,:,:,:]
            darr = da.empty(darr_base)
        elif img_type is ImageType.NIFTI:
            darr_base = da.from_array(nib.load(path).get_fdata(),chunks=chunks)
            darr_empty = da.empty(darr_base,shape=(len(channels),darr_base.shape[-3],darr_base.shape[-2],darr_base.shape[-1]))
        else:
            print('unknown image type')
            return None
        
        #ensure chunks include all channels
        if darr_empty.chunksize[0] < len(channels):
            darr_empty.rechunk(chunks=(len(channels),darr_empty.chunksize[1],darr_empty.chunksize[2],darr_empty.chunksize[3]))

        return cls(darr_empty, 
                    ras2vox=TransformSpec.ras2vox_from_image(path),
                    vox2ras=TransformSpec.vox2ras_from_image(path))


       

    @classmethod
    def img_from_path(cls, path,level,channels,chunks):
        """ returns a dask array whether a nifti or ome_zarr is provided """
        img_type = cls.check_img_type(path)
        if img_type is ImageType.OME_ZARR:
            darr = da.from_zarr(path,component=f'/{level}')[channels,:,:,:]
        elif img_type is ImageType.NIFTI:
            darr = da.from_array(np.expand_dims(nib.load(path).get_fdata(),axis=0),chunks=chunks)
        else:
            print('unknown image type')
            return None

        return cls(darr, 
                    ras2vox=TransformSpec.ras2vox_from_image(path),
                    vox2ras=TransformSpec.vox2ras_from_image(path))



    @staticmethod
    def check_img_type(path) -> ImageType:
        suffixes=Path(path).suffixes

        if len(suffixes) >1:
            if suffixes[-2] == '.ome' and suffixes[-1] == '.zarr':
                return ImageType.OME_ZARR
            if suffixes[-2] == '.nii' and suffixes[-1] == '.gz':
                return ImageType.NIFTI
        if suffixes[-1] == '.zarr':
            return ImageType.ZARR
        elif suffixes[-1] == 'nii':
            return ImageType.NIFTI
        else:
            return ImageType.UNKNOWN
       






#this global variable is set by apply_transform, then used in map_blocks with interp_by_blocks
#global _flo_darr




def get_bounded_subregion(points: np.array,flo_img: ImageSpec ):

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
    max_extent=np.minimum(max_extent,(flo_img.darr.shape[-3],flo_img.darr.shape[-2],flo_img.darr.shape[-1]))


    subvol = flo_img.darr[:,min_extent[0]:max_extent[0],
                          min_extent[1]:max_extent[1],
                          min_extent[2]:max_extent[2]].compute()


     # along with grid points for interpolation
    grid_points = (np.arange(min_extent[0],max_extent[0]),
                   np.arange(min_extent[1],max_extent[1]),
                   np.arange(min_extent[2],max_extent[2]))

    return (grid_points,subvol)
    

def interp_by_block(x,
                    transform_specs:list[TransformSpec],
                    flo_img:ImageSpec,
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
    xv,yv,zv=np.meshgrid(np.arange(arr_location[-3][0],arr_location[-3][1]),
            np.arange(arr_location[-2][0],arr_location[-2][1]),
            np.arange(arr_location[-1][0],arr_location[-1][1]),indexing='ij')

    #reshape them into a vectors (x,y,z,1) for each point, so we can matrix multiply
    xvf=xv.reshape((1,np.product(xv.shape)))
    yvf=yv.reshape((1,np.product(yv.shape)))
    zvf=zv.reshape((1,np.product(zv.shape)))
    homog=np.ones(xvf.shape)
   

    xfm_vecs=np.vstack((xvf,yvf,zvf,homog))

#    print(xfm_vecs[:,0])

    #apply transforms one at a time (will need to edit this for warps)
    for tfm_spec in transform_specs:
        xfm_vecs = tfm_spec.apply_transform(xfm_vecs)
#        print(xfm_vecs[:,0])
    

    
    #find bounding box required for flo vol
    (grid_points,flo_vol) = get_bounded_subregion(xfm_vecs,flo_img)

    #then finally interpolate those points on the template dseg volume
    #need to interpolate for each channel

    interpolated = np.zeros(x.shape)

    for c in range(flo_vol.shape[0]):
        interpolated[c,:,:,:] = interpn(grid_points,flo_vol[c,:,:,:],
                        xfm_vecs[:3,:].T, #
                        method=interp_method,
                        bounds_error=False,
                        fill_value=0).reshape((x.shape[-3],x.shape[-2],x.shape[-1])).astype(block_info[None]['dtype'])
#        print(interpolated[c,0,0,0])


    

    return interpolated


def apply_transform(flo_img_path:str, #can be ome_zarr or nifti
                    ref_img_path:str, #can be ome_zarr or nifti
                    transform_specs:list[str],
                    channels=[0], #channels to use
#                    ref_channel=0,#channel to use if ref is ome_zarr
#                    flo_channel=0,#channel to use if flo is ome_zarr
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
#    global _flo_darr
#    _flo_darr = get_dask_array_from_path(flo_img_path,level=flo_level,channels=channels,chunks=flo_chunks)

    #load reference  (this will be ref space for now)
#    ref_darr = get_dask_array_from_path(ref_img_path,level=ref_level,channels=channels,chunks=ref_chunks)


    flo_img = ImageSpec.img_from_path(flo_img_path,level=flo_level,channels=channels,chunks=flo_chunks)
    ref_img = ImageSpec.ref_from_path(ref_img_path,level=ref_level,channels=channels,chunks=ref_chunks)


    #transform specs already has the transformations to apply, just need the conversion to/from vox/ras at start and end
    transform_specs.insert(0,ref_img.vox2ras) #prepend
    transform_specs.append(flo_img.ras2vox)


    #perform interpolation on each block in parallel
    darr_interp=da.map_blocks(interp_by_block,
                        ref_img.darr, dtype=np.float32,
                        transform_specs=transform_specs,
                        flo_img=flo_img)

    return darr_interp
                    







