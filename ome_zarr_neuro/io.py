import dask.array as da
from transform import DaskImage


# function to extract ome zarr subvolume around nifti label from e.g. downsampled nii
in_zarr = '/cifs/prado/Kate/AK_Pipeline_LSM/spimprep/bids/sub-o28/micr/sub-o28_sample-brain_acq-prestitched_SPIM.ome.zarr/'
tiny_zarr='/local/SPIMquant/test_out/results/sub-o28/micr/sub-o28_sample-brain_acq-prestitched_desc-tiny_SPIM.ome.zarr'
in_lbl_nii='/local/SPIMquant/test_out/results/sub-o28/micr/sub-o28_sample-brain_acq-prestitched_from-ABAv3_level-5_desc-deform_dseg.nii.gz'

def extract_subvol_ome_zarr():
    zarr_img = DaskImage.from_path(in_zarr) 
    nii_label = DaskImage.from_path(in_lbl_nii)

    print(zarr_img)
    print(nii_label.darr.shape)

    darr_label = da.zeros_like(nii_label.darr)
    darr_label[nii_label.darr==457] = 1

    indices = da.argwhere(darr_label).compute()
    # Compute the minimum and maximum extents in each dimension
    min_extent = indices.min(axis=0)
    max_extent = indices.max(axis=0)

    #get bounds
    print(min_extent)
    print(max_extent)

    zarr_img.darr = zarr_img.darr[min_extent[0]:max_extent[0]+1, min_extent[1]:max_extent[1]+1, min_extent[2]:max_extent[2]+1]

    zarr_img.to_ome_zarr('test_subvol.ome.zarr')

extract_subvol_ome_zarr()

