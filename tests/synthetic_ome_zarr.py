"""
Copied over and modified from
https://github.com/khanlab/cvpl_tools/blob/main/src/cvpl_tools/ome_zarr/io.py,
sole purpose of this file to generate synthetic ome zarr files for testing
"""

import dask.array as da
import numpy as np
import zarr
import ngff_zarr as nz


def generate_synthetic_dataset(
    ome_zarr_path: str,
    arr_sz: tuple = (2, 224, 1600, 2048)) -> None:
    """Generate a 4d synthetic test ome zarr image physically stored in ome_zarr_path.

    Args:
        ome_zarr_path: Where to store the generated test image
        arr_sz: The size of the synthetic image

    """


    arr: da.Array = da.zeros(arr_sz, dtype=np.uint16)

    def process_block(block, block_info=None):
        if block_info is not None:
            # calculate (global) indices array for each pixel
            block_slice = block_info[0]["array-location"]
            indices = np.indices(block.shape)
            for dim in range(indices.shape[0]):
                indices[dim] += block_slice[dim][0]
        else:
            return np.zeros(block.shape, dtype=np.uint16)
        # now, create balls in the block
        sq = np.zeros(block.shape)  # distance squared
        for dim in range(
            1, indices.shape[0]
        ):  # every dim except channel dim which does not have distance
            sq += np.power(indices[dim], 2.0) * 0.0002
        for dim in range(
            1, indices.shape[0]
        ):  # every dim except channel dim which does not have distance
            indices[dim] %= 32
            sq += np.power(indices[dim] - 15.5, 2.0)
        im = np.array(np.clip(1200.0 - sq * 15.0, 0.0, 1200.0), dtype=np.uint16)
        return im

    arr = arr.map_blocks(process_block, dtype=np.uint16)

    ngff_image = nz.to_ngff_image(arr)
    multiscales = nz.to_multiscales(ngff_image)
    nz.to_ngff_zarr(ome_zarr_path,multiscales)
    


