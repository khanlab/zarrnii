#!/usr/bin/env python3
"""
Example demonstrating 5D image support in ZarrNii with time dimension (T,C,Z,Y,X).

This example shows how to:
1. Load 5D OME-Zarr data with time and channel dimensions
2. Select specific timepoints and channels  
3. Apply spatial transformations that preserve T,C dimensions
4. Use the new select_timepoints() method
"""

import numpy as np
import dask.array as da
import ngff_zarr as nz
import tempfile
import os
from zarrnii import ZarrNii


def create_example_5d_dataset():
    """Create a synthetic 5D dataset for demonstration."""
    print("Creating synthetic 5D dataset...")
    
    # Create 5D data: (timepoints=3, z=8, y=16, x=16, channels=2)
    num_timepoints, num_channels = 3, 2
    data_shape = (num_timepoints, 8, 16, 16, num_channels)
    
    # Generate synthetic data with different patterns for each timepoint/channel
    data = np.zeros(data_shape, dtype=np.float32)
    
    for t in range(num_timepoints):
        for c in range(num_channels):
            # Create a unique pattern for each time/channel combination
            pattern_value = (t + 1) * 100 + (c + 1) * 10
            data[t, :, :, :, c] = pattern_value
            
            # Add some spatial variation
            z, y, x = np.meshgrid(
                np.arange(8), np.arange(16), np.arange(16), indexing='ij'
            )
            data[t, :, :, :, c] += z + y * 0.1 + x * 0.01
    
    # Convert to dask array
    dask_data = da.from_array(data, chunks=(1, 4, 8, 8, 1))
    
    # Create NgffImage with proper 5D dimensions
    ngff_image = nz.to_ngff_image(dask_data, dims=["t", "z", "y", "x", "c"])
    multiscales = nz.to_multiscales(ngff_image)
    
    # Save to temporary zarr file
    tmpdir = tempfile.mkdtemp()
    store_path = os.path.join(tmpdir, "example_5d.zarr")
    nz.to_ngff_zarr(store_path, multiscales)
    
    print(f"Created dataset at: {store_path}")
    print(f"Data shape: {data_shape}")
    print(f"Dimensions: {ngff_image.dims}")
    
    return store_path


def demonstrate_5d_functionality():
    """Demonstrate various 5D operations."""
    
    # Create test dataset
    dataset_path = create_example_5d_dataset()
    
    try:
        print("\n" + "="*50)
        print("1. Loading 5D data (all timepoints and channels)")
        znimg = ZarrNii.from_ome_zarr(dataset_path)
        print(f"   Loaded shape: {znimg.darr.shape}")
        print(f"   Dimensions: {znimg.ngff_image.dims}")
        
        print("\n" + "="*50)
        print("2. Selecting specific timepoints")
        znimg_t02 = ZarrNii.from_ome_zarr(dataset_path, timepoints=[0, 2])
        print(f"   Timepoints [0,2] shape: {znimg_t02.darr.shape}")
        
        # Verify the data values
        t0_avg = znimg_t02.darr[0, :, :, :, 0].mean().compute()
        t1_avg = znimg_t02.darr[1, :, :, :, 0].mean().compute()  # This is actually timepoint 2
        print(f"   Timepoint 0, channel 0 average: {t0_avg:.1f}")
        print(f"   Timepoint 2, channel 0 average: {t1_avg:.1f}")
        
        print("\n" + "="*50)
        print("3. Combining timepoint and channel selection")
        znimg_t1c1 = ZarrNii.from_ome_zarr(dataset_path, timepoints=[1], channels=[1])
        print(f"   Timepoint [1], channel [1] shape: {znimg_t1c1.darr.shape}")
        
        avg_value = znimg_t1c1.darr.mean().compute()
        print(f"   Data average: {avg_value:.1f}")
        
        print("\n" + "="*50)
        print("4. Using select_timepoints() method")
        selected_timepoints = znimg.select_timepoints([0, 2])
        print(f"   Selected timepoints shape: {selected_timepoints.darr.shape}")
        
        print("\n" + "="*50)
        print("5. Spatial operations preserving T,C dimensions")
        
        # Test cropping - should preserve time and channel dimensions
        cropped = znimg.crop([0, 0, 0], [4, 8, 8])  # Crop spatial dimensions only
        print(f"   Original shape: {znimg.darr.shape}")
        print(f"   Cropped shape: {cropped.darr.shape}")
        print(f"   Time/channel dims preserved: {cropped.darr.shape[0] == 3 and cropped.darr.shape[-1] == 2}")
        
        # Test downsampling - should preserve time and channel dimensions
        downsampled = znimg.downsample(factors=2)
        print(f"   Downsampled shape: {downsampled.darr.shape}")
        expected_shape = (3, 4, 8, 8, 2)  # (t, z/2, y/2, x/2, c)
        print(f"   Expected vs actual: {expected_shape} vs {downsampled.darr.shape}")
        
        print("\n" + "="*50)
        print("6. Backward compatibility with 4D data")
        
        # Create 4D data for comparison
        data_4d = da.ones((1, 8, 16, 16), chunks=(1, 4, 8, 8))
        znimg_4d = ZarrNii.from_darr(data_4d)
        print(f"   4D data shape: {znimg_4d.darr.shape}")
        print(f"   4D dimensions: {znimg_4d.ngff_image.dims}")
        
        # Spatial operations should work the same
        cropped_4d = znimg_4d.crop([0, 0, 0], [4, 8, 8])
        print(f"   4D cropped shape: {cropped_4d.darr.shape}")
        
        print("\n" + "="*50)
        print("✅ All 5D functionality demonstrated successfully!")
        
    finally:
        # Clean up
        import shutil
        shutil.rmtree(os.path.dirname(dataset_path))
        print(f"\nCleaned up temporary files at {os.path.dirname(dataset_path)}")


if __name__ == "__main__":
    demonstrate_5d_functionality()