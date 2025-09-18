#!/usr/bin/env python3
"""
Demonstration of ZipStore support for OME-Zarr files in zarrnii.

This script shows how to:
1. Create synthetic OME-Zarr data
2. Save it to a .ome.zarr.zip file
3. Load it back and verify integrity
4. Access different pyramid levels
"""

import os
import tempfile

import dask.array as da
import numpy as np

import zarrnii


def main():
    """Demonstrate ZipStore functionality."""
    print("ZarrNii ZipStore (.ome.zarr.zip) Support Demo")
    print("=" * 45)

    # Create synthetic 4D data (channel, z, y, x)
    print("\n1. Creating synthetic OME-Zarr data...")
    shape = (2, 32, 64, 64)  # 2 channels, 32 z-slices, 64x64 xy
    chunks = (1, 16, 32, 32)

    # Create synthetic data with gradients and patterns
    data = da.random.random(shape, chunks=chunks).astype(np.float32)
    data = (data * 1000).astype(np.uint16)  # Scale to uint16 range

    print(f"   Data shape: {shape}")
    print(f"   Data type: {data.dtype}")
    print(f"   Chunks: {chunks}")

    # Create ZarrNii object
    znimg = zarrnii.ZarrNii.from_darr(data, axes_order="ZYX", orientation="RAS")
    print(f"   ZarrNii created with {znimg.darr.shape} shape")

    # Use temporary directory for demo
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n2. Working in temporary directory: {tmpdir}")

        # Define file paths
        regular_path = os.path.join(tmpdir, "regular.ome.zarr")
        zip_path = os.path.join(tmpdir, "compressed.ome.zarr.zip")

        # Save to regular directory
        print("\n3. Saving to regular OME-Zarr directory...")
        znimg.to_ome_zarr(regular_path, max_layer=3)
        regular_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, _, filenames in os.walk(regular_path)
            for filename in filenames
        )
        print(f"   Regular directory size: {regular_size:,} bytes")

        # Save to ZIP file
        print("\n4. Saving to .ome.zarr.zip file...")
        znimg.to_ome_zarr(zip_path, max_layer=3)
        zip_size = os.path.getsize(zip_path)
        print(f"   ZIP file size: {zip_size:,} bytes")
        print(f"   Compression ratio: {zip_size/regular_size:.2%}")

        # Load from ZIP file and verify
        print("\n5. Loading from .ome.zarr.zip file...")
        loaded_znimg = zarrnii.ZarrNii.from_ome_zarr(zip_path, level=0)

        print(f"   Loaded shape: {loaded_znimg.darr.shape}")
        print(f"   Loaded axes order: {loaded_znimg.axes_order}")
        print(f"   Loaded orientation: {loaded_znimg.orientation}")

        # Verify data integrity
        print("\n6. Verifying data integrity...")
        original_data = znimg.darr.compute()
        loaded_data = loaded_znimg.darr.compute()

        if np.array_equal(original_data, loaded_data):
            print("   ✓ Data integrity verified - perfect match!")
        else:
            print("   ✗ Data integrity failed")
            return False

        # Test multiscale pyramid access
        print("\n7. Testing multiscale pyramid access...")
        for level in range(3):
            try:
                level_data = zarrnii.ZarrNii.from_ome_zarr(zip_path, level=level)
                print(f"   Level {level}: {level_data.darr.shape}")
            except IndexError:
                print(f"   Level {level}: Not available")
                break

        # Test channel selection (if multichannel)
        if znimg.darr.shape[0] > 1:
            print("\n8. Testing channel selection...")
            channel_0 = zarrnii.ZarrNii.from_ome_zarr(zip_path, level=0, channels=[0])
            print(f"   Channel 0 shape: {channel_0.darr.shape}")

        print("\n" + "=" * 45)
        print("✓ ZipStore demo completed successfully!")
        print("\nKey benefits of .ome.zarr.zip format:")
        print("- Compressed storage (smaller file size)")
        print("- Single file for easy sharing")
        print("- Preserves all OME-Zarr metadata and multiscale structure")
        print("- Compatible with zarr ecosystem")

        return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
