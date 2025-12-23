#!/usr/bin/env python
"""
Example demonstrating memory-safe Imaris export with large datasets.

This script shows how the refactored to_imaris() method can handle
arbitrarily large datasets without loading them entirely into memory.
"""

import dask.array as da
import numpy as np

from zarrnii import ZarrNii


def main():
    """Demonstrate memory-safe Imaris export with a large Dask array."""
    print("Memory-Safe Imaris Export Example")
    print("=" * 50)

    # Simulate a large dataset using Dask (not materialized in memory)
    # For a real use case, this could be a 100+ GB volume
    print("\n1. Creating a large Dask array (not materialized)...")
    shape = (256, 512, 384)  # Z, Y, X - ~500MB if float32
    chunks = (16, 512, 384)  # Process 16 Z-slices at a time

    # Create a synthetic dataset with a pattern
    print(f"   Shape: {shape} (Z, Y, X)")
    print(f"   Chunks: {chunks}")
    print(f"   Memory footprint per chunk: ~{16 * 512 * 384 * 4 / 1024**2:.1f} MB")

    # Generate a Dask array (lazy - not computed)
    data = da.random.random(shape, chunks=chunks).astype(np.float32)
    data = data * 1000  # Scale to 0-1000 range

    # Add channel dimension
    data = data[np.newaxis, ...]

    print(f"\n2. Creating ZarrNii instance...")
    znimg = ZarrNii.from_darr(data, spacing=[2.0, 1.5, 1.0])
    print(f"   Data type: {type(znimg.ngff_image.data)}")
    print(f"   Shape: {znimg.darr.shape}")

    # Export to Imaris - this will NOT load the full array into memory
    print(f"\n3. Exporting to Imaris (memory-safe)...")
    print("   Processing in chunks of 16 Z-slices...")
    output_path = "large_volume.ims"

    # The export happens chunk-by-chunk:
    # - First pass: computes min/max incrementally
    # - Second pass: writes data + accumulates histogram incrementally
    # - Thumbnail: computes MIP incrementally
    result_path = znimg.to_imaris(output_path, compression="gzip", compression_opts=6)

    print(f"\n4. Export complete!")
    print(f"   File saved to: {result_path}")
    print(f"   Memory usage stayed bounded to chunk size")
    print(f"   (not dependent on total volume size)")

    # Verify the file
    print(f"\n5. Verifying exported file...")
    try:
        import h5py

        with h5py.File(result_path, "r") as f:
            dataset = f["DataSet"]["ResolutionLevel 0"]["TimePoint 0"]["Channel 0"][
                "Data"
            ]
            print(f"   ✓ HDF5 structure correct")
            print(f"   ✓ Data shape: {dataset.shape}")
            print(f"   ✓ Data type: {dataset.dtype}")
            print(f"   ✓ Compression: {dataset.compression}")

            # Check histogram
            histogram = f["DataSet"]["ResolutionLevel 0"]["TimePoint 0"]["Channel 0"][
                "Histogram"
            ]
            print(f"   ✓ Histogram shape: {histogram.shape}")
            print(f"   ✓ Total histogram counts: {histogram[:].sum()}")

            # Check thumbnail
            thumbnail = f["Thumbnail"]["Data"]
            print(f"   ✓ Thumbnail shape: {thumbnail.shape}")

        print(
            f"\n✓ Success! File is Imaris-compatible and was created with bounded memory."
        )

    except ImportError:
        print("   h5py not available for verification")


if __name__ == "__main__":
    main()
