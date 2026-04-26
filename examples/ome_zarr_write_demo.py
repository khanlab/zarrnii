#!/usr/bin/env python3
"""
Demonstration of writing OME-Zarr from a dask array using zarrnii.

This script shows how to:
1. Create a ZarrNii object directly from a dask array (``from_darr``).
2. Attach OMERO channel metadata so that downstream viewers (e.g. napari,
   OMERO.web) can display each channel with the correct colour and contrast.
3. Write the data to an OME-Zarr store using the ``ome-zarr-py`` backend,
   including:
   - per-axis (z + y + x) downsampling for the multiscale pyramid
   - sharding via ``storage_options`` for cloud-optimised storage
4. Reload the file and verify that the data round-trips correctly.
"""

import os
import tempfile

import dask.array as da
import ngff_zarr as nz
import numpy as np

import zarrnii


def build_omero_metadata(n_channels: int) -> nz.Omero:
    """Return simple OMERO metadata for *n_channels* channels.

    Each channel gets:
    - a colour (RRGGBB hex string)
    - a display window computed from the data range
    - a human-readable label
    """
    colours = ["FF0000", "00FF00", "0000FF", "FFFF00"]  # R, G, B, Y
    labels = ["DAPI", "GFP", "mCherry", "BF"]
    channels = []
    for i in range(n_channels):
        window = nz.OmeroWindow(min=0, max=65535, start=100, end=2000)
        channel = nz.OmeroChannel(
            color=colours[i % len(colours)],
            window=window,
            label=labels[i % len(labels)],
        )
        channels.append(channel)
    return nz.Omero(channels=channels)


def main() -> bool:
    """Run the OME-Zarr write demo."""
    print("ZarrNii OME-Zarr write demo")
    print("=" * 40)

    # ------------------------------------------------------------------
    # 1. Create synthetic 4-D data (channel, z, y, x)
    # ------------------------------------------------------------------
    n_channels = 2
    shape = (n_channels, 16, 64, 64)
    chunks = (1, 8, 32, 32)

    print(f"\n1. Creating synthetic dask array  shape={shape}  chunks={chunks}")
    rng = da.from_array(
        np.random.default_rng(42).integers(0, 4096, size=shape, dtype=np.uint16),
        chunks=chunks,
    )

    # ------------------------------------------------------------------
    # 2. Build a ZarrNii from the dask array
    # ------------------------------------------------------------------
    print("\n2. Building ZarrNii via from_darr()")
    omero = build_omero_metadata(n_channels)

    znimg = zarrnii.ZarrNii.from_darr(
        rng,
        axes_order="ZYX",
        orientation="RAS",
        spacing=(5.0, 0.5, 0.5),  # z=5 µm, y=x=0.5 µm
        origin=(0.0, 0.0, 0.0),
        name="demo_image",
        omero=omero,
    )
    print(f"   shape : {znimg.darr.shape}")
    print(f"   axes  : {znimg.axes_order}")

    # ------------------------------------------------------------------
    # 3. Write to OME-Zarr with per-axis z+y+x downsampling and shards
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = os.path.join(tmpdir, "demo.ome.zarr")

        print(f"\n3. Writing to {zarr_path}")
        print("   backend        : ome-zarr-py")
        print("   zarr_format    : 3")
        print("   max_layer      : 3  (level 0 + 2 downsampled levels)")
        print("   scale_factors  : [{'z':2,'y':2,'x':2}, {'z':4,'y':4,'x':4}]")
        print("   storage_options: {'chunks': (1, 8, 32, 32)}")

        # scale_factors as dicts → z is also downsampled (not only xy).
        # storage_options lets you pass any zarr backend option; here we
        # explicitly set the chunk shape.  For sharded arrays (zarr v3) you
        # would add  "shards": (1, 8, 32, 32)  alongside "chunks".
        znimg.to_ome_zarr(
            zarr_path,
            backend="ome-zarr-py",
            zarr_format=3,
            max_layer=3,
            scale_factors=[
                {"z": 2, "y": 2, "x": 2},
                {"z": 4, "y": 4, "x": 4},
            ],
            storage_options={"chunks": (1, 8, 32, 32)},
        )
        print("   → write complete")

        # ------------------------------------------------------------------
        # 4. Reload and verify
        # ------------------------------------------------------------------
        print("\n4. Reloading from OME-Zarr and verifying …")
        loaded = zarrnii.ZarrNii.from_ome_zarr(zarr_path, level=0)
        print(f"   loaded shape : {loaded.darr.shape}")

        original_data = znimg.darr.compute()
        loaded_data = loaded.darr.compute()

        if np.array_equal(original_data, loaded_data):
            print("   ✓ Data integrity verified")
        else:
            print("   ✗ Data mismatch!")
            return False

        # ------------------------------------------------------------------
        # 5. Inspect multiscale levels
        # ------------------------------------------------------------------
        print("\n5. Multiscale pyramid levels:")
        for level in range(3):
            try:
                lvl = zarrnii.ZarrNii.from_ome_zarr(zarr_path, level=level)
                print(f"   level {level}: {lvl.darr.shape}")
            except (IndexError, KeyError):
                print(f"   level {level}: not available")
                break

    print("\n" + "=" * 40)
    print("✓ Demo completed successfully!")
    return True


if __name__ == "__main__":
    success = main()
    raise SystemExit(0 if success else 1)
