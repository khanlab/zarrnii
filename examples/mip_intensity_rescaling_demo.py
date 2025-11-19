#!/usr/bin/env python
"""
Example demonstrating the new intensity rescaling and OMERO features for MIP visualization.

This script shows how to:
1. Create MIPs with custom intensity ranges
2. Use OMERO metadata for automatic color and range extraction
3. Apply alpha transparency to channel overlays
"""

import dask.array as da

from zarrnii import ZarrNii
from zarrnii.analysis import create_mip_visualization

# Example 1: Custom intensity ranges
# ==================================
print("Example 1: Custom Intensity Ranges")
print("-" * 40)

# Create sample 3-channel data
data = da.random.random((3, 50, 100, 100), chunks=(1, 25, 50, 50)) * 10000
dims = ["c", "z", "y", "x"]
scale = {"z": 2.0, "y": 1.0, "x": 1.0}

# Create MIP with custom intensity ranges for each channel
# This gives you precise control over contrast and brightness
mips = create_mip_visualization(
    data,
    dims,
    scale,
    plane="axial",
    slab_thickness_um=50.0,
    channel_colors=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
    channel_ranges=[
        (0, 5000),  # Red channel: map 0-5000 to display range
        (2000, 8000),  # Green channel: map 2000-8000 to display range
        (1000, 6000),  # Blue channel: map 1000-6000 to display range
    ],
)

print(f"Created {len(mips)} MIP slabs")
print(f"Each MIP shape: {mips[0].shape}")
print()

# Example 2: Alpha transparency
# =============================
print("Example 2: Alpha Transparency")
print("-" * 40)

# Create MIP with semi-transparent channels for better overlay visualization
mips_alpha = create_mip_visualization(
    data,
    dims,
    scale,
    plane="axial",
    slab_thickness_um=50.0,
    channel_colors=[
        (1.0, 0.0, 0.0, 0.8),  # 80% opaque red
        (0.0, 1.0, 0.0, 0.6),  # 60% opaque green
        (0.0, 0.0, 1.0, 0.4),  # 40% opaque blue
    ],
)

print(f"Created {len(mips_alpha)} MIP slabs with alpha blending")
print()

# Example 3: Using OMERO metadata (conceptual example)
# ===================================================
print("Example 3: OMERO Metadata Integration")
print("-" * 40)
print("When loading data with OMERO metadata:")
print()
print("# Load OME-Zarr with OMERO metadata")
print("znimg = ZarrNii.from_ome_zarr('path/to/data.ome.zarr')")
print()
print("# Option A: Use OMERO colors and ranges automatically")
print("mips = znimg.create_mip(plane='axial')")
print()
print("# Option B: Select specific channels by OMERO label")
print("mips = znimg.create_mip(")
print("    plane='axial',")
print("    channel_labels=['DAPI', 'GFP', 'RFP']")
print(")")
print()
print("# Option C: Mix OMERO metadata with custom settings")
print("mips = znimg.create_mip(")
print("    plane='axial',")
print("    channel_labels=['DAPI', 'GFP'],  # Select by label")
print("    channel_ranges=[(0, 2000), None]  # Custom range for DAPI, auto for GFP")
print(")")
print()

# Example 4: Backward compatibility
# =================================
print("Example 4: Backward Compatibility")
print("-" * 40)

# Old API still works exactly as before
mips_legacy = create_mip_visualization(
    data,
    dims,
    scale,
    plane="axial",
    slab_thickness_um=50.0,
    channel_colors=["red", "green", "blue"],
)

print(f"Legacy API still works: {len(mips_legacy)} MIP slabs created")
print()

print("All examples completed successfully!")
