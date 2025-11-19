#!/usr/bin/env python3
"""
MIP (Maximum Intensity Projection) Visualization Demo

This script demonstrates the new MIP visualization functionality added to zarrnii
for creating maximum intensity projections across slabs with channel-specific colors.
"""

import dask.array as da
import ngff_zarr as nz
import numpy as np

from zarrnii import ZarrNii, create_mip_visualization


def create_sample_multichannel_data():
    """Create sample multi-channel 3D data for demonstration."""
    print("🔬 Creating sample multi-channel 3D data...")

    np.random.seed(42)

    # Create a 3-channel dataset with interesting structures
    # Dimensions: channels, z, y, x
    shape = (3, 80, 100, 120)
    data_array = np.zeros(shape, dtype=np.float32)

    # Channel 0 (Red): Create vertical structures
    for i in range(20, 100, 20):
        data_array[0, :, :, i : i + 5] = np.random.uniform(0.6, 1.0, (80, 100, 5))

    # Channel 1 (Green): Create horizontal layers
    for j in range(10, 80, 15):
        data_array[1, j : j + 3, :, :] = np.random.uniform(0.5, 0.9, (3, 100, 120))

    # Channel 2 (Blue): Create spherical structures
    for z, y, x in [(20, 30, 40), (50, 60, 80), (70, 80, 100)]:
        for dz in range(-5, 6):
            for dy in range(-8, 9):
                for dx in range(-8, 9):
                    if dz**2 + dy**2 + dx**2 <= 64:
                        nz_idx, ny_idx, nx_idx = z + dz, y + dy, x + dx
                        if 0 <= nz_idx < 80 and 0 <= ny_idx < 100 and 0 <= nx_idx < 120:
                            data_array[2, nz_idx, ny_idx, nx_idx] = np.random.uniform(
                                0.7, 1.0
                            )

    # Convert to dask array
    darr = da.from_array(data_array, chunks=(1, 40, 50, 60))

    # Create ZarrNii object
    dims = ["c", "z", "y", "x"]
    # Note: In real NGFF/NIfTI files, scale would be in mm (e.g., 0.002 for 2um)
    # For this demo, we use micron values directly and specify scale_units='um'
    scale = {"z": 2.0, "y": 1.0, "x": 1.0}  # 2um z-spacing, 1um x/y
    translation = {"z": 0.0, "y": 0.0, "x": 0.0}

    ngff_image = nz.NgffImage(
        data=darr, dims=dims, scale=scale, translation=translation, name="demo_3channel"
    )

    znimg = ZarrNii(ngff_image=ngff_image, axes_order="ZYX", orientation="RAS")

    print(f"✅ Created 3-channel image with shape: {znimg.shape}")
    print(f"   Spatial extent: {80 * 2.0} x {100} x {120} microns")
    print(f"   Channels: {znimg.shape[0]} (Red, Green, Blue)")
    return znimg


def demonstrate_standalone_function():
    """Demonstrate the standalone create_mip_visualization function."""
    print("\n📊 Testing standalone create_mip_visualization function...")

    # Create simple test data
    data = da.random.random((2, 50, 60, 70), chunks=(1, 25, 30, 35))
    dims = ["c", "z", "y", "x"]
    # Note: In real NGFF/NIfTI files, scale would be in mm (e.g., 0.002 for 2um)
    # For this demo, we use micron values and specify scale_units='um'
    scale = {"z": 2.0, "y": 1.0, "x": 1.0}  # 2um z-spacing

    # Create MIPs with metadata
    mips, slab_info = create_mip_visualization(
        image=data,
        dims=dims,
        scale=scale,
        plane="axial",
        slab_thickness_um=40.0,
        slab_spacing_um=40.0,
        channel_colors=["red", "green"],
        return_slabs=True,
        scale_units="um",  # Specify that our scale is in microns
    )

    print(f"✅ Created {len(mips)} MIP slabs")
    print(f"   Each MIP shape: {mips[0].shape}")
    print(f"   Slab info for first slab:")
    print(f"     - Position: {slab_info[0]['center_um']:.1f} microns")
    print(
        f"     - Thickness: {slab_info[0]['end_um'] - slab_info[0]['start_um']:.1f} microns"
    )

    return mips, slab_info


def demonstrate_zarrnii_method(znimg):
    """Demonstrate the ZarrNii create_mip convenience method."""
    print("\n🔧 Testing ZarrNii create_mip method...")

    # Create axial MIPs
    print("  📸 Creating axial MIPs...")
    axial_mips = znimg.create_mip(
        plane="axial",
        slab_thickness_um=40.0,
        slab_spacing_um=40.0,
        channel_colors=["red", "green", "blue"],
        scale_units="um",  # Demo uses micron-scale values
    )
    print(f"     ✅ Created {len(axial_mips)} axial MIPs, shape: {axial_mips[0].shape}")

    # Create coronal MIPs
    print("  📸 Creating coronal MIPs...")
    coronal_mips = znimg.create_mip(
        plane="coronal",
        slab_thickness_um=30.0,
        slab_spacing_um=30.0,
        channel_colors=["red", "green", "blue"],
        scale_units="um",  # Demo uses micron-scale values
    )
    print(
        f"     ✅ Created {len(coronal_mips)} coronal MIPs, shape: {coronal_mips[0].shape}"
    )

    # Create sagittal MIPs
    print("  📸 Creating sagittal MIPs...")
    sagittal_mips = znimg.create_mip(
        plane="sagittal",
        slab_thickness_um=30.0,
        slab_spacing_um=30.0,
        channel_colors=["red", "green", "blue"],
        scale_units="um",  # Demo uses micron-scale values
    )
    print(
        f"     ✅ Created {len(sagittal_mips)} sagittal MIPs, shape: {sagittal_mips[0].shape}"
    )

    return axial_mips, coronal_mips, sagittal_mips


def demonstrate_custom_colors():
    """Demonstrate custom color specifications."""
    print("\n🎨 Testing custom color specifications...")

    # Create test data
    data = da.random.random((2, 40, 50, 60), chunks=(1, 20, 25, 30))
    dims = ["c", "z", "y", "x"]
    scale = {"z": 1.0, "y": 1.0, "x": 1.0}  # 1um spacing

    # Using RGB tuples
    print("  🎨 Using RGB tuples...")
    mips_rgb = create_mip_visualization(
        data,
        dims,
        scale,
        plane="axial",
        slab_thickness_um=20.0,
        channel_colors=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],  # Pure red, pure green
        scale_units="um",
    )
    print(f"     ✅ Created {len(mips_rgb)} MIPs with RGB tuple colors")

    # Using named colors
    print("  🎨 Using named colors...")
    mips_named = create_mip_visualization(
        data,
        dims,
        scale,
        plane="axial",
        slab_thickness_um=20.0,
        channel_colors=["cyan", "magenta"],
        scale_units="um",
    )
    print(f"     ✅ Created {len(mips_named)} MIPs with named colors")


def demonstrate_saving_mips(mips):
    """Demonstrate saving MIP images."""
    print("\n💾 Demonstrating MIP saving...")

    try:
        import matplotlib.pyplot as plt

        # Save first few MIPs as PNG files
        n_to_save = min(3, len(mips))
        print(f"  📁 Saving first {n_to_save} MIPs as PNG files...")

        import tempfile

        temp_dir = tempfile.gettempdir()

        for i in range(n_to_save):
            filename = f"{temp_dir}/mip_demo_{i:03d}.png"
            plt.imsave(filename, mips[i])
            print(f"     ✅ Saved: {filename}")

        print(f"\n  💡 Tip: MIPs are saved in {temp_dir}/")
    except ImportError:
        print("  ⚠️  matplotlib not available for saving images")


def demonstrate_thick_vs_thin_slabs(znimg):
    """Demonstrate difference between thick and thin slabs."""
    print("\n📏 Comparing thick vs. thin slabs...")

    # Thin slabs (20 microns)
    thin_mips = znimg.create_mip(
        plane="axial",
        slab_thickness_um=20.0,
        slab_spacing_um=40.0,
        channel_colors=["red", "green", "blue"],
        scale_units="um",
    )
    print(f"  🔍 Thin slabs (20 µm): {len(thin_mips)} MIPs")

    # Thick slabs (80 microns)
    thick_mips = znimg.create_mip(
        plane="axial",
        slab_thickness_um=80.0,
        slab_spacing_um=80.0,
        channel_colors=["red", "green", "blue"],
        scale_units="um",
    )
    print(f"  🔬 Thick slabs (80 µm): {len(thick_mips)} MIPs")

    print("\n  💡 Tip: Thicker slabs show more depth information but fewer total slabs")


def demonstrate_slab_metadata():
    """Demonstrate using slab metadata."""
    print("\n📋 Using slab metadata...")

    # Create simple data
    data = da.random.random((1, 100, 60, 80), chunks=(1, 50, 30, 40))
    dims = ["c", "z", "y", "x"]
    scale = {"z": 2.0, "y": 1.0, "x": 1.0}  # 2um z-spacing

    mips, slab_info = create_mip_visualization(
        data,
        dims,
        scale,
        plane="axial",
        slab_thickness_um=40.0,
        slab_spacing_um=40.0,
        return_slabs=True,
        scale_units="um",
    )

    print(f"  📊 Generated {len(slab_info)} slabs:")
    for i, info in enumerate(slab_info[:5]):  # Show first 5
        print(
            f"     Slab {i}: center={info['center_um']:.1f}µm, "
            f"range=[{info['start_um']:.1f}, {info['end_um']:.1f}]µm"
        )
    if len(slab_info) > 5:
        print(f"     ... and {len(slab_info) - 5} more slabs")


def main():
    """Run the complete MIP visualization demonstration."""
    print("🎉 ZarrNii MIP Visualization Demo")
    print("=" * 70)

    # Create sample data
    znimg = create_sample_multichannel_data()

    # Demonstrate standalone function
    mips, slab_info = demonstrate_standalone_function()

    # Demonstrate ZarrNii methods
    axial_mips, coronal_mips, sagittal_mips = demonstrate_zarrnii_method(znimg)

    # Demonstrate custom colors
    demonstrate_custom_colors()

    # Demonstrate saving
    demonstrate_saving_mips(axial_mips)

    # Demonstrate thick vs thin slabs
    demonstrate_thick_vs_thin_slabs(znimg)

    # Demonstrate slab metadata
    demonstrate_slab_metadata()

    print("\n" + "=" * 70)
    print("🎉 Demo completed successfully!")
    print("\n📚 Key features demonstrated:")
    print("  • Maximum intensity projection along axial, coronal, sagittal planes")
    print("  • Configurable slab thickness and spacing (in microns)")
    print("  • Multi-channel support with custom colors per channel")
    print("  • Efficient dask array operations for large datasets")
    print("  • Slab metadata for tracking positions")
    print("  • RGB visualization output ready for saving")
    print("\n💡 Use cases:")
    print("  • Whole brain lightsheet microscopy visualization")
    print("  • Ultra-high field MRI analysis")
    print("  • Multi-channel fluorescence imaging")
    print("  • Quick quality assessment of 3D datasets")


if __name__ == "__main__":
    main()
