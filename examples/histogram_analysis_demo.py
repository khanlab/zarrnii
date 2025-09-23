#!/usr/bin/env python3
"""
Histogram and Otsu Threshold Analysis Demo

This script demonstrates the new histogram and Otsu threshold functionality
added to zarrnii for streamlined image analysis workflows.
"""

import numpy as np
import dask.array as da
import ngff_zarr as nz
from zarrnii import ZarrNii, compute_histogram, compute_otsu_thresholds


def create_sample_data():
    """Create sample multi-modal image data for demonstration."""
    print("🔬 Creating sample multi-modal image data...")
    
    # Create a synthetic image with multiple intensity regions
    np.random.seed(42)
    base_shape = (1, 64, 64, 64)
    
    # Background region (low intensity)
    background = np.random.normal(0.1, 0.05, base_shape)
    
    # Add some bright spots (high intensity) 
    bright_spots = np.zeros(base_shape)
    bright_spots[:, 20:40, 20:40, 20:40] = np.random.normal(0.8, 0.1, (1, 20, 20, 20))
    
    # Add medium intensity region
    medium_region = np.zeros(base_shape)
    medium_region[:, 10:30, 40:60, 10:30] = np.random.normal(0.4, 0.08, (1, 20, 20, 20))
    
    # Combine regions
    image_data = np.clip(background + bright_spots + medium_region, 0, 1)
    
    # Convert to dask array
    darr = da.from_array(image_data, chunks=(1, 32, 32, 32))
    
    # Create ZarrNii object
    ngff_image = nz.NgffImage(
        data=darr,
        dims=['c', 'z', 'y', 'x'],
        scale={'z': 1.0, 'y': 1.0, 'x': 1.0},
        translation={'z': 0.0, 'y': 0.0, 'x': 0.0},
        name='multi_modal_sample'
    )
    
    znimg = ZarrNii(
        ngff_image=ngff_image,
        axes_order='ZYX',
        orientation='RAS'
    )
    
    print(f"✅ Created sample image with shape: {znimg.shape}")
    return znimg


def demonstrate_standalone_functions(znimg):
    """Demonstrate the standalone analysis functions."""
    print("\n📊 Testing standalone analysis functions...")
    
    # Compute histogram using standalone function
    hist, bin_edges = compute_histogram(
        znimg.darr, 
        bins=64, 
        range=(0.0, 1.0)
    )
    
    print(f"✅ Computed histogram: {hist.shape} bins, total pixels: {hist.sum().compute()}")
    
    # Test multi-level Otsu thresholding
    max_k = 4
    thresholds = {}
    
    for k in range(2, max_k + 1):
        thresholds[k] = compute_otsu_thresholds(
            hist, 
            classes=k, 
            bin_edges=bin_edges
        )
        print(f"  📈 {k} classes: thresholds = {[f'{t:.3f}' for t in thresholds[k]]}")
    
    return thresholds


def demonstrate_zarrnii_methods(znimg):
    """Demonstrate the ZarrNii convenience methods."""
    print("\n🔧 Testing ZarrNii convenience methods...")
    
    # Using ZarrNii histogram method
    hist, bin_edges = znimg.compute_histogram(bins=64, range=(0.0, 1.0))
    print(f"✅ ZarrNii histogram method: {hist.shape} bins")
    
    # Using ZarrNii Otsu threshold method
    thresholds_2 = znimg.compute_otsu_thresholds(classes=2, bins=64, range=(0.0, 1.0))
    thresholds_3 = znimg.compute_otsu_thresholds(classes=3, bins=64, range=(0.0, 1.0))
    
    print(f"  📈 Binary thresholds: {[f'{t:.3f}' for t in thresholds_2]}")
    print(f"  📈 3-class thresholds: {[f'{t:.3f}' for t in thresholds_3]}")
    
    return thresholds_2, thresholds_3


def demonstrate_threshold_segmentation(znimg, thresholds_2, thresholds_3):
    """Demonstrate threshold-based segmentation."""
    print("\n🎯 Testing threshold segmentation...")
    
    # Binary threshold segmentation
    binary_seg = znimg.segment_threshold(thresholds_2[1])  # Use middle threshold
    print(f"✅ Binary segmentation: shape={binary_seg.shape}")
    
    unique_binary = np.unique(binary_seg.darr.compute())
    print(f"  🏷️  Binary labels: {unique_binary}")
    
    # Multi-level threshold segmentation (exclude min/max values)
    multi_seg = znimg.segment_threshold(thresholds_3[1:-1])  
    print(f"✅ Multi-level segmentation: shape={multi_seg.shape}")
    
    unique_multi = np.unique(multi_seg.darr.compute())
    print(f"  🏷️  Multi-level labels: {unique_multi}")
    
    # Test with different inclusive settings
    exclusive_seg = znimg.segment_threshold(thresholds_2[1], inclusive=False)
    print(f"✅ Exclusive threshold segmentation: shape={exclusive_seg.shape}")


def demonstrate_masked_analysis(znimg):
    """Demonstrate histogram analysis with masking."""
    print("\n🎭 Testing masked histogram analysis...")
    
    # Create a mask (only analyze bright regions)
    mask_data = znimg.darr > 0.3
    mask_ngff = nz.NgffImage(
        data=mask_data.astype(np.uint8),
        dims=znimg.ngff_image.dims,
        scale=znimg.ngff_image.scale,
        translation=znimg.ngff_image.translation,
        name='mask'
    )
    mask_znimg = ZarrNii(
        ngff_image=mask_ngff,
        axes_order='ZYX',
        orientation='RAS'
    )
    
    # Compute histogram with mask
    masked_hist, masked_bin_edges = znimg.compute_histogram(
        bins=32, 
        range=(0.0, 1.0), 
        mask=mask_znimg
    )
    
    print(f"✅ Masked histogram: {masked_hist.shape} bins, total pixels: {masked_hist.sum()}")
    
    # Compute thresholds from masked data
    masked_thresholds = compute_otsu_thresholds(
        masked_hist, 
        classes=2, 
        bin_edges=masked_bin_edges
    )
    print(f"  📈 Masked Otsu thresholds: {[f'{t:.3f}' for t in masked_thresholds]}")


def demonstrate_backward_compatibility(znimg):
    """Demonstrate that existing functionality still works."""
    print("\n🔄 Testing backward compatibility...")
    
    # Test that old OtsuSegmentation still works
    from zarrnii import OtsuSegmentation, LocalOtsuSegmentation
    
    # Should be the same class
    assert OtsuSegmentation is LocalOtsuSegmentation
    print("✅ OtsuSegmentation alias works correctly")
    
    # Test existing segment_otsu method
    otsu_seg = znimg.segment_otsu(nbins=64)
    print(f"✅ segment_otsu method: shape={otsu_seg.shape}")
    
    unique_otsu = np.unique(otsu_seg.darr.compute())
    print(f"  🏷️  Otsu labels: {unique_otsu}")


def main():
    """Run the complete demonstration."""
    print("🎉 ZarrNii Histogram and Otsu Threshold Analysis Demo")
    print("=" * 60)
    
    # Create sample data
    znimg = create_sample_data()
    
    # Demonstrate standalone functions
    thresholds = demonstrate_standalone_functions(znimg)
    
    # Demonstrate ZarrNii methods
    thresholds_2, thresholds_3 = demonstrate_zarrnii_methods(znimg)
    
    # Demonstrate threshold segmentation
    demonstrate_threshold_segmentation(znimg, thresholds_2, thresholds_3)
    
    # Demonstrate masked analysis
    demonstrate_masked_analysis(znimg)
    
    # Demonstrate backward compatibility
    demonstrate_backward_compatibility(znimg)
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed successfully!")
    print("\n📚 This demo showcased:")
    print("  • Histogram computation with compute_histogram()")
    print("  • Multi-level Otsu thresholding with compute_otsu_thresholds()")
    print("  • ZarrNii convenience methods")
    print("  • Threshold-based segmentation")
    print("  • Masked histogram analysis")
    print("  • Full backward compatibility")
    print("\n💡 These functions enable efficient image analysis workflows")
    print("   as described in the original issue!")


if __name__ == "__main__":
    main()