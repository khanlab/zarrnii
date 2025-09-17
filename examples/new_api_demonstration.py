#!/usr/bin/env python3
"""
Demonstration of the new NgffImage-based ZarrNii API.

This script showcases the key features of the architectural overhaul,
including the new NgffZarrNii class and function-based API.
"""

import os
import sys
import tempfile
import numpy as np
import dask.array as da
import ngff_zarr as nz

# Add the parent directory to the path so we can import zarrnii
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zarrnii import (
    # Legacy API
    ZarrNii, AffineTransform,
    
    # New NgffImage-based API  
    NgffZarrNii, crop_ngff_image, downsample_ngff_image,
    create_reference_ngff_image, resample_ngff_image,
    compose_transforms
)


def demonstrate_basic_usage():
    """Demonstrate basic usage of the new API."""
    print("=== Basic Usage Demo ===")
    
    # Create test data
    data = da.random.random((1, 64, 128, 128), chunks=(1, 32, 64, 64))
    
    # Create NgffImage directly
    ngff_image = nz.NgffImage(
        data=data,
        dims=["c", "z", "y", "x"],
        scale={"z": 2.0, "y": 1.0, "x": 1.0},
        translation={"z": 0.0, "y": 0.0, "x": 0.0},
        name="demo_image"
    )
    
    # Wrap in NgffZarrNii
    znimg = NgffZarrNii(ngff_image=ngff_image)
    
    print(f"Shape: {znimg.shape}")
    print(f"Dimensions: {znimg.dims}")
    print(f"Scale: {znimg.scale}")
    print(f"Translation: {znimg.translation}")
    
    # Get affine transformation
    affine = znimg.get_affine_transform()
    print(f"Affine matrix:\n{affine.matrix}")
    print()
    
    return znimg


def demonstrate_function_api(znimg):
    """Demonstrate the function-based API."""
    print("=== Function-based API Demo ===")
    
    # Crop the image
    cropped = crop_ngff_image(
        znimg.ngff_image,
        bbox_min=(10, 20, 30),  # Z, Y, X
        bbox_max=(40, 80, 90),
        spatial_dims=["z", "y", "x"]
    )
    print(f"Original shape: {znimg.shape}")
    print(f"Cropped shape: {cropped.data.shape}")
    print(f"Cropped translation: {cropped.translation}")
    
    # Downsample the image
    downsampled = downsample_ngff_image(
        znimg.ngff_image,
        factors=2,  # Isotropic downsampling
        spatial_dims=["z", "y", "x"]
    )
    print(f"Downsampled shape: {downsampled.data.shape}")
    print(f"Downsampled scale: {downsampled.scale}")
    
    # Resample to different resolution
    resampled = resample_ngff_image(
        znimg.ngff_image,
        target_scale={"z": 1.0, "y": 0.5, "x": 0.5},
        spatial_dims=["z", "y", "x"]
    )
    print(f"Resampled shape: {resampled.data.shape}")
    print(f"Resampled scale: {resampled.scale}")
    print()
    
    return cropped, downsampled, resampled


def demonstrate_transformations():
    """Demonstrate transformation composition."""
    print("=== Transformation Demo ===")
    
    # Create multiple transformations
    transform1 = AffineTransform.from_array(np.array([
        [1.0, 0.0, 0.0, 5.0],   # Translation in Z
        [0.0, 1.0, 0.0, 10.0],  # Translation in Y
        [0.0, 0.0, 1.0, 15.0],  # Translation in X
        [0.0, 0.0, 0.0, 1.0]
    ]))
    
    transform2 = AffineTransform.from_array(np.array([
        [2.0, 0.0, 0.0, 0.0],   # Scale in Z
        [0.0, 1.5, 0.0, 0.0],   # Scale in Y
        [0.0, 0.0, 1.2, 0.0],   # Scale in X
        [0.0, 0.0, 0.0, 1.0]
    ]))
    
    # Compose transformations
    composed = compose_transforms(transform1, transform2)
    print(f"Transform 1:\n{transform1.matrix}")
    print(f"Transform 2:\n{transform2.matrix}")
    print(f"Composed transform:\n{composed.matrix}")
    print()


def demonstrate_compatibility():
    """Demonstrate compatibility between old and new APIs."""
    print("=== Compatibility Demo ===")
    
    # Start with legacy ZarrNii
    data = da.ones((1, 32, 64, 64), chunks=(1, 16, 32, 32))
    affine_matrix = np.array([
        [2.0, 0.0, 0.0, 10.0],
        [0.0, 1.0, 0.0, 20.0],
        [0.0, 0.0, 1.0, 30.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    affine = AffineTransform.from_array(affine_matrix)
    
    legacy_znimg = ZarrNii(darr=data, affine=affine, axes_order="ZYX")
    print(f"Legacy ZarrNii shape: {legacy_znimg.darr.shape}")
    print(f"Legacy affine:\n{legacy_znimg.affine.matrix}")
    
    # Convert to new API
    new_znimg = legacy_znimg.to_ngff_zarrnii("converted")
    print(f"New NgffZarrNii shape: {new_znimg.shape}")
    print(f"New scale: {new_znimg.scale}")
    print(f"New translation: {new_znimg.translation}")
    
    # Convert NgffImage back to legacy API
    ngff_image = new_znimg.ngff_image
    back_to_legacy = ZarrNii.from_ngff_image(ngff_image, axes_order="ZYX")
    print(f"Back to legacy shape: {back_to_legacy.darr.shape}")
    print(f"Preserved affine:\n{back_to_legacy.affine.matrix}")
    print()


def demonstrate_zarr_io():
    """Demonstrate I/O with the new API."""
    print("=== Zarr I/O Demo ===") 
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data
        data = da.random.random((2, 32, 64, 64), chunks=(1, 16, 32, 32))
        ngff_image = nz.NgffImage(
            data=data,
            dims=["c", "z", "y", "x"],
            scale={"z": 2.0, "y": 1.0, "x": 1.0},
            translation={"z": 5.0, "y": 10.0, "x": 15.0},
            name="io_test"
        )
        
        znimg = NgffZarrNii(ngff_image=ngff_image)
        
        # Save to OME-Zarr
        zarr_path = os.path.join(tmpdir, "test.zarr")
        znimg.to_ome_zarr(zarr_path, max_layer=3)
        print(f"Saved to: {zarr_path}")
        
        # Load back
        loaded = NgffZarrNii.from_ome_zarr(zarr_path, level=0)
        print(f"Loaded shape: {loaded.shape}")
        print(f"Loaded scale: {loaded.scale}")
        print(f"Loaded translation: {loaded.translation}")
        
        # Load different pyramid level
        level1 = NgffZarrNii.from_ome_zarr(zarr_path, level=1)
        print(f"Level 1 shape: {level1.shape}")
        print(f"Level 1 scale: {level1.scale}")
        
        # Access full multiscale structure
        if loaded.multiscales:
            print(f"Available pyramid levels: {len(loaded.multiscales.images)}")
            for i, img in enumerate(loaded.multiscales.images):
                print(f"  Level {i}: shape={img.data.shape}, scale={img.scale}")
    
    print()


def main():
    """Run all demonstrations."""
    print("ZarrNii NgffImage-based API Demonstration")
    print("=" * 50)
    print()
    
    # Basic usage
    znimg = demonstrate_basic_usage()
    
    # Function-based API
    cropped, downsampled, resampled = demonstrate_function_api(znimg)
    
    # Transformations
    demonstrate_transformations()
    
    # Compatibility
    demonstrate_compatibility()
    
    # I/O operations
    demonstrate_zarr_io()
    
    print("Demo completed successfully!")
    print("\nKey Benefits of New Architecture:")
    print("- Direct NgffImage support with better multiscale handling")
    print("- Function-based API for cleaner, more modular operations")
    print("- Better metadata preservation throughout workflows")
    print("- Seamless compatibility with legacy ZarrNii class")
    print("- More efficient operations without unnecessary conversions")


if __name__ == "__main__":
    main()