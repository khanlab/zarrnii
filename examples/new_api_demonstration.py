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
    
    # New NgffImage-based function API  
    crop_ngff_image, downsample_ngff_image,
    create_reference_ngff_image, resample_ngff_image,
    compose_transforms, load_ngff_image, save_ngff_image
)
from zarrnii.ngff_core import get_affine_matrix, get_affine_transform, get_multiscales


def demonstrate_basic_usage():
    """Demonstrate basic usage of the new API."""
    print("=== Basic Usage Demo ===")
    
    # Create test data
    data = da.random.random((1, 64, 128, 128), chunks=(1, 32, 64, 64))
    
    # Create NgffImage directly - no wrapper needed!
    ngff_image = nz.NgffImage(
        data=data,
        dims=["c", "z", "y", "x"],
        scale={"z": 2.0, "y": 1.0, "x": 1.0},
        translation={"z": 0.0, "y": 0.0, "x": 0.0},
        name="demo_image"
    )
    
    print(f"Shape: {ngff_image.data.shape}")
    print(f"Dimensions: {ngff_image.dims}")
    print(f"Scale: {ngff_image.scale}")
    print(f"Translation: {ngff_image.translation}")
    
    # Get affine transformation using helper functions
    affine_matrix = get_affine_matrix(ngff_image)
    affine_transform = get_affine_transform(ngff_image)
    print(f"Affine matrix:\n{affine_matrix}")
    print()
    
    return ngff_image


def demonstrate_function_api(ngff_image):
    """Demonstrate the function-based API."""
    print("=== Function-based API Demo ===")
    
    # Crop the image
    cropped = crop_ngff_image(
        ngff_image,
        bbox_min=(10, 20, 30),  # Z, Y, X
        bbox_max=(40, 80, 90),
        spatial_dims=["z", "y", "x"]
    )
    print(f"Original shape: {ngff_image.data.shape}")
    print(f"Cropped shape: {cropped.data.shape}")
    print(f"Cropped translation: {cropped.translation}")
    
    # Downsample the image
    downsampled = downsample_ngff_image(
        ngff_image,
        factors=2,  # Isotropic downsampling
        spatial_dims=["z", "y", "x"]
    )
    print(f"Downsampled shape: {downsampled.data.shape}")
    print(f"Downsampled scale: {downsampled.scale}")
    
    # Resample to different resolution
    resampled = resample_ngff_image(
        ngff_image,
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
    
    # Convert to new API - now just NgffImage directly
    ngff_image = legacy_znimg.to_ngff_image("converted")
    print(f"New NgffImage shape: {ngff_image.data.shape}")
    print(f"New scale: {ngff_image.scale}")
    print(f"New translation: {ngff_image.translation}")
    
    # Convert NgffImage back to legacy API
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
        
        # Save to OME-Zarr using function API
        zarr_path = os.path.join(tmpdir, "test.zarr")
        save_ngff_image(ngff_image, zarr_path, max_layer=3)
        print(f"Saved to: {zarr_path}")
        
        # Load back using function API
        loaded = load_ngff_image(zarr_path, level=0)
        print(f"Loaded shape: {loaded.data.shape}")
        print(f"Loaded scale: {loaded.scale}")
        print(f"Loaded translation: {loaded.translation}")
        
        # Load different pyramid level
        level1 = load_ngff_image(zarr_path, level=1)
        print(f"Level 1 shape: {level1.data.shape}")
        print(f"Level 1 scale: {level1.scale}")
        
        # Access full multiscale structure
        multiscales = get_multiscales(zarr_path)
        print(f"Available pyramid levels: {len(multiscales.images)}")
        for i, img in enumerate(multiscales.images):
            print(f"  Level {i}: shape={img.data.shape}, scale={img.scale}")
    
    print()


def main():
    """Run all demonstrations."""
    print("ZarrNii NgffImage-based API Demonstration")
    print("=" * 50)
    print()
    
    # Basic usage
    ngff_image = demonstrate_basic_usage()
    
    # Function-based API
    cropped, downsampled, resampled = demonstrate_function_api(ngff_image)
    
    # Transformations
    demonstrate_transformations()
    
    # Compatibility
    demonstrate_compatibility()
    
    # I/O operations
    demonstrate_zarr_io()
    
    print("Demo completed successfully!")
    print("\nKey Benefits of New Function-based Architecture:")
    print("- Direct NgffImage support without unnecessary wrapper classes")
    print("- Function-based API for cleaner, more modular operations")
    print("- Better metadata preservation throughout workflows")
    print("- Seamless compatibility with legacy ZarrNii class")
    print("- More efficient operations without unnecessary conversions")
    print("- Simpler and more aligned with ngff_zarr ecosystem")


if __name__ == "__main__":
    main()