"""
Tests for the new NgffImage-based function API.
"""

import os
import tempfile
import pytest
import numpy as np
import dask.array as da
import ngff_zarr as nz
from numpy.testing import assert_array_almost_equal, assert_array_equal

from zarrnii.ngff_core import (
    load_ngff_image, save_ngff_image, get_multiscales,
    crop_ngff_image, downsample_ngff_image, apply_transform_to_ngff_image,
    get_affine_matrix, get_affine_transform
)


class TestNgffImageFunctions:
    """Test the NgffImage-based function API."""
    
    @pytest.fixture
    def simple_ngff_image(self):
        """Create a simple NgffImage for testing."""
        # Create test data
        data = da.random.random((1, 32, 64, 64), chunks=(1, 16, 32, 32))
        
        # Create NgffImage
        ngff_image = nz.NgffImage(
            data=data,
            dims=["c", "z", "y", "x"],
            scale={"z": 2.0, "y": 1.0, "x": 1.0},
            translation={"z": 0.0, "y": 0.0, "x": 0.0},
            name="test_image"
        )
        
        return ngff_image
    
    @pytest.fixture
    def temp_zarr_store(self, simple_ngff_image):
        """Create a temporary OME-Zarr store for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, "test.zarr")
            
            # Create multiscales and save to zarr
            multiscales = nz.to_multiscales(simple_ngff_image, scale_factors=[2, 4])
            nz.to_ngff_zarr(zarr_path, multiscales)
            
            yield zarr_path
    
    def test_load_ngff_image(self, temp_zarr_store):
        """Test loading NgffImage from OME-Zarr store."""
        ngff_image = load_ngff_image(temp_zarr_store, level=0)
        
        assert ngff_image is not None
        assert ngff_image.data.shape == (1, 32, 64, 64)
        assert ngff_image.dims == ["c", "z", "y", "x"]
        assert ngff_image.scale["z"] == 2.0
        assert ngff_image.scale["y"] == 1.0
        assert ngff_image.scale["x"] == 1.0
    
    def test_load_ngff_image_different_level(self, temp_zarr_store):
        """Test loading different pyramid levels."""
        ngff_level0 = load_ngff_image(temp_zarr_store, level=0)
        ngff_level1 = load_ngff_image(temp_zarr_store, level=1)
        
        # Level 1 should be smaller due to downsampling
        assert ngff_level1.data.shape[1] < ngff_level0.data.shape[1]  # Z dimension
        assert ngff_level1.data.shape[2] < ngff_level0.data.shape[2]  # Y dimension
        assert ngff_level1.data.shape[3] < ngff_level0.data.shape[3]  # X dimension
    
    def test_save_ngff_image(self, simple_ngff_image):
        """Test saving NgffImage to OME-Zarr."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.zarr")
            save_ngff_image(simple_ngff_image, output_path, max_layer=3)
            
            # Verify the output exists and can be loaded
            assert os.path.exists(output_path)
            
            # Load back and verify
            reloaded = load_ngff_image(output_path, level=0)
            assert reloaded.data.shape == simple_ngff_image.data.shape
            assert reloaded.dims == simple_ngff_image.dims
    
    def test_get_multiscales(self, temp_zarr_store):
        """Test getting full multiscales object."""
        multiscales = get_multiscales(temp_zarr_store)
        
        assert multiscales is not None
        assert len(multiscales.images) == 3  # Original + 2 downsampled levels
        
        # Check that each level has appropriate shape
        for i, image in enumerate(multiscales.images):
            expected_scale_factor = 2**i
            if i == 0:
                assert image.data.shape == (1, 32, 64, 64)
            else:
                # Each level should be smaller
                assert image.data.shape[1] <= multiscales.images[i-1].data.shape[1]
    
    def test_get_affine_matrix(self, simple_ngff_image):
        """Test affine matrix construction from NgffImage."""
        affine = get_affine_matrix(simple_ngff_image)
        
        expected = np.eye(4)
        expected[0, 0] = 2.0  # Z scale
        expected[1, 1] = 1.0  # Y scale
        expected[2, 2] = 1.0  # X scale
        
        assert_array_equal(affine, expected)
    
    def test_get_affine_transform(self, simple_ngff_image):
        """Test AffineTransform object creation from NgffImage."""
        transform = get_affine_transform(simple_ngff_image)
        
        from zarrnii.transform import AffineTransform
        assert isinstance(transform, AffineTransform)
        
        expected_matrix = np.eye(4)
        expected_matrix[0, 0] = 2.0
        expected_matrix[1, 1] = 1.0
        expected_matrix[2, 2] = 1.0
        
        assert_array_equal(transform.matrix, expected_matrix)
    
    def test_crop_ngff_image(self):
        """Test cropping an NgffImage."""
        # Create test image
        data = da.ones((1, 32, 64, 64), chunks=(1, 16, 32, 32))
        ngff_image = nz.NgffImage(
            data=data,
            dims=["c", "z", "y", "x"],
            scale={"z": 2.0, "y": 1.0, "x": 1.0},
            translation={"z": 10.0, "y": 20.0, "x": 30.0},
            name="test_image"
        )
        
        # Crop to a smaller region
        bbox_min = (5, 10, 15)  # Z, Y, X
        bbox_max = (15, 30, 35)  # Z, Y, X
        
        cropped = crop_ngff_image(ngff_image, bbox_min, bbox_max)
        
        # Check new shape
        expected_shape = (1, 10, 20, 20)  # C unchanged, Z, Y, X cropped
        assert cropped.data.shape == expected_shape
        
        # Check translation is updated
        assert cropped.translation["z"] == 10.0 + (5 * 2.0)  # original + offset * scale
        assert cropped.translation["y"] == 20.0 + (10 * 1.0)
        assert cropped.translation["x"] == 30.0 + (15 * 1.0)
        
        # Scale should be unchanged
        assert cropped.scale == ngff_image.scale
    
    def test_downsample_ngff_image_isotropic(self):
        """Test isotropic downsampling of an NgffImage."""
        # Create test image
        data = da.ones((1, 32, 64, 64), chunks=(1, 16, 32, 32))
        ngff_image = nz.NgffImage(
            data=data,
            dims=["c", "z", "y", "x"],
            scale={"z": 2.0, "y": 1.0, "x": 1.0},
            translation={"z": 10.0, "y": 20.0, "x": 30.0},
            name="test_image"
        )
        
        downsampled = downsample_ngff_image(ngff_image, factors=2)
        
        # Check new shape (every spatial dimension should be halved)
        expected_shape = (1, 16, 32, 32)  # C unchanged, Z, Y, X downsampled by 2
        assert downsampled.data.shape == expected_shape
        
        # Check scale is updated
        assert downsampled.scale["z"] == 4.0  # 2.0 * 2
        assert downsampled.scale["y"] == 2.0  # 1.0 * 2
        assert downsampled.scale["x"] == 2.0  # 1.0 * 2
        
        # Translation should be unchanged
        assert downsampled.translation == ngff_image.translation
    
    def test_downsample_ngff_image_anisotropic(self):
        """Test anisotropic downsampling of an NgffImage."""
        # Create test image
        data = da.ones((1, 32, 64, 64), chunks=(1, 16, 32, 32))
        ngff_image = nz.NgffImage(
            data=data,
            dims=["c", "z", "y", "x"],
            scale={"z": 2.0, "y": 1.0, "x": 1.0},
            translation={"z": 10.0, "y": 20.0, "x": 30.0},
            name="test_image"
        )
        
        factors = [1, 2, 4]  # Different factors for Z, Y, X
        downsampled = downsample_ngff_image(ngff_image, factors=factors)
        
        # Check new shape
        expected_shape = (1, 32, 32, 16)  # C unchanged, Z same, Y/2, X/4
        assert downsampled.data.shape == expected_shape
        
        # Check scale is updated appropriately
        assert downsampled.scale["z"] == 2.0  # 2.0 * 1
        assert downsampled.scale["y"] == 2.0  # 1.0 * 2
        assert downsampled.scale["x"] == 4.0  # 1.0 * 4
    
    def test_apply_transform_to_ngff_image_placeholder(self):
        """Test the transform function placeholder."""
        from zarrnii.transform import AffineTransform
        
        # Create test images
        data = da.ones((1, 32, 64, 64), chunks=(1, 16, 32, 32))
        test_ngff_image = nz.NgffImage(
            data=data,
            dims=["c", "z", "y", "x"],
            scale={"z": 2.0, "y": 1.0, "x": 1.0},
            translation={"z": 10.0, "y": 20.0, "x": 30.0},
            name="test_image"
        )
        
        # Create a simple identity transform
        identity_transform = AffineTransform.identity()
        
        # Create a reference image (same as test image for now)
        reference = test_ngff_image
        
        # Apply transform (currently a placeholder)
        transformed = apply_transform_to_ngff_image(
            test_ngff_image, identity_transform, reference
        )
        
        # For now, this is just a placeholder that returns reference data
        assert transformed.data.shape == reference.data.shape
        assert transformed.dims == reference.dims
        assert transformed.scale == reference.scale