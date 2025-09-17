"""
Tests for the new NgffZarrNii class and function-based API.
"""

import os
import tempfile
import pytest
import numpy as np
import dask.array as da
import ngff_zarr as nz
from numpy.testing import assert_array_almost_equal, assert_array_equal

from zarrnii.ngff_core import NgffZarrNii, crop_ngff_image, downsample_ngff_image, apply_transform_to_ngff_image


class TestNgffZarrNii:
    """Test the new NgffZarrNii class."""
    
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
    
    def test_create_from_ngff_image(self, simple_ngff_image):
        """Test creating NgffZarrNii from an NgffImage."""
        znimg = NgffZarrNii(ngff_image=simple_ngff_image)
        
        assert znimg.ngff_image is simple_ngff_image
        assert znimg.multiscales is None
        assert znimg.shape == (1, 32, 64, 64)
        assert znimg.dims == ["c", "z", "y", "x"]
        assert znimg.scale == {"z": 2.0, "y": 1.0, "x": 1.0}
        assert znimg.translation == {"z": 0.0, "y": 0.0, "x": 0.0}
    
    def test_from_ome_zarr(self, temp_zarr_store):
        """Test loading NgffZarrNii from OME-Zarr store."""
        znimg = NgffZarrNii.from_ome_zarr(temp_zarr_store, level=0)
        
        assert znimg.ngff_image is not None
        assert znimg.multiscales is not None
        assert znimg.shape == (1, 32, 64, 64)
        assert znimg.dims == ["c", "z", "y", "x"]
        # The scale includes all dimensions from the NgffImage
        expected_scale = {"z": 2.0, "y": 1.0, "x": 1.0}
        for key, value in expected_scale.items():
            assert znimg.scale[key] == value
    
    def test_from_ome_zarr_different_level(self, temp_zarr_store):
        """Test loading different pyramid levels."""
        znimg_level0 = NgffZarrNii.from_ome_zarr(temp_zarr_store, level=0)
        znimg_level1 = NgffZarrNii.from_ome_zarr(temp_zarr_store, level=1)
        
        # Level 1 should be smaller due to downsampling
        assert znimg_level1.shape[1] < znimg_level0.shape[1]  # Z dimension
        assert znimg_level1.shape[2] < znimg_level0.shape[2]  # Y dimension
        assert znimg_level1.shape[3] < znimg_level0.shape[3]  # X dimension
    
    def test_get_affine_matrix(self, simple_ngff_image):
        """Test affine matrix construction."""
        znimg = NgffZarrNii(ngff_image=simple_ngff_image)
        affine = znimg.get_affine_matrix()
        
        expected = np.eye(4)
        expected[0, 0] = 2.0  # Z scale
        expected[1, 1] = 1.0  # Y scale
        expected[2, 2] = 1.0  # X scale
        
        assert_array_equal(affine, expected)
    
    def test_get_affine_transform(self, simple_ngff_image):
        """Test AffineTransform object creation."""
        znimg = NgffZarrNii(ngff_image=simple_ngff_image)
        transform = znimg.get_affine_transform()
        
        from zarrnii.transform import AffineTransform
        assert isinstance(transform, AffineTransform)
        
        expected_matrix = np.eye(4)
        expected_matrix[0, 0] = 2.0
        expected_matrix[1, 1] = 1.0
        expected_matrix[2, 2] = 1.0
        
        assert_array_equal(transform.matrix, expected_matrix)
    
    def test_to_ome_zarr(self, simple_ngff_image):
        """Test writing NgffZarrNii to OME-Zarr."""
        znimg = NgffZarrNii(ngff_image=simple_ngff_image)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.zarr")
            znimg.to_ome_zarr(output_path, max_layer=3)
            
            # Verify the output exists and can be loaded
            assert os.path.exists(output_path)
            
            # Load back and verify
            reloaded = NgffZarrNii.from_ome_zarr(output_path, level=0)
            assert reloaded.shape == znimg.shape
            assert reloaded.dims == znimg.dims


class TestNgffImageFunctions:
    """Test the function-based API for NgffImage operations."""
    
    @pytest.fixture
    def test_ngff_image(self):
        """Create a test NgffImage."""
        data = da.ones((1, 32, 64, 64), chunks=(1, 16, 32, 32))
        
        return nz.NgffImage(
            data=data,
            dims=["c", "z", "y", "x"],
            scale={"z": 2.0, "y": 1.0, "x": 1.0},
            translation={"z": 10.0, "y": 20.0, "x": 30.0},
            name="test_image"
        )
    
    def test_crop_ngff_image(self, test_ngff_image):
        """Test cropping an NgffImage."""
        # Crop to a smaller region
        bbox_min = (5, 10, 15)  # Z, Y, X
        bbox_max = (15, 30, 35)  # Z, Y, X
        
        cropped = crop_ngff_image(test_ngff_image, bbox_min, bbox_max)
        
        # Check new shape
        expected_shape = (1, 10, 20, 20)  # C unchanged, Z, Y, X cropped
        assert cropped.data.shape == expected_shape
        
        # Check translation is updated
        assert cropped.translation["z"] == 10.0 + (5 * 2.0)  # original + offset * scale
        assert cropped.translation["y"] == 20.0 + (10 * 1.0)
        assert cropped.translation["x"] == 30.0 + (15 * 1.0)
        
        # Scale should be unchanged
        assert cropped.scale == test_ngff_image.scale
    
    def test_downsample_ngff_image_isotropic(self, test_ngff_image):
        """Test isotropic downsampling of an NgffImage."""
        downsampled = downsample_ngff_image(test_ngff_image, factors=2)
        
        # Check new shape (every spatial dimension should be halved)
        expected_shape = (1, 16, 32, 32)  # C unchanged, Z, Y, X downsampled by 2
        assert downsampled.data.shape == expected_shape
        
        # Check scale is updated
        assert downsampled.scale["z"] == 4.0  # 2.0 * 2
        assert downsampled.scale["y"] == 2.0  # 1.0 * 2
        assert downsampled.scale["x"] == 2.0  # 1.0 * 2
        
        # Translation should be unchanged
        assert downsampled.translation == test_ngff_image.translation
    
    def test_downsample_ngff_image_anisotropic(self, test_ngff_image):
        """Test anisotropic downsampling of an NgffImage."""
        factors = [1, 2, 4]  # Different factors for Z, Y, X
        downsampled = downsample_ngff_image(test_ngff_image, factors=factors)
        
        # Check new shape
        expected_shape = (1, 32, 32, 16)  # C unchanged, Z same, Y/2, X/4
        assert downsampled.data.shape == expected_shape
        
        # Check scale is updated appropriately
        assert downsampled.scale["z"] == 2.0  # 2.0 * 1
        assert downsampled.scale["y"] == 2.0  # 1.0 * 2
        assert downsampled.scale["x"] == 4.0  # 1.0 * 4
    
    def test_apply_transform_to_ngff_image_placeholder(self, test_ngff_image):
        """Test the transform function placeholder."""
        from zarrnii.transform import AffineTransform
        
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