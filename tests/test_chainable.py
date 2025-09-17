"""
Tests for the ChainableNgffImage API.
"""

import os
import tempfile
import pytest
import numpy as np
import dask.array as da
import ngff_zarr as nz
from numpy.testing import assert_array_almost_equal, assert_array_equal

from zarrnii import ChainableNgffImage, chainable_from_zarrnii, ZarrNii, AffineTransform


class TestChainableNgffImage:
    """Test the chainable NgffImage wrapper."""
    
    @pytest.fixture
    def simple_ngff_image(self):
        """Create a simple NgffImage for testing."""
        data = da.random.random((1, 32, 64, 64), chunks=(1, 16, 32, 32))
        
        return nz.NgffImage(
            data=data,
            dims=["c", "z", "y", "x"],
            scale={"z": 2.0, "y": 1.0, "x": 1.0},
            translation={"z": 0.0, "y": 0.0, "x": 0.0},
            name="test_image"
        )
    
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
        """Test creating ChainableNgffImage from NgffImage."""
        chainable = ChainableNgffImage.from_ngff_image(simple_ngff_image)
        
        assert chainable.ngff_image is simple_ngff_image
        assert chainable.shape == (1, 32, 64, 64)
        assert chainable.dims == ["c", "z", "y", "x"]
        assert chainable.scale == {"z": 2.0, "y": 1.0, "x": 1.0}
        assert chainable.translation == {"z": 0.0, "y": 0.0, "x": 0.0}
        assert chainable.name == "test_image"
    
    def test_from_ome_zarr(self, temp_zarr_store):
        """Test loading from OME-Zarr store."""
        chainable = ChainableNgffImage.from_ome_zarr(temp_zarr_store, level=0)
        
        assert chainable.shape == (1, 32, 64, 64)
        assert chainable.dims == ["c", "z", "y", "x"]
        assert chainable.scale["z"] == 2.0
        assert chainable.scale["y"] == 1.0
        assert chainable.scale["x"] == 1.0
    
    def test_chainable_crop(self, simple_ngff_image):
        """Test chainable crop operation."""
        chainable = ChainableNgffImage.from_ngff_image(simple_ngff_image)
        
        cropped = chainable.crop(bbox_min=(5, 10, 15), bbox_max=(15, 30, 35))
        
        # Should return a new ChainableNgffImage
        assert isinstance(cropped, ChainableNgffImage)
        assert cropped is not chainable  # Should be a new instance
        
        # Check properties
        expected_shape = (1, 10, 20, 20)  # C unchanged, Z, Y, X cropped
        assert cropped.shape == expected_shape
        
        # Check translation is updated
        assert cropped.translation["z"] == 0.0 + (5 * 2.0)  # original + offset * scale
        assert cropped.translation["y"] == 0.0 + (10 * 1.0)
        assert cropped.translation["x"] == 0.0 + (15 * 1.0)
    
    def test_chainable_downsample(self, simple_ngff_image):
        """Test chainable downsample operation."""
        chainable = ChainableNgffImage.from_ngff_image(simple_ngff_image)
        
        downsampled = chainable.downsample(factors=2)
        
        # Should return a new ChainableNgffImage
        assert isinstance(downsampled, ChainableNgffImage)
        assert downsampled is not chainable
        
        # Check properties
        expected_shape = (1, 16, 32, 32)  # Each spatial dim halved
        assert downsampled.shape == expected_shape
        
        # Check scale is updated
        assert downsampled.scale["z"] == 4.0  # 2.0 * 2
        assert downsampled.scale["y"] == 2.0  # 1.0 * 2
        assert downsampled.scale["x"] == 2.0  # 1.0 * 2
    
    def test_method_chaining(self, simple_ngff_image):
        """Test chaining multiple operations."""
        chainable = ChainableNgffImage.from_ngff_image(simple_ngff_image)
        
        # Chain multiple operations
        result = (chainable
                  .downsample(factors=2)
                  .crop(bbox_min=(2, 4, 6), bbox_max=(10, 20, 26)))
        
        # Should be a ChainableNgffImage
        assert isinstance(result, ChainableNgffImage)
        
        # Check final shape after chaining
        # After downsample: (1, 16, 32, 32)
        # After crop: (1, 8, 16, 20)
        expected_shape = (1, 8, 16, 20)
        assert result.shape == expected_shape
        
        # Check that intermediate steps didn't modify original
        assert chainable.shape == (1, 32, 64, 64)
    
    def test_get_affine_operations(self, simple_ngff_image):
        """Test affine matrix and transform methods."""
        chainable = ChainableNgffImage.from_ngff_image(simple_ngff_image)
        
        affine_matrix = chainable.get_affine_matrix()
        affine_transform = chainable.get_affine_transform()
        
        expected_matrix = np.eye(4)
        expected_matrix[0, 0] = 2.0  # Z scale
        expected_matrix[1, 1] = 1.0  # Y scale
        expected_matrix[2, 2] = 1.0  # X scale
        
        assert_array_equal(affine_matrix, expected_matrix)
        assert isinstance(affine_transform, AffineTransform)
        assert_array_equal(affine_transform.matrix, expected_matrix)
    
    def test_save_operation(self, simple_ngff_image):
        """Test chainable save operation."""
        chainable = ChainableNgffImage.from_ngff_image(simple_ngff_image)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.zarr")
            
            # Save should return self for continued chaining
            result = chainable.save(output_path, max_layer=3)
            assert result is chainable
            
            # Verify file was created
            assert os.path.exists(output_path)
    
    def test_copy_operation(self, simple_ngff_image):
        """Test copy operation."""
        chainable = ChainableNgffImage.from_ngff_image(simple_ngff_image)
        
        copied = chainable.copy()
        
        # Should be a new instance
        assert isinstance(copied, ChainableNgffImage)
        assert copied is not chainable
        assert copied.ngff_image is not chainable.ngff_image
        
        # But should have same properties
        assert copied.shape == chainable.shape
        assert copied.dims == chainable.dims
        assert copied.scale == chainable.scale
        assert copied.translation == chainable.translation
    
    def test_compute_operation(self, simple_ngff_image):
        """Test compute operation."""
        chainable = ChainableNgffImage.from_ngff_image(simple_ngff_image)
        
        computed = chainable.compute()
        
        # Should return NgffImage (not chainable)
        assert isinstance(computed, nz.NgffImage)
        assert not isinstance(computed, ChainableNgffImage)
        
        # Data should be computed (numpy array instead of dask)
        assert isinstance(computed.data, np.ndarray)
        assert computed.data.shape == simple_ngff_image.data.shape
    
    def test_to_zarrnii_conversion(self, simple_ngff_image):
        """Test conversion to legacy ZarrNii."""
        chainable = ChainableNgffImage.from_ngff_image(simple_ngff_image)
        
        zarrnii = chainable.to_zarrnii()
        
        assert isinstance(zarrnii, ZarrNii)
        assert zarrnii.darr.shape == chainable.shape
        assert_array_equal(zarrnii.darr.compute(), chainable.data.compute())
    
    def test_repr(self, simple_ngff_image):
        """Test string representation."""
        chainable = ChainableNgffImage.from_ngff_image(simple_ngff_image)
        
        repr_str = repr(chainable)
        
        assert "ChainableNgffImage" in repr_str
        assert "test_image" in repr_str
        assert "(1, 32, 64, 64)" in repr_str
        assert "['c', 'z', 'y', 'x']" in repr_str


class TestChainableConversion:
    """Test conversion functions for chainable API."""
    
    def test_chainable_from_zarrnii(self):
        """Test creating chainable from legacy ZarrNii."""
        # Create legacy ZarrNii
        data = da.ones((1, 16, 32, 32), chunks=(1, 8, 16, 16))
        affine_matrix = np.array([
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        affine = AffineTransform.from_array(affine_matrix)
        zarrnii = ZarrNii(darr=data, affine=affine, axes_order="ZYX")
        
        # Convert to chainable
        chainable = chainable_from_zarrnii(zarrnii)
        
        assert isinstance(chainable, ChainableNgffImage)
        assert chainable.shape == zarrnii.darr.shape
        assert chainable.scale["z"] == 2.0
        assert chainable.scale["y"] == 1.0
        assert chainable.scale["x"] == 1.0
        assert chainable.translation["z"] == 10.0
        assert chainable.translation["y"] == 20.0
        assert chainable.translation["x"] == 30.0
    
    def test_round_trip_conversion(self):
        """Test round-trip conversion between ZarrNii and ChainableNgffImage."""
        # Start with ZarrNii
        data = da.random.random((1, 16, 32, 32), chunks=(1, 8, 16, 16))
        affine = AffineTransform.identity()
        original_zarrnii = ZarrNii(darr=data, affine=affine, axes_order="ZYX")
        
        # Convert to chainable and back
        chainable = chainable_from_zarrnii(original_zarrnii)
        back_to_zarrnii = chainable.to_zarrnii()
        
        # Should preserve data
        assert_array_equal(
            original_zarrnii.darr.compute(),
            back_to_zarrnii.darr.compute()
        )
        assert_array_almost_equal(
            original_zarrnii.affine.matrix,
            back_to_zarrnii.affine.matrix,
            decimal=10
        )


class TestChainableAdvancedOperations:
    """Test advanced chainable operations."""
    
    def test_complex_chaining_workflow(self):
        """Test a complex chaining workflow."""
        # Create test data
        data = da.random.random((2, 64, 128, 128), chunks=(1, 32, 64, 64))
        ngff_image = nz.NgffImage(
            data=data,
            dims=["c", "z", "y", "x"],
            scale={"z": 2.0, "y": 1.0, "x": 1.0},
            translation={"z": 0.0, "y": 0.0, "x": 0.0},
            name="complex_test"
        )
        
        # Complex workflow with chaining
        result = (ChainableNgffImage.from_ngff_image(ngff_image)
                  .downsample(factors=[1, 2, 2])  # Anisotropic downsampling
                  .crop(bbox_min=(10, 15, 20), bbox_max=(40, 45, 50))
                  .copy())  # Make a copy of the result
        
        # Verify final result
        assert isinstance(result, ChainableNgffImage)
        
        # After downsample: (2, 64, 64, 64) - only Y,X downsampled by 2
        # After crop: (2, 30, 30, 30)
        expected_shape = (2, 30, 30, 30)
        assert result.shape == expected_shape
        
        # Check scale was updated correctly
        assert result.scale["z"] == 2.0  # Unchanged (factor 1)
        assert result.scale["y"] == 2.0  # 1.0 * 2
        assert result.scale["x"] == 2.0  # 1.0 * 2
        
        # Check translation was updated for crop
        # After downsample, translation unchanged
        # After crop: translation += offset * scale
        assert result.translation["z"] == 0.0 + (10 * 2.0)  # 20.0
        assert result.translation["y"] == 0.0 + (15 * 2.0)  # 30.0  
        assert result.translation["x"] == 0.0 + (20 * 2.0)  # 40.0
    
    def test_chaining_with_io(self):
        """Test chaining that includes I/O operations."""
        # Create test data - make it larger to avoid downsampling issues
        data = da.ones((1, 32, 64, 64), chunks=(1, 16, 32, 32))
        ngff_image = nz.NgffImage(
            data=data,
            dims=["c", "z", "y", "x"],
            scale={"z": 1.0, "y": 1.0, "x": 1.0},
            translation={"z": 0.0, "y": 0.0, "x": 0.0},
            name="io_test"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            intermediate_path = os.path.join(tmpdir, "intermediate.zarr")
            final_path = os.path.join(tmpdir, "final.zarr")
            
            # Chain operations with I/O
            result = (ChainableNgffImage.from_ngff_image(ngff_image)
                      .downsample(factors=2)
                      .save(intermediate_path, max_layer=2)  # Fewer layers for smaller data
                      .crop(bbox_min=(2, 4, 6), bbox_max=(10, 20, 26))
                      .save(final_path, max_layer=2))  # Fewer layers for smaller data
            
            # Both files should exist
            assert os.path.exists(intermediate_path)
            assert os.path.exists(final_path)
            
            # Result should still be chainable for further operations
            assert isinstance(result, ChainableNgffImage)
            
            # Can load back the intermediate result
            reloaded = ChainableNgffImage.from_ome_zarr(intermediate_path, level=0)
            assert reloaded.shape == (1, 16, 32, 32)  # After downsample by 2