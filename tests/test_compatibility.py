"""
Tests for compatibility methods between legacy ZarrNii and new NgffImage-based API.
"""

import os
import tempfile
import pytest
import numpy as np
import dask.array as da
import ngff_zarr as nz
from numpy.testing import assert_array_almost_equal, assert_array_equal

from zarrnii import ZarrNii, NgffZarrNii, AffineTransform


class TestCompatibilityMethods:
    """Test compatibility methods between ZarrNii and NgffZarrNii."""
    
    @pytest.fixture
    def sample_zarrnii(self):
        """Create a sample ZarrNii instance for testing."""
        # Create test data
        data = da.random.random((1, 32, 64, 64), chunks=(1, 16, 32, 32))
        
        # Create affine transform
        affine_matrix = np.array([
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, 20.0], 
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        affine = AffineTransform.from_array(affine_matrix)
        
        return ZarrNii(
            darr=data,
            affine=affine,
            axes_order="ZYX"
        )
    
    @pytest.fixture
    def sample_ngff_image(self):
        """Create a sample NgffImage for testing."""
        data = da.random.random((1, 32, 64, 64), chunks=(1, 16, 32, 32))
        
        return nz.NgffImage(
            data=data,
            dims=["c", "z", "y", "x"],
            scale={"z": 2.0, "y": 1.0, "x": 1.0},
            translation={"z": 10.0, "y": 20.0, "x": 30.0},
            name="test_image"
        )
    
    def test_zarrnii_to_ngff_image(self, sample_zarrnii):
        """Test converting ZarrNii to NgffImage."""
        ngff_image = sample_zarrnii.to_ngff_image("converted_image")
        
        # Check data is preserved
        assert ngff_image.data.shape == sample_zarrnii.darr.shape
        assert_array_equal(ngff_image.data.compute(), sample_zarrnii.darr.compute())
        
        # Check dimensions
        assert ngff_image.dims == ["c", "z", "y", "x"]
        
        # Check scale and translation extracted from affine
        assert ngff_image.scale["z"] == 2.0
        assert ngff_image.scale["y"] == 1.0
        assert ngff_image.scale["x"] == 1.0
        
        assert ngff_image.translation["z"] == 10.0
        assert ngff_image.translation["y"] == 20.0
        assert ngff_image.translation["x"] == 30.0
        
        # Check name
        assert ngff_image.name == "converted_image"
    
    def test_zarrnii_to_ngff_zarrnii(self, sample_zarrnii):
        """Test converting ZarrNii to NgffZarrNii."""
        ngff_zarrnii = sample_zarrnii.to_ngff_zarrnii("converted_image")
        
        # Check it's the right type
        assert isinstance(ngff_zarrnii, NgffZarrNii)
        
        # Check data properties
        assert ngff_zarrnii.shape == sample_zarrnii.darr.shape
        assert_array_equal(ngff_zarrnii.data.compute(), sample_zarrnii.darr.compute())
        
        # Check scale and translation
        assert ngff_zarrnii.scale["z"] == 2.0
        assert ngff_zarrnii.scale["y"] == 1.0
        assert ngff_zarrnii.scale["x"] == 1.0
        
        assert ngff_zarrnii.translation["z"] == 10.0
        assert ngff_zarrnii.translation["y"] == 20.0
        assert ngff_zarrnii.translation["x"] == 30.0
    
    def test_ngff_image_to_zarrnii(self, sample_ngff_image):
        """Test converting NgffImage to ZarrNii."""
        zarrnii = ZarrNii.from_ngff_image(sample_ngff_image, axes_order="ZYX")
        
        # Check data is preserved
        assert zarrnii.darr.shape == sample_ngff_image.data.shape
        assert_array_equal(zarrnii.darr.compute(), sample_ngff_image.data.compute())
        
        # Check affine matrix constructed correctly
        expected_affine = np.array([
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0], 
            [0.0, 0.0, 0.0, 1.0]
        ])
        assert_array_almost_equal(zarrnii.affine.matrix, expected_affine)
        
        # Check axes order
        assert zarrnii.axes_order == "ZYX"
    
    def test_round_trip_zarrnii_to_ngff_and_back(self, sample_zarrnii):
        """Test round-trip conversion: ZarrNii -> NgffImage -> ZarrNii."""
        # Convert to NgffImage
        ngff_image = sample_zarrnii.to_ngff_image()
        
        # Convert back to ZarrNii
        zarrnii_back = ZarrNii.from_ngff_image(ngff_image, axes_order="ZYX")
        
        # Check data preservation
        assert_array_equal(
            sample_zarrnii.darr.compute(), 
            zarrnii_back.darr.compute()
        )
        
        # Check affine preservation (approximately, due to floating point)
        assert_array_almost_equal(
            sample_zarrnii.affine.matrix,
            zarrnii_back.affine.matrix,
            decimal=10
        )
        
        # Check axes order
        assert sample_zarrnii.axes_order == zarrnii_back.axes_order
    
    def test_round_trip_ngff_to_zarrnii_and_back(self, sample_ngff_image):
        """Test round-trip conversion: NgffImage -> ZarrNii -> NgffImage."""
        # Convert to ZarrNii
        zarrnii = ZarrNii.from_ngff_image(sample_ngff_image)
        
        # Convert back to NgffImage
        ngff_back = zarrnii.to_ngff_image("round_trip_test")
        
        # Check data preservation
        assert_array_equal(
            sample_ngff_image.data.compute(),
            ngff_back.data.compute()
        )
        
        # Check scale preservation
        for dim in ["z", "y", "x"]:
            assert_array_almost_equal(
                sample_ngff_image.scale[dim],
                ngff_back.scale[dim],
                decimal=10
            )
        
        # Check translation preservation
        for dim in ["z", "y", "x"]:
            assert_array_almost_equal(
                sample_ngff_image.translation[dim],
                ngff_back.translation[dim], 
                decimal=10
            )
    
    def test_zarrnii_without_affine(self):
        """Test ZarrNii to NgffImage conversion when no affine is set."""
        data = da.ones((1, 16, 32, 32), chunks=(1, 8, 16, 16))
        zarrnii = ZarrNii(darr=data, axes_order="ZYX")  # No affine provided
        
        ngff_image = zarrnii.to_ngff_image()
        
        # Should use default scale and translation
        assert ngff_image.scale == {"z": 1.0, "y": 1.0, "x": 1.0}
        assert ngff_image.translation == {"z": 0.0, "y": 0.0, "x": 0.0}
    
    def test_different_axes_orders(self):
        """Test conversion with different axes orders."""
        data = da.ones((1, 16, 32, 32), chunks=(1, 8, 16, 16))
        
        # Test ZYX order
        zarrnii_zyx = ZarrNii(darr=data, axes_order="ZYX")
        ngff_zyx = zarrnii_zyx.to_ngff_image()
        assert ngff_zyx.dims == ["c", "z", "y", "x"]
        
        # Convert back with different order should work
        zarrnii_back = ZarrNii.from_ngff_image(ngff_zyx, axes_order="XYZ")
        assert zarrnii_back.axes_order == "XYZ"


class TestMigrationWorkflow:
    """Test workflows for migrating between old and new APIs."""
    
    def test_legacy_to_new_workflow(self):
        """Test a typical migration workflow from legacy to new API."""
        # Start with legacy ZarrNii workflow
        data = da.random.random((1, 32, 64, 64), chunks=(1, 16, 32, 32))
        affine_matrix = np.eye(4)
        affine_matrix[0, 0] = 2.0  # Z scale
        affine = AffineTransform.from_array(affine_matrix)
        
        legacy_znimg = ZarrNii(darr=data, affine=affine, axes_order="ZYX")
        
        # Migrate to new API
        new_znimg = legacy_znimg.to_ngff_zarrnii("migrated")
        
        # Verify the migration preserved essential properties
        assert new_znimg.shape == legacy_znimg.darr.shape
        assert new_znimg.scale["z"] == 2.0
        assert new_znimg.scale["y"] == 1.0
        assert new_znimg.scale["x"] == 1.0
        
        # Use new API methods
        cropped = new_znimg.ngff_image
        assert cropped.name == "migrated"
    
    def test_interoperability(self):
        """Test that both APIs can work together."""
        # Create data with legacy API
        data = da.ones((1, 16, 32, 32), chunks=(1, 8, 16, 16))
        legacy_znimg = ZarrNii(darr=data, axes_order="ZYX")
        
        # Convert to new API for processing
        new_znimg = legacy_znimg.to_ngff_zarrnii()
        
        # Process with new API
        from zarrnii import crop_ngff_image
        cropped_ngff = crop_ngff_image(
            new_znimg.ngff_image,
            bbox_min=(2, 4, 6),
            bbox_max=(10, 20, 26)
        )
        
        # Convert back to legacy API if needed
        cropped_legacy = ZarrNii.from_ngff_image(cropped_ngff)
        
        # Verify the result
        expected_shape = (1, 8, 16, 20)  # After cropping
        assert cropped_legacy.darr.shape == expected_shape