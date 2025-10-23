"""Tests for metadata-safe delegation of Dask operations on ZarrNii objects.

This module tests the delegation of Dask array operations to ZarrNii objects,
ensuring that metadata is preserved when safe and errors are raised when
operations would invalidate metadata.
"""

import dask.array as da
import ngff_zarr as nz
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from zarrnii import MetadataInvalidError, ZarrNii


class TestSafeOperations:
    """Test operations that preserve metadata and return ZarrNii objects."""

    def test_rechunk_operation(self):
        """Test that rechunk preserves metadata."""
        # Create a ZarrNii object
        data = da.from_array(np.random.rand(1, 10, 20, 30), chunks=(1, 5, 10, 15))
        znii = ZarrNii.from_darr(
            data, axes_order="ZYX", spacing=(1.0, 1.0, 1.0), name="test"
        )

        # Rechunk the data
        rechunked = znii.rechunk((1, 10, 10, 10))

        # Should return a ZarrNii object
        assert isinstance(rechunked, ZarrNii)

        # Should preserve shape
        assert rechunked.shape == znii.shape

        # Should preserve metadata
        assert rechunked.scale == znii.scale
        assert rechunked.translation == znii.translation
        assert rechunked.dims == znii.dims

        # Should have different chunks
        assert rechunked.data.chunks != znii.data.chunks

    def test_persist_operation(self):
        """Test that persist preserves metadata."""
        data = da.from_array(np.random.rand(1, 10, 20, 30), chunks=(1, 5, 10, 15))
        znii = ZarrNii.from_darr(data, axes_order="ZYX")

        # Persist the data
        persisted = znii.persist()

        # Should return a ZarrNii object
        assert isinstance(persisted, ZarrNii)

        # Should preserve shape and metadata
        assert persisted.shape == znii.shape
        assert persisted.scale == znii.scale

    def test_astype_operation(self):
        """Test that astype raises error since it changes dtype."""
        data = da.from_array(
            np.random.rand(1, 10, 20, 30).astype(np.float32), chunks=(1, 5, 10, 15)
        )
        znii = ZarrNii.from_darr(data, axes_order="ZYX")

        # astype changes dtype, so it should raise MetadataInvalidError
        with pytest.raises(MetadataInvalidError, match="changes shape or dtype"):
            znii.astype(np.float64)

    def test_map_blocks_operation(self):
        """Test that map_blocks preserves metadata when output has same shape."""
        data = da.from_array(np.random.rand(1, 10, 20, 30), chunks=(1, 5, 10, 15))
        znii = ZarrNii.from_darr(data, axes_order="ZYX")

        # map_blocks with same output shape
        def add_one(block):
            return block + 1

        mapped = znii.map_blocks(add_one, dtype=znii.data.dtype)

        # Should return a ZarrNii object
        assert isinstance(mapped, ZarrNii)

        # Should preserve shape and metadata
        assert mapped.shape == znii.shape
        assert mapped.scale == znii.scale


class TestUnsafeOperations:
    """Test operations that invalidate metadata and should raise errors."""

    def test_transpose_raises_error(self):
        """Test that transpose raises MetadataInvalidError."""
        data = da.from_array(np.random.rand(1, 10, 20, 30), chunks=(1, 5, 10, 15))
        znii = ZarrNii.from_darr(data, axes_order="ZYX")

        # Transpose changes shape, so it should raise
        with pytest.raises(MetadataInvalidError, match="transpose"):
            znii.transpose((0, 2, 1, 3))

    def test_sum_axis_raises_error(self):
        """Test that sum along axis raises MetadataInvalidError."""
        data = da.from_array(np.random.rand(1, 10, 20, 30), chunks=(1, 5, 10, 15))
        znii = ZarrNii.from_darr(data, axes_order="ZYX")

        # Sum reduces dimensions, so it should raise
        with pytest.raises(MetadataInvalidError, match="sum"):
            znii.sum(axis=0)

    def test_reshape_raises_error(self):
        """Test that reshape raises MetadataInvalidError."""
        data = da.from_array(np.random.rand(1, 10, 20, 30), chunks=(1, 5, 10, 15))
        znii = ZarrNii.from_darr(data, axes_order="ZYX")

        # Reshape changes shape, so it should raise
        with pytest.raises(MetadataInvalidError, match="reshape"):
            znii.reshape((1, 10, 600))

    def test_mean_axis_raises_error(self):
        """Test that mean along axis raises MetadataInvalidError."""
        data = da.from_array(np.random.rand(1, 10, 20, 30), chunks=(1, 5, 10, 15))
        znii = ZarrNii.from_darr(data, axes_order="ZYX")

        # Mean reduces dimensions, so it should raise
        with pytest.raises(MetadataInvalidError, match="mean"):
            znii.mean(axis=1)


class TestNumpyUfuncs:
    """Test NumPy universal function support."""

    def test_sqrt_ufunc(self):
        """Test that np.sqrt works on ZarrNii objects."""
        data = da.from_array(np.array([[[[4.0, 9.0, 16.0]]]]), chunks=(1, 1, 1, 3))
        znii = ZarrNii.from_darr(data, axes_order="ZYX")

        # Apply sqrt
        result = np.sqrt(znii)

        # Should return a ZarrNii object
        assert isinstance(result, ZarrNii)

        # Should preserve shape and metadata
        assert result.shape == znii.shape
        assert result.scale == znii.scale

        # Check result values
        expected = np.array([[[[2.0, 3.0, 4.0]]]])
        assert_array_equal(result.data.compute(), expected)

    def test_add_ufunc(self):
        """Test that addition works between ZarrNii objects."""
        data1 = da.from_array(np.array([[[[1.0, 2.0, 3.0]]]]), chunks=(1, 1, 1, 3))
        data2 = da.from_array(np.array([[[[4.0, 5.0, 6.0]]]]), chunks=(1, 1, 1, 3))

        znii1 = ZarrNii.from_darr(data1, axes_order="ZYX")
        znii2 = ZarrNii.from_darr(data2, axes_order="ZYX")

        # Add two ZarrNii objects
        result = np.add(znii1, znii2)

        # Should return a ZarrNii object
        assert isinstance(result, ZarrNii)

        # Should preserve shape and metadata
        assert result.shape == znii1.shape
        assert result.scale == znii1.scale

        # Check result values
        expected = np.array([[[[5.0, 7.0, 9.0]]]])
        assert_array_equal(result.data.compute(), expected)

    def test_operator_overloading_add(self):
        """Test that + operator works on ZarrNii objects."""
        data = da.from_array(np.array([[[[1.0, 2.0, 3.0]]]]), chunks=(1, 1, 1, 3))
        znii = ZarrNii.from_darr(data, axes_order="ZYX")

        # Add scalar
        result = znii + 10

        # Should return a ZarrNii object
        assert isinstance(result, ZarrNii)

        # Check result values
        expected = np.array([[[[11.0, 12.0, 13.0]]]])
        assert_array_equal(result.data.compute(), expected)

    def test_operator_overloading_multiply(self):
        """Test that * operator works on ZarrNii objects."""
        data = da.from_array(np.array([[[[2.0, 3.0, 4.0]]]]), chunks=(1, 1, 1, 3))
        znii = ZarrNii.from_darr(data, axes_order="ZYX")

        # Multiply by scalar
        result = znii * 2

        # Should return a ZarrNii object
        assert isinstance(result, ZarrNii)

        # Check result values
        expected = np.array([[[[4.0, 6.0, 8.0]]]])
        assert_array_equal(result.data.compute(), expected)

    def test_operator_overloading_subtract(self):
        """Test that - operator works on ZarrNii objects."""
        data = da.from_array(np.array([[[[10.0, 20.0, 30.0]]]]), chunks=(1, 1, 1, 3))
        znii = ZarrNii.from_darr(data, axes_order="ZYX")

        # Subtract scalar
        result = znii - 5

        # Should return a ZarrNii object
        assert isinstance(result, ZarrNii)

        # Check result values
        expected = np.array([[[[5.0, 15.0, 25.0]]]])
        assert_array_equal(result.data.compute(), expected)

    def test_operator_overloading_divide(self):
        """Test that / operator works on ZarrNii objects."""
        data = da.from_array(np.array([[[[10.0, 20.0, 30.0]]]]), chunks=(1, 1, 1, 3))
        znii = ZarrNii.from_darr(data, axes_order="ZYX")

        # Divide by scalar
        result = znii / 10

        # Should return a ZarrNii object
        assert isinstance(result, ZarrNii)

        # Check result values
        expected = np.array([[[[1.0, 2.0, 3.0]]]])
        assert_array_equal(result.data.compute(), expected)


class TestNumpyFunctionProtocol:
    """Test NumPy __array_function__ protocol support."""

    def test_absolute_preserves_metadata(self):
        """Test that np.abs works on ZarrNii objects."""
        data = da.from_array(np.array([[[[-1.0, 2.0, -3.0]]]]), chunks=(1, 1, 1, 3))
        znii = ZarrNii.from_darr(data, axes_order="ZYX")

        # Apply absolute value
        result = np.abs(znii)

        # Should return a ZarrNii object
        assert isinstance(result, ZarrNii)

        # Check result values
        expected = np.array([[[[1.0, 2.0, 3.0]]]])
        assert_array_equal(result.data.compute(), expected)


class TestMetadataPreservation:
    """Test that metadata is correctly preserved across operations."""

    def test_chained_operations(self):
        """Test chaining multiple safe operations."""
        data = da.from_array(np.random.rand(1, 10, 20, 30), chunks=(1, 5, 10, 15))
        znii = ZarrNii.from_darr(
            data,
            axes_order="ZYX",
            spacing=(2.0, 1.5, 1.0),
            origin=(10.0, 20.0, 30.0),
            name="original",
        )

        # Chain operations
        result = znii.rechunk((1, 10, 10, 10))
        result = result + 10
        result = np.sqrt(result)
        result = result.persist()

        # Should still be ZarrNii
        assert isinstance(result, ZarrNii)

        # Should preserve all metadata
        assert result.shape == znii.shape
        assert result.scale == znii.scale
        assert result.translation == znii.translation
        assert result.dims == znii.dims
        assert result.name == znii.name

    def test_metadata_immutability(self):
        """Test that original object is not modified by operations."""
        data = da.from_array(np.random.rand(1, 10, 20, 30), chunks=(1, 5, 10, 15))
        znii = ZarrNii.from_darr(data, axes_order="ZYX", spacing=(2.0, 1.5, 1.0))

        # Store original chunks
        original_chunks = znii.data.chunks

        # Perform operation
        result = znii.rechunk((1, 10, 10, 10))

        # Original should be unchanged
        assert znii.data.chunks == original_chunks

        # Result should have new chunks
        assert result.data.chunks != original_chunks

    def test_scale_and_translation_preserved(self):
        """Test that scale and translation are preserved."""
        data = da.from_array(np.random.rand(1, 10, 20, 30), chunks=(1, 5, 10, 15))
        znii = ZarrNii.from_darr(
            data,
            axes_order="ZYX",
            spacing=(2.5, 1.8, 0.5),
            origin=(100.0, 200.0, 300.0),
        )

        # Perform operation
        result = znii + 5

        # Check scale preservation
        assert result.scale["z"] == 2.5
        assert result.scale["y"] == 1.8
        assert result.scale["x"] == 0.5

        # Check translation preservation
        assert result.translation["z"] == 100.0
        assert result.translation["y"] == 200.0
        assert result.translation["x"] == 300.0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_non_dask_array_result(self):
        """Test that non-dask results are returned as-is."""
        data = da.from_array(np.random.rand(1, 10, 20, 30), chunks=(1, 5, 10, 15))
        znii = ZarrNii.from_darr(data, axes_order="ZYX")

        # compute() returns NgffImage (not numpy array) via __getattr__ delegation
        # It doesn't get wrapped since it's not a dask array
        result = znii.compute()

        # Should return NgffImage, not ZarrNii
        assert not isinstance(result, ZarrNii)
        # The result should be NgffImage
        assert isinstance(result, nz.NgffImage)

    def test_attribute_access_non_dask_property(self):
        """Test accessing non-dask attributes raises AttributeError."""
        data = da.from_array(np.random.rand(1, 10, 20, 30), chunks=(1, 5, 10, 15))
        znii = ZarrNii.from_darr(data, axes_order="ZYX")

        # Accessing a non-existent attribute should raise
        with pytest.raises(AttributeError):
            znii.nonexistent_attribute

    def test_element_wise_operations_preserve_chunks(self):
        """Test that element-wise operations preserve chunk structure."""
        data = da.from_array(np.random.rand(1, 10, 20, 30), chunks=(1, 5, 10, 15))
        znii = ZarrNii.from_darr(data, axes_order="ZYX")

        # Element-wise operation
        result = znii * 2

        # Chunks should be preserved
        assert result.data.chunks == znii.data.chunks
