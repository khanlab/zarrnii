"""
Tests for image analysis functions.

This module tests the histogram and threshold analysis functions
to ensure they work correctly with ZarrNii images.
"""

import dask.array as da
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from zarrnii import ZarrNii
from zarrnii.analysis import compute_histogram, compute_otsu_thresholds


class TestStandaloneAnalysisFunctions:
    """Test the standalone analysis functions."""

    def test_compute_histogram_basic(self):
        """Test basic histogram computation."""
        # Create test data with known properties
        data = da.from_array(
            np.array([0.0, 0.25, 0.5, 0.75, 1.0] * 100).reshape(20, 25), chunks=(10, 25)
        )

        hist, bin_edges = compute_histogram(data, bins=4)

        assert hist.shape == (4,)
        assert bin_edges.shape == (5,)

        # Check that histogram sums to total number of pixels
        assert hist.sum() == 500  # 20 * 25 pixels

    def test_compute_histogram_with_range(self):
        """Test histogram computation with specified range."""
        data = da.from_array(np.linspace(0, 1, 100).reshape(10, 10), chunks=(5, 5))

        hist, bin_edges = compute_histogram(data, bins=10, range=(0.2, 0.8))

        assert hist.shape == (10,)
        assert bin_edges.shape == (11,)
        assert float(bin_edges[0]) == 0.2
        assert float(bin_edges[-1]) == 0.8

    def test_compute_histogram_with_mask(self):
        """Test histogram computation with mask."""
        data = da.ones((10, 10), chunks=(5, 5))
        mask = da.zeros((10, 10), chunks=(5, 5))
        mask[:5, :5] = 1  # Only count first quadrant

        hist, bin_edges = compute_histogram(data, bins=2, mask=mask)

        # Should only count 25 pixels (5x5 quadrant)
        assert hist.sum() == 25

    def test_compute_histogram_mask_shape_mismatch(self):
        """Test error when mask shape doesn't match image shape."""
        data = da.ones((10, 10), chunks=(5, 5))
        mask = da.ones((5, 5), chunks=(5, 5))

        with pytest.raises(
            ValueError, match="Mask shape .* does not match image shape"
        ):
            compute_histogram(data, mask=mask)

    def test_compute_otsu_thresholds_basic(self):
        """Test basic Otsu threshold computation."""
        # Create bimodal histogram
        hist = np.array([100, 80, 60, 10, 5, 10, 60, 80, 100])

        thresholds = compute_otsu_thresholds(hist, classes=2)

        assert len(thresholds) == 3  # [min, threshold, max]
        assert thresholds[0] == 0.0  # min
        assert thresholds[-1] == float(len(hist))  # max
        assert 0 < thresholds[1] < len(hist)  # threshold in valid range

    def test_compute_otsu_thresholds_multi_level(self):
        """Test multi-level Otsu threshold computation."""
        # Create multi-modal histogram
        hist = np.array([100, 50, 20, 5, 20, 50, 100, 50, 20, 5])

        thresholds = compute_otsu_thresholds(hist, classes=3)

        assert len(thresholds) == 4  # [min, thresh1, thresh2, max]
        assert thresholds[0] == 0.0
        assert thresholds[-1] == float(len(hist))
        # Thresholds should be in ascending order
        assert all(
            thresholds[i] <= thresholds[i + 1] for i in range(len(thresholds) - 1)
        )

    def test_compute_otsu_thresholds_with_bin_edges(self):
        """Test Otsu threshold computation with bin edges."""
        hist = np.array([100, 50, 20, 50, 100])
        bin_edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        thresholds = compute_otsu_thresholds(hist, classes=2, bin_edges=bin_edges)

        assert len(thresholds) == 3
        assert thresholds[0] == 0.0  # min edge
        assert thresholds[-1] == 1.0  # max edge
        assert 0.0 <= thresholds[1] <= 1.0  # threshold in range

    def test_compute_otsu_thresholds_dask_input(self):
        """Test Otsu threshold computation with dask arrays."""
        hist = da.from_array(np.array([100, 50, 20, 50, 100]), chunks=3)
        bin_edges = da.from_array(np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]), chunks=3)

        thresholds = compute_otsu_thresholds(hist, classes=2, bin_edges=bin_edges)

        assert len(thresholds) == 3
        assert isinstance(thresholds[0], float)

    def test_compute_otsu_thresholds_invalid_classes(self):
        """Test error for invalid number of classes."""
        hist = np.array([100, 50, 100])

        with pytest.raises(ValueError, match="Number of classes must be >= 2"):
            compute_otsu_thresholds(hist, classes=1)

    def test_compute_otsu_thresholds_empty_histogram(self):
        """Test error for empty histogram."""
        hist = np.array([])

        with pytest.raises(ValueError, match="Histogram is empty"):
            compute_otsu_thresholds(hist, classes=2)

    def test_compute_otsu_thresholds_large_histogram(self):
        """Test that large histograms are handled efficiently without memory issues."""
        # Create a histogram representing a very large image (e.g., 1000^3 pixels)
        # This simulates the case where old implementation would fail with OOM
        # The histogram is small (256 bins) but represents billions of pixels
        large_pixel_count = 10**9  # 1 billion pixels
        hist = np.array([large_pixel_count // 256] * 256)  # Uniform distribution
        bin_edges = np.linspace(0.0, 1.0, 257)

        # This should work without memory issues (old implementation would try to
        # create an array with 1 billion elements)
        thresholds = compute_otsu_thresholds(hist, classes=2, bin_edges=bin_edges)

        assert len(thresholds) == 3  # [min, threshold, max]
        assert thresholds[0] == 0.0
        assert thresholds[-1] == 1.0
        # For uniform distribution, threshold should be near middle
        assert 0.4 < thresholds[1] < 0.6


class TestZarrNiiAnalysisMethods:
    """Test the ZarrNii analysis methods."""

    def setup_method(self):
        """Set up test data."""
        import ngff_zarr as nz

        # Create test data
        np.random.seed(42)  # For reproducible tests
        data = da.random.random((1, 20, 20, 20), chunks=(1, 10, 10, 10))
        dims = ["c", "z", "y", "x"]
        scale = {"z": 1.0, "y": 1.0, "x": 1.0}
        translation = {"z": 0.0, "y": 0.0, "x": 0.0}

        ngff_image = nz.NgffImage(
            data=data, dims=dims, scale=scale, translation=translation, name="test"
        )

        self.znimg = ZarrNii(ngff_image=ngff_image, axes_order="ZYX", orientation="RAS")

    def test_compute_histogram_method(self):
        """Test ZarrNii histogram computation method."""
        hist, bin_edges = self.znimg.compute_histogram(bins=10)

        assert hist.shape == (10,)
        assert bin_edges.shape == (11,)
        # Should count all pixels
        assert hist.sum() == 20 * 20 * 20

    def test_compute_histogram_method_with_mask(self):
        """Test ZarrNii histogram computation with mask."""
        import ngff_zarr as nz

        # Create a simple mask
        mask_data = self.znimg.darr > 0.5
        mask_ngff_image = nz.NgffImage(
            data=mask_data.astype(np.uint8),
            dims=self.znimg.ngff_image.dims,
            scale=self.znimg.ngff_image.scale,
            translation=self.znimg.ngff_image.translation,
            name="mask",
        )
        mask_znimg = ZarrNii(
            ngff_image=mask_ngff_image, axes_order="ZYX", orientation="RAS"
        )

        hist, bin_edges = self.znimg.compute_histogram(bins=5, mask=mask_znimg)

        assert hist.shape == (5,)
        # Should count fewer pixels due to mask
        assert hist.sum() < 20 * 20 * 20

    def test_compute_otsu_thresholds_method(self):
        """Test ZarrNii Otsu threshold computation method."""
        thresholds = self.znimg.compute_otsu_thresholds(classes=2, bins=10)

        assert len(thresholds) == 3  # [min, threshold, max]
        assert isinstance(thresholds[0], float)
        assert isinstance(thresholds[1], float)
        assert isinstance(thresholds[2], float)
        # Thresholds should be in ascending order
        assert thresholds[0] <= thresholds[1] <= thresholds[2]

    def test_compute_otsu_thresholds_multi_level(self):
        """Test ZarrNii multi-level Otsu threshold computation."""
        thresholds = self.znimg.compute_otsu_thresholds(classes=3, bins=8)

        assert len(thresholds) == 4  # [min, thresh1, thresh2, max]
        # Thresholds should be in ascending order
        assert all(
            thresholds[i] <= thresholds[i + 1] for i in range(len(thresholds) - 1)
        )

    def test_segment_threshold_method(self):
        """Test ZarrNii threshold segmentation method."""
        # Test binary threshold
        segmented = self.znimg.segment_threshold(0.5)

        assert segmented.shape == self.znimg.shape
        assert segmented.darr.dtype == np.uint8
        # Should have only 0 and 1 values
        unique_values = np.unique(segmented.darr.compute())
        assert set(unique_values).issubset({0, 1})

    def test_segment_threshold_multi_level(self):
        """Test ZarrNii multi-level threshold segmentation."""
        thresholds = self.znimg.compute_otsu_thresholds(classes=3, bins=8)
        # Use the computed thresholds excluding min/max
        segmented = self.znimg.segment_threshold(thresholds[1:-1])

        assert segmented.shape == self.znimg.shape
        assert segmented.darr.dtype == np.uint8
        # Should have values 0, 1, 2
        unique_values = np.unique(segmented.darr.compute())
        assert set(unique_values).issubset({0, 1, 2})

    def test_segment_threshold_inclusive_parameter(self):
        """Test threshold segmentation inclusive parameter."""
        import ngff_zarr as nz

        # Create test data with known threshold value
        test_data = da.from_array(
            np.array([0.4, 0.5, 0.6]).reshape(1, 1, 1, 3), chunks=(1, 1, 1, 3)
        )
        test_ngff_image = nz.NgffImage(
            data=test_data,
            dims=self.znimg.ngff_image.dims,
            scale=self.znimg.ngff_image.scale,
            translation=self.znimg.ngff_image.translation,
            name="test",
        )
        test_znimg = ZarrNii(
            ngff_image=test_ngff_image, axes_order="ZYX", orientation="RAS"
        )

        # Test inclusive (>=)
        seg_inclusive = test_znimg.segment_threshold(0.5, inclusive=True)
        result_inclusive = seg_inclusive.darr.compute().flatten()

        # Test exclusive (>)
        seg_exclusive = test_znimg.segment_threshold(0.5, inclusive=False)
        result_exclusive = seg_exclusive.darr.compute().flatten()

        # With threshold 0.5: [0.4, 0.5, 0.6]
        # Inclusive: [0, 1, 1]
        # Exclusive: [0, 0, 1]
        assert_array_equal(result_inclusive, [0, 1, 1])
        assert_array_equal(result_exclusive, [0, 0, 1])


class TestIntegrationWithExistingCode:
    """Test integration with existing ZarrNii functionality."""

    def test_backward_compatibility_segment_otsu(self):
        """Test that existing segment_otsu method still works."""
        import ngff_zarr as nz

        # Create test data
        data = da.random.random((1, 10, 10, 10), chunks=(1, 5, 5, 5))
        dims = ["c", "z", "y", "x"]
        scale = {"z": 1.0, "y": 1.0, "x": 1.0}
        translation = {"z": 0.0, "y": 0.0, "x": 0.0}

        ngff_image = nz.NgffImage(
            data=data, dims=dims, scale=scale, translation=translation, name="test"
        )

        znimg = ZarrNii(ngff_image=ngff_image, axes_order="ZYX", orientation="RAS")

        # Should still work as before
        segmented = znimg.segment_otsu(nbins=32)

        assert segmented.shape == znimg.shape
        assert segmented.darr.dtype == np.uint8
        unique_values = np.unique(segmented.darr.compute())
        assert set(unique_values).issubset({0, 1})

    def test_analysis_workflow_example(self):
        """Test the complete analysis workflow from the issue description."""
        import ngff_zarr as nz

        # Create test data similar to the issue example
        np.random.seed(123)
        data = da.random.random((1, 50, 50, 50), chunks=(1, 25, 25, 25))
        dims = ["c", "z", "y", "x"]
        scale = {"z": 1.0, "y": 1.0, "x": 1.0}
        translation = {"z": 0.0, "y": 0.0, "x": 0.0}

        ngff_image = nz.NgffImage(
            data=data, dims=dims, scale=scale, translation=translation, name="test"
        )

        znimg_hires = ZarrNii(
            ngff_image=ngff_image, axes_order="ZYX", orientation="RAS"
        )

        # Test the workflow described in the issue
        histogram_opts = {"bins": 128, "range": (0.0, 1.0)}
        hist, bin_edges = compute_histogram(znimg_hires.darr, **histogram_opts)

        max_k = 4
        thresholds = {}
        for k in range(2, max_k + 1):
            thresholds[k] = compute_otsu_thresholds(
                hist, classes=k, bin_edges=bin_edges
            )

        # Verify results
        assert len(thresholds) == 3  # k = 2, 3, 4
        for k, thresh_list in thresholds.items():
            assert len(thresh_list) == k + 1  # [min, thresh1, ..., thresh_k-1, max]
            # First and last should be min and max
            assert thresh_list[0] == 0.0
            assert thresh_list[-1] == 1.0
