"""
Tests for MultiOtsuSegmentation plugin.

This module tests the multi-level Otsu thresholding segmentation plugin
including saving histogram, thresholds, and visualization figures.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from zarrnii.plugins.segmentation import MultiOtsuSegmentation


class TestMultiOtsuSegmentation:
    """Test the MultiOtsuSegmentation plugin."""

    def test_basic_binary_segmentation(self):
        """Test basic binary (2-class) segmentation."""
        # Create bimodal test image
        np.random.seed(42)
        image = np.concatenate(
            [np.random.normal(0.2, 0.05, 500), np.random.normal(0.8, 0.05, 500)]
        ).reshape(10, 100)

        plugin = MultiOtsuSegmentation(classes=2, nbins=128)
        result = plugin.segment(image)

        assert result.shape == image.shape
        assert result.dtype == np.uint8
        # Should have 2 classes: 0 and 1
        unique_values = np.unique(result)
        assert set(unique_values).issubset({0, 1})
        assert len(unique_values) == 2

    def test_multi_class_segmentation(self):
        """Test multi-class (3-class) segmentation."""
        # Create trimodal test image
        np.random.seed(42)
        image = np.concatenate(
            [
                np.random.normal(0.2, 0.05, 300),
                np.random.normal(0.5, 0.05, 300),
                np.random.normal(0.8, 0.05, 300),
            ]
        ).reshape(30, 30)

        plugin = MultiOtsuSegmentation(classes=3, nbins=128)
        result = plugin.segment(image)

        assert result.shape == image.shape
        assert result.dtype == np.uint8
        # Should have 3 classes: 0, 1, and 2
        unique_values = np.unique(result)
        assert set(unique_values).issubset({0, 1, 2})

    def test_save_histogram(self):
        """Test saving histogram to file."""
        np.random.seed(42)
        image = np.random.random((10, 10))

        with tempfile.TemporaryDirectory() as tmpdir:
            hist_path = Path(tmpdir) / "histogram.npz"

            plugin = MultiOtsuSegmentation(
                classes=2, nbins=64, save_histogram=hist_path
            )
            plugin.segment(image)

            # Check file was created
            assert hist_path.exists()

            # Load and verify contents
            data = np.load(hist_path)
            assert "counts" in data
            assert "bin_edges" in data
            assert data["counts"].shape == (64,)
            assert data["bin_edges"].shape == (65,)  # bins + 1

            # Verify histogram sums to total pixels
            assert data["counts"].sum() == image.size

    def test_save_thresholds(self):
        """Test saving thresholds to JSON file."""
        np.random.seed(42)
        image = np.random.random((10, 10))

        with tempfile.TemporaryDirectory() as tmpdir:
            thresh_path = Path(tmpdir) / "thresholds.json"

            plugin = MultiOtsuSegmentation(
                classes=3, nbins=64, save_thresholds=thresh_path
            )
            plugin.segment(image)

            # Check file was created
            assert thresh_path.exists()

            # Load and verify contents
            with open(thresh_path, "r") as f:
                data = json.load(f)

            assert "classes" in data
            assert "thresholds" in data
            assert "min_value" in data
            assert "max_value" in data

            assert data["classes"] == 3
            assert len(data["thresholds"]) == 2  # 3 classes = 2 thresholds
            assert all(isinstance(t, float) for t in data["thresholds"])

            # Thresholds should be in ascending order
            assert data["thresholds"][0] < data["thresholds"][1]

            # Thresholds should be within data range
            assert data["min_value"] <= data["thresholds"][0] <= data["max_value"]
            assert data["min_value"] <= data["thresholds"][1] <= data["max_value"]

    def test_save_figure(self):
        """Test saving visualization figure as SVG."""
        pytest.importorskip("matplotlib")

        np.random.seed(42)
        image = np.random.random((10, 10))

        with tempfile.TemporaryDirectory() as tmpdir:
            fig_path = Path(tmpdir) / "histogram.svg"

            plugin = MultiOtsuSegmentation(classes=2, nbins=64, save_figure=fig_path)
            plugin.segment(image)

            # Check file was created
            assert fig_path.exists()

            # Verify it's an SVG file
            with open(fig_path, "r") as f:
                content = f.read()
                assert content.startswith("<?xml") or content.startswith("<svg")
                assert "svg" in content.lower()

    def test_save_all_outputs(self):
        """Test saving all outputs simultaneously."""
        pytest.importorskip("matplotlib")

        np.random.seed(42)
        image = np.random.random((10, 10))

        with tempfile.TemporaryDirectory() as tmpdir:
            hist_path = Path(tmpdir) / "histogram.npz"
            thresh_path = Path(tmpdir) / "thresholds.json"
            fig_path = Path(tmpdir) / "histogram.svg"

            plugin = MultiOtsuSegmentation(
                classes=2,
                nbins=64,
                save_histogram=hist_path,
                save_thresholds=thresh_path,
                save_figure=fig_path,
            )
            result = plugin.segment(image)

            # All files should be created
            assert hist_path.exists()
            assert thresh_path.exists()
            assert fig_path.exists()

            # Result should still be valid
            assert result.shape == image.shape

    def test_get_thresholds(self):
        """Test retrieving computed thresholds."""
        np.random.seed(42)
        image = np.random.random((10, 10))

        plugin = MultiOtsuSegmentation(classes=3, nbins=64)

        # Before segmentation
        assert plugin.get_thresholds() is None

        # After segmentation
        plugin.segment(image)
        thresholds = plugin.get_thresholds()

        assert thresholds is not None
        assert len(thresholds) == 2  # 3 classes = 2 thresholds
        assert all(isinstance(t, float) for t in thresholds)
        assert thresholds[0] < thresholds[1]

    def test_get_histogram(self):
        """Test retrieving computed histogram."""
        np.random.seed(42)
        image = np.random.random((10, 10))

        plugin = MultiOtsuSegmentation(classes=2, nbins=64)

        # Before segmentation
        assert plugin.get_histogram() is None

        # After segmentation
        plugin.segment(image)
        hist_data = plugin.get_histogram()

        assert hist_data is not None
        counts, bin_edges = hist_data
        assert counts.shape == (64,)
        assert bin_edges.shape == (65,)
        assert counts.sum() == image.size

    def test_plugin_properties(self):
        """Test plugin name and description properties."""
        plugin_binary = MultiOtsuSegmentation(classes=2)
        assert "Binary" in plugin_binary.name
        assert "2" in plugin_binary.description

        plugin_multi = MultiOtsuSegmentation(classes=4)
        assert "Multi-level" in plugin_multi.name
        assert "4" in plugin_multi.description

    def test_invalid_classes(self):
        """Test error for invalid number of classes."""
        with pytest.raises(ValueError, match="Number of classes must be >= 2"):
            MultiOtsuSegmentation(classes=1)

    def test_empty_image(self):
        """Test error for empty image."""
        plugin = MultiOtsuSegmentation(classes=2)

        with pytest.raises(ValueError, match="Input image is empty"):
            plugin.segment(np.array([]))

    def test_1d_image(self):
        """Test error for 1D image."""
        plugin = MultiOtsuSegmentation(classes=2)

        with pytest.raises(ValueError, match="Input image must be at least 2D"):
            plugin.segment(np.array([1, 2, 3]))

    def test_boolean_image(self):
        """Test segmentation of boolean image."""
        image = np.random.random((10, 10)) > 0.5

        plugin = MultiOtsuSegmentation(classes=2)
        result = plugin.segment(image)

        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_3d_image(self):
        """Test segmentation of 3D image."""
        np.random.seed(42)
        image = np.random.random((5, 10, 10))

        plugin = MultiOtsuSegmentation(classes=2, nbins=64)
        result = plugin.segment(image)

        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_different_nbins(self):
        """Test segmentation with different histogram bin counts."""
        np.random.seed(42)
        image = np.random.random((10, 10))

        for nbins in [32, 64, 128, 256]:
            plugin = MultiOtsuSegmentation(classes=2, nbins=nbins)
            result = plugin.segment(image)

            assert result.shape == image.shape
            hist_data = plugin.get_histogram()
            counts, _ = hist_data
            assert len(counts) == nbins

    def test_constant_image(self):
        """Test segmentation of constant (uniform) image."""
        image = np.ones((10, 10)) * 0.5

        plugin = MultiOtsuSegmentation(classes=2, nbins=64)
        result = plugin.segment(image)

        # Should still produce a result (likely all same class)
        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_plugin_repr(self):
        """Test plugin string representation."""
        plugin = MultiOtsuSegmentation(classes=3, nbins=128)
        repr_str = repr(plugin)

        assert "MultiOtsuSegmentation" in repr_str
        assert "classes=3" in repr_str
        assert "nbins=128" in repr_str
