"""
Tests for StarDist segmentation plugin.

These tests check the StarDist plugin functionality, including initialization,
parameter handling, and integration with ZarrNii. Some tests are skipped if
StarDist dependencies are not installed.
"""

from unittest.mock import Mock, patch

import dask.array as da
import numpy as np
import pytest

# Try to import StarDist components
try:
    from zarrnii.plugins.segmentation import StarDistSegmentation

    STARDIST_AVAILABLE = True
except ImportError:
    STARDIST_AVAILABLE = False


@pytest.mark.skipif(not STARDIST_AVAILABLE, reason="StarDist not available")
class TestStarDistSegmentation:
    """Test the StarDist segmentation plugin."""

    def test_stardist_plugin_initialization(self):
        """Test StarDist plugin can be initialized with different parameters."""
        # Default initialization
        plugin = StarDistSegmentation()
        assert plugin.model_name == "2D_versatile_fluo"
        assert plugin.prob_thresh == 0.5
        assert plugin.nms_thresh == 0.4
        assert plugin.use_dask_relabeling is False
        assert plugin.overlap == 64
        assert plugin.name == "StarDist (2D_versatile_fluo)"
        assert "StarDist deep learning instance segmentation" in plugin.description

        # Custom initialization
        plugin_custom = StarDistSegmentation(
            model_name="3D_demo",
            prob_thresh=0.6,
            nms_thresh=0.3,
            use_gpu=True,
            use_dask_relabeling=False,
            overlap=32,
        )
        assert plugin_custom.model_name == "3D_demo"
        assert plugin_custom.prob_thresh == 0.6
        assert plugin_custom.nms_thresh == 0.3
        assert plugin_custom.use_gpu is True
        assert plugin_custom.use_dask_relabeling is False
        assert plugin_custom.overlap == 32

    def test_stardist_plugin_properties(self):
        """Test plugin properties."""
        plugin = StarDistSegmentation(model_name="2D_versatile_he")

        assert plugin.name == "StarDist (2D_versatile_he)"
        assert "StarDist deep learning" in plugin.description
        assert "2D_versatile_he" in plugin.description

        # Test string representation
        repr_str = repr(plugin)
        assert "StarDistSegmentation" in repr_str
        assert "2D_versatile_he" in repr_str

    @patch("zarrnii.plugins.segmentation.stardist.StarDist2D")
    def test_stardist_model_loading_2d(self, mock_stardist2d):
        """Test 2D model loading."""
        mock_model = Mock()
        mock_stardist2d.from_pretrained.return_value = mock_model

        plugin = StarDistSegmentation(model_name="2D_versatile_fluo")
        plugin._load_model()

        mock_stardist2d.from_pretrained.assert_called_once_with("2D_versatile_fluo")
        assert plugin._model == mock_model
        assert plugin._model_loaded is True

    @patch("zarrnii.plugins.segmentation.stardist.StarDist3D")
    def test_stardist_model_loading_3d(self, mock_stardist3d):
        """Test 3D model loading."""
        mock_model = Mock()
        mock_stardist3d.from_pretrained.return_value = mock_model

        plugin = StarDistSegmentation(model_name="3D_demo")
        plugin._load_model()

        mock_stardist3d.from_pretrained.assert_called_once_with("3D_demo")
        assert plugin._model == mock_model
        assert plugin._model_loaded is True

    @patch("zarrnii.plugins.segmentation.stardist.StarDist2D")
    def test_stardist_custom_model_path(self, mock_stardist2d):
        """Test loading custom model from path."""
        mock_model = Mock()
        mock_stardist2d.return_value = mock_model

        plugin = StarDistSegmentation(model_path="/path/to/custom/model")
        plugin._load_model()

        mock_stardist2d.assert_called_once_with(None, name="/path/to/custom/model")
        assert plugin._model == mock_model

    def test_stardist_should_use_dask_relabeling(self):
        """Test logic for determining when to use dask_relabeling."""
        plugin = StarDistSegmentation()

        # Small 2D image - should not use dask_relabeling
        small_2d = np.random.rand(512, 512)
        assert not plugin._should_use_dask_relabeling(small_2d)

        # Large 2D image - should use dask_relabeling
        large_2d = np.random.rand(3000, 3000)
        assert plugin._should_use_dask_relabeling(large_2d)

        # Small 3D image - should not use dask_relabeling
        small_3d = np.random.rand(64, 256, 256)
        assert not plugin._should_use_dask_relabeling(small_3d)

        # Large 3D image - should use dask_relabeling
        large_3d = np.random.rand(100, 600, 600)
        assert plugin._should_use_dask_relabeling(large_3d)

    @patch("zarrnii.plugins.segmentation.stardist.StarDist2D")
    def test_stardist_direct_segmentation(self, mock_stardist2d):
        """Test direct segmentation without dask_relabeling."""
        # Mock the model and its predict_instances method
        mock_model = Mock()
        labels = np.zeros((100, 100), dtype=np.uint32)
        labels[20:40, 20:40] = 1
        labels[60:80, 60:80] = 2
        mock_model.predict_instances.return_value = (labels, None)
        mock_stardist2d.from_pretrained.return_value = mock_model

        plugin = StarDistSegmentation(model_name="2D_versatile_fluo")
        plugin.use_dask_relabeling = False  # Force direct segmentation

        # Test with 2D image
        test_image = np.random.rand(100, 100).astype(np.float32)
        result = plugin.segment(test_image)

        assert result.shape == test_image.shape
        assert result.dtype == np.uint32
        assert np.max(result) == 2
        assert np.sum(result > 0) > 0  # Should have some segmented objects

        # Verify the model was called correctly
        mock_model.predict_instances.assert_called_once()

    def test_stardist_empty_image_handling(self):
        """Test handling of empty images."""
        plugin = StarDistSegmentation()

        # Empty image should raise ValueError
        empty_image = np.array([])
        with pytest.raises(ValueError, match="Input image is empty"):
            plugin.segment(empty_image)

    def test_stardist_invalid_dimensions(self):
        """Test handling of images with invalid dimensions."""
        plugin = StarDistSegmentation()

        # 1D image should raise ValueError
        image_1d = np.random.rand(100)
        with pytest.raises(ValueError, match="Input image must be at least 2D"):
            plugin.segment(image_1d)

    @patch("zarrnii.plugins.segmentation.stardist.StarDist2D")
    def test_stardist_multichannel_handling(self, mock_stardist2d):
        """Test handling of multi-channel images."""
        # Mock the model
        mock_model = Mock()
        labels = np.zeros((50, 50), dtype=np.uint32)
        mock_model.predict_instances.return_value = (labels, None)
        mock_stardist2d.from_pretrained.return_value = mock_model

        plugin = StarDistSegmentation()
        plugin.use_dask_relabeling = False

        # Test with multi-channel 2D image (C, H, W)
        multichannel_image = np.random.rand(3, 50, 50).astype(np.float32)
        result = plugin.segment(multichannel_image)

        # Should process first channel and return 2D result
        assert result.shape == (50, 50)
        mock_model.predict_instances.assert_called_once()

        # Check that first channel was passed to model
        call_args = mock_model.predict_instances.call_args[0]
        assert call_args[0].shape == (50, 50)

    @patch("zarrnii.plugins.segmentation.stardist.StarDist2D")
    def test_stardist_get_model_info(self, mock_stardist2d):
        """Test get_model_info method."""
        mock_model = Mock()
        mock_config = Mock()
        mock_config.n_dim = 2
        mock_model.config = mock_config
        mock_stardist2d.from_pretrained.return_value = mock_model

        plugin = StarDistSegmentation(
            model_name="2D_versatile_fluo",
            prob_thresh=0.6,
            nms_thresh=0.3,
            use_gpu=True,
        )

        info = plugin.get_model_info()

        expected_info = {
            "model_name": "2D_versatile_fluo",
            "model_path": None,
            "is_3d": False,
            "prob_thresh": 0.6,
            "nms_thresh": 0.3,
            "use_gpu": True,
        }

        assert info == expected_info

    def test_stardist_import_error_handling(self):
        """Test handling when StarDist is not installed."""
        # This test simulates the import error by monkeypatching
        with patch(
            "zarrnii.plugins.segmentation.stardist.StarDist2D", side_effect=ImportError
        ):
            plugin = StarDistSegmentation()

            test_image = np.random.rand(100, 100).astype(np.float32)

            with pytest.raises(ImportError, match="StarDist is not installed"):
                plugin.segment(test_image)


@pytest.mark.skipif(not STARDIST_AVAILABLE, reason="StarDist not available")
class TestZarrNiiStarDistIntegration:
    """Test StarDist integration with ZarrNii."""

    def test_stardist_convenience_method_import_error(self):
        """Test StarDist convenience method when dependencies missing."""
        from zarrnii import ZarrNii

        # Create test data
        test_data = np.random.rand(1, 50, 100, 100).astype(np.float32)
        darr = da.from_array(test_data, chunks=(1, 25, 50, 50))
        znimg = ZarrNii.from_darr(darr, axes_order="CZYX", orientation="RAS")

        # Mock the import to simulate missing dependencies
        with patch("zarrnii.core.StarDistSegmentation", side_effect=ImportError):
            with pytest.raises(ImportError, match="StarDist is not available"):
                znimg.segment_stardist()

    @patch("zarrnii.plugins.segmentation.stardist.StarDist2D")
    def test_stardist_convenience_method_success(self, mock_stardist2d):
        """Test StarDist convenience method success case."""
        from zarrnii import ZarrNii

        # Mock the model
        mock_model = Mock()
        labels = np.zeros((50, 100), dtype=np.uint32)
        labels[10:20, 10:30] = 1
        mock_model.predict_instances.return_value = (labels, None)
        mock_stardist2d.from_pretrained.return_value = mock_model

        # Create test data
        test_data = np.random.rand(1, 50, 100).astype(np.float32)
        darr = da.from_array(test_data, chunks=(1, 25, 50))
        znimg = ZarrNii.from_darr(darr, axes_order="ZYX", orientation="RAS")

        # Test convenience method
        result = znimg.segment_stardist(
            model_name="2D_versatile_fluo",
            prob_thresh=0.6,
            use_dask_relabeling=False,
        )

        # Check result
        assert isinstance(result, ZarrNii)
        assert result.shape == znimg.shape
        assert result.data.dtype == np.uint8  # segment() returns uint8
        assert np.max(result.data.compute()) == 1  # Should have segmented objects

    def test_stardist_plugin_repr(self):
        """Test plugin string representation."""
        plugin = StarDistSegmentation(
            model_name="3D_demo",
            prob_thresh=0.7,
            use_gpu=False,
        )
        repr_str = repr(plugin)

        assert "StarDistSegmentation" in repr_str
        assert "3D_demo" in repr_str
        assert "prob_thresh=0.7" in repr_str
        assert "use_gpu=False" in repr_str
