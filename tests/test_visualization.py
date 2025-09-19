"""Tests for visualization functionality."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import dask.array as da
import numpy as np
import pytest

from zarrnii import ZarrNii


class TestVisualizationModule:
    """Test the visualization module directly."""

    def test_visualization_not_available_by_default(self):
        """Test that visualization raises ImportError without vizarr."""
        with patch.dict("sys.modules", {"vizarr": None}):
            from zarrnii import visualization
            
            # Reload the module to get the ImportError behavior
            import importlib
            importlib.reload(visualization)
            
            assert not visualization.is_available()
            
            with pytest.raises(ImportError, match="vizarr is required"):
                visualization.visualize("test.zarr")

    def test_visualize_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        from zarrnii import visualization
        
        with patch.object(visualization, 'vizarr', Mock()):
            with pytest.raises(ValueError, match="Invalid mode"):
                visualization.visualize("test.zarr", mode="invalid")

    def test_visualize_nonexistent_path(self):
        """Test that nonexistent path raises FileNotFoundError."""
        from zarrnii import visualization
        
        with patch.object(visualization, 'vizarr', Mock()):
            with pytest.raises(FileNotFoundError, match="does not exist"):
                visualization.visualize("nonexistent.zarr")

    @patch('zarrnii.visualization.vizarr')
    @patch('zarrnii.visualization.webbrowser')
    def test_generate_html_mode(self, mock_webbrowser, mock_vizarr):
        """Test HTML generation mode."""
        from zarrnii import visualization
        
        # Mock vizarr
        mock_viewer = Mock()
        mock_viewer.to_html.return_value = "<html>test content</html>"
        mock_vizarr.Viewer.return_value = mock_viewer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"
            zarr_path.mkdir()
            
            output_path = Path(tmpdir) / "output.html"
            
            result = visualization.visualize(
                zarr_path=zarr_path,
                mode="html",
                output_path=output_path,
                open_browser=False
            )
            
            assert result == str(output_path)
            assert output_path.exists()
            
            # Check content
            with open(output_path, 'r') as f:
                content = f.read()
            assert content == "<html>test content</html>"
            
            # Verify calls
            mock_vizarr.Viewer.assert_called_once()
            mock_viewer.add_image.assert_called_once_with(source=str(zarr_path))
            mock_viewer.to_html.assert_called_once()

    @patch('zarrnii.visualization.vizarr')
    def test_generate_html_with_browser_open(self, mock_vizarr):
        """Test HTML generation with browser opening."""
        from zarrnii import visualization
        
        mock_viewer = Mock()
        mock_viewer.to_html.return_value = "<html>test</html>"
        mock_vizarr.Viewer.return_value = mock_viewer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"
            zarr_path.mkdir()
            
            with patch('zarrnii.visualization.webbrowser.open') as mock_open:
                result = visualization.visualize(
                    zarr_path=zarr_path,
                    mode="html",
                    open_browser=True
                )
                
                # Should have called webbrowser.open
                mock_open.assert_called_once()
                assert result.endswith('.html')

    @patch('zarrnii.visualization.vizarr')
    def test_server_mode(self, mock_vizarr):
        """Test server mode."""
        from zarrnii import visualization
        
        mock_viewer = Mock()
        mock_vizarr.Viewer.return_value = mock_viewer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"
            zarr_path.mkdir()
            
            with patch('zarrnii.visualization.webbrowser.open') as mock_open:
                with patch('builtins.print') as mock_print:
                    result = visualization.visualize(
                        zarr_path=zarr_path,
                        mode="server",
                        port=8080,
                        open_browser=True
                    )
                    
                    assert result is None
                    mock_viewer.show.assert_called_once_with(port=8080)
                    mock_open.assert_called_once_with("http://localhost:8080")


class TestZarrNiiVisualization:
    """Test visualization through ZarrNii class."""

    def create_test_zarrnii(self) -> ZarrNii:
        """Create a test ZarrNii instance."""
        data = da.random.random((10, 20, 30), chunks=(5, 10, 15))
        return ZarrNii.from_darr(data, axes_order="ZYX", orientation="RAS")

    def test_visualize_method_exists(self):
        """Test that ZarrNii has visualize method."""
        znimg = self.create_test_zarrnii()
        assert hasattr(znimg, 'visualize')
        assert callable(znimg.visualize)

    @patch('zarrnii.visualization.vizarr')
    def test_zarrnii_visualize_html_mode(self, mock_vizarr):
        """Test ZarrNii.visualize in HTML mode."""
        # Mock vizarr
        mock_viewer = Mock()
        mock_viewer.to_html.return_value = "<html>test</html>"
        mock_vizarr.Viewer.return_value = mock_viewer
        
        znimg = self.create_test_zarrnii()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.html"
            
            with patch('zarrnii.visualization.webbrowser.open'):
                result = znimg.visualize(
                    mode="html",
                    output_path=output_path,
                    open_browser=False
                )
                
                assert result == str(output_path)
                assert output_path.exists()

    @patch('zarrnii.visualization.vizarr')
    def test_zarrnii_visualize_server_mode(self, mock_vizarr):
        """Test ZarrNii.visualize in server mode."""
        mock_viewer = Mock()
        mock_vizarr.Viewer.return_value = mock_viewer
        
        znimg = self.create_test_zarrnii()
        
        with patch('zarrnii.visualization.webbrowser.open'):
            with patch('builtins.print'):
                result = znimg.visualize(
                    mode="server",
                    port=8080,
                    open_browser=False
                )
                
                assert result is None
                mock_viewer.show.assert_called_once_with(port=8080)

    def test_zarrnii_visualize_without_vizarr(self):
        """Test that ZarrNii.visualize raises ImportError without vizarr."""
        znimg = self.create_test_zarrnii()
        
        with patch.dict("sys.modules", {"zarrnii.visualization.vizarr": None}):
            # Need to reload the visualization module
            import zarrnii.visualization
            import importlib
            importlib.reload(zarrnii.visualization)
            
            with pytest.raises(ImportError, match="vizarr is required"):
                znimg.visualize()

    @patch('zarrnii.visualization.vizarr')
    def test_zarrnii_visualize_cleanup_on_error(self, mock_vizarr):
        """Test that temporary files are cleaned up on error."""
        mock_vizarr.Viewer.side_effect = Exception("Test error")
        
        znimg = self.create_test_zarrnii()
        
        with pytest.raises(Exception, match="Test error"):
            znimg.visualize(mode="html")
        
        # Temp files should be cleaned up (hard to test directly due to tempfile cleanup)

    @patch('zarrnii.visualization.vizarr')
    def test_zarrnii_visualize_with_kwargs(self, mock_vizarr):
        """Test that kwargs are passed through to vizarr."""
        mock_viewer = Mock()
        mock_viewer.to_html.return_value = "<html>test</html>"
        mock_vizarr.Viewer.return_value = mock_viewer
        
        znimg = self.create_test_zarrnii()
        
        with patch('zarrnii.visualization.webbrowser.open'):
            znimg.visualize(
                mode="html",
                open_browser=False,
                # Custom kwargs that should be passed to vizarr
                name="test_image",
                colormap="viridis"
            )
            
            # Check that kwargs were passed to add_image
            mock_viewer.add_image.assert_called_once()
            call_args = mock_viewer.add_image.call_args
            assert 'name' in call_args.kwargs
            assert 'colormap' in call_args.kwargs
            assert call_args.kwargs['name'] == "test_image"
            assert call_args.kwargs['colormap'] == "viridis"


class TestVisualizationAvailability:
    """Test visualization availability checking."""

    def test_is_available_with_vizarr(self):
        """Test is_available returns True when vizarr is available."""
        from zarrnii import visualization
        
        with patch.object(visualization, 'vizarr', Mock()):
            assert visualization.is_available()

    def test_is_available_without_vizarr(self):
        """Test is_available returns False when vizarr is not available."""
        from zarrnii import visualization
        
        with patch.object(visualization, 'vizarr', None):
            assert not visualization.is_available()