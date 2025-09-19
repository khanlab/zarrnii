"""Visualization module for ZarrNii using vizarr and VolumeViewer for interactive OME-Zarr viewing."""

import os
import socket
import tempfile
import threading
import time
import webbrowser
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Optional, Union

try:
    import vizarr
except ImportError:
    vizarr = None


class CORSHTTPRequestHandler(SimpleHTTPRequestHandler):
    """HTTP request handler that enables CORS for cross-origin requests."""

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.end_headers()


def _find_free_port() -> int:
    """Find a free port for the HTTP server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def visualize(
    zarr_path: Union[str, Path],
    mode: str = "vol",
    output_path: Optional[Union[str, Path]] = None,
    port: int = 8080,
    open_browser: bool = True,
    **kwargs,
) -> Union[str, None, "vizarr.Viewer"]:
    """
    Visualize OME-Zarr data using vizarr or volumeviewer.

    This function provides interactive web-based visualization of OME-Zarr images
    using different visualization backends.

    Args:
        zarr_path: Path to the OME-Zarr dataset to visualize
        mode: Visualization mode - 'widget', 'vol', or 'server' (default: 'vol')
              - 'widget': Return vizarr widget (for Jupyter notebooks)
              - 'vol': Open dataset in VolumeViewer web viewer
              - 'server': Not supported in vizarr 0.1.0
        output_path: Not used (deprecated, kept for compatibility)
        port: Port number for HTTP server (used in 'vol' mode)
        open_browser: Whether to automatically open the visualization in browser
        **kwargs: Additional arguments passed to visualization backend

    Returns:
        For 'widget' mode: vizarr.Viewer widget object
        For 'vol' mode: URL to the VolumeViewer
        For 'server' mode: None (not supported)

    Raises:
        ImportError: If vizarr is not installed (for widget mode)
        ValueError: If invalid mode is specified
        FileNotFoundError: If zarr_path does not exist
        NotImplementedError: If requested mode is not supported

    Examples:
        >>> # Open in VolumeViewer web viewer (default)
        >>> url = visualize("data.ome.zarr")
        >>> print(f"View at: {url}")

        >>> # Return widget for Jupyter notebook
        >>> widget = visualize("data.ome.zarr", mode="widget")
        >>> widget  # Display in notebook
    """
    # Validate inputs
    zarr_path = Path(zarr_path)
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr path does not exist: {zarr_path}")

    if mode not in ["widget", "vol", "server"]:
        raise ValueError(
            f"Invalid mode '{mode}'. Must be 'widget', 'vol', or 'server'"
        )

    if mode == "widget" and vizarr is None:
        raise ImportError(
            "vizarr is required for widget visualization. "
            "Install with: pip install zarrnii[viz]"
        )

    if mode == "widget":
        return _create_widget(zarr_path, **kwargs)
    elif mode == "vol":
        return _launch_volumeviewer(zarr_path, port, open_browser, **kwargs)
    elif mode == "server":
        raise NotImplementedError(
            "Server mode is not supported in vizarr 0.1.0. "
            "Use 'widget' mode in Jupyter notebooks or 'vol' mode for web viewing."
        )


def _launch_volumeviewer(zarr_path: Path, port: int, open_browser: bool, **kwargs) -> str:
    """Launch VolumeViewer web viewer with the OME-Zarr dataset."""
    # Find available port if specified port is in use
    if port == 8080:  # Default port
        port = _find_free_port()

    # Start HTTP server in a separate thread
    server_dir = zarr_path.parent

    # Wrap your handler with the directory you want
    handler = partial(CORSHTTPRequestHandler, directory=server_dir)
    httpd = HTTPServer(("localhost", port), handler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    # Give server time to start
    time.sleep(1)

    # Build URLs
    dataset_filename = zarr_path.name
    local_url = f"http://localhost:{port}/{dataset_filename}"
    volumeviewer_url = f"http://volumeviewer.allencell.org/viewer?url={local_url}"

    print(f"🚀 Starting HTTP server on port {port}")
    print(f"📁 Serving directory: {server_dir}")
    print(f"🔗 Dataset URL: {local_url}")
    print(f"👁️  VolumeViewer: {volumeviewer_url}")
    print(f"⚠️  Keep this Python session running to maintain the server")

    if open_browser:
        print(f"🌐 Opening VolumeViewer in browser...")
        webbrowser.open(volumeviewer_url)

    # Store server reference for potential cleanup
    if not hasattr(_launch_volumeviewer, "_servers"):
        _launch_volumeviewer._servers = []
    _launch_volumeviewer._servers.append(httpd)

    return volumeviewer_url


def _create_widget(zarr_path: Path, **kwargs) -> "vizarr.Viewer":
    """Create a vizarr widget for use in Jupyter notebooks."""
    try:
        viewer = vizarr.Viewer()
        # Convert path to file:// URL to ensure proper handling by the browser
        zarr_url = zarr_path.absolute().as_uri()
        viewer.add_image(source=zarr_url, **kwargs)
        return viewer
    except Exception as e:
        raise RuntimeError(f"Failed to create vizarr widget: {e}") from e



def is_available() -> bool:
    """Check if visualization functionality is available."""
    return vizarr is not None


def stop_servers():
    """Stop all running HTTP servers started by vol mode."""
    if hasattr(_launch_volumeviewer, "_servers"):
        for server in _launch_volumeviewer._servers:
            try:
                server.shutdown()
                server.server_close()
            except:
                pass
        _launch_volumeviewer._servers.clear()
        print("🛑 All HTTP servers stopped")
    else:
        print("ℹ️  No active servers to stop")
