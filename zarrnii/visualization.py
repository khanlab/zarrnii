"""Visualization module for ZarrNii using vizarr and avivator for interactive OME-Zarr viewing."""

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
    mode: str = "widget",
    output_path: Optional[Union[str, Path]] = None,
    port: int = 8080,
    open_browser: bool = True,
    **kwargs,
) -> Union[str, None, "vizarr.Viewer"]:
    """
    Visualize OME-Zarr data using vizarr or avivator.

    This function provides interactive web-based visualization of OME-Zarr images
    using different visualization backends.

    Args:
        zarr_path: Path to the OME-Zarr dataset to visualize
        mode: Visualization mode - 'widget', 'html', 'avivator', or 'server' (default: 'widget')
              - 'widget': Return vizarr widget (for Jupyter notebooks)
              - 'html': Generate informational HTML file (limited functionality)
              - 'avivator': Open dataset in Avivator web viewer
              - 'server': Not supported in vizarr 0.1.0
        output_path: Output path for HTML file (only used in 'html' mode).
                    If None, creates a temporary file.
        port: Port number for HTTP server (used in 'avivator' mode)
        open_browser: Whether to automatically open the visualization in browser
        **kwargs: Additional arguments passed to visualization backend

    Returns:
        For 'widget' mode: vizarr.Viewer widget object
        For 'html' mode: Path to the generated HTML file (if successful)
        For 'avivator' mode: URL to the Avivator viewer
        For 'server' mode: None (not supported)

    Raises:
        ImportError: If vizarr is not installed (for widget/html modes)
        ValueError: If invalid mode is specified
        FileNotFoundError: If zarr_path does not exist
        NotImplementedError: If requested mode is not supported

    Examples:
        >>> # Return widget for Jupyter notebook
        >>> widget = visualize("data.ome.zarr", mode="widget")
        >>> widget  # Display in notebook

        >>> # Open in Avivator web viewer
        >>> url = visualize("data.ome.zarr", mode="avivator")
        >>> print(f"View at: {url}")

        >>> # Generate informational HTML
        >>> html_path = visualize("data.ome.zarr", mode="html")
    """
    # Validate inputs
    zarr_path = Path(zarr_path)
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr path does not exist: {zarr_path}")

    if mode not in ["widget", "html", "server", "avivator"]:
        raise ValueError(
            f"Invalid mode '{mode}'. Must be 'widget', 'html', 'avivator', or 'server'"
        )

    if mode in ["widget", "html"] and vizarr is None:
        raise ImportError(
            "vizarr is required for visualization. "
            "Install with: pip install zarrnii[viz]"
        )

    if mode == "widget":
        return _create_widget(zarr_path, **kwargs)
    elif mode == "html":
        return _generate_html_fallback(zarr_path, output_path, open_browser, **kwargs)
    elif mode == "avivator":
        return _launch_avivator(zarr_path, port, open_browser, **kwargs)
    elif mode == "server":
        raise NotImplementedError(
            "Server mode is not supported in vizarr 0.1.0. "
            "Use 'widget' mode in Jupyter notebooks or 'avivator' mode for web viewing."
        )


def _launch_avivator(zarr_path: Path, port: int, open_browser: bool, **kwargs) -> str:
    """Launch Avivator web viewer with the OME-Zarr dataset."""
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
    avivator_url = f"http://volumeviewer.allencell.org/viewer?url={local_url}"

    print(f"🚀 Starting HTTP server on port {port}")
    print(f"📁 Serving directory: {server_dir}")
    print(f"🔗 Dataset URL: {local_url}")
    print(f"👁️  Avivator viewer: {avivator_url}")
    print(f"⚠️  Keep this Python session running to maintain the server")

    if open_browser:
        print(f"🌐 Opening Avivator in browser...")
        webbrowser.open(avivator_url)

    # Store server reference for potential cleanup
    if not hasattr(_launch_avivator, "_servers"):
        _launch_avivator._servers = []
    _launch_avivator._servers.append(httpd)

    return avivator_url


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


def _generate_html_fallback(
    zarr_path: Path,
    output_path: Optional[Union[str, Path]],
    open_browser: bool,
    **kwargs,
) -> str:
    """
    Generate HTML fallback using widget embedding.

    Note: This is a fallback implementation since vizarr 0.1.0 doesn't
    directly support HTML generation. This creates a basic HTML page
    that embeds the widget, but may have limited functionality.
    """
    if output_path is None:
        # Create temporary HTML file
        fd, output_path = tempfile.mkstemp(suffix=".html", prefix="zarrnii_viz_")
        os.close(fd)  # Close file descriptor, but keep the path
    else:
        output_path = Path(output_path)
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path = Path(output_path)

    try:
        # Create widget
        viewer = vizarr.Viewer()
        # Convert path to file:// URL to ensure proper handling by the browser
        zarr_url = zarr_path.absolute().as_uri()
        viewer.add_image(source=zarr_url, **kwargs)

        # Generate a basic HTML page with instructions
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>ZarrNii Visualization</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
        }}
        .info {{
            background-color: #f0f8ff;
            border: 1px solid #0066cc;
            border-radius: 5px;
            padding: 20px;
            margin: 20px 0;
        }}
        .error {{
            background-color: #fff0f0;
            border: 1px solid #cc0000;
            border-radius: 5px;
            padding: 20px;
            margin: 20px 0;
        }}
        code {{
            background-color: #f5f5f5;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ZarrNii Visualization</h1>

        <div class="info">
            <h3>Visualization Not Available in HTML Mode</h3>
            <p>The vizarr library (version 0.1.0) is designed primarily for Jupyter notebooks
            and doesn't support standalone HTML generation.</p>

            <h4>Recommended Usage:</h4>
            <ol>
                <li><strong>Jupyter Notebook:</strong> Use the widget mode for interactive visualization:
                    <br><code>widget = znimg.visualize(mode="widget")</code>
                    <br>Then display the widget in your notebook.
                </li>
                <li><strong>Web Browser:</strong> Use Avivator for full-featured web viewing:
                    <br><code>url = znimg.visualize(mode="avivator")</code>
                    <br>This will start a local server and open Avivator.
                </li>
                <li><strong>Alternative viewers:</strong> Consider using other OME-Zarr viewers like:
                    <ul>
                        <li>Napari with OME-Zarr plugin</li>
                        <li>ImageJ/Fiji with Bio-Formats</li>
                        <li>Direct zarr array visualization with matplotlib</li>
                    </ul>
                </li>
            </ol>
        </div>

        <div class="error">
            <h4>Dataset Information:</h4>
            <p><strong>Zarr Path:</strong> {zarr_path}</p>
            <p><strong>File exists:</strong> {zarr_path.exists()}</p>
            <p>To view this dataset, please use a Jupyter notebook with the vizarr widget or Avivator mode.</p>
        </div>

        <h3>Example Usage:</h3>
        <pre><code>from zarrnii import ZarrNii

# Load your data
znimg = ZarrNii.from_ome_zarr("{zarr_path}")

# For Jupyter notebooks
widget = znimg.visualize(mode="widget")
widget

# For web browser viewing
url = znimg.visualize(mode="avivator")
print(f"View at: {{url}}")</code></pre>
    </div>
</body>
</html>"""

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        if open_browser:
            webbrowser.open(f"file://{output_path.absolute()}")

        return str(output_path)

    except Exception as e:
        # Clean up temporary file on error
        if output_path.exists():
            output_path.unlink()
        raise RuntimeError(f"Failed to generate HTML fallback: {e}") from e


def is_available() -> bool:
    """Check if visualization functionality is available."""
    return vizarr is not None


def stop_servers():
    """Stop all running HTTP servers started by avivator mode."""
    if hasattr(_launch_avivator, "_servers"):
        for server in _launch_avivator._servers:
            try:
                server.shutdown()
                server.server_close()
            except:
                pass
        _launch_avivator._servers.clear()
        print("🛑 All HTTP servers stopped")
    else:
        print("ℹ️  No active servers to stop")
