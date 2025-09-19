"""Visualization module for ZarrNii using vizarr for interactive OME-Zarr viewing."""

import os
import tempfile
import webbrowser
from pathlib import Path
from typing import Optional, Union

try:
    import vizarr
except ImportError:
    vizarr = None


def visualize(
    zarr_path: Union[str, Path],
    mode: str = "html",
    output_path: Optional[Union[str, Path]] = None,
    port: int = 8080,
    open_browser: bool = True,
    **kwargs
) -> Union[str, None]:
    """
    Visualize OME-Zarr data using vizarr.

    This function provides interactive web-based visualization of OME-Zarr images
    using the vizarr library. It supports two modes:
    1. 'html': Generate a self-contained HTML file
    2. 'server': Start a local HTTP server

    Args:
        zarr_path: Path to the OME-Zarr dataset to visualize
        mode: Visualization mode - 'html' or 'server' (default: 'html')
        output_path: Output path for HTML file (only used in 'html' mode).
                    If None, creates a temporary file.
        port: Port number for HTTP server (only used in 'server' mode)
        open_browser: Whether to automatically open the visualization in browser
        **kwargs: Additional arguments passed to vizarr

    Returns:
        For 'html' mode: Path to the generated HTML file
        For 'server' mode: None (server runs until interrupted)

    Raises:
        ImportError: If vizarr is not installed
        ValueError: If invalid mode is specified
        FileNotFoundError: If zarr_path does not exist

    Examples:
        >>> # Generate HTML file
        >>> html_path = visualize("data.ome.zarr", mode="html")
        >>> print(f"Visualization saved to: {html_path}")

        >>> # Start HTTP server
        >>> visualize("data.ome.zarr", mode="server", port=8080)
    """
    if vizarr is None:
        raise ImportError(
            "vizarr is required for visualization. "
            "Install with: pip install zarrnii[viz]"
        )

    # Validate inputs
    zarr_path = Path(zarr_path)
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr path does not exist: {zarr_path}")

    if mode not in ["html", "server"]:
        raise ValueError(f"Invalid mode '{mode}'. Must be 'html' or 'server'")

    if mode == "html":
        return _generate_html(zarr_path, output_path, open_browser, **kwargs)
    elif mode == "server":
        return _start_server(zarr_path, port, open_browser, **kwargs)


def _generate_html(
    zarr_path: Path,
    output_path: Optional[Union[str, Path]],
    open_browser: bool,
    **kwargs
) -> str:
    """Generate a self-contained HTML file for visualization."""
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
        # Use vizarr to create the visualization
        viewer = vizarr.Viewer()
        viewer.add_image(source=str(zarr_path), **kwargs)
        
        # Generate HTML
        html_content = viewer.to_html()
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        if open_browser:
            webbrowser.open(f"file://{output_path.absolute()}")

        return str(output_path)

    except Exception as e:
        # Clean up temporary file on error
        if output_path.exists():
            output_path.unlink()
        raise RuntimeError(f"Failed to generate HTML visualization: {e}") from e


def _start_server(
    zarr_path: Path,
    port: int,
    open_browser: bool,
    **kwargs
) -> None:
    """Start a local HTTP server for visualization."""
    try:
        # Create viewer
        viewer = vizarr.Viewer()
        viewer.add_image(source=str(zarr_path), **kwargs)
        
        # Start server
        url = f"http://localhost:{port}"
        print(f"Starting vizarr server at: {url}")
        print("Press Ctrl+C to stop the server")
        
        if open_browser:
            webbrowser.open(url)
        
        # Start the server (this will block)
        viewer.show(port=port)
        
    except Exception as e:
        raise RuntimeError(f"Failed to start visualization server: {e}") from e


def is_available() -> bool:
    """Check if visualization functionality is available."""
    return vizarr is not None