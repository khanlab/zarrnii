#!/usr/bin/env python3
"""
Demonstration of visualization functionality in zarrnii.

This script shows how to:
1. Create or load OME-Zarr data 
2. Use the visualization features with vizarr
3. Handle different visualization modes
"""

import tempfile
import numpy as np
import dask.array as da
from pathlib import Path

import zarrnii


def main():
    """Demonstrate visualization functionality."""
    print("ZarrNii Visualization Demo")
    print("=" * 40)
    
    # Check if visualization is available
    from zarrnii import visualization
    if visualization is None:
        print("❌ Visualization not available. Install with: pip install zarrnii[viz]")
        return False
    
    print(f"✅ Visualization available: {visualization.is_available()}")
    
    # Create synthetic data
    print("\n1. Creating synthetic OME-Zarr data...")
    np.random.seed(42)
    
    # Create 3D data with some structure
    z, y, x = 20, 64, 64
    data = np.zeros((z, y, x), dtype=np.float32)
    
    # Add some interesting patterns
    for i in range(z):
        # Create concentric circles that change with z
        yy, xx = np.ogrid[:y, :x]
        center_y, center_x = y//2, x//2
        radius = 15 + i * 0.5
        mask = (yy - center_y)**2 + (xx - center_x)**2 < radius**2
        data[i, mask] = 0.8 + 0.2 * np.sin(i * 0.5)
        
        # Add some noise
        data[i] += 0.1 * np.random.random((y, x))
    
    # Convert to dask array
    dask_data = da.from_array(data, chunks=(10, 32, 32))
    znimg = zarrnii.ZarrNii.from_darr(dask_data, axes_order="ZYX", orientation="RAS")
    
    print(f"   Created ZarrNii with shape: {znimg.shape}")
    print(f"   Axes order: {znimg.axes_order}")
    print(f"   Orientation: {znimg.orientation}")

    # Test widget mode (primary use case)
    print("\n2. Testing widget mode (for Jupyter notebooks)...")
    try:
        widget = znimg.visualize(mode="widget")
        print(f"   ✅ Widget created: {type(widget)}")
        print(f"   Widget ready for display in Jupyter notebook")
        print(f"   In a notebook, simply run: widget")
    except Exception as e:
        print(f"   ❌ Widget mode failed: {e}")

    # Test HTML mode (generates informational page)
    print("\n3. Testing HTML mode (informational page)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        html_file = Path(tmpdir) / "visualization_demo.html"
        try:
            result = znimg.visualize(
                mode="html",
                output_path=html_file,
                open_browser=False
            )
            print(f"   ✅ HTML file generated: {result}")
            
            # Check file properties
            if html_file.exists():
                size = html_file.stat().st_size
                print(f"   File size: {size:,} bytes")
                
                # Show first few lines
                with open(html_file, 'r') as f:
                    lines = f.readlines()[:5]
                    print("   First few lines:")
                    for line in lines:
                        print(f"     {line.strip()}")
                        
        except Exception as e:
            print(f"   ❌ HTML mode failed: {e}")

    # Test server mode (should show limitation)
    print("\n4. Testing server mode (not supported in vizarr 0.1.0)...")
    try:
        znimg.visualize(mode="server")
        print("   ❌ Server mode should have failed")
    except NotImplementedError as e:
        print(f"   ✅ Server mode correctly unavailable: {e}")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")

    # Show recommended usage
    print("\n" + "=" * 40)
    print("📋 RECOMMENDED USAGE")
    print("=" * 40)
    print("""
For interactive visualization:

1. In Jupyter Notebook/Lab:
   ```python
   from zarrnii import ZarrNii
   
   # Load your data
   znimg = ZarrNii.from_ome_zarr("your_data.ome.zarr")
   
   # Create and display widget
   widget = znimg.visualize(mode="widget")
   widget  # This displays the interactive viewer
   ```

2. For standalone visualization, consider alternatives:
   - Napari with OME-Zarr plugin
   - ImageJ/Fiji with Bio-Formats
   - Custom matplotlib/plotly visualizations

3. The HTML mode provides usage instructions and dataset info.
""")
    
    print("\n✅ Demo completed successfully!")
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)