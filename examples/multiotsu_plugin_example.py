#!/usr/bin/env python
"""
Example demonstrating the MultiOtsuSegmentation plugin with histogram and threshold saving.

This script shows how to use the MultiOtsuSegmentation plugin to:
1. Segment an image using multi-level Otsu thresholding
2. Save the histogram data to a file
3. Save the computed thresholds to a JSON file
4. Save a visualization figure as SVG
"""

import tempfile
from pathlib import Path

import numpy as np

from zarrnii.plugins.segmentation import MultiOtsuSegmentation

# Create a synthetic trimodal image
np.random.seed(42)
image = np.concatenate(
    [
        np.random.normal(0.2, 0.05, 1000),  # Dark pixels
        np.random.normal(0.5, 0.05, 1000),  # Medium pixels
        np.random.normal(0.8, 0.05, 1000),  # Bright pixels
    ]
).reshape(30, 100)

print("Created synthetic trimodal image with shape:", image.shape)
print("Image intensity range:", image.min(), "-", image.max())
print()

# Create temporary directory for output files
with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)

    # Initialize the plugin with file saving options
    plugin = MultiOtsuSegmentation(
        classes=3,  # Segment into 3 classes
        nbins=128,  # Use 128 histogram bins
        save_histogram=tmpdir / "histogram.npz",
        save_thresholds=tmpdir / "thresholds.json",
        save_figure=tmpdir / "histogram_visualization.svg",
    )

    print("Plugin initialized with:")
    print(f"  - Name: {plugin.name}")
    print(f"  - Description: {plugin.description}")
    print()

    # Perform segmentation
    print("Performing segmentation...")
    result = plugin.segment(image)

    print("Segmentation complete!")
    print(f"  - Result shape: {result.shape}")
    print(f"  - Result dtype: {result.dtype}")
    print(f"  - Classes found: {np.unique(result)}")
    print()

    # Retrieve computed values
    thresholds = plugin.get_thresholds()
    hist_data = plugin.get_histogram()

    print("Computed thresholds:", thresholds)
    print(
        f"Histogram data: {hist_data[0].shape[0]} bins, "
        f"{hist_data[0].sum()} total counts"
    )
    print()

    # Verify saved files
    print("Saved files:")
    for filepath in [
        tmpdir / "histogram.npz",
        tmpdir / "thresholds.json",
        tmpdir / "histogram_visualization.svg",
    ]:
        if filepath.exists():
            print(f"  ✓ {filepath.name} ({filepath.stat().st_size} bytes)")
        else:
            print(f"  ✗ {filepath.name} (not found)")

print()
print("Example completed successfully!")
