# MultiOtsuSegmentation Plugin

The `MultiOtsuSegmentation` plugin implements multi-level Otsu's automatic threshold selection method for multi-class image segmentation. It provides options to save intermediate results including histograms, computed thresholds, and visualization figures.

## Features

- **Multi-class segmentation**: Segment images into 2 or more classes
- **Histogram saving**: Save computed histogram as `.npz` file
- **Threshold saving**: Save computed thresholds as `.json` file with metadata
- **Visualization**: Generate and save histogram plot with threshold lines as `.svg`
- **Data retrieval**: Access computed histogram and thresholds programmatically

## Basic Usage

```python
from zarrnii.plugins.segmentation import MultiOtsuSegmentation
import numpy as np

# Create or load your image
image = np.random.random((100, 100))

# Create plugin instance
plugin = MultiOtsuSegmentation(classes=3, nbins=128)

# Perform segmentation
result = plugin.segment(image)

# Retrieve computed values
thresholds = plugin.get_thresholds()
histogram_data = plugin.get_histogram()
```

## Saving Results

```python
# Create plugin with file saving options
plugin = MultiOtsuSegmentation(
    classes=3,
    nbins=128,
    save_histogram="output/histogram.npz",
    save_thresholds="output/thresholds.json",
    save_figure="output/visualization.svg"
)

# Segment image - files will be saved automatically
result = plugin.segment(image)
```

## Parameters

- `classes` (int): Number of classes to segment into (default: 2, minimum: 2)
- `nbins` (int): Number of histogram bins (default: 256)
- `save_histogram` (str/Path, optional): Path to save histogram data as `.npz`
- `save_thresholds` (str/Path, optional): Path to save thresholds as `.json`
- `save_figure` (str/Path, optional): Path to save visualization as `.svg`

## Output Files

### Histogram File (.npz)
Contains two arrays:
- `counts`: Histogram bin counts
- `bin_edges`: Histogram bin edge values

```python
import numpy as np
data = np.load("histogram.npz")
counts = data['counts']
bin_edges = data['bin_edges']
```

### Thresholds File (.json)
Contains:
- `classes`: Number of classes
- `thresholds`: List of computed threshold values
- `min_value`: Minimum value in data range
- `max_value`: Maximum value in data range

```json
{
  "classes": 3,
  "thresholds": [0.333, 0.667],
  "min_value": 0.0,
  "max_value": 1.0
}
```

### Visualization File (.svg)
SVG figure showing:
- Histogram bars
- Vertical red dashed lines at threshold locations
- Legend with threshold values
- Axis labels and title

## Examples

### Binary Segmentation (2 classes)
```python
plugin = MultiOtsuSegmentation(classes=2, nbins=64)
result = plugin.segment(image)
# result contains values [0, 1]
```

### Multi-class Segmentation (3+ classes)
```python
plugin = MultiOtsuSegmentation(classes=4, nbins=128)
result = plugin.segment(image)
# result contains values [0, 1, 2, 3]
```

### Complete Workflow with Saving
```python
from zarrnii.plugins.segmentation import MultiOtsuSegmentation
from pathlib import Path
import numpy as np

# Prepare output directory
output_dir = Path("analysis_output")
output_dir.mkdir(exist_ok=True)

# Create plugin
plugin = MultiOtsuSegmentation(
    classes=3,
    nbins=128,
    save_histogram=output_dir / "histogram.npz",
    save_thresholds=output_dir / "thresholds.json",
    save_figure=output_dir / "histogram_viz.svg"
)

# Segment image
result = plugin.segment(image)

# Access computed values
thresholds = plugin.get_thresholds()
print(f"Computed thresholds: {thresholds}")

# Process segmented result
for class_id in range(plugin.classes):
    mask = result == class_id
    pixel_count = mask.sum()
    print(f"Class {class_id}: {pixel_count} pixels")
```

## Edge Cases

The plugin handles edge cases gracefully:

- **Constant images**: Returns all pixels as class 0
- **Insufficient unique values**: Assigns pixels to class 0
- **Missing matplotlib**: Figure saving is skipped silently if matplotlib is not installed
