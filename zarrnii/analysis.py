"""
Image analysis functions for zarrnii.

This module provides functions for image analysis operations such as
histogram computation and threshold calculation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
from skimage.filters import threshold_multiotsu


def compute_histogram(
    image: da.Array,
    bins: int = 256,
    range: Optional[Tuple[float, float]] = None,
    mask: Optional[da.Array] = None,
    **kwargs: Any,
) -> Tuple[da.Array, da.Array]:
    """
    Compute histogram of a dask array image.

    This function computes the histogram of image intensities, optionally using
    a mask to weight the computation. The histogram is computed using dask for
    efficient processing of large datasets.

    Args:
        image: Input dask array image
        bins: Number of histogram bins (default: 256)
        range: Optional tuple (min, max) defining histogram range. If None,
            uses the full range of the data
        mask: Optional dask array mask of same shape as image. Only pixels
            where mask > 0 are included in histogram computation
        **kwargs: Additional arguments passed to dask.array.histogram

    Returns:
        Tuple of (histogram_counts, bin_edges) where:
        - histogram_counts: dask array of histogram bin counts
        - bin_edges: dask array of bin edge values (length = bins + 1)

    Examples:
        >>> import dask.array as da
        >>> from zarrnii import compute_histogram
        >>>
        >>> # Create test image
        >>> image = da.random.random((100, 100, 100), chunks=(50, 50, 50))
        >>>
        >>> # Compute histogram
        >>> hist, bin_edges = compute_histogram(image, bins=128)
        >>>
        >>> # Compute histogram with mask
        >>> mask = image > 0.5
        >>> hist_masked, _ = compute_histogram(image, mask=mask)
    """
    if mask is not None:
        if mask.shape != image.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match image shape {image.shape}"
            )
        # Apply mask by flattening both arrays and selecting valid pixels
        flat_image = image.flatten()
        flat_mask = mask.flatten()
        # Only include pixels where mask > 0
        valid_indices = flat_mask > 0
        valid_data = flat_image[valid_indices]

        # For dask histogram, we need to provide a range
        if range is None:
            data_min = da.min(valid_data).compute()
            data_max = da.max(valid_data).compute()
            range = (data_min, data_max)
        return da.histogram(valid_data, bins=bins, range=range, **kwargs)
    else:
        # For dask histogram, we need to provide a range
        if range is None:
            data_min = da.min(image).compute()
            data_max = da.max(image).compute()
            range = (data_min, data_max)
        return da.histogram(image, bins=bins, range=range, **kwargs)


def compute_otsu_thresholds(
    histogram_counts: Union[np.ndarray, da.Array],
    classes: int = 2,
    bin_edges: Optional[Union[np.ndarray, da.Array]] = None,
) -> List[float]:
    """
    Compute Otsu multi-level thresholds from histogram data.

    This function uses scikit-image's threshold_multiotsu to compute optimal
    threshold values that separate the histogram into the specified number of
    classes.

    Args:
        histogram_counts: Histogram bin counts as numpy or dask array
        classes: Number of classes to separate data into (default: 2).
            Must be >= 2. For classes=2, returns 1 threshold. For classes=k,
            returns k-1 thresholds.
        bin_edges: Optional bin edges corresponding to histogram. If provided,
            used to determine the min/max range for the output format. If None,
            assumes bin edges from 0 to len(histogram_counts)

    Returns:
        List of threshold values. For classes=k, returns k+1 values:
        [min_value, threshold1, threshold2, ..., threshold_k-1, max_value]
        where min_value and max_value are the data range bounds.

    Raises:
        ValueError: If classes < 2 or if histogram is empty

    Examples:
        >>> import numpy as np
        >>> from zarrnii import compute_otsu_thresholds
        >>>
        >>> # Create sample histogram (bimodal distribution)
        >>> hist = np.array([100, 50, 20, 5, 2, 5, 20, 50, 100])
        >>>
        >>> # Compute binary threshold (2 classes)
        >>> thresholds = compute_otsu_thresholds(hist, classes=2)
        >>> print(f"Binary thresholds: {thresholds}")
        >>>
        >>> # Compute multi-level thresholds (3 classes)
        >>> thresholds = compute_otsu_thresholds(hist, classes=3)
        >>> print(f"Multi-level thresholds: {thresholds}")
    """
    if classes < 2:
        raise ValueError("Number of classes must be >= 2")

    # Convert dask arrays to numpy if needed
    if hasattr(histogram_counts, "compute"):
        histogram_counts = histogram_counts.compute()
    if bin_edges is not None and hasattr(bin_edges, "compute"):
        bin_edges = bin_edges.compute()

    if len(histogram_counts) == 0:
        raise ValueError("Histogram is empty")

    # Use scikit-image's threshold_multiotsu
    # When using hist parameter, thresholds are returned in the same scale as the data
    # We need to reconstruct the intensity values from bin centers
    if bin_edges is not None:
        # Calculate bin centers from edges
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # threshold_multiotsu with hist returns indices into the histogram
        threshold_indices = threshold_multiotsu(hist=histogram_counts, classes=classes)
        # Convert indices to actual intensity values using bin centers
        # But threshold_multiotsu actually returns intensity values when hist is provided
        # Let's use it directly on the reconstructed data

        # Reconstruct data from histogram for threshold_multiotsu
        data_points = []
        for i, count in enumerate(histogram_counts):
            if count > 0:
                # Add 'count' number of points at bin_centers[i]
                data_points.extend([bin_centers[i]] * int(count))

        if len(data_points) == 0:
            # Empty histogram case
            min_val = float(bin_edges[0])
            max_val = float(bin_edges[-1])
            # Return evenly spaced thresholds
            mid_thresholds = np.linspace(min_val, max_val, classes + 1)[1:-1].tolist()
        else:
            data_array = np.array(data_points)
            otsu_thresholds = threshold_multiotsu(data_array, classes=classes)
            mid_thresholds = otsu_thresholds.tolist()

        min_val = float(bin_edges[0])
        max_val = float(bin_edges[-1])
    else:
        # If no bin_edges provided, work in histogram bin index space
        # Reconstruct data points from histogram
        data_points = []
        for i, count in enumerate(histogram_counts):
            if count > 0:
                data_points.extend([i] * int(count))

        if len(data_points) == 0:
            # Empty histogram case
            min_val = 0.0
            max_val = float(len(histogram_counts))
            # Return evenly spaced thresholds
            mid_thresholds = np.linspace(min_val, max_val, classes + 1)[1:-1].tolist()
        else:
            data_array = np.array(data_points)
            otsu_thresholds = threshold_multiotsu(data_array, classes=classes)
            mid_thresholds = otsu_thresholds.tolist()

        min_val = 0.0
        max_val = float(len(histogram_counts))

    # Format as requested in the issue: [min, threshold1, ..., threshold_k-1, max]
    result = [min_val] + mid_thresholds + [max_val]
    return result
