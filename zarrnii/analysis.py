"""
Image analysis functions for zarrnii.

This module provides functions for image analysis operations such as
histogram computation and threshold calculation.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

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
    return_histogram: bool = False,
    return_figure: bool = False,
) -> Union[
    List[float],
    Tuple[List[float], Tuple[np.ndarray, np.ndarray]],
    Tuple[List[float], Any],
]:
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
        return_histogram: If True, returns a tuple containing thresholds and
            histogram data (histogram_counts, bin_edges) (default: False)
        return_figure: If True, returns a tuple containing thresholds and a
            matplotlib figure with the histogram and annotated threshold lines
            (default: False). Cannot be combined with return_histogram=True.

    Returns:
        If both return_histogram and return_figure are False (default):
            List of threshold values. For classes=k, returns k+1 values:
            [min_value, threshold1, threshold2, ..., threshold_k-1, max_value]
            where min_value and max_value are the data range bounds.

        If return_histogram is True:
            Tuple of (thresholds, (histogram_counts, bin_edges)) where
            histogram_counts is a numpy array of bin counts and bin_edges
            is a numpy array of bin edge values.

        If return_figure is True:
            Tuple of (thresholds, figure) where figure is a matplotlib Figure
            object showing the histogram with annotated threshold lines.

    Raises:
        ValueError: If classes < 2 or if histogram is empty, or if both
            return_histogram and return_figure are True

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
        >>>
        >>> # Return histogram data along with thresholds
        >>> thresholds, (hist, bin_edges) = compute_otsu_thresholds(
        ...     hist, classes=2, return_histogram=True
        ... )
        >>>
        >>> # Generate a figure with annotated thresholds
        >>> thresholds, fig = compute_otsu_thresholds(
        ...     hist, classes=2, return_figure=True
        ... )
        >>> fig.savefig('otsu_thresholds.png')
    """
    if classes < 2:
        raise ValueError("Number of classes must be >= 2")

    if return_histogram and return_figure:
        raise ValueError(
            "Cannot return both histogram and figure. "
            "Set only one of return_histogram or return_figure to True."
        )

    # Convert dask arrays to numpy if needed
    if hasattr(histogram_counts, "compute"):
        histogram_counts = histogram_counts.compute()
    if bin_edges is not None and hasattr(bin_edges, "compute"):
        bin_edges = bin_edges.compute()

    if len(histogram_counts) == 0:
        raise ValueError("Histogram is empty")

    # Use scikit-image's threshold_multiotsu with histogram directly
    # threshold_multiotsu accepts hist parameter as
    # (histogram_counts, bin_centers) tuple
    # This avoids reconstructing millions of data points from histogram
    if bin_edges is not None:
        # Calculate bin centers from edges
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        min_val = float(bin_edges[0])
        max_val = float(bin_edges[-1])

        # Check if histogram has any data
        if histogram_counts.sum() == 0:
            # Empty histogram case
            mid_thresholds = np.linspace(min_val, max_val, classes + 1)[1:-1].tolist()
        else:
            # Pass histogram and bin centers directly to threshold_multiotsu
            # This is memory-efficient as it doesn't reconstruct data points
            # The hist parameter expects a 2-tuple: (histogram_counts, bin_centers)
            otsu_thresholds = threshold_multiotsu(
                hist=(histogram_counts, bin_centers), classes=classes
            )
            mid_thresholds = otsu_thresholds.tolist()
    else:
        # If no bin_edges provided, work in histogram bin index space
        # Create bin centers as integer indices (0 to len-1)
        bin_centers = np.arange(len(histogram_counts))
        min_val = 0.0
        # max_val is set to len(histogram_counts) for backward compatibility
        # This represents the upper bound of the histogram range
        max_val = float(len(histogram_counts))

        # Check if histogram has any data
        if histogram_counts.sum() == 0:
            # Empty histogram case
            mid_thresholds = np.linspace(min_val, max_val, classes + 1)[1:-1].tolist()
        else:
            # Pass histogram and bin centers directly to threshold_multiotsu
            # The hist parameter expects a 2-tuple: (histogram_counts, bin_centers)
            otsu_thresholds = threshold_multiotsu(
                hist=(histogram_counts, bin_centers), classes=classes
            )
            mid_thresholds = otsu_thresholds.tolist()

    # Format as requested in the issue: [min, threshold1, ..., threshold_k-1, max]
    result = [min_val] + mid_thresholds + [max_val]

    # Handle different return options
    if return_histogram:
        # Return thresholds and histogram data
        return result, (
            histogram_counts,
            (
                bin_edges
                if bin_edges is not None
                else np.arange(len(histogram_counts) + 1)
            ),
        )
    elif return_figure:
        # Import matplotlib here to avoid requiring it as a hard dependency
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for return_figure=True. "
                "Install it with: pip install matplotlib"
            )

        # Create figure with histogram and threshold annotations
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histogram
        if bin_edges is not None:
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ax.bar(
                bin_centers,
                histogram_counts,
                width=np.diff(bin_edges),
                alpha=0.7,
                color="skyblue",
                edgecolor="black",
                label="Histogram",
            )
        else:
            ax.bar(
                np.arange(len(histogram_counts)),
                histogram_counts,
                alpha=0.7,
                color="skyblue",
                edgecolor="black",
                label="Histogram",
            )

        # Plot threshold lines (exclude min and max)
        colors = plt.cm.Set1(np.linspace(0, 1, len(mid_thresholds)))
        for i, (thresh, color) in enumerate(zip(mid_thresholds, colors)):
            ax.axvline(
                thresh,
                color=color,
                linestyle="--",
                linewidth=2,
                label=f"Threshold {i+1}: {thresh:.3f}",
            )

        ax.set_xlabel("Intensity Value", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(f"Histogram with {classes}-class Otsu Thresholds", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        return result, fig
    else:
        # Default: return only thresholds
        return result
