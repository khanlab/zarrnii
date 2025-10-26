"""
Multi-level Otsu thresholding segmentation plugin.

This module implements multi-level Otsu's automatic threshold selection method
for multi-class image segmentation, with options to save intermediate histogram,
computed thresholds, and visualization figures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from zarrnii.analysis import compute_otsu_thresholds

from .base import SegmentationPlugin


class MultiOtsuSegmentation(SegmentationPlugin):
    """
    Multi-level Otsu thresholding segmentation plugin.

    This plugin uses multi-level Otsu's method to automatically determine optimal
    thresholds for multi-class image segmentation. The method finds thresholds
    that minimize intra-class variance across multiple classes.

    Parameters:
        classes: Number of classes to segment into (default: 2). Must be >= 2.
            For classes=k, returns k-1 thresholds creating k labeled regions.
        nbins: Number of bins for histogram computation (default: 256)
        save_histogram: Optional path to save histogram data. Saves as .npz file
            with 'counts' and 'bin_edges' arrays (default: None)
        save_thresholds: Optional path to save computed thresholds. Saves as .json
            file with threshold values (default: None)
        save_figure: Optional path to save visualization figure as SVG. Shows
            histogram with threshold lines (default: None)
    """

    def __init__(
        self,
        classes: int = 2,
        nbins: int = 256,
        save_histogram: Optional[Union[str, Path]] = None,
        save_thresholds: Optional[Union[str, Path]] = None,
        save_figure: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """
        Initialize multi-level Otsu segmentation plugin.

        Args:
            classes: Number of classes to segment into
            nbins: Number of bins for histogram computation
            save_histogram: Optional path to save histogram data (.npz)
            save_thresholds: Optional path to save thresholds (.json)
            save_figure: Optional path to save visualization figure (.svg)
            **kwargs: Additional parameters passed to parent class
        """
        super().__init__(
            classes=classes,
            nbins=nbins,
            save_histogram=save_histogram,
            save_thresholds=save_thresholds,
            save_figure=save_figure,
            **kwargs,
        )

        if classes < 2:
            raise ValueError("Number of classes must be >= 2")

        self.classes = classes
        self.nbins = nbins
        self.save_histogram = Path(save_histogram) if save_histogram else None
        self.save_thresholds = Path(save_thresholds) if save_thresholds else None
        self.save_figure = Path(save_figure) if save_figure else None

        # Store last computed values for inspection
        self._last_histogram_counts = None
        self._last_bin_edges = None
        self._last_thresholds = None

    def segment(
        self, image: np.ndarray, metadata: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Segment image using multi-level Otsu thresholding.

        Args:
            image: Input image as numpy array
            metadata: Optional metadata (unused in Otsu method)

        Returns:
            Labeled segmentation mask as numpy array with same shape as input.
            Values are 0, 1, 2, ..., classes-1 for each class.

        Raises:
            ValueError: If input image is empty or has invalid dimensions
        """
        if image.size == 0:
            raise ValueError("Input image is empty")

        if image.ndim < 2:
            raise ValueError("Input image must be at least 2D")

        # Store original shape for output
        original_shape = image.shape

        # Ensure image is in a suitable format for histogram computation
        if image.dtype == bool:
            image = image.astype(np.float32)

        # Compute histogram
        hist_counts, bin_edges = np.histogram(
            image.flatten(), bins=self.nbins, range=(image.min(), image.max())
        )

        # Store for optional saving
        self._last_histogram_counts = hist_counts
        self._last_bin_edges = bin_edges

        # Compute thresholds using the analysis module
        try:
            threshold_list = compute_otsu_thresholds(
                hist_counts, classes=self.classes, bin_edges=bin_edges
            )
            # Extract middle thresholds (excluding min and max)
            thresholds = threshold_list[1:-1]
        except ValueError as e:
            # Handle constant images or images with insufficient unique values
            if "different values" in str(e) or "cannot be thresholded" in str(e):
                # For constant/near-constant images where thresholding is impossible,
                # assign all pixels to class 0 (background) as a sensible default.
                # This is preferable to arbitrary thresholding since all pixels
                # have essentially the same value.
                self._last_thresholds = []
                self._save_outputs()
                return np.zeros(original_shape, dtype=np.uint8)
            else:
                raise

        self._last_thresholds = thresholds

        # Save outputs if requested
        self._save_outputs()

        # Apply thresholds to create labeled regions
        # Use np.digitize for proper multi-class classification:
        # - It assigns each value to exactly one bin/class based on threshold intervals
        # - For thresholds [t1, t2], values are classified as:
        #   * class 0: x < t1
        #   * class 1: t1 <= x < t2
        #   * class 2: x >= t2
        # This avoids the issue of overlapping assignments from iterative thresholding
        result = np.digitize(image, bins=thresholds).astype(np.uint8)

        return result

    def _save_outputs(self) -> None:
        """Save histogram, thresholds, and figure if paths are provided."""
        if self.save_histogram and self._last_histogram_counts is not None:
            np.savez(
                self.save_histogram,
                counts=self._last_histogram_counts,
                bin_edges=self._last_bin_edges,
            )

        if self.save_thresholds and self._last_thresholds is not None:
            threshold_data = {
                "classes": self.classes,
                "thresholds": [float(t) for t in self._last_thresholds],
                "min_value": float(self._last_bin_edges[0]),
                "max_value": float(self._last_bin_edges[-1]),
            }
            with open(self.save_thresholds, "w") as f:
                json.dump(threshold_data, f, indent=2)

        if self.save_figure and self._last_histogram_counts is not None:
            self._save_visualization()

    def _save_visualization(self) -> None:
        """Create and save histogram visualization with threshold lines as SVG."""
        try:
            import matplotlib

            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            # Silently skip if matplotlib not available
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histogram
        bin_centers = (self._last_bin_edges[:-1] + self._last_bin_edges[1:]) / 2
        ax.bar(
            bin_centers,
            self._last_histogram_counts,
            width=(self._last_bin_edges[1] - self._last_bin_edges[0]),
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )

        # Plot threshold lines
        if self._last_thresholds is not None:
            for i, threshold in enumerate(self._last_thresholds):
                ax.axvline(
                    threshold,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Threshold {i+1}: {threshold:.3f}",
                )

        ax.set_xlabel("Intensity Value")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Multi-Otsu Histogram ({self.classes} classes)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save as SVG
        fig.savefig(self.save_figure, format="svg", bbox_inches="tight")
        plt.close(fig)

    @property
    def name(self) -> str:
        """Return the name of the segmentation algorithm."""
        if self.classes == 2:
            return "Binary Otsu Thresholding"
        else:
            return f"Multi-level Otsu Thresholding ({self.classes} classes)"

    @property
    def description(self) -> str:
        """Return a description of the segmentation algorithm."""
        return (
            f"Multi-level Otsu's automatic threshold selection method for "
            f"{self.classes}-class segmentation. Finds {self.classes-1} thresholds "
            f"that minimize intra-class variance across classes. "
            f"Uses {self.nbins} histogram bins for computation."
        )

    def get_thresholds(self) -> Optional[List[float]]:
        """
        Get the last computed threshold values.

        Returns:
            List of threshold values from last segmentation, or None if not yet computed
        """
        return (
            self._last_thresholds.copy() if self._last_thresholds is not None else None
        )

    def get_histogram(self) -> Optional[tuple]:
        """
        Get the last computed histogram data.

        Returns:
            Tuple of (counts, bin_edges) from last segmentation,
            or None if not yet computed
        """
        if self._last_histogram_counts is not None:
            return (
                self._last_histogram_counts.copy(),
                self._last_bin_edges.copy(),
            )
        return None
