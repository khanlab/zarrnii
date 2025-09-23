"""
Threshold segmentation plugin.

This module implements threshold-based segmentation that can use either
manual threshold values or computed thresholds (e.g., from Otsu analysis).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np

from .base import SegmentationPlugin


class ThresholdSegmentation(SegmentationPlugin):
    """
    Threshold-based segmentation plugin.

    This plugin applies threshold-based segmentation using either a single
    threshold value or multiple threshold values to create labeled regions.
    It can use manually specified thresholds or thresholds computed from
    analysis functions like Otsu multi-thresholding.

    Parameters:
        thresholds: Single threshold value or list of threshold values.
            For single threshold, creates binary segmentation (0/1).
            For multiple thresholds, creates multi-class segmentation (0/1/2/...).
        inclusive: Whether thresholds are inclusive (default: True).
            If True, pixels >= threshold are labeled as foreground.
            If False, pixels > threshold are labeled as foreground.
    """

    def __init__(
        self, thresholds: Union[float, List[float]], inclusive: bool = True, **kwargs
    ):
        """
        Initialize threshold segmentation plugin.

        Args:
            thresholds: Single threshold or list of thresholds
            inclusive: Whether thresholds are inclusive (>= vs >)
            **kwargs: Additional parameters passed to parent class
        """
        super().__init__(thresholds=thresholds, inclusive=inclusive, **kwargs)

        # Normalize thresholds to always be a list
        if isinstance(thresholds, (int, float)):
            self.thresholds = [float(thresholds)]
        else:
            self.thresholds = [float(t) for t in thresholds]

        # Sort thresholds to ensure proper ordering
        self.thresholds = sorted(self.thresholds)
        self.inclusive = inclusive

    def segment(
        self, image: np.ndarray, metadata: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Segment image using threshold values.

        Args:
            image: Input image as numpy array
            metadata: Optional metadata (unused in threshold method)

        Returns:
            Labeled segmentation mask as numpy array with same shape as input.
            Values are 0 (background), 1, 2, ... up to len(thresholds) classes.

        Raises:
            ValueError: If input image is empty
        """
        if image.size == 0:
            raise ValueError("Input image is empty")

        # Initialize result with zeros (background class)
        result = np.zeros(image.shape, dtype=np.uint8)

        # Apply thresholds to create labeled regions
        for i, threshold in enumerate(self.thresholds):
            if self.inclusive:
                mask = image >= threshold
            else:
                mask = image > threshold

            # Assign class label (i+1) to pixels above threshold
            result[mask] = i + 1

        return result

    @property
    def name(self) -> str:
        """Return the name of the segmentation algorithm."""
        if len(self.thresholds) == 1:
            return "Binary Threshold"
        else:
            return "Multi-level Threshold"

    @property
    def description(self) -> str:
        """Return a description of the segmentation algorithm."""
        if len(self.thresholds) == 1:
            op = ">=" if self.inclusive else ">"
            return (
                f"Binary threshold segmentation using threshold = {self.thresholds[0]}. "
                f"Pixels {op} threshold are labeled as foreground (1), others as background (0)."
            )
        else:
            op = ">=" if self.inclusive else ">"
            return (
                f"Multi-level threshold segmentation using {len(self.thresholds)} thresholds: "
                f"{self.thresholds}. Creates {len(self.thresholds) + 1} labeled regions based on "
                f"which thresholds each pixel exceeds (using {op} comparison)."
            )

    def get_thresholds(self) -> List[float]:
        """
        Get the threshold values used by this plugin.

        Returns:
            List of threshold values
        """
        return self.thresholds.copy()
