"""
Local Otsu thresholding segmentation plugin.

This module implements Otsu's automatic threshold selection method for binary
image segmentation, applied locally to each processing block.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from skimage.filters import threshold_otsu

from .base import SegmentationPlugin


class LocalOtsuSegmentation(SegmentationPlugin):
    """
    Local Otsu thresholding segmentation plugin.

    This plugin uses Otsu's method to automatically determine an optimal threshold
    for binary image segmentation. The method assumes a bimodal histogram and
    finds the threshold that minimizes intra-class variance. The threshold is
    computed locally for each processing block, making it suitable for images
    with varying illumination or contrast.

    Parameters:
        nbins: Number of bins for histogram computation (default: 256)
    """

    def __init__(self, nbins: int = 256, **kwargs):
        """
        Initialize local Otsu segmentation plugin.

        Args:
            nbins: Number of bins for histogram computation
            **kwargs: Additional parameters passed to parent class
        """
        super().__init__(nbins=nbins, **kwargs)
        self.nbins = nbins

    def segment(
        self, image: np.ndarray, metadata: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Segment image using local Otsu thresholding.

        Args:
            image: Input image as numpy array
            metadata: Optional metadata (unused in Otsu method)

        Returns:
            Binary segmentation mask as numpy array with same shape as input.
            Values are 0 (background) and 1 (foreground).

        Raises:
            ValueError: If input image is empty or has invalid dimensions
        """
        if image.size == 0:
            raise ValueError("Input image is empty")

        if image.ndim < 2:
            raise ValueError("Input image must be at least 2D")

        # Store original shape for output
        original_shape = image.shape

        # Ensure image is in a suitable format for Otsu thresholding
        if image.dtype == bool:
            # Already binary, return as-is
            return image.astype(np.uint8)

        # Handle multi-dimensional images
        work_image = image

        # For images with more than 3 dimensions, flatten extra dimensions
        if work_image.ndim > 3:
            # Reshape to 3D by flattening leading dimensions
            leading_dims = work_image.shape[:-2]  # All dimensions except last 2
            work_image = work_image.reshape(-1, *work_image.shape[-2:])

        if work_image.ndim == 3:
            # For 3D images, process the first slice/channel
            if work_image.shape[0] <= 4:  # Likely channels-first format
                work_image = work_image[0]
            else:
                # If first dimension is large, likely depth dimension, take middle slice
                mid_slice = work_image.shape[0] // 2
                work_image = work_image[mid_slice]

        # Now work_image should be 2D
        try:
            threshold = threshold_otsu(work_image, nbins=self.nbins)
        except ValueError as e:
            # Handle edge cases where Otsu fails (e.g., constant image)
            if "all values are identical" in str(e).lower() or np.all(
                work_image == work_image.flat[0]
            ):
                # Return all zeros for constant images with original shape
                return np.zeros(original_shape, dtype=np.uint8)
            else:
                raise e

        # Apply threshold to original image
        binary_mask = image > threshold

        return binary_mask.astype(np.uint8)

    @property
    def name(self) -> str:
        """Return the name of the segmentation algorithm."""
        return "Local Otsu Thresholding"

    @property
    def description(self) -> str:
        """Return a description of the segmentation algorithm."""
        return (
            "Local Otsu's automatic threshold selection method for binary segmentation. "
            "Finds the threshold that minimizes intra-class variance assuming a "
            "bimodal intensity distribution. Threshold is computed locally for each "
            "processing block, suitable for images with varying illumination."
        )

    def get_threshold(self, image: np.ndarray) -> float:
        """
        Get the Otsu threshold value without applying segmentation.

        Args:
            image: Input image as numpy array

        Returns:
            Computed Otsu threshold value
        """
        if image.size == 0:
            raise ValueError("Input image is empty")

        # Handle multi-channel images by taking the first channel if needed
        if image.ndim > 3:
            while image.ndim > 3:
                image = image[0]

        if image.ndim == 3 and image.shape[0] <= 4:
            image = image[0]

        return threshold_otsu(image, nbins=self.nbins)


# Keep the old name for backward compatibility
OtsuSegmentation = LocalOtsuSegmentation
