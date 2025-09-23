"""
Bias field correction plugin using multi-resolution processing.

This module implements a bias field correction plugin that estimates the bias field
at low resolution and applies it to full resolution data.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import dask.array as da
import numpy as np
from scipy import ndimage

from .base import ScaledProcessingPlugin


class BiasFieldCorrection(ScaledProcessingPlugin):
    """
    Bias field correction plugin using multi-resolution processing.

    This plugin estimates a smooth bias field at low resolution using
    simple smoothing (as a placeholder for more sophisticated methods like N4)
    and applies the correction to full resolution data by division.

    Parameters:
        sigma: Standard deviation for Gaussian smoothing (default: 5.0)
        mode: Boundary condition for smoothing (default: 'reflect')
    """

    def __init__(self, sigma: float = 5.0, mode: str = "reflect", **kwargs):
        """
        Initialize bias field correction plugin.

        Args:
            sigma: Standard deviation for Gaussian smoothing
            mode: Boundary condition for smoothing
            **kwargs: Additional parameters passed to parent class
        """
        super().__init__(sigma=sigma, mode=mode, **kwargs)
        self.sigma = sigma
        self.mode = mode

    def lowres_func(self, lowres_array: np.ndarray) -> np.ndarray:
        """
        Estimate bias field from low-resolution data.

        This is a simplified bias field estimation using Gaussian smoothing.
        In practice, this could be replaced with more sophisticated methods
        like N4 bias field correction.

        Args:
            lowres_array: Downsampled input image

        Returns:
            Estimated bias field at low resolution
        """
        if lowres_array.size == 0:
            raise ValueError("Input array is empty")

        # Handle different array dimensions
        if lowres_array.ndim < 2:
            raise ValueError("Input array must be at least 2D")

        # Store original shape
        original_shape = lowres_array.shape

        # For multi-dimensional arrays, work with the last 3 dimensions
        work_array = lowres_array
        if work_array.ndim > 3:
            # Flatten leading dimensions and work with spatial dimensions
            leading_shape = work_array.shape[:-3]
            spatial_shape = work_array.shape[-3:]
            work_array = work_array.reshape(-1, *spatial_shape)

        # Ensure we're working with float data
        if work_array.dtype.kind in ["i", "u"]:  # integer types
            work_array = work_array.astype(np.float32)

        # Simple bias field estimation: smooth the image to get low-frequency components
        if work_array.ndim == 2:
            smoothed = ndimage.gaussian_filter(
                work_array, sigma=self.sigma, mode=self.mode
            )
        else:
            # Apply smoothing to each volume if we have multiple
            if work_array.ndim == 4:  # batched 3D volumes
                smoothed = np.zeros_like(work_array)
                for i in range(work_array.shape[0]):
                    smoothed[i] = ndimage.gaussian_filter(
                        work_array[i], sigma=self.sigma, mode=self.mode
                    )
            else:  # single 3D volume
                smoothed = ndimage.gaussian_filter(
                    work_array, sigma=self.sigma, mode=self.mode
                )

        # Reshape back to original shape if needed
        if original_shape != smoothed.shape:
            smoothed = smoothed.reshape(original_shape)

        # Avoid division by zero by adding small epsilon
        smoothed = np.maximum(smoothed, np.finfo(smoothed.dtype).eps)

        return smoothed

    def highres_func(
        self, fullres_array: da.Array, upsampled_output: da.Array
    ) -> da.Array:
        """
        Apply bias field correction to full-resolution data.

        This function takes the upsampled bias field (same size as fullres_array)
        and applies it to the full-resolution data by division.

        Args:
            fullres_array: Full-resolution dask array
            upsampled_output: Upsampled bias field (same shape as fullres_array)

        Returns:
            Bias-corrected full-resolution array
        """
        # Apply bias field correction by division using dask operations
        # Avoid division by zero by adding small epsilon
        corrected_array = fullres_array / da.maximum(
            upsampled_output, np.finfo(fullres_array.dtype).eps
        )

        return corrected_array

    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return "Bias Field Correction"

    @property
    def description(self) -> str:
        """Return a description of the algorithm."""
        return (
            "Multi-resolution bias field correction. Estimates smooth bias field "
            "at low resolution using Gaussian smoothing and applies correction "
            "to full resolution data by division."
        )
