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
        self, fullres_array: da.Array, lowres_output: np.ndarray
    ) -> da.Array:
        """
        Apply bias field correction to full-resolution data.

        This function upsamples the low-resolution bias field and applies
        it to the full-resolution data by division.

        Args:
            fullres_array: Full-resolution dask array
            lowres_output: Low-resolution bias field from lowres_func

        Returns:
            Bias-corrected full-resolution array
        """
        # Get shapes for upsampling calculation
        fullres_shape = fullres_array.shape
        lowres_shape = lowres_output.shape

        # Calculate zoom factors for upsampling
        # Handle potential dimension mismatches
        if len(fullres_shape) == len(lowres_shape):
            zoom_factors = [f / l for f, l in zip(fullres_shape, lowres_shape)]
        else:
            # If shapes have different number of dimensions, assume spatial dimensions align
            # and handle leading dimensions appropriately
            spatial_dims = min(len(fullres_shape), len(lowres_shape))
            zoom_factors = [1.0] * len(lowres_shape)

            for i in range(spatial_dims):
                zoom_factors[-(i + 1)] = (
                    fullres_shape[-(i + 1)] / lowres_shape[-(i + 1)]
                )

        # Upsample the bias field to match full resolution
        upsampled_bias = ndimage.zoom(
            lowres_output, zoom_factors, order=1, mode=self.mode  # Linear interpolation
        )

        # Ensure upsampled bias has the same shape as full-res data
        if upsampled_bias.shape != fullres_shape:
            # Handle small shape mismatches due to rounding in zoom
            upsampled_bias = np.resize(upsampled_bias, fullres_shape)

        # Apply bias field correction by division
        # Use map_blocks for efficient dask processing
        def apply_correction(block, bias_block):
            # Ensure bias block has same shape as data block
            if bias_block.shape != block.shape:
                bias_block = np.resize(bias_block, block.shape)

            # Apply correction, avoiding division by zero
            corrected = block / np.maximum(bias_block, np.finfo(bias_block.dtype).eps)
            return corrected.astype(block.dtype)

        # Apply correction using dask map_blocks
        corrected_array = da.map_blocks(
            apply_correction,
            fullres_array,
            upsampled_bias,
            dtype=fullres_array.dtype,
            meta=np.array([], dtype=fullres_array.dtype),
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
