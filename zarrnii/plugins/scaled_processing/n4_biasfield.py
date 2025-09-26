"""
N4 Bias field correction plugin using multi-resolution processing.

This module implements a bias field correction plugin that estimates the bias
field at low resolution using N4 algorithm from ANTsPy and applies it to full
resolution data.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import dask.array as da
import numpy as np

from .base import ScaledProcessingPlugin

try:
    import ants

    HAS_ANTSPYX = True
except ImportError:
    HAS_ANTSPYX = False


class N4BiasFieldCorrection(ScaledProcessingPlugin):
    """
    N4 bias field correction plugin using multi-resolution processing.

    This plugin estimates a smooth bias field at low resolution using
    the N4 bias field correction algorithm from ANTsPy and applies the
    correction to full resolution data by division.

    Parameters:
        spline_param: Spacing between knots for spline fitting (default: 200)
        convergence: Convergence criteria [iters, tol] (default: [50, 0.001])
        shrink_factor: Shrink factor for processing (default: 1)
    """

    def __init__(
        self,
        spline_param: tuple[int, int, int] = [2, 2, 2],
        convergence: Optional[Dict[str, Any]] = {
            "iters": [50, 50, 50, 50],
            "tol": 1e-07,
        },
        shrink_factor: int = 1,
        **kwargs,
    ):
        """
        Initialize N4 bias field correction plugin.

        Args:
            spline_param: Spacing between knots for spline fitting
            convergence: Convergence criteria dict with 'iters' (list), 'tol'
            shrink_factor: Shrink factor for processing
            **kwargs: Additional parameters passed to parent class

        Raises:
            ImportError: If antspyx is not installed
        """
        if not HAS_ANTSPYX:
            raise ImportError(
                "antspyx is required for N4BiasFieldCorrection. "
                "Install it with: pip install 'zarrnii[n4]' "
                "or pip install antspyx"
            )

        super().__init__(
            spline_param=spline_param,
            convergence=convergence,
            shrink_factor=shrink_factor,
            **kwargs,
        )
        self.spline_param = spline_param
        self.convergence = convergence
        self.shrink_factor = shrink_factor

    def lowres_func(self, lowres_array: np.ndarray) -> np.ndarray:
        """
        Estimate bias field from low-resolution data using N4 algorithm.

        This function uses ANTsPy's N4 bias field correction to estimate
        the bias field at low resolution.

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
            spatial_shape = work_array.shape[-3:]
            work_array = work_array.reshape(-1, *spatial_shape)

        # Ensure we're working with float data
        if work_array.dtype.kind in ["i", "u"]:  # integer types
            work_array = work_array.astype(np.float32)

        # Apply N4 bias field correction
        if work_array.ndim == 2:
            # For 2D data
            ants_img = ants.from_numpy(work_array)
            # Use return_bias_field=True to get the bias field
            bias_result = ants.n4_bias_field_correction(
                ants_img,
                return_bias_field=True,
                spline_param=self.spline_param,
                convergence=self.convergence,
                shrink_factor=self.shrink_factor,
            )
            # Extract the bias field
            bias_field = bias_result.numpy()
        else:
            # Apply N4 to each volume if we have multiple or single 3D volume
            if work_array.ndim == 4:  # batched 3D volumes
                bias_field = np.zeros_like(work_array)
                for i in range(work_array.shape[0]):
                    ants_img = ants.from_numpy(work_array[i])
                    bias_result = ants.n4_bias_field_correction(
                        ants_img,
                        return_bias_field=True,
                        spline_param=self.spline_param,
                        convergence=self.convergence,
                        shrink_factor=self.shrink_factor,
                    )
                    bias_field[i] = bias_result.numpy()
            else:  # single 3D volume
                ants_img = ants.from_numpy(work_array)
                bias_result = ants.n4_bias_field_correction(
                    ants_img,
                    return_bias_field=True,
                    spline_param=self.spline_param,
                    convergence=self.convergence,
                    shrink_factor=self.shrink_factor,
                )
                bias_field = bias_result.numpy()

        # Reshape back to original shape if needed
        if original_shape != bias_field.shape:
            bias_field = bias_field.reshape(original_shape)

        # Avoid division by zero by adding small epsilon
        bias_field = np.maximum(bias_field, np.finfo(bias_field.dtype).eps)

        return bias_field

    def highres_func(
        self, fullres_array: da.Array, upsampled_output: da.Array
    ) -> da.Array:
        """
        Apply bias field correction to full-resolution data.

        This function takes the upsampled bias field (same size as
        fullres_array) and applies it to the full-resolution data by division.

        Args:
            fullres_array: Full-resolution dask array
            upsampled_output: Upsampled bias field (same shape as fullres)

        Returns:
            Bias-corrected full-resolution array
        """
        # Apply bias field correction by division using dask operations
        # Avoid division by zero by adding small epsilon
        epsilon = np.finfo(np.float32).eps
        corrected_array = fullres_array / da.maximum(upsampled_output, epsilon)

        return corrected_array

    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return "N4 Bias Field Correction"

    @property
    def description(self) -> str:
        """Return a description of the algorithm."""
        return (
            "Multi-resolution N4 bias field correction. Estimates smooth bias "
            "field at low resolution using ANTsPy N4 algorithm and applies "
            "correction to full resolution data by division."
        )
