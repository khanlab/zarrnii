"""
N4 Bias field correction plugin using multi-resolution processing.

This module implements a bias field correction plugin that estimates the bias
field at low resolution using N4 algorithm from ANTsPy and applies it to full
resolution data.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from zarrnii.plugins.markers import hookimpl

try:
    import ants

    HAS_ANTSPYX = True
except ImportError:
    HAS_ANTSPYX = False


class N4BiasFieldCorrection:
    """
    N4 bias field correction plugin using multi-resolution processing.

    This plugin estimates a smooth bias field at low resolution using
    the N4 bias field correction algorithm from ANTsPy and applies the
    correction to full resolution data by division.

    The bias field is estimated in log space: ``lowres_func`` returns
    ``log(bias_field)``, which is upsampled via interpolation, and
    ``highres_func`` exponentiates the result before applying the correction.
    Performing interpolation in log space preserves the multiplicative
    structure of the bias field.

    Parameters:
        spline_param: Spacing between knots for spline fitting (default: [2,2,2])
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
    ):
        """
        Initialize N4 bias field correction plugin.

        Args:
            spline_param: Spacing between knots for spline fitting
            convergence: Convergence criteria dict with 'iters' (list), 'tol'
            shrink_factor: Shrink factor for processing

        Raises:
            ImportError: If antspyx is not installed
        """
        if not HAS_ANTSPYX:
            raise ImportError(
                "antspyx is required for N4BiasFieldCorrection. "
                "Install it with: pip install 'zarrnii[n4]' "
                "or pip install antspyx"
            )

        self.spline_param = spline_param
        self.convergence = convergence
        self.shrink_factor = shrink_factor

    @hookimpl
    def lowres_func(
        self, lowres_array: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Estimate bias field from low-resolution data using N4 algorithm.

        This function uses ANTsPy's N4 bias field correction to estimate
        the bias field at low resolution.  The result is returned in log
        space so that subsequent upsampling interpolation is performed on
        a quantity that varies smoothly in an additive sense.

        Args:
            lowres_array: Downsampled input image
            mask: Optional binary mask (same spatial shape as ``lowres_array``)
                indicating voxels to use for bias field estimation.  Voxels
                where mask is zero are excluded from the N4 fit.

        Returns:
            Log of the estimated bias field at low resolution
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

        # Prepare spatial mask for ANTs (common to all batches)
        ants_mask = None
        if mask is not None:
            # Use only spatial dimensions of the mask
            mask_spatial = mask
            if mask_spatial.ndim > 3:
                mask_spatial = mask_spatial.reshape(-1, *mask_spatial.shape[-3:])[0]
            ants_mask = ants.from_numpy(np.asarray(mask_spatial, dtype=np.float32))

        # Apply N4 bias field correction
        if work_array.ndim == 2:
            # For 2D data
            ants_img = ants.from_numpy(work_array)
            self.spline_param = [2, 2]
            # Use return_bias_field=True to get the bias field
            bias_result = ants.n4_bias_field_correction(
                ants_img,
                mask=ants_mask,
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
                        mask=ants_mask,
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
                    mask=ants_mask,
                    return_bias_field=True,
                    spline_param=self.spline_param,
                    convergence=self.convergence,
                    shrink_factor=self.shrink_factor,
                )
                bias_field = bias_result.numpy()

        # Reshape back to original shape if needed
        if original_shape != bias_field.shape:
            bias_field = bias_field.reshape(original_shape)

        # Clamp to positive values before taking log
        bias_field = np.maximum(bias_field, np.finfo(np.float32).eps)

        # Return log of bias field so upsampling interpolation is in log space
        return np.log(bias_field.astype(np.float32))

    @hookimpl
    def highres_func(self, fullres_array, upsampled_output):
        """
        Apply bias field correction to full-resolution data.

        This function takes the upsampled log bias field (same size as
        ``fullres_array``), exponentiates it to recover the bias field, and
        applies the correction to the full-resolution data by division.

        Works with both dask arrays (``"default"`` method) and plain NumPy arrays
        (``"map_blocks"`` method) because only NumPy-compatible operations are used.

        Args:
            fullres_array: Full-resolution array (dask or NumPy)
            upsampled_output: Upsampled log bias field (same shape as fullres;
                dask or NumPy).  This is the value returned by ``lowres_func``
                after upsampling.

        Returns:
            Bias-corrected full-resolution array (same type as inputs)
        """
        # Exponentiate from log space to recover the bias field, then divide.
        # np.exp and np.maximum work for both dask arrays and NumPy.
        epsilon = np.finfo(np.float32).eps
        bias_field = np.exp(upsampled_output)
        corrected_array = fullres_array / np.maximum(bias_field, epsilon)

        return corrected_array

    @hookimpl
    def scaled_processing_plugin_name(self) -> str:
        """Return the name of the algorithm."""
        return "N4 Bias Field Correction"

    @hookimpl
    def scaled_processing_plugin_description(self) -> str:
        """Return a description of the algorithm."""
        return (
            "Multi-resolution N4 bias field correction. Estimates smooth bias "
            "field at low resolution using ANTsPy N4 algorithm and applies "
            "correction to full resolution data by division."
        )

    def __repr__(self) -> str:
        """Return string representation of the plugin."""
        return (
            f"N4BiasFieldCorrection("
            f"spline_param={self.spline_param}, "
            f"convergence={self.convergence}, "
            f"shrink_factor={self.shrink_factor})"
        )
