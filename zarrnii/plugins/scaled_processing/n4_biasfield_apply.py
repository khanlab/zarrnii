"""
N4 Bias Field Apply plugin for pre-computed bias fields.

This module implements a bias field correction plugin that accepts a
pre-computed (externally N4-corrected) downsampled image and applies it to
full-resolution data.  Unlike :mod:`n4_biasfield`, this plugin has no ANTsPy
dependency — N4 is assumed to have been run beforehand and the result is
supplied via the ``lowres_znimg`` argument of
:meth:`~zarrnii.core.ZarrNii.apply_scaled_processing`.

Typical usage::

    import zarrnii as znii

    fullres = znii.ZarrNii.from_ome_zarr("fullres.ome.zarr")
    # Load the already-N4-corrected bias field at low resolution
    bias_field = znii.ZarrNii.from_ome_zarr("n4_bias_field.ome.zarr")

    corrected = fullres.apply_scaled_processing(
        znii.plugins.N4BiasFieldApply(),
        method="map_blocks",
        lowres_znimg=bias_field,
    )

Log-space upsampling (recommended for smoother interpolation)::

    corrected = fullres.apply_scaled_processing(
        znii.plugins.N4BiasFieldApply(log_space=True),
        method="map_blocks",
        lowres_znimg=bias_field,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from zarrnii.plugins.markers import hookimpl

if TYPE_CHECKING:
    import dask.array as da


class N4BiasFieldApply:
    """
    Bias field correction plugin for pre-computed N4 bias fields.

    This plugin is a companion to the full
    :class:`~zarrnii.plugins.N4BiasFieldCorrection` plugin.  Instead of
    running N4 itself it expects the user to supply the already-corrected
    downsampled image through the ``lowres_znimg`` parameter of
    :meth:`~zarrnii.core.ZarrNii.apply_scaled_processing`.

    The ``lowres_func`` is a passthrough — it validates, float-casts, and
    zero-clamps the input, then optionally log-transforms it before upsampling.
    The ``highres_func`` divides the full-resolution image by the (optionally
    exponentiated) upsampled bias field to produce a corrected image.

    Because no ANTsPy call is made this plugin has **no antspyx dependency**
    and works with the base ``zarrnii`` install.

    Parameters:
        log_space: When ``True`` the bias field is log-transformed before
            upsampling and exponentiated before the correction is applied.
            This produces smoother interpolation because the bias field is
            approximately log-linear.  Default: ``False``.
        log_offset: Small positive constant added to the bias field before
            taking the logarithm to guard against ``log(0)``.  Only used
            when ``log_space=True``.  Default: ``1e-6``.

    Example (linear space, default)::

        corrected = fullres.apply_scaled_processing(
            N4BiasFieldApply(),
            method="map_blocks",
            lowres_znimg=precomputed_bias_field_znimg,
        )

    Example (log space)::

        corrected = fullres.apply_scaled_processing(
            N4BiasFieldApply(log_space=True),
            method="map_blocks",
            lowres_znimg=precomputed_bias_field_znimg,
        )
    """

    def __init__(self, log_space: bool = False, log_offset: float = 1e-6):
        """
        Initialize the N4BiasFieldApply plugin.

        Args:
            log_space: If ``True``, log-transform the bias field before
                upsampling and exponentiate after upsampling before dividing.
            log_offset: Small positive constant added before ``log`` for
                numerical stability.  Only used when ``log_space=True``.

        Raises:
            ValueError: If ``log_offset`` is not strictly positive.
        """
        if log_offset <= 0:
            raise ValueError(
                f"log_offset must be strictly positive, got {log_offset!r}"
            )
        self.log_space = log_space
        self.log_offset = log_offset

    @hookimpl
    def lowres_func(self, lowres_array: np.ndarray) -> np.ndarray:
        """
        Return the pre-computed bias field, optionally log-transformed.

        The downsampled input is expected to already contain the N4 bias
        field values (i.e. the output of a prior N4 run).  When
        ``log_space=True`` the field is transformed as
        ``log(field + log_offset)`` before being returned so that the
        subsequent upsampling step interpolates in log space.

        Args:
            lowres_array: Pre-computed bias field at low resolution.

        Returns:
            The bias field, cast to float32, zero-clamped, and optionally
            log-transformed.
        """
        if lowres_array.size == 0:
            raise ValueError("Input array is empty")
        if lowres_array.ndim < 2:
            raise ValueError("Input array must be at least 2D")

        # Ensure float so that subsequent division in highres_func is safe.
        if lowres_array.dtype.kind in ("i", "u"):
            lowres_array = lowres_array.astype(np.float32)

        if self.log_space:
            # Clamp to zero first, then add log_offset to guarantee the
            # argument to log is strictly positive even if the bias field
            # contains negative values (e.g. from integer underflow).
            return np.log(np.maximum(lowres_array, 0.0) + self.log_offset)
        else:
            # Clamp to a small positive value to guard against division by zero
            # when the field is later upsampled and applied.
            return np.maximum(lowres_array, np.finfo(np.float32).eps)

    @hookimpl
    def highres_func(
        self,
        fullres_array: np.ndarray | da.Array,
        upsampled_output: np.ndarray | da.Array,
    ) -> np.ndarray | da.Array:
        """
        Apply the upsampled bias field to full-resolution data.

        Divides the full-resolution image by the (optionally exponentiated)
        upsampled bias field to produce a bias-corrected image.

        When ``log_space=True`` the upsampled field is first exponentiated
        (``exp(upsampled_log_bias)``) before dividing, recovering the linear
        bias field that was log-transformed in :meth:`lowres_func`.

        Works with both dask arrays (``"default"`` method) and plain NumPy
        arrays (``"map_blocks"`` method) because only NumPy-compatible
        operations are used.

        Args:
            fullres_array: Full-resolution array (dask or NumPy).
            upsampled_output: Upsampled bias field with the same shape as
                ``fullres_array`` (dask or NumPy).  Contains log-transformed
                values when ``log_space=True``.

        Returns:
            Bias-corrected full-resolution array (same type as inputs).
        """
        epsilon = np.finfo(np.float32).eps
        if self.log_space:
            bias = np.exp(upsampled_output)
        else:
            bias = upsampled_output
        return fullres_array / np.maximum(bias, epsilon)

    @hookimpl
    def scaled_processing_plugin_name(self) -> str:
        """Return the name of the algorithm."""
        return "N4 Bias Field Apply"

    @hookimpl
    def scaled_processing_plugin_description(self) -> str:
        """Return a description of the algorithm."""
        log_note = (
            " Bias field is interpolated in log space for smoother upsampling."
            if self.log_space
            else ""
        )
        return (
            "Apply a pre-computed N4 bias field to full-resolution data. "
            "The bias field must be supplied as a pre-computed downsampled "
            "image via the ``lowres_znimg`` argument; no ANTsPy dependency "
            f"is required.{log_note}"
        )

    def __repr__(self) -> str:
        """Return string representation of the plugin."""
        return (
            f"N4BiasFieldApply(log_space={self.log_space!r}, "
            f"log_offset={self.log_offset!r})"
        )
