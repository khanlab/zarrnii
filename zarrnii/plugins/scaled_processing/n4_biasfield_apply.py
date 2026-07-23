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

Intensity rescaling after bias-field correction::

    # scale and offset are computed from the low-resolution images beforehand
    corrected = fullres.apply_scaled_processing(
        znii.plugins.N4BiasFieldApply(scale=0.5, offset=10.0),
        method="map_blocks",
        lowres_znimg=bias_field,
    )

Masked application (N4 run with ``-x`` mask option)::

    # Set bias field values OUTSIDE the mask to a negative sentinel (e.g. -1)
    # before saving to lowres_znimg.  The plugin detects those negative values
    # and restricts all corrections to the masked-in region only.
    corrected = fullres.apply_scaled_processing(
        znii.plugins.N4BiasFieldApply(scale=0.5, offset=10.0),
        method="map_blocks",
        lowres_znimg=bias_field_with_negative_outside_mask,
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
    exponentiated) upsampled bias field to produce a corrected image, then
    applies an optional linear rescaling ``scale * x + offset``.

    Because no ANTsPy call is made this plugin has **no antspyx dependency**
    and works with the base ``zarrnii`` install.

    **Masked application (N4 run with** ``-x`` **option)**

    When N4 is run with a mask (``-x``), the bias field is only meaningful
    inside that mask.  To restrict correction to the masked region, set bias
    field voxels *outside* the mask to any negative value (e.g. ``-1``) before
    passing the field to this plugin.  :meth:`lowres_func` preserves those
    negative values so that, after upsampling, :meth:`highres_func` can detect
    them (any upsampled value ``< 0`` is treated as outside the mask).
    Full-resolution voxels outside the mask are left completely unchanged — no
    bias-field division, no scale/offset is applied to them.

    Parameters:
        log_space: When ``True`` the bias field is log-transformed before
            upsampling and exponentiated before the correction is applied.
            This produces smoother interpolation because the bias field is
            approximately log-linear.  Default: ``False``.
        log_offset: Small positive constant added to the bias field before
            taking the logarithm to guard against ``log(0)``.  Only used
            when ``log_space=True``.  Default: ``1e-6``.
        scale: Multiplicative factor applied to the bias-corrected image
            as ``scale * corrected + offset``.  Computed from the low-res
            images beforehand (e.g. to match global intensity statistics).
            Only applied inside the mask when negative sentinel values are
            present.  Default: ``1.0`` (no rescaling).
        offset: Additive constant applied after ``scale`` as
            ``scale * corrected + offset``.  Default: ``0.0``.

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

    Example (with rescaling)::

        corrected = fullres.apply_scaled_processing(
            N4BiasFieldApply(scale=0.5, offset=10.0),
            method="map_blocks",
            lowres_znimg=precomputed_bias_field_znimg,
        )

    Example (masked, N4 run with ``-x``)::

        # bias_field_masked has values set to -1 outside the brain mask
        corrected = fullres.apply_scaled_processing(
            N4BiasFieldApply(scale=0.5, offset=10.0),
            method="map_blocks",
            lowres_znimg=bias_field_masked,
        )
    """

    def __init__(
        self,
        log_space: bool = False,
        log_offset: float = 1e-6,
        scale: float = 1.0,
        offset: float = 0.0,
    ):
        """
        Initialize the N4BiasFieldApply plugin.

        Args:
            log_space: If ``True``, log-transform the bias field before
                upsampling and exponentiate after upsampling before dividing.
            log_offset: Small positive constant added before ``log`` for
                numerical stability.  Only used when ``log_space=True``.
            scale: Multiplicative factor for the linear rescaling applied
                after bias-field division: ``corrected = scale * (x / bias)
                + offset``.  Default ``1.0`` leaves intensity unchanged.
            offset: Additive constant for the linear rescaling applied after
                bias-field division.  Default ``0.0`` leaves intensity
                unchanged.

        Raises:
            ValueError: If ``log_offset`` is not strictly positive.
        """
        if log_offset <= 0:
            raise ValueError(
                f"log_offset must be strictly positive, got {log_offset!r}"
            )
        self.log_space = log_space
        self.log_offset = log_offset
        self.scale = float(scale)
        self.offset = float(offset)

    @hookimpl
    def lowres_func(self, lowres_array: np.ndarray) -> np.ndarray:
        """
        Return the pre-computed bias field, optionally log-transformed.

        The downsampled input is expected to already contain the N4 bias
        field values (i.e. the output of a prior N4 run).  When
        ``log_space=True`` the field is transformed as
        ``log(field + log_offset)`` before being returned so that the
        subsequent upsampling step interpolates in log space.

        **Mask encoding via negative values**

        If any voxel in ``lowres_array`` is negative (e.g. set to ``-1`` by
        the caller to mark regions outside an N4 mask), those voxels are left
        unchanged.  Only non-negative voxels are clamped / log-transformed.
        The negative sentinel values survive upsampling and allow
        :meth:`highres_func` to reconstruct the mask (``upsampled >= 0``).

        Args:
            lowres_array: Pre-computed bias field at low resolution.  Voxels
                outside the N4 mask should be set to a negative value (e.g.
                ``-1``) by the caller before passing the array here.

        Returns:
            The bias field, cast to float32, with inside-mask values clamped
            (or log-transformed) and outside-mask (negative) values preserved
            as-is.
        """
        if lowres_array.size == 0:
            raise ValueError("Input array is empty")
        if lowres_array.ndim < 2:
            raise ValueError("Input array must be at least 2D")

        # Ensure float so that subsequent division in highres_func is safe.
        if lowres_array.dtype.kind in ("i", "u"):
            lowres_array = lowres_array.astype(np.float32)

        # Negative values encode "outside mask"; preserve them so that
        # highres_func can reconstruct the mask after upsampling.
        inside_mask = lowres_array >= 0

        if self.log_space:
            # Log-transform inside-mask values; leave outside-mask values
            # (negative) unchanged so the sign is preserved through upsampling.
            return np.where(
                inside_mask,
                np.log(np.maximum(lowres_array, 0.0) + self.log_offset),
                lowres_array,
            )
        else:
            # Clamp inside-mask to a small positive value to guard against
            # division by zero; leave outside-mask values unchanged.
            return np.where(
                inside_mask,
                np.maximum(lowres_array, np.finfo(np.float32).eps),
                lowres_array,
            )

    @hookimpl
    def highres_func(
        self,
        fullres_array: np.ndarray | da.Array,
        upsampled_output: np.ndarray | da.Array,
    ) -> np.ndarray | da.Array:
        """
        Apply the upsampled bias field to full-resolution data.

        Divides the full-resolution image by the (optionally exponentiated)
        upsampled bias field to produce a bias-corrected image, then applies
        the linear rescaling ``scale * corrected + offset``.

        When ``log_space=True`` the upsampled field is first exponentiated
        (``exp(upsampled_log_bias)``) before dividing, recovering the linear
        bias field that was log-transformed in :meth:`lowres_func`.

        **Mask handling**

        Voxels where ``upsampled_output < 0`` are treated as outside the mask
        (they correspond to voxels set to a negative sentinel — e.g. ``-1`` —
        before being passed to :meth:`lowres_func`).  For those voxels the
        original ``fullres_array`` value is returned unchanged; neither the
        bias-field division nor the scale/offset rescaling is applied.

        Works with both dask arrays (``"default"`` method) and plain NumPy
        arrays (``"map_blocks"`` method) because only NumPy-compatible
        operations are used.

        Args:
            fullres_array: Full-resolution array (dask or NumPy).
            upsampled_output: Upsampled bias field with the same shape as
                ``fullres_array`` (dask or NumPy).  Contains log-transformed
                values when ``log_space=True``.  Negative values indicate
                voxels outside the N4 mask.

        Returns:
            Bias-corrected (and optionally rescaled) full-resolution array
            (same type as inputs).  Voxels outside the mask are returned
            unchanged.
        """
        epsilon = np.finfo(np.float32).eps

        # Derive mask: negative values in the upsampled bias field indicate
        # voxels that were outside the N4 mask (set to a negative sentinel
        # by the caller before lowres_func was invoked).
        mask = upsampled_output >= 0

        if self.log_space:
            bias = np.exp(upsampled_output)
        else:
            bias = upsampled_output

        corrected = fullres_array / np.maximum(bias, epsilon)
        if self.scale != 1.0 or self.offset != 0.0:
            corrected = self.scale * corrected + self.offset

        # Outside-mask voxels are left as-is; inside-mask get the corrected value.
        return np.where(mask, corrected, fullres_array)

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
        rescale_note = (
            f" Intensity rescaling applied: scale={self.scale}, offset={self.offset}."
            if (self.scale != 1.0 or self.offset != 0.0)
            else ""
        )
        return (
            "Apply a pre-computed N4 bias field to full-resolution data. "
            "The bias field must be supplied as a pre-computed downsampled "
            "image via the ``lowres_znimg`` argument; no ANTsPy dependency "
            f"is required.{log_note}{rescale_note}"
        )

    def __repr__(self) -> str:
        """Return string representation of the plugin."""
        return (
            f"N4BiasFieldApply(log_space={self.log_space!r}, "
            f"log_offset={self.log_offset!r}, "
            f"scale={self.scale!r}, "
            f"offset={self.offset!r})"
        )
