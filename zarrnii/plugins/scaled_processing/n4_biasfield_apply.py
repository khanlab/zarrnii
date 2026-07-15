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

    The ``lowres_func`` is a passthrough — it returns the downsampled array
    unchanged.  The ``highres_func`` then divides the full-resolution image
    by the upsampled bias field to produce a corrected image.

    Because no ANTsPy call is made this plugin has **no antspyx dependency**
    and works with the base ``zarrnii`` install.

    Example::

        corrected = fullres.apply_scaled_processing(
            N4BiasFieldApply(),
            method="map_blocks",
            lowres_znimg=precomputed_bias_field_znimg,
        )
    """

    @hookimpl
    def lowres_func(self, lowres_array: np.ndarray) -> np.ndarray:
        """
        Return the pre-computed bias field unchanged.

        The downsampled input is expected to already contain the N4 bias
        field values (i.e. the output of a prior N4 run).  This function
        simply passes the array through so that it can be upsampled and
        applied to the full-resolution data by :meth:`highres_func`.

        Args:
            lowres_array: Pre-computed bias field at low resolution.

        Returns:
            The same array, cast to float32 if necessary to ensure safe
            subsequent division.
        """
        if lowres_array.size == 0:
            raise ValueError("Input array is empty")
        if lowres_array.ndim < 2:
            raise ValueError("Input array must be at least 2D")

        # Ensure float so that subsequent division in highres_func is safe.
        if lowres_array.dtype.kind in ("i", "u"):
            lowres_array = lowres_array.astype(np.float32)

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

        Divides the full-resolution image by the upsampled bias field to
        produce a bias-corrected image.

        Works with both dask arrays (``"default"`` method) and plain NumPy
        arrays (``"map_blocks"`` method) because only NumPy-compatible
        operations are used.

        Args:
            fullres_array: Full-resolution array (dask or NumPy).
            upsampled_output: Upsampled bias field with the same shape as
                ``fullres_array`` (dask or NumPy).

        Returns:
            Bias-corrected full-resolution array (same type as inputs).
        """
        epsilon = np.finfo(np.float32).eps
        return fullres_array / np.maximum(upsampled_output, epsilon)

    @hookimpl
    def scaled_processing_plugin_name(self) -> str:
        """Return the name of the algorithm."""
        return "N4 Bias Field Apply"

    @hookimpl
    def scaled_processing_plugin_description(self) -> str:
        """Return a description of the algorithm."""
        return (
            "Apply a pre-computed N4 bias field to full-resolution data. "
            "The bias field must be supplied as a pre-computed downsampled "
            "image via the ``lowres_znimg`` argument; no ANTsPy dependency "
            "is required."
        )

    def __repr__(self) -> str:
        """Return string representation of the plugin."""
        return "N4BiasFieldApply()"
