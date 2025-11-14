"""
Plugin hook specifications for ZarrNii plugins.

This module defines the hook specifications that plugins must implement
using the pluggy framework.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import dask.array as da
import numpy as np
import pluggy

hookspec = pluggy.HookspecMarker("zarrnii")


@hookspec
def segmentation_plugin_name() -> str:
    """
    Return the name of the segmentation algorithm.

    Returns:
        String name of the algorithm
    """


@hookspec
def segmentation_plugin_description() -> str:
    """
    Return a description of the segmentation algorithm.

    Returns:
        String description of the algorithm
    """


@hookspec
def segment(image: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Segment an image and return a binary or labeled mask.

    Args:
        image: Input image as numpy array
        metadata: Optional metadata dictionary containing image information

    Returns:
        Segmented image as numpy array. Should be binary (0/1) for binary
        segmentation or labeled (0, 1, 2, ...) for multi-class segmentation.
    """


@hookspec
def scaled_processing_plugin_name() -> str:
    """
    Return the name of the scaled processing algorithm.

    Returns:
        String name of the algorithm
    """


@hookspec
def scaled_processing_plugin_description() -> str:
    """
    Return a description of the scaled processing algorithm.

    Returns:
        String description of the algorithm
    """


@hookspec
def lowres_func(lowres_array: np.ndarray) -> np.ndarray:
    """
    Process low-resolution data and return the result.

    This function operates on a downsampled numpy array and computes
    the algorithm output that will be upsampled and applied to the
    full-resolution data.

    Args:
        lowres_array: Downsampled input image as numpy array

    Returns:
        Low-resolution output array (e.g., bias field, correction map)
    """


@hookspec
def highres_func(fullres_array: da.Array, upsampled_output: da.Array) -> da.Array:
    """
    Apply upsampled output to full-resolution data blockwise.

    This function receives the full-resolution dask array and the
    upsampled output (same size as fullres_array), and applies the operation.
    The upsampling is handled internally by the apply_scaled_processing method.

    Args:
        fullres_array: Full-resolution dask array
        upsampled_output: Upsampled output (same shape as fullres_array)

    Returns:
        Processed full-resolution dask array
    """
