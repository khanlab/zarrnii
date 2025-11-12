"""
Base class for scaled processing plugins.

This module defines the interface that all scaled processing plugins must implement
using the pluggy framework.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import dask.array as da
import numpy as np
import pluggy

hookimpl = pluggy.HookimplMarker("zarrnii")


class ScaledProcessingPlugin:
    """
    Base class for scaled processing plugins using pluggy.

    All scaled processing plugins should inherit from this class and implement the
    required methods. This architecture enables efficient multi-resolution processing
    where an algorithm is computed at low resolution and then applied to full
    resolution data.

    The plugin methods are decorated with @hookimpl to work with pluggy.
    """

    def __init__(self, **kwargs):
        """
        Initialize the scaled processing plugin.

        Args:
            **kwargs: Plugin-specific parameters
        """
        self.params = kwargs

    @hookimpl
    def lowres_func(self, lowres_array: np.ndarray) -> np.ndarray:
        """
        Process low-resolution data and return the result.

        This function operates on a downsampled numpy array and computes
        the algorithm output that will be upsampled and applied to the
        full-resolution data.

        Args:
            lowres_array: Downsampled input image as numpy array

        Returns:
            Low-resolution output array (e.g., bias field, correction map)

        Raises:
            NotImplementedError: If the method is not implemented by the subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement lowres_func method"
        )

    @hookimpl
    def highres_func(
        self, fullres_array: da.Array, upsampled_output: da.Array
    ) -> da.Array:
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

        Raises:
            NotImplementedError: If the method is not implemented by the subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement highres_func method"
        )

    @hookimpl
    def scaled_processing_plugin_name(self) -> str:
        """
        Return the name of the scaled processing algorithm.

        Returns:
            String name of the algorithm

        Raises:
            NotImplementedError: If the method is not implemented by the subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement scaled_processing_plugin_name method"
        )

    @hookimpl
    def scaled_processing_plugin_description(self) -> str:
        """
        Return a description of the scaled processing algorithm.

        Returns:
            String description of the algorithm

        Raises:
            NotImplementedError: If the method is not implemented by the subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement scaled_processing_plugin_description method"
        )

    @property
    def name(self) -> str:
        """
        Return the name of the scaled processing algorithm.

        This property provides backward compatibility.

        Returns:
            String name of the algorithm
        """
        return self.scaled_processing_plugin_name()

    @property
    def description(self) -> str:
        """
        Return a description of the scaled processing algorithm.

        This property provides backward compatibility.

        Returns:
            String description of the algorithm
        """
        return self.scaled_processing_plugin_description()

    def __repr__(self) -> str:
        """Return string representation of the plugin."""
        return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in self.params.items())})"
