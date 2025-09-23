"""
Base class for scaled processing plugins.

This module defines the abstract interface that all scaled processing plugins must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import dask.array as da
import numpy as np


class ScaledProcessingPlugin(ABC):
    """
    Abstract base class for scaled processing plugins.

    All scaled processing plugins must inherit from this class and implement the
    lowres_func and highres_func methods. This architecture enables efficient
    multi-resolution processing where an algorithm is computed at low resolution
    and then applied to full resolution data.
    """

    def __init__(self, **kwargs):
        """
        Initialize the scaled processing plugin.

        Args:
            **kwargs: Plugin-specific parameters
        """
        self.params = kwargs

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the name of the scaled processing algorithm.

        Returns:
            String name of the algorithm
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Return a description of the scaled processing algorithm.

        Returns:
            String description of the algorithm
        """
        pass

    def __repr__(self) -> str:
        """Return string representation of the plugin."""
        return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in self.params.items())})"
