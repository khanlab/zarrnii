"""
Base class for segmentation plugins.

This module defines the abstract interface that all segmentation plugins must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class SegmentationPlugin(ABC):
    """
    Abstract base class for segmentation plugins.

    All segmentation plugins must inherit from this class and implement the
    segment method. This ensures a consistent interface across different
    segmentation algorithms.
    """

    def __init__(self, **kwargs):
        """
        Initialize the segmentation plugin.

        Args:
            **kwargs: Plugin-specific parameters
        """
        self.params = kwargs

    @abstractmethod
    def segment(
        self, image: np.ndarray, metadata: dict[str, Any] | None = None
    ) -> np.ndarray:
        """
        Segment an image and return a binary or labeled mask.

        Args:
            image: Input image as numpy array
            metadata: Optional metadata dictionary containing image information

        Returns:
            Segmented image as numpy array. Should be binary (0/1) for binary
            segmentation or labeled (0, 1, 2, ...) for multi-class segmentation.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the name of the segmentation algorithm.

        Returns:
            String name of the algorithm
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Return a description of the segmentation algorithm.

        Returns:
            String description of the algorithm
        """
        pass

    def __repr__(self) -> str:
        """Return string representation of the plugin."""
        return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in self.params.items())})"
