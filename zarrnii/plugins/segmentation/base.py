"""
Base class for segmentation plugins.

This module defines the interface that all segmentation plugins must implement
using the pluggy framework.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pluggy

hookimpl = pluggy.HookimplMarker("zarrnii")


class SegmentationPlugin:
    """
    Base class for segmentation plugins using pluggy.

    All segmentation plugins should inherit from this class and implement the
    required methods. This ensures a consistent interface across different
    segmentation algorithms.

    The plugin methods are decorated with @hookimpl to work with pluggy.
    """

    def __init__(self, **kwargs):
        """
        Initialize the segmentation plugin.

        Args:
            **kwargs: Plugin-specific parameters
        """
        self.params = kwargs

    @hookimpl
    def segment(
        self, image: np.ndarray, metadata: Optional[Dict[str, Any]] = None
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
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement segment method"
        )

    @hookimpl
    def segmentation_plugin_name(self) -> str:
        """
        Return the name of the segmentation algorithm.

        Returns:
            String name of the algorithm

        Raises:
            NotImplementedError: If the method is not implemented by the subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement segmentation_plugin_name method"
        )

    @hookimpl
    def segmentation_plugin_description(self) -> str:
        """
        Return a description of the segmentation algorithm.

        Returns:
            String description of the algorithm

        Raises:
            NotImplementedError: If the method is not implemented by the subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement segmentation_plugin_description method"
        )

    @property
    def name(self) -> str:
        """
        Return the name of the segmentation algorithm.

        This property provides backward compatibility.

        Returns:
            String name of the algorithm
        """
        return self.segmentation_plugin_name()

    @property
    def description(self) -> str:
        """
        Return a description of the segmentation algorithm.

        This property provides backward compatibility.

        Returns:
            String description of the algorithm
        """
        return self.segmentation_plugin_description()

    def __repr__(self) -> str:
        """Return string representation of the plugin."""
        return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in self.params.items())})"
