"""
Plugin hook specifications for ZarrNii plugins.

This module defines the hook specifications that plugins must implement.
"""

from __future__ import annotations

from .markers import hookspec


class ZarrNiiSpec:
    """Hook specifications for ZarrNii plugins.

    Plugin authors should implement any subset of these hooks as plain methods
    decorated with ``@hookimpl`` from :mod:`zarrnii.plugins`.

    Example::

        from zarrnii.plugins import hookimpl

        class MyPlugin:
            @hookimpl
            def segment(self, image, metadata=None):
                ...
    """

    @hookspec
    def segment(self, image, metadata=None):
        """Segment an image and return a binary or labeled mask.

        Args:
            image: Input image as numpy array.
            metadata: Optional metadata dictionary containing image information.

        Returns:
            Segmented image as numpy array.
        """

    @hookspec
    def segmentation_plugin_name(self) -> str:
        """Return the name of the segmentation algorithm.

        Returns:
            String name of the algorithm.
        """

    @hookspec
    def segmentation_plugin_description(self) -> str:
        """Return a description of the segmentation algorithm.

        Returns:
            String description of the algorithm.
        """

    @hookspec
    def scaled_processing_plugin_name(self) -> str:
        """Return the name of the scaled processing algorithm.

        Returns:
            String name of the algorithm.
        """

    @hookspec
    def scaled_processing_plugin_description(self) -> str:
        """Return a description of the scaled processing algorithm.

        Returns:
            String description of the algorithm.
        """

    @hookspec
    def lowres_func(self, lowres_array):
        """Process low-resolution data and return the result.

        This function operates on a downsampled numpy array and computes
        the algorithm output that will be upsampled and applied to the
        full-resolution data.

        Args:
            lowres_array: Downsampled input image as numpy array.

        Returns:
            Low-resolution output array (e.g., bias field, correction map).
        """

    @hookspec
    def highres_func(self, fullres_array, upsampled_output):
        """Apply upsampled output to full-resolution data blockwise.

        Args:
            fullres_array: Full-resolution dask array.
            upsampled_output: Upsampled output (same shape as fullres_array).

        Returns:
            Processed full-resolution dask array.
        """


__all__ = ["hookspec", "ZarrNiiSpec"]
