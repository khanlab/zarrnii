"""
Segmentation cleaning plugin using multi-resolution processing.

This module implements a segmentation cleaning plugin that removes artifactual
objects from a segmentation by performing connected components analysis at low
resolution and filtering by extent, then applying the exclusion mask to full
resolution data.
"""

from __future__ import annotations

import dask.array as da
import numpy as np
from skimage.measure import label, regionprops

from zarrnii.logging import get_logger

from .base import ScaledProcessingPlugin, hookimpl

# Module-level logger for this plugin
logger = get_logger(__name__)


class SegmentationCleaner(ScaledProcessingPlugin):
    """
    Segmentation cleaning plugin using multi-resolution processing.

    This plugin removes artifactual objects from segmentations by:
    1. At low resolution: performing connected components analysis on a
       thresholded mask and identifying objects with extent below a threshold
    2. At high resolution: applying an exclusion mask to remove these objects
       from the full-resolution segmentation

    The extent of a region is defined as the ratio of pixels in the region to
    pixels in the total bounding box. Objects with low extent (e.g., < 0.15)
    are typically large artifactual objects with sparse coverage.

    Parameters:
        mask_threshold: Threshold for creating initial binary mask (default: 50)
        max_extent: Maximum extent threshold for exclusion (default: 0.15)
        exclusion_threshold: Threshold for upsampled exclusion mask (default: 50)
    """

    def __init__(
        self,
        mask_threshold: float = 50,
        max_extent: float = 0.15,
        exclusion_threshold: float = 50,
        **kwargs,
    ):
        """
        Initialize segmentation cleaner plugin.

        Args:
            mask_threshold: Threshold for creating initial binary mask from input
            max_extent: Maximum extent to include in exclusion mask (objects with
                       extent < max_extent are considered artifactual)
            exclusion_threshold: Threshold for applying upsampled exclusion mask
            **kwargs: Additional parameters passed to parent class
        """
        super().__init__(
            mask_threshold=mask_threshold,
            max_extent=max_extent,
            exclusion_threshold=exclusion_threshold,
            **kwargs,
        )
        self.mask_threshold = mask_threshold
        self.max_extent = max_extent
        self.exclusion_threshold = exclusion_threshold
        logger.debug(
            "Initialized SegmentationCleaner with mask_threshold=%.2f, "
            "max_extent=%.2f, exclusion_threshold=%.2f",
            mask_threshold,
            max_extent,
            exclusion_threshold,
        )

    @hookimpl
    def lowres_func(self, lowres_array: np.ndarray) -> np.ndarray:
        """
        Create exclusion mask from low-resolution segmentation data.

        This function performs connected components analysis on thresholded
        low-resolution data and creates an exclusion mask containing objects
        with extent below the threshold.

        Args:
            lowres_array: Downsampled segmentation image

        Returns:
            Exclusion mask at low resolution (uint8, values 0 or 100)
        """
        logger.debug(
            "SegmentationCleaner.lowres_func - input shape: %s, dtype: %s",
            lowres_array.shape,
            lowres_array.dtype,
        )

        if lowres_array.size == 0:
            raise ValueError("Input array is empty")

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
            logger.debug(
                "Reshaped array for processing - new shape: %s", work_array.shape
            )

        # Process connected components and create exclusion mask
        logger.debug(
            "Running connected components analysis with mask_threshold=%.2f",
            self.mask_threshold,
        )

        if work_array.ndim == 2:
            # 2D case
            mask_img = work_array > self.mask_threshold
            conncomp, nlabels = label(mask_img, return_num=True)
            props = regionprops(conncomp)
            keep_labels = {r.label for r in props if r.extent < self.max_extent}
            exclude_mask = np.isin(conncomp, list(keep_labels))
            exclude_mask = exclude_mask.astype("uint8") * 100
            logger.debug(
                "2D analysis - found %d labels, %d marked for exclusion (extent < %.2f)",
                nlabels,
                len(keep_labels),
                self.max_extent,
            )
        else:
            # 3D or batched 3D case
            if work_array.ndim == 4:
                # Batched 3D volumes - process each separately
                exclude_mask = np.zeros_like(work_array, dtype="uint8")
                for i in range(work_array.shape[0]):
                    mask_img = work_array[i] > self.mask_threshold
                    conncomp, nlabels = label(mask_img, return_num=True)
                    props = regionprops(conncomp)
                    keep_labels = {r.label for r in props if r.extent < self.max_extent}
                    batch_exclude = np.isin(conncomp, list(keep_labels))
                    exclude_mask[i] = batch_exclude.astype("uint8") * 100
                    logger.debug(
                        "Volume %d/%d - found %d labels, %d marked for exclusion",
                        i + 1,
                        work_array.shape[0],
                        nlabels,
                        len(keep_labels),
                    )
            else:
                # Single 3D volume
                mask_img = work_array > self.mask_threshold
                conncomp, nlabels = label(mask_img, return_num=True)
                props = regionprops(conncomp)
                keep_labels = {r.label for r in props if r.extent < self.max_extent}
                exclude_mask = np.isin(conncomp, list(keep_labels))
                exclude_mask = exclude_mask.astype("uint8") * 100
                logger.debug(
                    "3D analysis - found %d labels, %d marked for exclusion (extent < %.2f)",
                    nlabels,
                    len(keep_labels),
                    self.max_extent,
                )

        # Reshape back to original shape if needed
        if original_shape != exclude_mask.shape:
            exclude_mask = exclude_mask.reshape(original_shape)
            logger.debug("Reshaped output back to: %s", original_shape)

        logger.debug(
            "SegmentationCleaner.lowres_func - output shape: %s, "
            "excluded pixels: %d (%.2f%%)",
            exclude_mask.shape,
            np.sum(exclude_mask > 0),
            100.0 * np.sum(exclude_mask > 0) / exclude_mask.size,
        )

        return exclude_mask

    @hookimpl
    def highres_func(
        self, fullres_array: da.Array, upsampled_output: da.Array
    ) -> da.Array:
        """
        Apply exclusion mask to full-resolution segmentation data.

        This function takes the upsampled exclusion mask and applies it to
        the full-resolution segmentation by zeroing out the excluded regions.

        Args:
            fullres_array: Full-resolution segmentation dask array
            upsampled_output: Upsampled exclusion mask (same shape as fullres)

        Returns:
            Cleaned full-resolution segmentation array
        """
        logger.debug(
            "SegmentationCleaner.highres_func - fullres shape: %s, "
            "chunks: %s, upsampled shape: %s, chunks: %s",
            fullres_array.shape,
            fullres_array.chunksize,
            upsampled_output.shape,
            upsampled_output.chunksize,
        )

        # Threshold the upsampled exclusion mask
        # Values >= exclusion_threshold (e.g., 50) indicate regions to exclude
        exclusion_mask = upsampled_output >= self.exclusion_threshold

        # Apply mask: set excluded regions to zero
        cleaned_array = da.where(exclusion_mask, 0, fullres_array)

        logger.debug(
            "Exclusion mask applied - output chunks: %s, npartitions: %d",
            cleaned_array.chunksize,
            cleaned_array.npartitions,
        )

        return cleaned_array

    @hookimpl
    def scaled_processing_plugin_name(self) -> str:
        """Return the name of the algorithm."""
        return "Segmentation Cleaner"

    @hookimpl
    def scaled_processing_plugin_description(self) -> str:
        """Return a description of the algorithm."""
        return (
            "Multi-resolution segmentation cleaning. Identifies and removes "
            "artifactual objects by performing connected components analysis at "
            "low resolution, filtering by extent, and applying exclusion mask to "
            "full resolution data."
        )

    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return self.scaled_processing_plugin_name()

    @property
    def description(self) -> str:
        """Return a description of the algorithm."""
        return self.scaled_processing_plugin_description()
