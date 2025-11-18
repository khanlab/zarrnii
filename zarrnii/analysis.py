"""
Image analysis functions for zarrnii.

This module provides functions for image analysis operations such as
histogram computation and threshold calculation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import dask.array as da

import numpy as np

from skimage.filters import threshold_multiotsu
from skimage.measure import label, regionprops


def compute_histogram(
    image: da.Array,
    bins: int = 256,
    range: Optional[Tuple[float, float]] = None,
    mask: Optional[da.Array] = None,
    **kwargs: Any,
) -> Tuple[da.Array, da.Array]:
    """
    Compute histogram of a dask array image.

    This function computes the histogram of image intensities, optionally using
    a mask to weight the computation. The histogram is computed using dask for
    efficient processing of large datasets.

    Args:
        image: Input dask array image
        bins: Number of histogram bins (default: 256)
        range: Optional tuple (min, max) defining histogram range. If None,
            uses the full range of the data
        mask: Optional dask array mask of same shape as image. Only pixels
            where mask > 0 are included in histogram computation
        **kwargs: Additional arguments passed to dask.array.histogram

    Returns:
        Tuple of (histogram_counts, bin_edges) where:
        - histogram_counts: dask array of histogram bin counts
        - bin_edges: dask array of bin edge values (length = bins + 1)

    Examples:
        >>> import dask.array as da
        >>> from zarrnii import compute_histogram
        >>>
        >>> # Create test image
        >>> image = da.random.random((100, 100, 100), chunks=(50, 50, 50))
        >>>
        >>> # Compute histogram
        >>> hist, bin_edges = compute_histogram(image, bins=128)
        >>>
        >>> # Compute histogram with mask
        >>> mask = image > 0.5
        >>> hist_masked, _ = compute_histogram(image, mask=mask)
    """
    if mask is not None:
        if mask.shape != image.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match image shape {image.shape}"
            )
        # Apply mask by flattening both arrays and selecting valid pixels
        flat_image = image.flatten()
        flat_mask = mask.flatten()
        # Only include pixels where mask > 0
        valid_indices = flat_mask > 0
        valid_data = flat_image[valid_indices]

        # For dask histogram, we need to provide a range
        if range is None:
            data_min = da.min(valid_data).compute()
            data_max = da.max(valid_data).compute()
            range = (data_min, data_max)
        return da.histogram(valid_data, bins=bins, range=range, **kwargs)
    else:
        # For dask histogram, we need to provide a range
        if range is None:
            data_min = da.min(image).compute()
            data_max = da.max(image).compute()
            range = (data_min, data_max)
        return da.histogram(image, bins=bins, range=range, **kwargs)


def compute_otsu_thresholds(
    histogram_counts: Union[np.ndarray, da.Array],
    classes: int = 2,
    bin_edges: Optional[Union[np.ndarray, da.Array]] = None,
    return_figure: bool = False,
) -> Union[
    List[float],
    Tuple[List[float], Any],
]:
    """
    Compute Otsu multi-level thresholds from histogram data.

    This function uses scikit-image's threshold_multiotsu to compute optimal
    threshold values that separate the histogram into the specified number of
    classes.

    Args:
        histogram_counts: Histogram bin counts as numpy or dask array
        classes: Number of classes to separate data into (default: 2).
            Must be >= 2. For classes=2, returns 1 threshold. For classes=k,
            returns k-1 thresholds.
        bin_edges: Optional bin edges corresponding to histogram. If provided,
            used to determine the min/max range for the output format. If None,
            assumes bin edges from 0 to len(histogram_counts)
        return_figure: If True, returns a tuple containing thresholds and a
            matplotlib figure with the histogram and annotated threshold lines
            (default: False).

    Returns:
        If return_figure is False (default):
            List of threshold values. For classes=k, returns k+1 values:
            [min_value, threshold1, threshold2, ..., threshold_k-1, max_value]
            where min_value and max_value are the data range bounds.

        If return_figure is True:
            Tuple of (thresholds, figure) where figure is a matplotlib Figure
            object showing the histogram with annotated threshold lines.

    Raises:
        ValueError: If classes < 2 or if histogram is empty

    Examples:
        >>> import numpy as np
        >>> from zarrnii import compute_otsu_thresholds
        >>>
        >>> # Create sample histogram (bimodal distribution)
        >>> hist = np.array([100, 50, 20, 5, 2, 5, 20, 50, 100])
        >>>
        >>> # Compute binary threshold (2 classes)
        >>> thresholds = compute_otsu_thresholds(hist, classes=2)
        >>> print(f"Binary thresholds: {thresholds}")
        >>>
        >>> # Compute multi-level thresholds (3 classes)
        >>> thresholds = compute_otsu_thresholds(hist, classes=3)
        >>> print(f"Multi-level thresholds: {thresholds}")
        >>>
        >>> # Generate a figure with annotated thresholds
        >>> thresholds, fig = compute_otsu_thresholds(
        ...     hist, classes=2, return_figure=True
        ... )
        >>> fig.savefig('otsu_thresholds.png')
    """
    if classes < 2:
        raise ValueError("Number of classes must be >= 2")

    # Convert dask arrays to numpy if needed
    if hasattr(histogram_counts, "compute"):
        histogram_counts = histogram_counts.compute()
    if bin_edges is not None and hasattr(bin_edges, "compute"):
        bin_edges = bin_edges.compute()

    if len(histogram_counts) == 0:
        raise ValueError("Histogram is empty")

    # Use scikit-image's threshold_multiotsu with histogram directly
    # threshold_multiotsu accepts hist parameter as
    # (histogram_counts, bin_centers) tuple
    # This avoids reconstructing millions of data points from histogram
    if bin_edges is not None:
        # Calculate bin centers from edges
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        min_val = float(bin_edges[0])
        max_val = float(bin_edges[-1])

        # Check if histogram has any data
        if histogram_counts.sum() == 0:
            # Empty histogram case
            mid_thresholds = np.linspace(min_val, max_val, classes + 1)[1:-1].tolist()
        else:
            # Pass histogram and bin centers directly to threshold_multiotsu
            # This is memory-efficient as it doesn't reconstruct data points
            # The hist parameter expects a 2-tuple: (histogram_counts, bin_centers)
            otsu_thresholds = threshold_multiotsu(
                hist=(histogram_counts, bin_centers), classes=classes
            )
            mid_thresholds = otsu_thresholds.tolist()
    else:
        # If no bin_edges provided, work in histogram bin index space
        # Create bin centers as integer indices (0 to len-1)
        bin_centers = np.arange(len(histogram_counts))
        min_val = 0.0
        # max_val is set to len(histogram_counts) for backward compatibility
        # This represents the upper bound of the histogram range
        max_val = float(len(histogram_counts))

        # Check if histogram has any data
        if histogram_counts.sum() == 0:
            # Empty histogram case
            mid_thresholds = np.linspace(min_val, max_val, classes + 1)[1:-1].tolist()
        else:
            # Pass histogram and bin centers directly to threshold_multiotsu
            # The hist parameter expects a 2-tuple: (histogram_counts, bin_centers)
            otsu_thresholds = threshold_multiotsu(
                hist=(histogram_counts, bin_centers), classes=classes
            )
            mid_thresholds = otsu_thresholds.tolist()

    # Format as requested in the issue: [min, threshold1, ..., threshold_k-1, max]
    result = [min_val] + mid_thresholds + [max_val]

    if return_figure:
        # Import matplotlib here to avoid requiring it as a hard dependency
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for return_figure=True. "
                "Install it with: pip install matplotlib"
            )

        # Create figure with histogram and threshold annotations
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histogram
        if bin_edges is not None:
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ax.bar(
                bin_centers,
                histogram_counts,
                width=np.diff(bin_edges),
                alpha=0.7,
                color="skyblue",
                edgecolor="black",
                label="Histogram",
            )
        else:
            ax.bar(
                np.arange(len(histogram_counts)),
                histogram_counts,
                alpha=0.7,
                color="skyblue",
                edgecolor="black",
                label="Histogram",
            )

        # Plot threshold lines (exclude min and max)
        colors = plt.cm.Set1(np.linspace(0, 1, len(mid_thresholds)))
        for i, (thresh, color) in enumerate(zip(mid_thresholds, colors)):
            ax.axvline(
                thresh,
                color=color,
                linestyle="--",
                linewidth=2,
                label=f"Threshold {i+1}: {thresh:.3f}",
            )

        ax.set_xlabel("Intensity Value", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(f"Histogram with {classes}-class Otsu Thresholds", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        return result, fig
    else:
        # Default: return only thresholds
        return result


def compute_centroids(
    image: da.Array,
    affine: np.ndarray,
    depth: Union[int, Tuple[int, ...], Dict[int, int]] = 10,
    boundary: str = "none",
    rechunk: Optional[Union[int, Tuple[int, ...]]] = None,
) -> np.ndarray:
    """
    Compute centroids of binary segmentation objects in physical coordinates.

    This function processes a binary segmentation image (typically output from
    a segmentation plugin) to identify connected components and compute their
    centroids in physical coordinates. It processes the image chunk-by-chunk
    with overlap to handle objects that span chunk boundaries efficiently.

    Args:
        image: Input binary dask array (typically 0/1 values) at highest resolution.
            Should be 3D with shape (z, y, x) or (x, y, z) depending on axes order,
            or 4D with shape (c, z, y, x) where c=1 (single channel). Multi-channel
            images (c>1) are not supported - process each channel separately.
        affine: 4x4 affine transformation matrix to convert voxel coordinates
            to physical coordinates. Can be a numpy array or AffineTransform object.
        depth: Number of elements of overlap between chunks. Can be:
            - int: same depth for all dimensions
            - tuple: different depth per dimension
            - dict: mapping dimension index to depth
            Default is 10 voxels of overlap.
        boundary: How to handle boundaries when adding overlap. Currently not used
            (always uses 'none' behavior). Reserved for future compatibility.
        rechunk: Optional rechunking specification before processing. Can be:
            - int: target chunk size for all dimensions
            - tuple: target chunk size per dimension
            - None: use existing chunks
            Default is None (use existing chunks).

    Returns:
        numpy.ndarray: Nx3 array of physical coordinates for N detected objects,
            where each row is [x, y, z] in physical space. The array has dtype
            float64.

    Notes:
        - Objects with centroids in the overlap regions are filtered out to
          avoid duplicate detections across chunks.
        - The function uses scikit-image's label() with connectivity=3 (26-connectivity
          in 3D) to identify connected components.
        - Empty chunks (no objects detected) contribute empty arrays to the result.
        - This function computes the result immediately (not lazy) to return a
          concrete numpy array.

    Examples:
        >>> import dask.array as da
        >>> import numpy as np
        >>> from zarrnii import compute_centroids
        >>>
        >>> # Create a binary segmentation image
        >>> binary_seg = da.from_array(
        ...     np.random.random((100, 100, 100)) > 0.95,
        ...     chunks=(50, 50, 50)
        ... )
        >>>
        >>> # Create an affine transform (e.g., 1mm isotropic voxels)
        >>> affine = np.eye(4)
        >>>
        >>> # Compute centroids
        >>> centroids = compute_centroids(binary_seg, affine, depth=5)
        >>> print(f"Found {len(centroids)} objects with shape {centroids.shape}")
    """
    # Import required modules
    from dask.array.overlap import overlap, trim_overlap

    from .transform import AffineTransform

    # Convert affine to numpy array if it's an AffineTransform object
    if isinstance(affine, AffineTransform):
        affine_matrix = affine.matrix
    else:
        affine_matrix = np.asarray(affine)

    # Validate affine matrix shape
    if affine_matrix.shape != (4, 4):
        raise ValueError(f"Affine matrix must be 4x4, got shape {affine_matrix.shape}")

    # Handle 4D images with channel dimension (CZYX format)
    if image.ndim == 4:
        # Check if first dimension is channel dimension (size 1 is common)
        if image.shape[0] == 1:
            # Squeeze the channel dimension to get 3D image
            image = image[0]
        else:
            # Multiple channels - raise informative error for now
            raise ValueError(
                f"Image has {image.ndim}D shape {image.shape} with "
                f"{image.shape[0]} channels. "
                "compute_centroids only supports 3D images or 4D images "
                "with a single channel (channel dimension size = 1). "
                "For multi-channel images, please process each channel "
                "separately or squeeze/select a single channel before "
                "calling this function."
            )
    elif image.ndim not in [1, 2, 3]:
        raise ValueError(
            f"Image must be 1D, 2D, 3D, or 4D (with single channel), "
            f"got {image.ndim}D with shape {image.shape}"
        )

    # Rechunk if requested
    if rechunk is not None:
        image = image.rechunk(rechunk)

    # Parse depth parameter
    ndim = image.ndim
    if isinstance(depth, int):
        depth_tuple = tuple([depth] * ndim)
    elif isinstance(depth, dict):
        depth_tuple = tuple([depth.get(i, 0) for i in range(ndim)])
    else:
        depth_tuple = tuple(depth)

    # Add overlap to the image
    expanded = overlap(image, depth=depth_tuple, boundary=boundary)

    # Define the processing function to run on each block
    def detect_centroids(block, block_info=None):
        """
        Detect centroids in a block and store them with coordinates.

        This function labels connected components in the block, computes their
        centroids, and stores the centroid coordinates in a structured output array.
        Centroids in the overlap regions are excluded.

        Args:
            block: Input block with overlap
            block_info: Block metadata from map_blocks

        Returns:
            Array with same shape as block, with last dimension containing
            centroid coordinates (padded with zeros for non-centroid locations)
        """
        # Get block location in the array
        if block_info is None:
            # No block info means we're testing the function
            output_shape = block.shape + (
                ndim + 1,
            )  # Extra dimension for coords + marker
            return np.zeros(output_shape, dtype=np.float32)

        # Label connected components
        labeled = label(block, connectivity=3)

        # Create output array: same spatial dims + 1 extra dim for storing coordinates
        # Shape: (...spatial..., ndim+1) where last dim is [marker, coord1, coord2, ...]
        output_shape = block.shape + (ndim + 1,)
        centroid_data = np.zeros(output_shape, dtype=np.float32)

        # If no objects, return empty array
        if labeled.max() == 0:
            return centroid_data

        # Get region properties
        regions = regionprops(labeled)

        # Determine the core region (exclude overlap)
        # Core region starts at depth_tuple and ends at shape - depth_tuple
        core_slices = tuple(
            slice(depth_tuple[i], block.shape[i] - depth_tuple[i]) for i in range(ndim)
        )

        # Process each region
        for region in regions:
            centroid = region.centroid

            # Check if centroid is in the core region
            in_core = all(
                core_slices[i].start <= centroid[i] < core_slices[i].stop
                for i in range(ndim)
            )

            if in_core:
                # Store centroid at rounded location with marker=1 and local coordinates
                # Round to nearest integer for indexing
                centroid_idx = tuple(int(round(centroid[i])) for i in range(ndim))
                # Ensure index is within bounds
                if all(0 <= centroid_idx[i] < block.shape[i] for i in range(ndim)):
                    # Mark this location as a centroid
                    centroid_data[centroid_idx + (0,)] = 1.0
                    # Store the LOCAL fractional coordinates
                    # After trim_overlap, these will map to the correct global positions
                    for i in range(ndim):
                        centroid_data[centroid_idx + (i + 1,)] = centroid[i]

        return centroid_data

    # Apply the detection function to all blocks
    centroid_array = expanded.map_blocks(
        detect_centroids,
        dtype=np.float32,
        meta=np.array((), dtype=np.float32),
        new_axis=ndim,  # Add one extra dimension for coordinates
        chunks=expanded.chunks + ((ndim + 1,),),  # Chunks for spatial dims + coords dim
    )

    # Trim the overlap from the spatial dimensions (keep the coordinate dimension)
    # Build the depth parameter for trim_overlap - only trim spatial dimensions
    trim_depth = {i: depth_tuple[i] for i in range(ndim)}

    centroid_array_trimmed = trim_overlap(
        centroid_array, depth=trim_depth, boundary=boundary
    )

    # Compute the result
    centroid_data_computed = centroid_array_trimmed.compute()

    # Extract centroids: find all locations where marker (first channel) is > 0
    marker_channel = centroid_data_computed[..., 0]
    centroid_mask = marker_channel > 0

    # Get integer indices where centroids are marked
    centroid_integer_locations = np.argwhere(centroid_mask)

    if len(centroid_integer_locations) == 0:
        return np.empty((0, 3), dtype=np.float64)

    # For each centroid, get the stored fractional offset and compute actual
    # position. The stored coordinates are LOCAL to the overlapped block, but
    # after trim_overlap, the integer indices give us the position in the
    # trimmed array. We need to compute:
    # trimmed_integer_index + (stored_local_coord - rounded(stored_local_coord))
    voxel_coords = []
    for int_loc in centroid_integer_locations:
        # Get the stored local fractional coordinates
        stored_coords = centroid_data_computed[
            tuple(int_loc) + (slice(1, ndim + 1),)
        ]

        # The integer location int_loc is where we marked the centroid
        # (rounded position). The actual centroid position should be
        # reconstructed from int_loc. After trim_overlap, int_loc is in the
        # coordinate system of the original (non-overlapped) array.
        # The stored_coords contain the fractional part relative to the
        # overlapped block. But we need to adjust: the true position is where
        # the marker is in the trimmed array plus the fractional offset from
        # rounding.

        # Actually, after trim_overlap, the array indices directly correspond
        # to original array indices. The stored_coords are in local
        # (overlapped block) coordinates. We marked at round(local_coord),
        # which after trim becomes the int_loc we found. The fractional
        # offset is: stored_coords - round(stored_coords)
        fractional_offset = stored_coords - np.round(stored_coords)
        actual_coord = int_loc.astype(np.float64) + fractional_offset
        voxel_coords.append(actual_coord)

    voxel_coords = np.array(voxel_coords, dtype=np.float64)

    # Convert to physical coordinates
    n_points = voxel_coords.shape[0]
    voxel_homogeneous = np.column_stack(
        [voxel_coords, np.ones((n_points, 1), dtype=np.float64)]
    )

    physical_homogeneous = voxel_homogeneous @ affine_matrix.T
    physical_coords = physical_homogeneous[:, :3]

    return physical_coords
