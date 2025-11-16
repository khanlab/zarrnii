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
    centroids in physical coordinates. It uses dask's map_overlap to efficiently
    process large images in chunks with overlap to handle objects that span
    chunk boundaries.

    Args:
        image: Input binary dask array (typically 0/1 values) at highest resolution.
            Should be 3D with shape (z, y, x) or (x, y, z) depending on axes order.
        affine: 4x4 affine transformation matrix to convert voxel coordinates
            to physical coordinates. Can be a numpy array or AffineTransform object.
        depth: Number of elements of overlap between chunks. Can be:
            - int: same depth for all dimensions
            - tuple: different depth per dimension
            - dict: mapping dimension index to depth
            Default is 10 voxels of overlap.
        boundary: How to handle boundaries when adding overlap. Options include
            'none', 'reflect', 'periodic', 'nearest', or constant values.
            Default is 'none' (no padding at array boundaries).
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
    # Import AffineTransform to handle both numpy arrays and AffineTransform objects
    from .transform import AffineTransform

    # Convert affine to numpy array if it's an AffineTransform object
    if isinstance(affine, AffineTransform):
        affine_matrix = affine.matrix
    else:
        affine_matrix = np.asarray(affine)

    # Validate affine matrix shape
    if affine_matrix.shape != (4, 4):
        raise ValueError(
            f"Affine matrix must be 4x4, got shape {affine_matrix.shape}"
        )

    # Rechunk if requested
    if rechunk is not None:
        image = image.rechunk(rechunk)

    # Ensure chunks are large enough to avoid automatic rechunking by map_overlap
    # map_overlap will rechunk if depth >= chunk_size
    # To avoid this, we need chunks to be at least 2*depth in each dimension
    min_chunk_size = 2 * (depth if isinstance(depth, int) else max(overlap_sizes if isinstance(overlap_sizes, (list, tuple)) else depth.values()))
    
    # Check if rechunking is needed
    needs_rechunk = False
    for dim_chunks in image.chunks:
        if any(c < min_chunk_size for c in dim_chunks):
            needs_rechunk = True
            break
    
    if needs_rechunk:
        # Rechunk to ensure minimum chunk size
        target_chunks = tuple(
            max(min_chunk_size, image.shape[i] // 4)  # At least 4 chunks per dimension if possible
            for i in range(image.ndim)
        )
        image = image.rechunk(target_chunks)

    # Store original shape before overlap
    original_shape = image.shape

    # Parse depth parameter
    ndim = image.ndim
    if isinstance(depth, int):
        overlap_sizes = [depth] * ndim
    elif isinstance(depth, dict):
        overlap_sizes = [depth.get(i, 0) for i in range(ndim)]
    else:
        overlap_sizes = list(depth)

    def _process_chunk(block, block_info=None):
        """
        Process a single chunk with overlap to find centroids.

        This function receives a block with overlap added and returns
        centroids in physical coordinates for objects whose centroids
        fall within the core (non-overlap) region.

        Args:
            block: The image block with overlap added
            block_info: Dictionary containing block location information

        Returns:
            2D numpy array of shape (n_objects, 3) with physical coordinates
        """
        if block_info is None:
            return np.empty((0, 3), dtype=np.float64)

        # Get the block location in the original array (with overlap included)
        # array_location gives us the range in original array coordinates
        # that this block covers (including the overlap regions)
        array_location = block_info[0]["array-location"]

        # Determine the core region (excluding overlap)
        # The core region is the part that belongs to this chunk, not borrowed from neighbors
        core_slices = []
        core_start_global = []  # Start of core in global coords

        for i in range(ndim):
            #  array_location[i] = (start, end) in original array coords
            block_start = array_location[i][0]
            block_end = array_location[i][1]

            # Determine if this block is at the start or end of the array
            is_at_start = block_start == 0
            is_at_end = block_end >= original_shape[i]

            # Overlap is only added where there are neighboring chunks
            # If at start, no overlap at beginning; if at end, no overlap at end
            overlap_at_start = 0 if is_at_start else overlap_sizes[i]
            overlap_at_end = 0 if is_at_end else overlap_sizes[i]

            # Core region in local block coordinates
            core_start_local = overlap_at_start
            core_end_local = block.shape[i] - overlap_at_end

            core_slices.append(slice(core_start_local, core_end_local))

            # Core start in global coordinates
            core_start_global.append(block_start + overlap_at_start)

        # Label connected components in the block
        labeled = label(block, connectivity=3)

        # If no objects, return empty array
        if labeled.max() == 0:
            return np.empty((0, 3), dtype=np.float64)

        # Get region properties
        regions = regionprops(labeled)

        # Collect centroids that fall within the core region
        valid_centroids = []

        for region in regions:
            # Get centroid in local block coordinates
            centroid = region.centroid

            # Check if centroid is in the core region (not in overlap)
            in_core = True
            for i, (coord, core_slice) in enumerate(zip(centroid, core_slices)):
                if not (core_slice.start <= coord < core_slice.stop):
                    in_core = False
                    break

            if in_core:
                # Convert to global voxel coordinates in original array
                # The block starts at array_location[i][0] in global coords
                # The centroid at local position x corresponds to global position:
                # array_location[i][0] + x
                global_coords = [
                    array_location[i][0] + centroid[i] for i in range(ndim)
                ]

                valid_centroids.append(global_coords)

        if len(valid_centroids) == 0:
            return np.empty((0, 3), dtype=np.float64)

        voxel_coords = np.array(valid_centroids, dtype=np.float64)

        # Convert to physical coordinates using affine transform
        n_points = voxel_coords.shape[0]
        voxel_homogeneous = np.column_stack(
            [voxel_coords, np.ones((n_points, 1), dtype=np.float64)]
        )

        # Apply affine transform
        physical_homogeneous = voxel_homogeneous @ affine_matrix.T
        physical_coords = physical_homogeneous[:, :3]

        return physical_coords

    # Use map_overlap to process chunks with overlap
    # The output is object dtype because each chunk returns variable-sized arrays
    result = da.map_overlap(
        _process_chunk,
        image,
        depth=depth,
        boundary=boundary,
        dtype=object,
        drop_axis=list(range(ndim)),
        new_axis=0,
        trim=True,  # Trim overlap after processing
    )

    # Compute all blocks and collect results
    computed_results = result.compute()

    # Handle the result structure - could be a scalar or array depending on chunks
    all_coords = []

    def collect_coords(item):
        """Recursively collect coordinate arrays."""
        if isinstance(item, np.ndarray):
            if item.dtype == object:
                # This is an array of arrays
                for sub_item in item.flat:
                    collect_coords(sub_item)
            elif len(item.shape) == 2 and item.shape[1] == 3 and item.shape[0] > 0:
                # This is a coordinate array
                all_coords.append(item)
        # else: ignore empty or invalid items

    collect_coords(computed_results)

    if len(all_coords) == 0:
        return np.empty((0, 3), dtype=np.float64)

    return np.vstack(all_coords)
