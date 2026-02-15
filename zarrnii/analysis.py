"""
Image analysis functions for zarrnii.

This module provides functions for image analysis operations such as
histogram computation, threshold calculation, and MIP visualization.
"""

from __future__ import annotations

import operator as op
from typing import Any, Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
from skimage.filters import threshold_multiotsu
from skimage.measure import label, regionprops


def compute_histogram(
    image: da.Array,
    bins: Optional[int] = None,
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
        bins: Number of histogram bins (default: bin width 1, bins=max - min + 1)
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
        if range is None or bins is None:
            data_min = da.min(valid_data).compute()
            data_max = da.max(valid_data).compute()
        if range is None:
            range = (data_min, data_max)
        if bins is None:
            calculated_bins = int(data_max - data_min + 1)
            # Cap at a reasonable maximum to avoid memory issues
            bins = min(calculated_bins, 65536)
            # Ensure at least 2 bins for meaningful histogram
            if bins < 2:
                bins = 2

        return da.histogram(valid_data, bins=bins, range=range, **kwargs)
    else:
        # For dask histogram, we need to provide a range
        if range is None or bins is None:
            data_min = da.min(image).compute()
            data_max = da.max(image).compute()
        if range is None:
            range = (data_min, data_max)
        if bins is None:
            calculated_bins = int(data_max - data_min + 1)
            # Cap at a reasonable maximum to avoid memory issues
            bins = min(calculated_bins, 65536)
            # Ensure at least 2 bins for meaningful histogram
            if bins < 2:
                bins = 2

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


def create_mip_visualization(
    image: da.Array,
    dims: List[str],
    scale: dict,
    plane: str = "axial",
    slab_thickness_um: float = 100.0,
    slab_spacing_um: float = 100.0,
    channel_colors: Optional[
        List[Union[str, Tuple[float, float, float], Tuple[float, float, float, float]]]
    ] = None,
    channel_ranges: Optional[List[Tuple[float, float]]] = None,
    omero_metadata: Optional[Any] = None,
    channel_labels: Optional[List[str]] = None,
    return_slabs: bool = False,
    scale_units: str = "mm",
) -> Union[List[da.Array], Tuple[List[da.Array], List[dict]]]:
    """
    Create Maximum Intensity Projection (MIP) visualizations across slabs.

    This function generates MIP visualizations by dividing the volume into slabs
    along the specified plane, computing the maximum intensity projection within
    each slab using dask operations, then rendering with channel-specific colors.

    Args:
        image: Input dask array image with shape matching dims. Should include
            spatial dimensions (x, y, z) and optionally channel dimension (c).
        dims: List of dimension names matching image shape (e.g., ['c', 'z', 'y', 'x'])
        scale: Dictionary mapping dimension names to spacing values. Units determined
            by scale_units parameter (e.g., {'x': 1.0, 'y': 1.0, 'z': 2.0})
        plane: Projection plane - one of 'axial', 'coronal', 'sagittal'.
            - 'axial': projects along z-axis (creates xy slices)
            - 'coronal': projects along y-axis (creates xz slices)
            - 'sagittal': projects along x-axis (creates yz slices)
        slab_thickness_um: Thickness of each slab in microns (default: 100.0)
        slab_spacing_um: Spacing between slab centers in microns (default: 100.0)
        channel_colors: Optional list of colors for each channel. Each color can be:
            - Color name string (e.g., 'red', 'green', 'blue')
            - RGB tuple with values 0-1 (e.g., (1.0, 0.0, 0.0) for red)
            - RGBA tuple with values 0-1 (e.g., (1.0, 0.0, 0.0, 0.5) for
                semi-transparent red)
            If None and omero_metadata is provided, uses OMERO channel
                colors.
            Otherwise uses default colors: ['red', 'green', 'blue', 'cyan',
                'magenta', 'yellow']
        channel_ranges: Optional list of (min, max) tuples specifying
            intensity range for each channel. If None and omero_metadata is
            provided, uses OMERO window settings. Otherwise uses auto-scaling
            based on data min/max.
        omero_metadata: Optional OMERO metadata object containing channel
            information. Used to extract default colors and intensity ranges
            when channel_colors or channel_ranges are not explicitly provided.
        channel_labels: Optional list of channel label names to use for
            selecting channels from OMERO metadata. If provided, channels are
            filtered and reordered to match these labels. Requires
            omero_metadata to be provided.
        return_slabs: If True, returns tuple of (mip_list, slab_info_list) where
            slab_info_list contains metadata about each slab. If False (default),
            returns only the mip_list.
        scale_units: Units for scale values. Either "mm" (millimeters, default) or
            "um" (microns). When "mm", scale values are converted to microns internally
            (multiplied by 1000). This parameter reflects the NGFF/NIfTI convention
            where scale values are typically in millimeters.

    Returns:
        If return_slabs is False (default):
            List of 2D dask arrays, each containing an RGB MIP
            visualization for one slab. Each array has shape (height, width,
            3) with RGB values in range [0, 1]. Arrays are lazy and will
            only be computed when explicitly requested.

        If return_slabs is True:
            Tuple of (mip_list, slab_info_list) where:
            - mip_list: List of 2D RGB dask arrays as described above
            - slab_info_list: List of dictionaries with slab metadata
                including:
                - 'start_um': Start position of slab in microns
                - 'end_um': End position of slab in microns
                - 'center_um': Center position of slab in microns
                - 'start_idx': Start index in array coordinates
                - 'end_idx': End index in array coordinates

    Raises:
        ValueError: If plane is not one of 'axial', 'coronal', 'sagittal'
        ValueError: If required spatial dimensions are not in dims
        ValueError: If number of channels exceeds number of colors and
            channel_colors not provided
        ValueError: If channel_labels specified but omero_metadata not
            provided
        ValueError: If channel_labels contains labels not found in
            omero_metadata

    Examples:
        >>> import dask.array as da
        >>> from zarrnii.analysis import create_mip_visualization
        >>>
        >>> # Create test data with 2 channels
        >>> data = da.random.random((2, 100, 100, 100), chunks=(1, 50, 50, 50))
        >>> dims = ['c', 'z', 'y', 'x']
        >>> scale = {'z': 0.002, 'y': 0.001, 'x': 0.001}  # 2mm z, 1mm x/y in mm
        >>>
        >>> # Create axial MIPs with custom intensity ranges and 100 micron slabs (scale in mm by default)
        >>> mips = create_mip_visualization(
        ...     data, dims, scale,
        ...     plane='axial',
        ...     slab_thickness_um=100.0,
        ...     slab_spacing_um=100.0,
        ...     channel_colors=['red', 'green'],
        ...     channel_ranges=[(0.0, 0.8), (0.2, 1.0)]
        ... )
        >>>
        >>> # Or if scale is already in microns, specify scale_units='um'
        >>> scale_um = {'z': 2.0, 'y': 1.0, 'x': 1.0}  # 2um z, 1um x/y
        >>> mips = create_mip_visualization(
        ...     data, dims, scale_um,
        ...     plane='axial',
        ...     slab_thickness_um=100.0,
        ...     scale_units='um'
        ... )
        >>>
        >>> # Use OMERO metadata for colors and ranges
        >>> mips = create_mip_visualization(
        ...     data, dims, scale,
        ...     plane='axial',
        ...     omero_metadata=omero,
        ...     channel_labels=['DAPI', 'GFP']
        ... )
        >>>
        >>> # Use alpha transparency
        >>> mips = create_mip_visualization(
        ...     data, dims, scale,
        ...     plane='axial',
        ...     channel_colors=[(1.0, 0.0, 0.0, 0.7), (0.0, 1.0, 0.0, 0.5)]
        ... )
    """
    # Validate plane
    valid_planes = ["axial", "coronal", "sagittal"]
    if plane not in valid_planes:
        raise ValueError(f"plane must be one of {valid_planes}, got '{plane}'")

    # Validate and handle scale units
    valid_units = ["mm", "um"]
    if scale_units not in valid_units:
        raise ValueError(
            f"scale_units must be one of {valid_units}, got '{scale_units}'"
        )

    # Convert scale to microns if needed
    # NGFF/NIfTI convention: scale values are in millimeters
    # But slab parameters are in microns for better precision
    if scale_units == "mm":
        # Convert mm to um (1 mm = 1000 um)
        scale_um = {k: v * 1000.0 for k, v in scale.items()}
    else:
        # Already in microns
        scale_um = scale

    # Map plane to axis dimension
    plane_axis_map = {
        "axial": "z",  # projects along z, shows xy
        "coronal": "y",  # projects along y, shows xz
        "sagittal": "x",  # projects along x, shows yz
    }
    projection_axis = plane_axis_map[plane]

    # Validate that required dimensions exist
    if projection_axis not in dims:
        raise ValueError(
            f"Projection axis '{projection_axis}' not found in dims {dims}"
        )

    # Check if image has channel dimension
    has_channel = "c" in dims
    if has_channel:
        channel_idx = dims.index("c")
        n_channels = image.shape[channel_idx]
    else:
        n_channels = 1

    # Handle channel_labels if provided
    channel_indices = None
    if channel_labels is not None:
        if omero_metadata is None:
            raise ValueError(
                "channel_labels specified but omero_metadata not provided. "
                "OMERO metadata is required to map channel labels to indices."
            )
        if not hasattr(omero_metadata, "channels"):
            raise ValueError(
                "omero_metadata does not have 'channels' attribute. "
                "Please provide valid OMERO metadata."
            )

        # Extract channel labels from OMERO metadata
        omero_channels = omero_metadata.channels
        available_labels = []
        for ch in omero_channels:
            if isinstance(ch, dict):
                available_labels.append(ch.get("label", ""))
            else:
                available_labels.append(getattr(ch, "label", ""))

        # Map requested labels to indices
        channel_indices = []
        for label in channel_labels:
            try:
                idx = available_labels.index(label)
                channel_indices.append(idx)
            except ValueError:
                raise ValueError(
                    f"Channel label '{label}' not found in OMERO metadata. "
                    f"Available labels: {available_labels}"
                )

        # Update n_channels to match requested channels
        n_channels = len(channel_indices)

    # Extract colors from OMERO metadata if not provided
    if (
        channel_colors is None
        and omero_metadata is not None
        and hasattr(omero_metadata, "channels")
    ):
        omero_channels = omero_metadata.channels
        channel_colors = []

        # Get indices to use (either filtered by channel_labels or all)
        indices_to_use = (
            channel_indices
            if channel_indices is not None
            else range(len(omero_channels))
        )

        for idx in indices_to_use:
            if idx < len(omero_channels):
                ch = omero_channels[idx]
                # Extract color from OMERO channel
                if isinstance(ch, dict):
                    color_hex = ch.get("color", "FFFFFF")
                else:
                    color_hex = getattr(ch, "color", "FFFFFF")

                # Convert hex color to RGB tuple
                if color_hex:
                    # Remove leading # if present
                    color_hex = color_hex.lstrip("#")
                    # Convert hex to RGB (0-1 range)
                    r = int(color_hex[0:2], 16) / 255.0
                    g = int(color_hex[2:4], 16) / 255.0
                    b = int(color_hex[4:6], 16) / 255.0
                    channel_colors.append((r, g, b))
                else:
                    # Fallback to default colors
                    default_colors = [
                        "red",
                        "green",
                        "blue",
                        "cyan",
                        "magenta",
                        "yellow",
                    ]
                    channel_colors.append(default_colors[idx % len(default_colors)])

    # Extract intensity ranges from OMERO metadata if not provided
    if (
        channel_ranges is None
        and omero_metadata is not None
        and hasattr(omero_metadata, "channels")
    ):
        omero_channels = omero_metadata.channels
        channel_ranges = []

        # Get indices to use (either filtered by channel_labels or all)
        indices_to_use = (
            channel_indices
            if channel_indices is not None
            else range(len(omero_channels))
        )

        for idx in indices_to_use:
            if idx < len(omero_channels):
                ch = omero_channels[idx]
                # Extract window from OMERO channel
                if isinstance(ch, dict):
                    window = ch.get("window", {})
                    if isinstance(window, dict):
                        window_start = window.get("start", None)
                        window_end = window.get("end", None)
                    else:
                        window_start = getattr(window, "start", None)
                        window_end = getattr(window, "end", None)
                else:
                    window = getattr(ch, "window", None)
                    if window is not None:
                        window_start = getattr(window, "start", None)
                        window_end = getattr(window, "end", None)
                    else:
                        window_start = None
                        window_end = None

                # Use window range if available
                if window_start is not None and window_end is not None:
                    channel_ranges.append((float(window_start), float(window_end)))
                else:
                    channel_ranges.append(None)  # Use auto-scaling

    # Set up default colors if still not provided
    if channel_colors is None:
        default_colors = ["red", "green", "blue", "cyan", "magenta", "yellow"]
        if n_channels > len(default_colors):
            raise ValueError(
                f"Image has {n_channels} channels but only {len(default_colors)} "
                f"default colors. Please provide channel_colors parameter."
            )
        channel_colors = default_colors[:n_channels]
    elif len(channel_colors) < n_channels:
        raise ValueError(
            f"Provided {len(channel_colors)} colors but image has {n_channels} channels"
        )

    # Validate channel_ranges if provided
    if channel_ranges is not None:
        if len(channel_ranges) < n_channels:
            raise ValueError(
                f"Provided {len(channel_ranges)} intensity ranges but image has "
                f"{n_channels} channels"
            )

    # Convert color names to RGBA tuples (with alpha support)
    def color_to_rgba(color):
        """Convert color name or tuple to RGBA tuple."""
        if isinstance(color, str):
            # Import matplotlib for color conversion
            try:
                import matplotlib.colors as mcolors

                rgb = mcolors.to_rgb(color)
                return rgb + (1.0,)  # Add full opacity
            except ImportError:
                # Fallback to basic colors if matplotlib not available
                basic_colors = {
                    "red": (1.0, 0.0, 0.0),
                    "green": (0.0, 1.0, 0.0),
                    "blue": (0.0, 0.0, 1.0),
                    "cyan": (0.0, 1.0, 1.0),
                    "magenta": (1.0, 0.0, 1.0),
                    "yellow": (1.0, 1.0, 0.0),
                    "white": (1.0, 1.0, 1.0),
                }
                if color.lower() in basic_colors:
                    rgb = basic_colors[color.lower()]
                    return rgb + (1.0,)  # Add full opacity
                else:
                    raise ValueError(
                        f"Color '{color}' not recognized. Install matplotlib for "
                        f"full color support or use: {list(basic_colors.keys())}"
                    )
        # Handle tuple colors (RGB or RGBA)
        if len(color) == 3:
            return color + (1.0,)  # Add full opacity to RGB
        elif len(color) == 4:
            return tuple(color)  # Already RGBA
        else:
            raise ValueError(
                f"Color tuple must have 3 (RGB) or 4 (RGBA) values, got {len(color)}"
            )

    rgba_colors = [color_to_rgba(c) for c in channel_colors]

    # Get projection axis index and size
    proj_axis_idx = dims.index(projection_axis)
    proj_axis_size = image.shape[proj_axis_idx]
    proj_axis_spacing_um = scale_um.get(projection_axis, 1000.0)  # Default 1mm = 1000um

    # Calculate slab parameters (now both in microns)
    slab_thickness_idx = int(np.ceil(slab_thickness_um / proj_axis_spacing_um))
    slab_spacing_idx = int(np.round(slab_spacing_um / proj_axis_spacing_um))

    # Ensure both are at least 1 to avoid issues
    # This can happen when slab thickness/spacing in microns is smaller than voxel spacing
    slab_thickness_idx = max(1, slab_thickness_idx)
    slab_spacing_idx = max(1, slab_spacing_idx)

    # Generate slab positions
    slab_centers_idx = []
    current_pos = slab_thickness_idx // 2  # Start from half slab thickness
    while current_pos < proj_axis_size:
        slab_centers_idx.append(current_pos)
        current_pos += slab_spacing_idx

    # If no slabs fit, create at least one centered slab
    if len(slab_centers_idx) == 0:
        slab_centers_idx = [proj_axis_size // 2]

    # Generate MIPs for each slab
    mip_list = []
    slab_info_list = []

    for center_idx in slab_centers_idx:
        # Calculate slab bounds
        half_thickness = slab_thickness_idx // 2
        start_idx = max(0, center_idx - half_thickness)
        end_idx = min(proj_axis_size, center_idx + half_thickness)

        # Ensure we have at least one slice (handle edge case where thickness=1)
        if end_idx <= start_idx:
            end_idx = min(start_idx + 1, proj_axis_size)

        # Store slab info (positions in microns)
        slab_info = {
            "start_um": start_idx * proj_axis_spacing_um,
            "end_um": end_idx * proj_axis_spacing_um,
            "center_um": center_idx * proj_axis_spacing_um,
            "start_idx": start_idx,
            "end_idx": end_idx,
        }
        slab_info_list.append(slab_info)

        # Extract slab from image using slice along projection axis
        slices = [slice(None)] * len(dims)
        slices[proj_axis_idx] = slice(start_idx, end_idx)
        slab_data = image[tuple(slices)]

        # Compute MIP along projection axis
        mip_data = slab_data.max(axis=proj_axis_idx)

        # Now mip_data should have spatial dimensions (and channel if present)
        # Shape after max: depends on dims without projection axis

        # Create RGB visualization
        if has_channel:
            # Process each channel and combine with colors - using lazy dask operations
            # Keep mip_data as dask array, don't compute yet

            # If channel_indices is specified, we need to select those channels
            if channel_indices is not None:
                # Move channel axis to first position for easier iteration
                channel_axis_after_max = (
                    channel_idx if channel_idx < proj_axis_idx else channel_idx - 1
                )
                mip_channels = da.moveaxis(mip_data, channel_axis_after_max, 0)
                # Select only the requested channels
                mip_channels = mip_channels[channel_indices]
            else:
                # Move channel axis to first position for easier iteration
                channel_axis_after_max = (
                    channel_idx if channel_idx < proj_axis_idx else channel_idx - 1
                )
                mip_channels = da.moveaxis(mip_data, channel_axis_after_max, 0)

            # Get spatial dimensions after removing projection axis
            spatial_shape = mip_channels.shape[1:]

            # Initialize RGB image as dask array
            rgb_image = da.zeros(
                spatial_shape + (3,),
                dtype=np.float32,
                chunks=mip_channels.chunks[1:] + (-1,),
            )

            # Combine channels with their colors
            for ch_idx in range(n_channels):
                channel_data = mip_channels[ch_idx]

                # Determine intensity range for normalization
                if channel_ranges is not None and channel_ranges[ch_idx] is not None:
                    # Use custom intensity range
                    ch_min, ch_max = channel_ranges[ch_idx]
                else:
                    # Use auto-scaling based on data - need to compute min/max
                    # This is a minimal computation that's necessary for auto-scaling
                    ch_min = float(channel_data.min().compute())
                    ch_max = float(channel_data.max().compute())

                # Normalize to [0, 1]
                if ch_max > ch_min:
                    channel_normalized = (channel_data - ch_min) / (ch_max - ch_min)
                    # Clip to [0, 1] range in case data extends beyond specified range
                    channel_normalized = da.clip(channel_normalized, 0.0, 1.0)
                elif ch_max > 0:
                    # Uniform non-zero values - keep them
                    channel_normalized = da.ones_like(channel_data)
                else:
                    # All zeros - keep as zeros
                    channel_normalized = da.zeros_like(channel_data)

                # Apply color with alpha blending (multiply by RGBA color values)
                color_rgba = rgba_colors[ch_idx]
                alpha = color_rgba[3]  # Extract alpha value
                for rgb_idx in range(3):
                    rgb_image[..., rgb_idx] = rgb_image[..., rgb_idx] + (
                        channel_normalized * color_rgba[rgb_idx] * alpha
                    )

            # Clip to [0, 1] range
            rgb_image = da.clip(rgb_image, 0.0, 1.0)

        else:
            # Single channel - use lazy dask operations

            # Determine intensity range for normalization
            if channel_ranges is not None and channel_ranges[0] is not None:
                # Use custom intensity range
                ch_min, ch_max = channel_ranges[0]
            else:
                # Use auto-scaling based on data - need to compute min/max
                # This is a minimal computation that's necessary for auto-scaling
                ch_min = float(mip_data.min().compute())
                ch_max = float(mip_data.max().compute())

            # Normalize to [0, 1]
            if ch_max > ch_min:
                normalized = (mip_data - ch_min) / (ch_max - ch_min)
                # Clip to [0, 1] range in case data extends beyond specified range
                normalized = da.clip(normalized, 0.0, 1.0)
            elif ch_max > 0:
                # Uniform non-zero values - keep them
                normalized = da.ones_like(mip_data)
            else:
                # All zeros - keep as zeros
                normalized = da.zeros_like(mip_data)

            # Apply first color with alpha
            color_rgba = rgba_colors[0]
            alpha = color_rgba[3]
            rgb_image = da.zeros(
                mip_data.shape + (3,), dtype=np.float32, chunks=mip_data.chunks + (-1,)
            )
            for rgb_idx in range(3):
                rgb_image[..., rgb_idx] = normalized * color_rgba[rgb_idx] * alpha

        mip_list.append(rgb_image)

    if return_slabs:
        return mip_list, slab_info_list
    else:
        return mip_list


def _apply_region_filter(region: Any, filters: Dict[str, Tuple[str, Any]]) -> bool:
    """
    Check if a region passes all specified filters.

    Args:
        region: A regionprops region object from scikit-image
        filters: Dictionary mapping property names to (operator, value) tuples

    Returns:
        True if the region passes ALL filters, False otherwise
    """
    # Map string operators to actual comparison functions
    operators = {
        ">": op.gt,
        ">=": op.ge,
        "<": op.lt,
        "<=": op.le,
        "==": op.eq,
        "!=": op.ne,
    }

    for prop_name, (operator_str, threshold) in filters.items():
        # Get the operator function
        if operator_str not in operators:
            raise ValueError(
                f"Invalid operator '{operator_str}'. "
                f"Must be one of: {list(operators.keys())}"
            )
        compare_func = operators[operator_str]

        # Get the property value from the region
        try:
            prop_value = getattr(region, prop_name)
        except AttributeError:
            raise ValueError(
                f"Invalid regionprops property '{prop_name}'. "
                "See scikit-image regionprops documentation for valid properties."
            )

        # Apply the comparison
        if not compare_func(prop_value, threshold):
            return False

    return True


# Properties that represent coordinates and need full affine transformation.
# This includes all regionprops properties whose values are voxel coordinates
# and should be mapped to physical space using the affine matrix.
COORDINATE_PROPERTIES = frozenset(
    [
        "centroid",
        "centroid_weighted",
        "bbox",  # bounding box coordinates (min_row, min_col, min_plane, max_row, ...)
        "centroid_local",  # centroid in local (bbox-relative) coordinates
        "coords",  # all voxel coordinates for the region
    ]
)

# Default output properties for compute_region_properties
DEFAULT_OUTPUT_PROPERTIES = ["centroid"]


def _transform_coordinate_to_physical(
    voxel_coords: np.ndarray, affine_matrix: np.ndarray
) -> np.ndarray:
    """
    Transform voxel coordinates to physical coordinates using affine matrix.

    Args:
        voxel_coords: Array of shape (N, 3) containing voxel coordinates
        affine_matrix: 4x4 affine transformation matrix

    Returns:
        Array of shape (N, 3) containing physical coordinates
    """
    if voxel_coords.size == 0:
        return voxel_coords

    n_points = voxel_coords.shape[0]
    voxel_homogeneous = np.column_stack(
        [voxel_coords, np.ones((n_points, 1), dtype=np.float64)]
    )
    physical_homogeneous = voxel_homogeneous @ affine_matrix.T
    return physical_homogeneous[:, :3]


def _extract_region_property(region: Any, prop_name: str) -> Any:
    """
    Extract a property value from a regionprops region object.

    Args:
        region: A regionprops region object from scikit-image
        prop_name: Name of the property to extract

    Returns:
        The property value (scalar, tuple, or array depending on property)

    Raises:
        ValueError: If the property name is invalid
    """
    try:
        return getattr(region, prop_name)
    except AttributeError:
        raise ValueError(
            f"Invalid regionprops property '{prop_name}'. "
            "See scikit-image regionprops documentation for valid properties."
        )


def compute_region_properties(
    image: da.Array,
    affine: np.ndarray,
    output_properties: Optional[Union[List[str], Dict[str, str]]] = None,
    depth: Union[int, Tuple[int, ...], Dict[int, int]] = 10,
    boundary: str = "none",
    rechunk: Optional[Union[int, Tuple[int, ...]]] = None,
    output_path: Optional[str] = None,
    region_filters: Optional[Dict[str, Tuple[str, Any]]] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Compute properties of binary segmentation objects with coordinate transformation.

    This function processes a binary segmentation image to identify connected
    components and compute their properties using scikit-image's regionprops.
    Coordinate-based properties (like centroid) are automatically transformed to
    physical coordinates. The function processes the image chunk-by-chunk with
    overlap to handle objects that span chunk boundaries efficiently.

    This is a generalized version of compute_centroids that allows extraction of
    any combination of regionprops properties, enabling downstream quantification
    and filtering based on the global Parquet output.

    Args:
        image: Input binary dask array (typically 0/1 values) at highest resolution.
            Should be 3D with shape (z, y, x) or (x, y, z) depending on axes order,
            4D with shape (c, z, y, x) where c=1 (single channel), or 5D with shape
            (t, c, z, y, x) where t=1 and c=1 (singleton time and channel).
            Multi-channel images (c>1) or multi-timepoint images (t>1) are not
            supported - process each channel/timepoint separately.
        affine: 4x4 affine transformation matrix to convert voxel coordinates
            to physical coordinates. Can be a numpy array or AffineTransform object.
        output_properties: Properties to extract. Can be either:
            - List of regionprops property names to extract. Property names are
              used as output keys.
            - Dict mapping regionprops property names to custom output names.
              Example: {'area': 'nvoxels', 'equivalent_diameter_area': 'equivdiam'}
            Coordinate properties ('centroid', 'centroid_weighted') are automatically
            transformed to physical coordinates and split into separate x, y, z
            columns. When using a dict, coordinate property output names are suffixed
            with '_x', '_y', '_z' (e.g., {'centroid': 'loc'} gives 'loc_x', 'loc_y',
            'loc_z').
            Default is ['centroid'] for backward compatibility.
            Example list: ['centroid', 'area', 'equivalent_diameter_area']
            Example dict: {'area': 'nvoxels', 'centroid': 'position'}
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
        output_path: Optional path to write properties to Parquet file instead of
            returning them in memory. If provided, properties are written to this
            file path and None is returned. Use this for large datasets to avoid
            memory issues. If None (default), properties are returned as a dict.
        region_filters: Optional dictionary specifying filters to apply to detected
            regions based on scikit-image regionprops properties. Each key is a
            property name (e.g., 'area', 'perimeter', 'eccentricity'), and the value
            is a tuple of (operator, threshold) where operator is one of:
            '>', '>=', '<', '<=', '==', '!='.
            Regions that don't satisfy ALL filters are excluded.
            Example: {'area': ('>=', 30), 'eccentricity': ('<', 0.9)}
            If None (default), no filtering is applied.

    Returns:
        Optional[Dict[str, numpy.ndarray]]: If output_path is None, returns a
            dictionary mapping property names (or custom names if dict was used)
            to numpy arrays. For coordinate properties like 'centroid', the keys
            are suffixed with _x, _y, _z (e.g., 'centroid_x' or 'custom_name_x')
            containing physical coordinates.
            Scalar properties have their name (or custom name) as the key.
            If output_path is provided, writes to Parquet file and returns None.

    Notes:
        - Objects with centroids in the overlap regions are filtered out to
          avoid duplicate detections across chunks.
        - The function uses scikit-image's label() with connectivity=3 (26-connectivity
          in 3D) to identify connected components.
        - Coordinate properties ('centroid', 'centroid_weighted') are transformed
          to physical coordinates and split into suffixed columns (e.g.,
          'centroid_x', 'centroid_y', 'centroid_z' or when renamed via dict,
          'custom_name_x', 'custom_name_y', 'custom_name_z').
        - Scalar properties are included directly without transformation.
        - Empty chunks (no objects detected) contribute empty arrays to the result.
        - This function computes the result immediately (not lazy).
        - Available regionprops properties include: 'area', 'area_bbox', 'centroid',
          'eccentricity', 'equivalent_diameter_area', 'euler_number', 'extent',
          'feret_diameter_max', 'axis_major_length', 'axis_minor_length',
          'moments', 'perimeter', 'solidity', and more.
          See scikit-image regionprops documentation for full list.

    Examples:
        >>> import dask.array as da
        >>> import numpy as np
        >>> from zarrnii import compute_region_properties
        >>>
        >>> # Create a binary segmentation image
        >>> binary_seg = da.from_array(
        ...     np.random.random((100, 100, 100)) > 0.95,
        ...     chunks=(50, 50, 50)
        ... )
        >>> affine = np.eye(4)
        >>>
        >>> # Extract centroid and area (default properties)
        >>> props = compute_region_properties(binary_seg, affine, depth=5)
        >>> print(f"Found {len(props['centroid_x'])} objects")
        >>>
        >>> # Extract multiple properties for downstream analysis
        >>> props = compute_region_properties(
        ...     binary_seg, affine, depth=5,
        ...     output_properties=['centroid', 'area', 'equivalent_diameter_area']
        ... )
        >>> print(f"Areas: {props['area']}")
        >>>
        >>> # With filtering and multiple properties
        >>> props = compute_region_properties(
        ...     binary_seg, affine, depth=5,
        ...     output_properties=['centroid', 'area', 'eccentricity'],
        ...     region_filters={'area': ('>=', 30)}
        ... )
        >>>
        >>> # Use dict to rename output columns
        >>> props = compute_region_properties(
        ...     binary_seg, affine, depth=5,
        ...     output_properties={'area': 'nvoxels', 'equivalent_diameter_area': 'equivdiam'}
        ... )
        >>> print(f"Number of voxels: {props['nvoxels']}")
        >>>
        >>> # Write to Parquet for large datasets
        >>> compute_region_properties(
        ...     binary_seg, affine, depth=5,
        ...     output_properties=['centroid', 'area', 'equivalent_diameter_area'],
        ...     output_path='region_props.parquet'
        ... )
    """
    # Import AffineTransform to handle both numpy arrays and AffineTransform objects
    from .transform import AffineTransform

    # Set default output properties
    if output_properties is None:
        output_properties = list(DEFAULT_OUTPUT_PROPERTIES)

    # Handle both list and dict input for output_properties
    # Extract property names (keys) and optional rename mapping (values)
    if isinstance(output_properties, dict):
        if len(output_properties) == 0:
            raise ValueError(
                "output_properties must be a non-empty list or dict. "
                f"Got: {output_properties}"
            )
        # Dict: keys are regionprops names, values are output names
        property_names = list(output_properties.keys())
        rename_mapping = output_properties
    elif isinstance(output_properties, list):
        if len(output_properties) == 0:
            raise ValueError(
                "output_properties must be a non-empty list or dict. "
                f"Got: {output_properties}"
            )
        property_names = output_properties
        # No renaming when list is used - use property names directly
        rename_mapping = {name: name for name in property_names}
    else:
        raise ValueError(
            "output_properties must be a non-empty list or dict. "
            f"Got: {output_properties}"
        )

    # Convert affine to numpy array if it's an AffineTransform object
    if isinstance(affine, AffineTransform):
        affine_matrix = affine.matrix
    else:
        affine_matrix = np.asarray(affine)

    # Validate affine matrix shape
    if affine_matrix.shape != (4, 4):
        raise ValueError(f"Affine matrix must be 4x4, got shape {affine_matrix.shape}")

    # Handle 5D images with time and channel dimensions (TCZYX format)
    if image.ndim == 5:
        # Check if time dimension is singleton (common for zarr data)
        if image.shape[0] == 1:
            # Squeeze singleton time dimension
            image = image[0]
            # Now image is 4D (CZYX) - fall through to 4D handler below
        elif image.shape[0] > 1:
            # Multiple timepoints - not supported
            raise ValueError(
                f"Image has 5D shape {image.shape} with {image.shape[0]} timepoints. "
                "compute_region_properties only supports 3D images or 4D/5D images with "
                "a single timepoint (t=1). Please select a single timepoint before "
                "calling this function."
            )

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
                f"{image.shape[0]} channels. compute_region_properties only supports "
                "3D images or 4D images with a single channel (channel dimension "
                "size = 1). For multi-channel images, please process each channel "
                "separately or squeeze/select a single channel before calling "
                "this function."
            )
    elif image.ndim not in [1, 2, 3]:
        raise ValueError(
            f"Image must be 1D, 2D, 3D, 4D (with single channel), or 5D (with "
            f"singleton time and channel), got {image.ndim}D with shape {image.shape}"
        )

    # Rechunk if requested
    if rechunk is not None:
        image = image.rechunk(rechunk)

    # Parse depth parameter to get overlap sizes
    ndim = image.ndim
    if isinstance(depth, int):
        overlap_sizes = tuple([depth] * ndim)
    elif isinstance(depth, dict):
        overlap_sizes = tuple([depth.get(i, 0) for i in range(ndim)])
    else:
        overlap_sizes = tuple(depth)

    def _block_properties(block, block_info=None):
        """
        Process a single block to find region properties.

        Args:
            block: numpy array for this chunk (binary mask) with overlap
            block_info: dict containing array location information from map_overlap

        Returns:
            Object array matching block shape, with properties stored in first element
        """
        # Create result array matching block shape
        result = np.empty(block.shape, dtype=object)

        # Handle empty blocks or no block_info
        if block.size == 0 or block_info is None:
            result.fill([])
            return result

        # Label connected components
        labeled = label(block > 0, connectivity=3)

        if labeled.max() == 0:
            # No objects found
            result.fill([])
            return result

        # Get original array location from block_info[None]
        original_info = block_info[None]
        array_location = original_info["array-location"]

        # Determine core region boundaries within the block
        core_slices = []
        for dim in range(len(array_location)):
            global_start, global_end = array_location[dim]
            core_size = global_end - global_start

            if global_start == 0:
                overlap_before = 0
            else:
                overlap_before = overlap_sizes[dim]

            core_start = overlap_before
            core_end = core_start + core_size

            core_slices.append((core_start, core_end))

        # Process regions and filter to core
        region_data_list = []
        for region in regionprops(labeled):
            # Apply region filters if specified
            if region_filters is not None:
                if not _apply_region_filter(region, region_filters):
                    continue

            centroid = np.array(region.centroid)

            # Check if centroid is in core region
            in_core = True
            for dim in range(len(centroid)):
                core_start, core_end = core_slices[dim]
                if not (core_start <= centroid[dim] < core_end):
                    in_core = False
                    break

            if in_core:
                # Extract all requested properties for this region
                region_data = {}

                for prop_name in property_names:
                    prop_value = _extract_region_property(region, prop_name)

                    if prop_name in COORDINATE_PROPERTIES:
                        # Convert coordinate properties to global voxel coordinates
                        coord = np.array(prop_value)
                        global_coord = []
                        for dim in range(len(coord)):
                            core_start, _ = core_slices[dim]
                            global_start, _ = array_location[dim]
                            global_coord.append(
                                global_start + (coord[dim] - core_start)
                            )
                        region_data[prop_name] = tuple(global_coord)
                    else:
                        # Non-coordinate properties are stored as-is
                        region_data[prop_name] = prop_value

                region_data_list.append(region_data)

        # Store all region data for this block in the first element
        result.fill([])
        if len(region_data_list) > 0:
            result.flat[0] = region_data_list

        return result

    # Apply block operation with overlap
    props_blocks = image.map_overlap(
        _block_properties,
        depth=overlap_sizes,
        boundary=boundary,
        trim=False,
        dtype=object,
        drop_axis=[],
    )

    # Handle Parquet output differently to avoid memory overflow
    if output_path is not None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from dask.delayed import delayed

        @delayed
        def process_block(block_result):
            """Process a single block and return properties with transformed coords."""
            # Extract region data from this block
            block_data = []
            if hasattr(block_result, "flat"):
                for item in block_result.flat:
                    if isinstance(item, list):
                        block_data.extend(item)
            elif isinstance(block_result, list):
                block_data = block_result

            if len(block_data) == 0:
                return None

            # Transform coordinate properties to physical coordinates
            processed_data = {}

            for prop_name in property_names:
                output_name = rename_mapping[prop_name]
                if prop_name in COORDINATE_PROPERTIES:
                    # Collect voxel coordinates
                    voxel_coords = np.array(
                        [region_data[prop_name] for region_data in block_data],
                        dtype=np.float64,
                    )
                    # Transform to physical coordinates
                    physical_coords = _transform_coordinate_to_physical(
                        voxel_coords, affine_matrix
                    )
                    # Split into suffixed x, y, z columns with renamed output
                    processed_data[f"{output_name}_x"] = physical_coords[:, 0]
                    processed_data[f"{output_name}_y"] = physical_coords[:, 1]
                    processed_data[f"{output_name}_z"] = physical_coords[:, 2]
                else:
                    # Non-coordinate properties
                    values = [region_data[prop_name] for region_data in block_data]
                    processed_data[output_name] = np.array(values, dtype=np.float64)

            return processed_data

        # Create delayed tasks for all blocks
        delayed_results = []
        for block_idx in np.ndindex(props_blocks.numblocks):
            block = props_blocks.blocks[block_idx]
            result_delayed = process_block(block)
            delayed_results.append(result_delayed)

        # Compute all blocks in parallel
        computed_results = da.compute(*delayed_results)

        # Build schema dynamically based on property_names and rename_mapping
        schema_fields = []
        for prop_name in property_names:
            output_name = rename_mapping[prop_name]
            if prop_name in COORDINATE_PROPERTIES:
                schema_fields.append(pa.field(f"{output_name}_x", pa.float64()))
                schema_fields.append(pa.field(f"{output_name}_y", pa.float64()))
                schema_fields.append(pa.field(f"{output_name}_z", pa.float64()))
            else:
                schema_fields.append(pa.field(output_name, pa.float64()))

        schema = pa.schema(schema_fields)

        # Write results to Parquet file
        writer = None
        for processed_data in computed_results:
            if processed_data is None:
                continue

            # Create PyArrow table for this batch
            table_data = {}
            for field in schema_fields:
                if field.name in processed_data:
                    table_data[field.name] = pa.array(
                        processed_data[field.name], type=pa.float64()
                    )

            table = pa.table(table_data)

            if writer is None:
                writer = pq.ParquetWriter(output_path, schema)

            writer.write_table(table)

        # Close the writer
        if writer is not None:
            writer.close()
        else:
            # No data found - write empty file
            empty_data = {
                field.name: pa.array([], type=pa.float64()) for field in schema_fields
            }
            empty_table = pa.table(empty_data)
            pq.write_table(empty_table, output_path)

        return None

    else:
        # In-memory path
        all_props_lists = props_blocks.compute()

        # Flatten and collect all region data
        all_region_data = []

        if hasattr(all_props_lists, "flat"):
            for item in all_props_lists.flat:
                if isinstance(item, list):
                    all_region_data.extend(item)
        else:
            if isinstance(all_props_lists, list):
                all_region_data = all_props_lists

        if len(all_region_data) == 0:
            # Return empty dict with correct structure
            result = {}
            for prop_name in property_names:
                output_name = rename_mapping[prop_name]
                if prop_name in COORDINATE_PROPERTIES:
                    result[f"{output_name}_x"] = np.empty((0,), dtype=np.float64)
                    result[f"{output_name}_y"] = np.empty((0,), dtype=np.float64)
                    result[f"{output_name}_z"] = np.empty((0,), dtype=np.float64)
                else:
                    result[output_name] = np.empty((0,), dtype=np.float64)
            return result

        # Transform coordinate properties and build result dict
        result = {}

        for prop_name in property_names:
            output_name = rename_mapping[prop_name]
            if prop_name in COORDINATE_PROPERTIES:
                # Collect voxel coordinates
                voxel_coords = np.array(
                    [region_data[prop_name] for region_data in all_region_data],
                    dtype=np.float64,
                )
                # Transform to physical coordinates
                physical_coords = _transform_coordinate_to_physical(
                    voxel_coords, affine_matrix
                )
                # Split into suffixed x, y, z columns with renamed output
                result[f"{output_name}_x"] = physical_coords[:, 0]
                result[f"{output_name}_y"] = physical_coords[:, 1]
                result[f"{output_name}_z"] = physical_coords[:, 2]
            else:
                # Non-coordinate properties
                values = [region_data[prop_name] for region_data in all_region_data]
                result[output_name] = np.array(values, dtype=np.float64)

        return result


def density_from_points(
    points: Union[np.ndarray, str],
    reference_zarrnii: "ZarrNii",
    in_physical_space: bool = True,
) -> "ZarrNii":
    """
    Create a density map from a set of points in the space of a reference ZarrNii image.

    This function takes a list of points (e.g., centroids from segmentation) and
    computes a 3D density map by binning the points into voxels of the reference
    image. The density map is returned as a new ZarrNii instance that can be
    written to OME-Zarr format for multiscale visualization.

    The function handles coordinate transformations automatically:
    - If points are in physical space (default), they are transformed to voxel
      indices using the inverse of the reference image's affine transformation
    - If points are already in voxel space, they are used directly
    - Uses dask.array.histogramdd for efficient computation on large datasets

    Args:
        points: Point coordinates to create density map from. Can be either:
            - numpy array of shape (N, 3) with coordinates [x, y, z]
            - str path to .npy file containing numpy array
            - str path to .parquet file with columns ['x', 'y', 'z']
        reference_zarrnii: ZarrNii instance defining the output image space
            (dimensions, spacing, origin). The density map will have the same
            spatial properties as this reference image.
        in_physical_space: Whether input points are in physical coordinates
            (default: True). If True, points are transformed to voxel indices
            using the inverse affine. If False, points are assumed to already
            be in voxel coordinates.

    Returns:
        ZarrNii: New ZarrNii instance containing the density map with the same
            spatial properties (shape, spacing, origin) as the reference image.
            The data type is float32, suitable for visualization and analysis.
            Values represent the number of points falling in each voxel.

    Raises:
        ValueError: If points array doesn't have shape (N, 3)
        ValueError: If reference_zarrnii is not 3D (after removing channel/time dims)
        FileNotFoundError: If points path doesn't exist
        ImportError: If pandas/pyarrow not installed for parquet support

    Examples:
        >>> import numpy as np
        >>> from zarrnii import ZarrNii, density_from_points
        >>>
        >>> # Load reference image
        >>> ref_img = ZarrNii.from_ome_zarr("reference.zarr")
        >>>
        >>> # Create density from centroids in physical space
        >>> centroids = np.load("centroids.npy")  # Shape: (N, 3)
        >>> density = density_from_points(centroids, ref_img)
        >>>
        >>> # Save as multiscale OME-Zarr
        >>> density.to_ome_zarr("density_map.zarr")
        >>>
        >>> # Load from parquet file (e.g., from compute_centroids output)
        >>> density = density_from_points("centroids.parquet", ref_img)
        >>>
        >>> # Use points already in voxel coordinates
        >>> voxel_coords = np.array([[10, 20, 30], [15, 25, 35]])
        >>> density = density_from_points(
        ...     voxel_coords, ref_img, in_physical_space=False
        ... )

    Notes:
        - The output density map has a single channel dimension (c=1)
        - Points outside the image bounds are ignored
        - Multiple points in the same voxel accumulate (sum)
        - The density map preserves the reference image's orientation and spacing
        - For large point sets, consider using parquet format for efficient I/O
        - The function uses dask arrays for memory-efficient computation
    """
    # Import here to avoid circular dependency
    from .core import ZarrNii
    from .transform import AffineTransform

    # Load points from file if string path provided
    if isinstance(points, str):
        if points.endswith(".npy"):
            points = np.load(points)
        elif points.endswith(".parquet"):
            try:
                import pandas as pd
            except ImportError:
                raise ImportError(
                    "pandas is required to read parquet files. "
                    "Install with: pip install pandas pyarrow"
                )
            df = pd.read_parquet(points)
            # Expect columns: x, y, z
            if not all(col in df.columns for col in ["x", "y", "z"]):
                raise ValueError(
                    f"Parquet file must contain columns 'x', 'y', 'z'. "
                    f"Found: {list(df.columns)}"
                )
            points = df[["x", "y", "z"]].values
        else:
            raise ValueError(
                f"Unsupported file format. Expected .npy or .parquet, got: {points}"
            )

    # Validate points shape
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Points must have shape (N, 3), got shape {points.shape}")

    # Get reference image properties
    ref_data = reference_zarrnii.data
    ref_dims = reference_zarrnii.dims
    ref_shape_dict = dict(zip(ref_dims, ref_data.shape))

    # Extract spatial dimensions (x, y, z)
    spatial_dims = ["x", "y", "z"]
    if not all(dim in ref_shape_dict for dim in spatial_dims):
        raise ValueError(
            f"Reference image must have spatial dimensions x, y, z. "
            f"Found dims: {ref_dims}"
        )

    # Get spatial shape
    Nx = ref_shape_dict["x"]
    Ny = ref_shape_dict["y"]
    Nz = ref_shape_dict["z"]

    # Transform points from physical to voxel coordinates if needed
    if in_physical_space:
        # Get affine transform and invert it
        # The affine maps from voxel coords in axes_order to physical (x, y, z)
        # So the inverse maps from physical (x, y, z) to voxel coords in axes_order
        affine = reference_zarrnii.get_affine_transform()
        affine_inv = affine.invert()

        # Convert points: (N, 3) array where each row is [x, y, z]
        # apply_transform expects points in shape (3, N) for batch processing
        points_transposed = points.T  # Shape: (3, N)
        voxel_coords_transposed = affine_inv.apply_transform(points_transposed)
        voxel_coords_axes_order = voxel_coords_transposed.T  # Shape: (N, 3)

        # The voxel_coords are now in axes_order
        # We need to reorder them to (x, y, z) for histogramdd
        axes_order = reference_zarrnii.axes_order
        if axes_order == "ZYX":
            # voxel_coords_axes_order is (z, y, x), need to reorder to (x, y, z)
            voxel_coords = np.column_stack(
                [
                    voxel_coords_axes_order[:, 2],  # x
                    voxel_coords_axes_order[:, 1],  # y
                    voxel_coords_axes_order[:, 0],  # z
                ]
            )
        elif axes_order == "XYZ":
            # Already in (x, y, z) order
            voxel_coords = voxel_coords_axes_order
        else:
            raise ValueError(
                f"Unsupported axes_order: {axes_order}. "
                "Only 'ZYX' and 'XYZ' are currently supported."
            )
    else:
        # Points are already in voxel coordinates
        # Assume they are provided in (x, y, z) order
        voxel_coords = points

    # Convert to dask array for histogram computation
    pts_dask = da.from_array(voxel_coords, chunks=(10000, 3))

    # Define voxel edges for histogram
    # histogramdd bins by [edge[i], edge[i+1]) so we need N+1 edges for N bins
    # The edges span from 0 to N for each dimension
    x_edges = np.linspace(0, Nx, Nx + 1)
    y_edges = np.linspace(0, Ny, Ny + 1)
    z_edges = np.linspace(0, Nz, Nz + 1)

    # Note: histogramdd expects sample in shape (N, D) where D is number of dimensions
    # and bins as a sequence of arrays defining bin edges for each dimension
    # The order should match the point coordinate order: [x, y, z]
    edges = [x_edges, y_edges, z_edges]

    # Compute density histogram
    # Returns tuple: (histogram, edges) where histogram has shape (Nx, Ny, Nz)
    density, _ = da.histogramdd(pts_dask, bins=edges)

    # Convert to float32 for better compatibility and smaller size
    density = density.astype(np.float32)

    # Create new ZarrNii instance with density data
    # Add channel dimension to make it 4D: (c, x, y, z) or (c, z, y, x)
    # depending on axes_order
    axes_order = reference_zarrnii.axes_order

    if axes_order == "XYZ":
        # Density is currently in (x, y, z) order
        # Need to add channel dimension: (c, x, y, z)
        density_with_channel = density[np.newaxis, :, :, :]
    elif axes_order == "ZYX":
        # Density is currently in (x, y, z) order
        # Need to reorder to (z, y, x) and add channel: (c, z, y, x)
        density_reordered = da.transpose(density, (2, 1, 0))  # (z, y, x)
        density_with_channel = density_reordered[np.newaxis, :, :, :]
    else:
        raise ValueError(
            f"Unsupported axes_order: {axes_order}. "
            "Only 'ZYX' and 'XYZ' are currently supported."
        )

    # Get spacing and origin from reference
    scale = reference_zarrnii.scale
    translation = reference_zarrnii.translation

    # Extract spacing and origin in axes_order
    spacing = tuple(scale.get(dim.lower(), 1.0) for dim in axes_order)
    origin = tuple(translation.get(dim.lower(), 0.0) for dim in axes_order)

    # Create new ZarrNii from the density array
    density_zarrnii = ZarrNii.from_darr(
        darr=density_with_channel,
        axes_order=axes_order,
        orientation=reference_zarrnii.xyz_orientation,
        spacing=spacing,
        origin=origin,
        name="density_map",
    )

    return density_zarrnii
