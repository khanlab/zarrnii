"""
Image analysis functions for zarrnii.

This module provides functions for image analysis operations such as
histogram computation, threshold calculation, and MIP visualization.
"""

from __future__ import annotations

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


def compute_centroids(
    image: da.Array,
    affine: np.ndarray,
    depth: Union[int, Tuple[int, ...], Dict[int, int]] = 10,
    boundary: str = "none",
    rechunk: Optional[Union[int, Tuple[int, ...]]] = None,
    output_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Compute centroids of binary segmentation objects in physical coordinates.

    This function processes a binary segmentation image (typically output from
    a segmentation plugin) to identify connected components and compute their
    centroids in physical coordinates. It processes the image chunk-by-chunk
    with overlap to handle objects that span chunk boundaries efficiently.

    For large datasets with many objects, use the output_path parameter to write
    centroids directly to a Parquet file on disk instead of returning them as a
    numpy array. This avoids memory issues when dealing with millions of objects.

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
        output_path: Optional path to write centroids to Parquet file instead of
            returning them in memory. If provided, centroids are written to this
            file path and None is returned. Use this for large datasets to avoid
            memory issues. The Parquet file will contain columns 'x', 'y', 'z' with
            physical coordinates. If None (default), centroids are returned as numpy
            array.

    Returns:
        Optional[numpy.ndarray]: If output_path is None, returns Nx3 array of
            physical coordinates for N detected objects, where each row is
            [x, y, z] in physical space. The array has dtype float64.
            If output_path is provided, writes to Parquet file and returns None.

    Notes:
        - Objects with centroids in the overlap regions are filtered out to
          avoid duplicate detections across chunks.
        - The function uses scikit-image's label() with connectivity=3 (26-connectivity
          in 3D) to identify connected components.
        - Empty chunks (no objects detected) contribute empty arrays to the result.
        - This function computes the result immediately (not lazy).
        - Uses Dask's map_overlap for efficient parallel processing across chunks.
        - When using output_path, centroids are written in batches to avoid
          memory overflow, making it suitable for datasets with millions of objects.

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
        >>> # Compute centroids and return as numpy array (default)
        >>> centroids = compute_centroids(binary_seg, affine, depth=5)
        >>> print(f"Found {len(centroids)} objects with shape {centroids.shape}")
        >>>
        >>> # For large datasets, write to Parquet file
        >>> compute_centroids(binary_seg, affine, depth=5,
        ...                   output_path='centroids.parquet')
        >>> # Read back with pandas or pyarrow
        >>> import pandas as pd
        >>> df = pd.read_parquet('centroids.parquet')
        >>> print(f"Found {len(df)} objects")
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
                f"{image.shape[0]} channels. compute_centroids only supports 3D "
                "images or 4D images with a single channel (channel dimension "
                "size = 1). For multi-channel images, please process each channel "
                "separately or squeeze/select a single channel before calling "
                "this function."
            )
    elif image.ndim not in [1, 2, 3]:
        raise ValueError(
            f"Image must be 1D, 2D, 3D, or 4D (with single channel), "
            f"got {image.ndim}D with shape {image.shape}"
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

    def _block_centroids(block, block_info=None):
        """
        Process a single block to find centroids.

        Args:
            block: numpy array for this chunk (binary mask) with overlap
            block_info: dict containing array location information from map_overlap

        Returns:
            Object array matching block shape, with centroids stored in first element
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
        # block_info[0] contains info about the overlapped/extended array
        # block_info[None] contains info about the ORIGINAL input array
        original_info = block_info[None]
        array_location = original_info["array-location"]

        # With map_overlap and trim=False, the block includes overlap regions
        # We need to calculate the "core" region (non-overlap) within this block
        # to avoid counting the same object multiple times across chunks

        # Determine core region boundaries within the block
        # The core region is the part that maps to the original array coordinates
        core_slices = []
        for dim in range(len(array_location)):
            global_start, global_end = array_location[dim]

            # The block may have overlap on either or both sides
            # Calculate where the "core" starts and ends within this block
            #
            # The core region size is (global_end - global_start)
            # The block size may be larger due to overlap

            core_size = global_end - global_start

            # Determine how much overlap is before vs after
            # With boundary='none', edge chunks don't get overlap on the edge side
            # We can infer this from whether global_start is 0 (no overlap before)

            if global_start == 0:
                # First chunk - no overlap before
                overlap_before = 0
            else:
                # Not first chunk - we have overlap before
                overlap_before = overlap_sizes[dim]

            core_start = overlap_before
            core_end = core_start + core_size

            core_slices.append((core_start, core_end))

        # Process regions and filter to core
        centroids = []
        for region in regionprops(labeled):
            centroid = np.array(region.centroid)

            # Check if centroid is in core region
            in_core = True
            for dim in range(len(centroid)):
                core_start, core_end = core_slices[dim]
                if not (core_start <= centroid[dim] < core_end):
                    in_core = False
                    break

            if in_core:
                # Convert to global voxel coordinates
                # The core region maps to [global_start, global_end) in the
                # original array. Centroid is at position
                # (centroid[dim] - core_start) within the core
                # So global position is:
                # global_start + (centroid[dim] - core_start)
                global_centroid = []
                for dim in range(len(centroid)):
                    core_start, _ = core_slices[dim]
                    global_start, _ = array_location[dim]
                    global_centroid.append(global_start + (centroid[dim] - core_start))

                centroids.append(tuple(global_centroid))

        # Store all centroids for this block in the first element
        # Fill rest with empty lists
        result.fill([])
        if len(centroids) > 0:
            result.flat[0] = centroids

        return result

    # Apply block operation with overlap
    cents_blocks = image.map_overlap(
        _block_centroids,
        depth=overlap_sizes,
        boundary=boundary,
        trim=False,  # Don't trim - we handle filtering internally
        dtype=object,
        drop_axis=[],  # Keep dimensions initially
    )

    # Handle Parquet output differently to avoid memory overflow
    if output_path is not None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Process blocks incrementally and write to Parquet
        # This avoids loading all centroids into memory at once
        writer = None
        schema = pa.schema(
            [
                pa.field("x", pa.float64()),
                pa.field("y", pa.float64()),
                pa.field("z", pa.float64()),
            ]
        )

        # Iterate over blocks and process them one at a time
        for block_idx in np.ndindex(cents_blocks.numblocks):
            # Compute only this block
            block_result = cents_blocks.blocks[block_idx].compute()

            # Extract centroids from this block
            block_centroids = []
            if hasattr(block_result, "flat"):
                for item in block_result.flat:
                    if isinstance(item, list):
                        block_centroids.extend(item)
                    elif isinstance(item, (tuple, np.ndarray)) and len(item) > 0:
                        block_centroids.append(tuple(item))
            elif isinstance(block_result, list):
                block_centroids = block_result

            if len(block_centroids) == 0:
                continue  # Skip empty blocks

            # Convert to numpy array
            voxel_coords = np.array(block_centroids, dtype=np.float64)

            # Convert to physical coordinates using affine transform
            n_points = voxel_coords.shape[0]
            voxel_homogeneous = np.column_stack(
                [voxel_coords, np.ones((n_points, 1), dtype=np.float64)]
            )

            physical_homogeneous = voxel_homogeneous @ affine_matrix.T
            physical_coords = physical_homogeneous[:, :3]

            # Create PyArrow table for this batch
            table = pa.table(
                {
                    "x": pa.array(physical_coords[:, 0], type=pa.float64()),
                    "y": pa.array(physical_coords[:, 1], type=pa.float64()),
                    "z": pa.array(physical_coords[:, 2], type=pa.float64()),
                }
            )

            # Write to Parquet file (append mode)
            if writer is None:
                writer = pq.ParquetWriter(output_path, schema)

            writer.write_table(table)

        # Close the writer
        if writer is not None:
            writer.close()
        else:
            # No centroids found - write empty file
            empty_table = pa.table(
                {
                    "x": pa.array([], type=pa.float64()),
                    "y": pa.array([], type=pa.float64()),
                    "z": pa.array([], type=pa.float64()),
                }
            )
            pq.write_table(empty_table, output_path)

        return None

    else:
        # Original in-memory path for backward compatibility
        # Compute to get all centroid lists
        all_centroid_lists = cents_blocks.compute()

        # Flatten and collect all centroids
        centroid_list = []

        # Handle different dimensionalities
        if hasattr(all_centroid_lists, "flat"):
            for item in all_centroid_lists.flat:
                if isinstance(item, list):
                    centroid_list.extend(item)
                elif isinstance(item, (tuple, np.ndarray)) and len(item) > 0:
                    centroid_list.append(tuple(item))
        else:
            # Single item
            if isinstance(all_centroid_lists, list):
                centroid_list = all_centroid_lists

        if len(centroid_list) == 0:
            return np.empty((0, 3), dtype=np.float64)

        # Convert to numpy array
        voxel_coords = np.array(centroid_list, dtype=np.float64)

        # Convert to physical coordinates using affine transform
        n_points = voxel_coords.shape[0]
        voxel_homogeneous = np.column_stack(
            [voxel_coords, np.ones((n_points, 1), dtype=np.float64)]
        )

        physical_homogeneous = voxel_homogeneous @ affine_matrix.T
        physical_coords = physical_homogeneous[:, :3]

        return physical_coords
