"""
Image analysis functions for zarrnii.

This module provides functions for image analysis operations such as
histogram computation, threshold calculation, and MIP visualization.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
from skimage.filters import threshold_multiotsu


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
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[dict]]]:
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
            List of 2D numpy arrays, each containing an RGB MIP
            visualization for one slab. Each array has shape (height, width,
            3) with RGB values in range [0, 1].

        If return_slabs is True:
            Tuple of (mip_list, slab_info_list) where:
            - mip_list: List of 2D RGB arrays as described above
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
        >>> # Get slab metadata
        >>> mips, slab_info = create_mip_visualization(
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
            # Process each channel and combine with colors
            mip_computed = mip_data.compute()  # Compute dask array

            # If channel_indices is specified, we need to select those channels
            if channel_indices is not None:
                # Move channel axis to first position for easier iteration
                channel_axis_after_max = (
                    channel_idx if channel_idx < proj_axis_idx else channel_idx - 1
                )
                mip_computed = np.moveaxis(mip_computed, channel_axis_after_max, 0)
                # Select only the requested channels
                mip_channels = mip_computed[channel_indices]
            else:
                # Move channel axis to first position for easier iteration
                channel_axis_after_max = (
                    channel_idx if channel_idx < proj_axis_idx else channel_idx - 1
                )
                mip_channels = np.moveaxis(mip_computed, channel_axis_after_max, 0)

            # Get spatial dimensions after removing projection axis
            spatial_shape = mip_channels.shape[1:]

            # Initialize RGB image
            rgb_image = np.zeros(spatial_shape + (3,), dtype=np.float32)

            # Combine channels with their colors
            for ch_idx in range(n_channels):
                channel_data = mip_channels[ch_idx]

                # Determine intensity range for normalization
                if channel_ranges is not None and channel_ranges[ch_idx] is not None:
                    # Use custom intensity range
                    ch_min, ch_max = channel_ranges[ch_idx]
                else:
                    # Use auto-scaling based on data
                    ch_min = channel_data.min()
                    ch_max = channel_data.max()

                # Normalize to [0, 1]
                if ch_max > ch_min:
                    channel_normalized = (channel_data - ch_min) / (ch_max - ch_min)
                    # Clip to [0, 1] range in case data extends beyond specified range
                    channel_normalized = np.clip(channel_normalized, 0.0, 1.0)
                elif ch_max > 0:
                    # Uniform non-zero values - keep them
                    channel_normalized = np.ones_like(channel_data)
                else:
                    # All zeros - keep as zeros
                    channel_normalized = np.zeros_like(channel_data)

                # Apply color with alpha blending (multiply by RGBA color values)
                color_rgba = rgba_colors[ch_idx]
                alpha = color_rgba[3]  # Extract alpha value
                for rgb_idx in range(3):
                    rgb_image[..., rgb_idx] += (
                        channel_normalized * color_rgba[rgb_idx] * alpha
                    )

            # Clip to [0, 1] range
            rgb_image = np.clip(rgb_image, 0.0, 1.0)

        else:
            # Single channel - compute and normalize
            mip_computed = mip_data.compute()

            # Determine intensity range for normalization
            if channel_ranges is not None and channel_ranges[0] is not None:
                # Use custom intensity range
                ch_min, ch_max = channel_ranges[0]
            else:
                # Use auto-scaling based on data
                ch_min = mip_computed.min()
                ch_max = mip_computed.max()

            # Normalize to [0, 1]
            if ch_max > ch_min:
                normalized = (mip_computed - ch_min) / (ch_max - ch_min)
                # Clip to [0, 1] range in case data extends beyond specified range
                normalized = np.clip(normalized, 0.0, 1.0)
            elif ch_max > 0:
                # Uniform non-zero values - keep them
                normalized = np.ones_like(mip_computed)
            else:
                # All zeros - keep as zeros
                normalized = np.zeros_like(mip_computed)

            # Apply first color with alpha
            color_rgba = rgba_colors[0]
            alpha = color_rgba[3]
            rgb_image = np.zeros(mip_computed.shape + (3,), dtype=np.float32)
            for rgb_idx in range(3):
                rgb_image[..., rgb_idx] = normalized * color_rgba[rgb_idx] * alpha

        mip_list.append(rgb_image)

    if return_slabs:
        return mip_list, slab_info_list
    else:
        return mip_list
