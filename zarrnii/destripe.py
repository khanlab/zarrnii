"""Destriping module for removing stripe artifacts from volumetric images."""

from __future__ import annotations

import dask.array as da
import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.signal import medfilt2d
from skimage.morphology import binary_dilation, disk, remove_small_objects
from skimage.transform import resize


def _odd(n: int) -> int:
    """Convert an integer to the nearest odd number.

    If the input is even, returns n + 1. If the input is already odd, returns n unchanged.

    Args:
        n: Integer value to convert to odd number.

    Returns:
        The nearest odd integer (n if odd, n+1 if even).
    """
    n = int(n)
    return n if n % 2 else n + 1


def phasecong(
    image: np.ndarray,
    nscale: int = 4,
    norient: int = 6,
    min_wave_length: int = 3,
    mult: int = 2,
    sigma_on_f: float = 0.55,
    d_theta_on_sigma: float = 1.2,
    k: float = 2.0,
    cut_off: float = 0.4,
    g: float = 10.0,
    epsilon: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute phase congruency for detecting image features.

    Python reimplementation of Kovesi's phase congruency algorithm.
    Phase congruency is a measure of feature significance based on local
    frequency and phase information, independent of image contrast.

    Parameters
    ----------
    image : np.ndarray
        Square grayscale image (N, N). Float32/64 recommended.
    nscale : int
        Number of wavelet scales to use in the analysis.
    norient : int
        Number of filter orientations to use (divides 180 degrees).
    min_wave_length : int
        Wavelength of smallest scale filter in pixels.
    mult : int
        Scaling factor between successive filter wavelengths.
    sigma_on_f : float
        Ratio of standard deviation of Gaussian describing log Gabor
        filter's transfer function in frequency domain to filter center frequency.
    d_theta_on_sigma : float
        Ratio of angular interval between filter orientations and
        standard deviation of angular Gaussian function used to construct filters.
    k : float
        Number of standard deviations of noise energy above mean at which
        we set threshold for phase congruency.
    cut_off : float
        Threshold used to determine significance of frequency spread weighting.
    g : float
        Controls sharpness of the frequency spread weighting sigmoid function.
    epsilon : float
        Small constant to prevent division by zero.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        phaseCongruency : np.ndarray
            Phase congruency values (N, N) as float32, representing feature strength.
        orientation_deg : np.ndarray
            Orientation in degrees (N, N) as float32, range 0-180, quantized by
            the winning orientation bin.

    Raises
    ------
    ValueError
        If image is not square 2D.
    """
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.ndim != 2 or image_array.shape[0] != image_array.shape[1]:
        raise ValueError("phasecong: image must be square 2D.")

    rows = cols = image_array.shape[0]
    thetaSigma = np.pi / norient / d_theta_on_sigma

    # Fourier transform of image
    imagefft = np.fft.fft2(image_array)

    total_energy = np.zeros_like(image_array, dtype=np.float32)
    total_sum_an = np.zeros_like(image_array, dtype=np.float32)
    orientation_idx = np.zeros_like(image_array, dtype=np.float32)  # stores (o-1)
    estMeanE2n_list = []

    # ----- precompute polar coords (unshifted layout) -----
    x = np.arange(-cols / 2, cols / 2, dtype=np.float32)
    y = np.arange(-rows / 2, rows / 2, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="xy")
    radius = np.sqrt(X**2 + Y**2).astype(np.float32)
    # avoid log(0) at center
    radius[rows // 2, cols // 2] = 1.0
    # +ve anticlockwise angles (note -Y to match MATLAB)
    theta = np.arctan2(-Y, X).astype(np.float32)

    maxEnergy = None

    for o in range(1, norient + 1):
        # Filter orientation
        angl = (o - 1) * np.pi / norient

        # Angular spread
        ds = np.sin(theta) * np.cos(angl) - np.cos(theta) * np.sin(angl)
        dc = np.cos(theta) * np.cos(angl) + np.sin(theta) * np.sin(angl)
        dtheta = np.abs(np.arctan2(ds, dc))
        spread = np.exp(-(dtheta**2) / (2 * thetaSigma**2)).astype(np.float32)

        wavelength = float(min_wave_length)

        sumE_ThisOrient = np.zeros_like(image_array, dtype=np.float32)
        sumO_ThisOrient = np.zeros_like(image_array, dtype=np.float32)
        sumAn_ThisOrient = np.zeros_like(image_array, dtype=np.float32)
        Energy_ThisOrient = np.zeros_like(image_array, dtype=np.float32)

        EO_list = []  # complex responses (per scale)
        ifftFilt_list = []  # real(ifft2(filter))*sqrt(N)

        EM_n = None  # mean squared filter value at smallest scale (for noise est)
        maxAn = None

        for s in range(1, nscale + 1):
            fo = 1.0 / wavelength  # centre frequency
            # MATLAB: rfo = fo/0.5 * (cols/2) == fo * cols
            rfo = fo * cols

            # Log-Gabor radial component (in unshifted frequency coords)
            log_term = np.log(radius / (rfo + 1e-12))
            logGabor = np.exp(
                -(log_term**2) / (2 * (np.log(sigma_on_f) ** 2))
            ).astype(np.float32)
            logGabor[rows // 2, cols // 2] = 0.0  # undo radius fudge

            # Full filter = radial * angular spread
            filt = (logGabor * spread).astype(np.float32)
            # Move zero-freq to corners to match convolution layout
            filt = np.fft.fftshift(filt)

            # Record ifft of filter (scaled like MATLAB)
            ifftFilt = np.real(np.fft.ifft2(filt)) * np.sqrt(rows * cols)
            ifftFilt_list.append(ifftFilt.astype(np.float32))

            # Convolution in Fourier domain
            EOfft = imagefft * filt
            EO = np.fft.ifft2(EOfft)  # complex
            EO_list.append(EO)

            An = np.abs(EO).astype(np.float32)
            sumAn_ThisOrient += An
            sumE_ThisOrient += EO.real.astype(np.float32)
            sumO_ThisOrient += EO.imag.astype(np.float32)

            maxAn = An if (maxAn is None) else np.maximum(maxAn, An)

            if s == 1:
                # mean squared filter value at smallest scale (for noise est)
                EM_n = np.sum(filt**2, dtype=np.float64)

            wavelength *= mult

        # Weighted mean phase angle
        XEnergy = np.sqrt(sumE_ThisOrient**2 + sumO_ThisOrient**2) + epsilon
        MeanE = sumE_ThisOrient / XEnergy
        MeanO = sumO_ThisOrient / XEnergy

        # Energy accumulation across scales (PC_2)
        for s in range(nscale):
            EOr = EO_list[s].real.astype(np.float32)
            EOi = EO_list[s].imag.astype(np.float32)
            Energy_ThisOrient += (
                EOr * MeanE + EOi * MeanO - np.abs(EOr * MeanO - EOi * MeanE)
            )

        # ----- Noise compensation (from smallest scale) -----
        E2_small = np.abs(EO_list[0]) ** 2
        medianE2n = float(np.median(E2_small))
        meanE2n = -medianE2n / np.log(0.5)
        estMeanE2n_list.append(meanE2n)

        noisePower = meanE2n / (EM_n + 1e-12)

        # Estimate total noise energy^2 over all scales
        EstSumAn2 = np.zeros_like(image_array, dtype=np.float32)
        for s in range(nscale):
            EstSumAn2 += ifftFilt_list[s] ** 2

        EstSumAiAj = np.zeros_like(image_array, dtype=np.float32)
        for si in range(nscale - 1):
            for sj in range(si + 1, nscale):
                EstSumAiAj += ifftFilt_list[si] * ifftFilt_list[sj]

        EstNoiseEnergy2 = 2 * noisePower * np.sum(
            EstSumAn2, dtype=np.float64
        ) + 4 * noisePower * np.sum(EstSumAiAj, dtype=np.float64)

        tau = np.sqrt(EstNoiseEnergy2 / 2.0 + 1e-12)
        EstNoiseEnergy = tau * np.sqrt(np.pi / 2.0)
        EstNoiseEnergySigma = np.sqrt((2.0 - np.pi / 2.0) * tau**2)

        T = (
            EstNoiseEnergy + k * EstNoiseEnergySigma
        ) / 1.7  # empirical scaling for PC_2

        # Apply threshold
        Energy_ThisOrient = np.maximum(Energy_ThisOrient - float(T), 0.0)

        # Sigmoidal weighting based on frequency width
        width = (sumAn_ThisOrient / (maxAn + epsilon)) / float(nscale)
        weight = 1.0 / (1.0 + np.exp((cut_off - width) * g))
        Energy_ThisOrient *= weight.astype(np.float32)

        # Accumulate
        total_sum_an += sumAn_ThisOrient
        total_energy += Energy_ThisOrient

        # Track best orientation index
        if o == 1:
            maxEnergy = Energy_ThisOrient.copy()
        else:
            change = Energy_ThisOrient > maxEnergy
            orientation_idx = np.where(change, float(o - 1), orientation_idx)
            maxEnergy = np.maximum(maxEnergy, Energy_ThisOrient)

    # Phase Congruency
    phaseCongruency = total_energy / (total_sum_an + epsilon)

    # Orientation in degrees (0..180)
    orientation_deg = orientation_idx * (180.0 / norient)

    return phaseCongruency.astype(np.float32), orientation_deg.astype(np.float32)


def destripe_block(
    block: np.ndarray,
    bg_thresh: float = 0.004,  # threshold for background mask (like T)
    factor: int = 16,  # down/upsampling grid factor
    diff_thresh: float = 0.007,  # threshold on D to split D0 / D1
    med_size_min: int = 9,  # deterministic median filter size range per tile
    med_size_max: int = 19,
    phase_size: int = 512,  # size for phasecong (square)
    ori_target_deg: float = 90.0,  # stripe orientation
    ori_tol_deg: float = 5.0,  # tolerance around target orientation
) -> np.ndarray:
    """
    De-stripe a single 2D block (typically a z-slice).

    The input block is normalized, background is masked, and stripe-like
    artifacts are detected via phase congruency in a downsampled grid and
    then corrected. The result is rescaled back to the original intensity
    range and has the same shape as the input.

    Parameters
    ----------
    block:
        Input image block as a NumPy array. It is squeezed to 2D
        (Y, X) before processing. Usually this is the array passed by
        ``dask.map_blocks``.
    bg_thresh:
        Background intensity threshold in the normalized [0, 1] domain.
        Pixels with intensity lower than this value are treated as
        background when building the background mask. Increasing this
        value makes the background mask more aggressive (more pixels
        are considered background); decreasing it is more conservative.
    factor:
        Integer down/upsampling factor for the internal processing grid.
        The image is divided into a coarse grid scaled by this factor
        for estimating and correcting stripe patterns. Larger values
        reduce computational cost and capture broader stripe structure
        but may miss very fine-scale artifacts; smaller values provide
        finer sampling at higher computational cost.
    diff_thresh:
        Threshold applied to an internal difference map ``D`` used to
        separate two regimes (e.g., ``D0``/``D1``) when estimating stripe
        contributions. Higher values make the split more selective
        (fewer pixels classified as high-difference), while lower values
        make it more inclusive.
    med_size_min:
        Minimum size (in pixels) of the median filter kernel applied per
        tile during destriping. The actual kernel size used for a given
        tile is deterministically selected between ``med_size_min`` and
        ``med_size_max`` (for example, based on the channel index), and
        odd sizes are enforced internally. Smaller values preserve more
        fine detail but may leave residual stripe noise.
    med_size_max:
        Maximum size (in pixels) of the median filter kernel applied per
        tile during destriping. Within the deterministically chosen range
        between ``med_size_min`` and ``med_size_max``, larger kernel
        sizes yield stronger smoothing and more aggressive stripe removal,
        at the risk of blurring small structures.
    phase_size:
        Size (in pixels) of the square region used for phase congruency
        analysis. This controls the spatial extent over which oriented
        features (such as stripes) are detected. Must be large enough to
        capture several stripe periods; increasing it may improve
        robustness for broad patterns but increases computation.
    ori_target_deg:
        Target stripe orientation in degrees for phase congruency
        detection (e.g., 90° for vertical stripes in image coordinates).
        Only features near this orientation are treated as stripe
        artifacts.
    ori_tol_deg:
        Angular tolerance (in degrees) around ``ori_target_deg`` within
        which features are considered stripe-like. A larger tolerance
        captures a wider range of orientations (more aggressive
        destriping), while a smaller tolerance focuses on a narrower
        band of orientations.

    Returns
    -------
    np.ndarray
        De-striped image block with the same shape as ``block``,
        rescaled to the original intensity range.

    Notes
    -----
    This function operates on in-memory NumPy arrays and is designed to
    be mapped over a larger volume using ``dask.map_blocks``.
    """

    image = np.squeeze(block)  # (Y,X)

    # ---------- normalize input to [0,1] ---------- #
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    norm_val = 1.0

    if np.issubdtype(image.dtype, np.integer):
        norm_val = max(1.0, float(np.iinfo(image.dtype).max))
        image = image.astype(np.float32) / norm_val
    else:
        image = image.astype(np.float32, copy=False)
        img_min = image.min()
        img_max = image.max()
        # Handle constant images
        if img_max == img_min:
            # Image is constant, no normalization needed
            pass
        elif img_max > 1.0:
            norm_val = img_max + 1e-8
            image = image / norm_val

    II0 = image

    # ---------- background mask & stacking ---------- #
    mask_full = np.zeros_like(II0, dtype=np.float32)
    mask_full[II0 < float(bg_thresh)] = 1.0

    mask_stack = downsample_grid(mask_full, factor)
    I_stack = downsample_grid(II0, factor)

    h_small, w_small, num_channels = I_stack.shape

    # ---------- tile-wise processing (NO parallel inside) ---------- #
    for idx in range(num_channels):
        I_tile = I_stack[:, :, idx]
        bg_mask = mask_stack[:, :, idx]

        # dilate background mask
        se_bg = disk(3)
        bg_mask = binary_dilation(bg_mask.astype(bool), se_bg).astype(np.float32)

        I0 = I_tile.copy()

        # deterministic odd med size in [med_size_min, med_size_max]
        lo = max(1, med_size_min)
        hi = max(lo, med_size_max)
        if hi == lo:
            med_size = lo
        else:
            span = hi - lo
            # use channel index to choose a value in [lo, hi] deterministically
            med_size = lo + (idx % (span + 1))
        med_size = _odd(med_size)

        # median filter along stripe direction (vertical kernel)
        I_med = medfilt2d(I_tile, kernel_size=(med_size, 1))

        D = I0 - I_med

        # apply background mask first
        I_b = I_med + D * bg_mask
        D_b = D * (1.0 - bg_mask)

        # split into D0 (small diff) and D1 (large)
        D0 = np.zeros_like(D_b, dtype=np.float32)
        D1 = np.zeros_like(D_b, dtype=np.float32)

        mask_big = np.abs(D_b) >= float(diff_thresh)
        D0[~mask_big] = D_b[~mask_big]
        D1[mask_big] = D_b[mask_big]

        II = I_b + D1

        # ----- phase congruency on D0 (resized to phase_size x phase_size) ----- #
        D0_resized = resize(
            D0.astype(np.float32),
            (phase_size, phase_size),
            order=1,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.float32)

        pc_small, ori_small_deg = phasecong(D0_resized)  # your CPU version

        # orientation mask: within ori_target_deg ± ori_tol_deg, wrap-safe
        ori = ori_small_deg.astype(np.float32)
        diff = np.abs(ori - float(ori_target_deg))
        diff = np.minimum(diff, 180.0 - diff)
        mask_ori = diff <= float(ori_tol_deg)

        # morphology to clean orientation mask
        mask_ori = remove_small_objects(mask_ori, min_size=50)
        mask_ori = binary_dilation(mask_ori, disk(3))
        mask_ori = binary_fill_holes(mask_ori)
        v = mask_ori.astype(np.float32)

        # gate PC by mask, scale to [0, 2] (like MATLAB)
        pc_gated = pc_small * v
        m = float(pc_gated.max()) if pc_gated.size else 0.0
        if m > 0.0:
            pc_gated = (pc_gated / m) * 2.0
        else:
            pc_gated = np.zeros_like(pc_small, dtype=np.float32)

        # upsample PC back to tile shape
        pc = resize(
            pc_gated.astype(np.float32),
            D0.shape,
            order=1,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.float32)

        # reconstruction: Y1 = D0 * (1 - pc), II += Y1
        Y1 = D0 * np.maximum(0.0, 1.0 - pc)
        II = II + Y1

        I_out_tile = II.astype(np.float32)
        I_stack[:, :, idx] = I_out_tile

    # ---------- reconstruct full image from tiles ---------- #
    img_recon = upsample_grid(I_stack, factor)

    # use original shape and dtype and undo normalization
    return (img_recon * norm_val).astype(block.dtype).reshape(block.shape)


def downsample_grid(img: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a 2D image into an interleaved grid stack.

    Divides the input image into a regular grid of non-overlapping tiles
    based on the downsampling factor. Each tile is extracted by taking every
    ``factor``-th pixel in both dimensions starting from different offsets.
    The function crops the image to ensure dimensions are multiples of
    ``factor``.

    Parameters
    ----------
    img : np.ndarray
        2D grayscale image with shape (H, W).
    factor : int
        Downsampling grid factor. The image is divided into ``factor**2``
        interleaved sub-grids.

    Returns
    -------
    np.ndarray
        3D stack with shape (h_small, w_small, factor**2) where
        ``h_small = H // factor`` and ``w_small = W // factor``.
        Each channel along the third axis corresponds to one interleaved
        sub-grid defined by offsets ``(i, j)`` for ``i, j in range(factor)``.
    """
    h, w = img.shape
    h_small = h // factor
    w_small = w // factor

    # Crop to multiples of factor to mimic MATLAB's floor(h/factor)
    img_c = img[: h_small * factor, : w_small * factor]

    num_channels = factor**2
    I_stack = np.zeros((h_small, w_small, num_channels), dtype=img_c.dtype)

    idx = 0
    for i in range(factor):
        for j in range(factor):
            I_stack[:, :, idx] = img_c[i::factor, j::factor][:h_small, :w_small]
            idx += 1

    return I_stack


def upsample_grid(I_stack: np.ndarray, factor: int) -> np.ndarray:
    """Reconstruct a high-resolution 2D image from a downsampled grid stack.

    This function reverses :func:`downsample_grid` by "unshuffling" the
    interleaved low-resolution tiles stored in ``I_stack`` back into their
    original pixel positions in a single 2D image. The input ``I_stack`` is
    expected to have been produced by ``downsample_grid(img, factor)`` on a
    2D grayscale image ``img``.

    Parameters
    ----------
    I_stack : np.ndarray
        3D stack of downsampled images with shape
        ``(h_small, w_small, factor**2)``, where
        ``h_small = floor(h / factor)`` and ``w_small = floor(w / factor)``.
        The third axis indexes the ``factor**2`` interleaved sub-grids,
        corresponding to all combinations of row/column offsets
        ``(i, j)`` in ``range(factor)`` used during downsampling.
    factor : int
        Downsampling / upsampling grid factor. Must be the same positive
        integer that was used in :func:`downsample_grid`. The reconstructed
        image will have spatial dimensions ``h = h_small * factor`` and
        ``w = w_small * factor``.

    Returns
    -------
    np.ndarray
        Reconstructed 2D image of shape ``(h_small * factor, w_small * factor)``.
        For each channel index ``idx`` corresponding to offsets
        ``(i, j)`` in ``range(factor)``, the slice ``I_stack[:, :, idx]`` is
        written into ``img_recon[i::factor, j::factor]``, reassembling the
        original interleaved pixel grid.
    """
    h_small, w_small, num_channels = I_stack.shape
    img_recon = np.zeros((h_small * factor, w_small * factor), dtype=I_stack.dtype)

    idx = 0
    for i in range(factor):
        for j in range(factor):
            img_recon[i::factor, j::factor] = I_stack[:, :, idx]
            idx += 1

    return img_recon


def _has_allowed_chunking(arr: da.Array) -> bool:
    """Validate that a Dask array has the correct chunking for destriping.

    Checks whether the input array satisfies the chunking requirements for
    the destripe operation: 3-5 dimensions with trailing (Z, Y, X) axes,
    where Z is chunked into single slices, Y and X are each in one full chunk,
    and any leading dimensions are also singleton-chunked.

    Args:
        arr: Dask array to validate.

    Returns:
        True if the array has valid chunking for destriping, False otherwise.
    """
    if arr.ndim < 3 or arr.ndim > 5:
        return False

    chunks = arr.chunks
    shape = arr.shape

    # Last 3 dims: (Nz, Ny, Nx)
    z_chunks, y_chunks, x_chunks = chunks[-3:]

    # Z must be sliced into 1s
    if not all(c == 1 for c in z_chunks):
        return False

    # Y, X must be exactly one chunk and full-sized
    if not (len(y_chunks) == 1 and y_chunks[0] == shape[-2]):
        return False
    if not (len(x_chunks) == 1 and x_chunks[0] == shape[-1]):
        return False

    # Leading dims (Nt, Nc) if present must be chunked as all 1s
    for dim_chunks in chunks[:-3]:
        if not all(c == 1 for c in dim_chunks):
            return False

    return True


def destripe(
    img: da.Array,  # must be 3–5D; last 3 axes Z Y X chunked as Z-slices, leading dims (if any) singleton-chunked
    bg_thresh: float = 0.004,  # threshold for background mask (like T)
    factor: int = 16,  # down/upsampling grid factor
    diff_thresh: float = 0.007,  # threshold on D to split D0 / D1
    med_size_min: int = 9,  # deterministic median filter size range per tile
    med_size_max: int = 19,
    phase_size: int = 512,  # size for phasecong (square)
    ori_target_deg: float = 90.0,  # stripe orientation
    ori_tol_deg: float = 5.0,  # tolerance around target orientation
) -> da.Array:
    """
    Reduce stripe artifacts in a volumetric image using a block-wise destriping algorithm.

    This function applies :func:`destripe_block` independently to each Z-slice
    (or to each Z-slice per leading index such as time or channel) of a Dask
    array. It is designed for large 3D imaging data (e.g. light-sheet
    microscopy volumes) stored as a stack of 2D planes, where striping arises
    from illumination or acquisition artifacts with a dominant orientation.

    The input must be a Dask array with 3 to 5 dimensions. The last three
    dimensions are interpreted as ``(Z, Y, X)``. Any leading dimensions (e.g.
    time ``T`` and/or channels ``C``) are preserved and processed independently.

    Chunking requirements
    ----------------------
    The destriping algorithm assumes that each block corresponds to a single
    Z-slice with the full in-plane field of view. The following chunking
    constraints are enforced (see :func:`_has_allowed_chunking`):

    * The array must have between 3 and 5 dimensions.
    * The last three axes must be ``(Z, Y, X)``.
    * Z chunks must have size 1 along the Z axis (i.e. one slice per chunk).
    * Y and X must each be a single chunk covering the full image extent
      (``chunk[-2] == shape[-2]`` and ``chunk[-1] == shape[-1]``).
    * Any leading axes (e.g. T, C) must also be chunked with size 1 along each
      of those axes.

    If these conditions are not met, a :class:`ValueError` is raised.

    Parameters
    ----------
    img:
        Dask array containing the input image data. The last three dimensions
        must be ``(Z, Y, X)`` with chunking as described above. The data type
        is preserved in the output.
    bg_thresh:
        Threshold used to define a background mask. Typical values are small
        positive fractions of the image dynamic range; pixels below this value
        are considered background and are down-weighted in the destriping
        process.
    factor:
        Down/upsampling grid factor used by the internal tiling scheme. Larger
        values correspond to finer tiling during destriping and may increase
        computation time.
    diff_thresh:
        Threshold applied to an internal difference image (``D``) to separate
        low- and high-intensity components before stripe estimation.
    med_size_min:
        Minimum size (in pixels) of the median filter kernel used per tile.
        The effective kernel size per tile is chosen within
        ``[med_size_min, med_size_max]``.
    med_size_max:
        Maximum size (in pixels) of the median filter kernel used per tile.
    phase_size:
        Size (in pixels) of the square region used for phase congruency
        analysis. Must be large enough to capture several stripe periods.
    ori_target_deg:
        Target stripe orientation in degrees. The default of ``90.0`` assumes
        vertical stripes in the image coordinate system.
    ori_tol_deg:
        Angular tolerance (in degrees) around ``ori_target_deg`` within which
        structures are considered stripe-like.

    Returns
    -------
    dask.array.Array
        A Dask array of the same shape and data type as ``img``, with stripe
        artifacts reduced. The chunking pattern is preserved.

    Raises
    ------
    ValueError
        If ``img`` does not have between 3 and 5 dimensions or if its chunking
        does not satisfy the constraints described above.

    Examples
    --------
    Create a synthetic volume and apply destriping::

        import dask.array as da
        import numpy as np

        # 3D volume with shape (Z, Y, X)
        vol = np.random.rand(16, 512, 512).astype("float32")

        # Chunk as one Z-slice per chunk, full XY
        darr = da.from_array(vol, chunks=(1, 512, 512))

        # Apply destriping lazily
        darr_destriped = destripe(darr)

        # Trigger computation
        result = darr_destriped.compute()

    """
    if _has_allowed_chunking(img):

        return img.map_blocks(
            destripe_block,
            dtype=img.dtype,
            bg_thresh=bg_thresh,
            factor=factor,
            diff_thresh=diff_thresh,
            med_size_min=med_size_min,
            med_size_max=med_size_max,
            phase_size=phase_size,
            ori_target_deg=ori_target_deg,
            ori_tol_deg=ori_tol_deg,
        )
    else:
        raise ValueError(
            "Incorrect shape or chunking in dask array for destripe.\n"
            f"Detected shape: {img.shape}, chunks: {img.chunks}.\n"
            "Required chunking:\n"
            "  - 3D–5D array with trailing axes (..., Z, Y, X)\n"
            "  - Z axis chunk size = 1 (one slice per chunk)\n"
            "  - Y and X axes each in a single chunk equal to the full image size\n"
            "  - Any leading axes (e.g. time, channel) chunked with size 1\n"
            "You can rechunk with dask before calling destripe, for example:\n"
            "  # for a 3D array (Z, Y, X)\n"
            "  img = img.rechunk((1, img.shape[1], img.shape[2]))\n"
            "  # for a 4D array (C, Z, Y, X)\n"
            "  img = img.rechunk((1, 1, img.shape[2], img.shape[3]))\n"
        )
