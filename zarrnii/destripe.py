import dask
import dask.array as da
import numpy as np
import zarr
from scipy.ndimage import binary_fill_holes
from scipy.signal import medfilt2d
from skimage.morphology import binary_dilation, disk, remove_small_objects
from skimage.transform import resize


def _odd(n):
    n = int(n)
    return n if n % 2 else n + 1


def phasecong(
    image,
    nscale=4,
    norient=6,
    minWaveLength=3,
    mult=2,
    sigmaOnf=0.55,
    dThetaOnSigma=1.2,
    k=2.0,
    cutOff=0.4,
    g=10.0,
    epsilon=1e-4,
):
    """
    Python reimplementation of Kovesi's phase congruency (as in your MATLAB code).

    Parameters
    ----------
    image : (N,N) array_like
        Square grayscale image. Float32/64 recommended.
    Returns
    -------
    phaseCongruency : (N,N) float
    orientation_deg : (N,N) float
        Orientation in degrees (0..180), quantized by the winning orientation bin.
    """
    I = np.asarray(image, dtype=np.float32)
    if I.ndim != 2 or I.shape[0] != I.shape[1]:
        raise ValueError("phasecong: image must be square 2D.")

    rows = cols = I.shape[0]
    thetaSigma = np.pi / norient / dThetaOnSigma

    # Fourier transform of image
    imagefft = np.fft.fft2(I)

    zero = np.zeros_like(I, dtype=np.float32)
    totalEnergy = np.zeros_like(I, dtype=np.float32)
    totalSumAn = np.zeros_like(I, dtype=np.float32)
    orientation_idx = np.zeros_like(I, dtype=np.float32)  # stores (o-1)
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

        wavelength = float(minWaveLength)

        sumE_ThisOrient = np.zeros_like(I, dtype=np.float32)
        sumO_ThisOrient = np.zeros_like(I, dtype=np.float32)
        sumAn_ThisOrient = np.zeros_like(I, dtype=np.float32)
        Energy_ThisOrient = np.zeros_like(I, dtype=np.float32)

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
            logGabor = np.exp(-(log_term**2) / (2 * (np.log(sigmaOnf) ** 2))).astype(
                np.float32
            )
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
        EstSumAn2 = np.zeros_like(I, dtype=np.float32)
        for s in range(nscale):
            EstSumAn2 += ifftFilt_list[s] ** 2

        EstSumAiAj = np.zeros_like(I, dtype=np.float32)
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
        weight = 1.0 / (1.0 + np.exp((cutOff - width) * g))
        Energy_ThisOrient *= weight.astype(np.float32)

        # Accumulate
        totalSumAn += sumAn_ThisOrient
        totalEnergy += Energy_ThisOrient

        # Track best orientation index
        if o == 1:
            maxEnergy = Energy_ThisOrient.copy()
        else:
            change = Energy_ThisOrient > maxEnergy
            orientation_idx = np.where(change, float(o - 1), orientation_idx)
            maxEnergy = np.maximum(maxEnergy, Energy_ThisOrient)

    # Phase Congruency
    phaseCongruency = totalEnergy / (totalSumAn + epsilon)

    # Orientation in degrees (0..180)
    orientation_deg = orientation_idx * (180.0 / norient)

    return phaseCongruency.astype(np.float32), orientation_deg.astype(np.float32)


# INPUT_PATH = "../sub-AS40F2_sample-brain_acq-imaris4x_SPIM.ome.zarr"
#
# CHANNEL_INDEX = 0  # 0-based: the 3rd channel
# CENTER_FRACTION = 1.0  # 10% per spatial dimension


def destripe_block(
    block: np.ndarray,
    bg_thresh: float = 0.004,  # threshold for background mask (like T)
    factor: int = 16,  # down/upsampling grid factor
    diff_thresh: float = 0.007,  # threshold on D to split D0 / D1
    med_size_min: int = 9,  # random median filter size range per tile
    med_size_max: int = 19,
    phase_size: int = 512,  # size for phasecong (square)
    ori_target_deg: float = 90.0,  # stripe orientation
    ori_tol_deg: float = 5.0,  # tolerance around target orientation
) -> np.ndarray:
    """
    De-stripe a single block, assumed to be a z-slice.

    Takes a plain numpy array (no Dask/Zarr), returns a numpy array
    with the same shape. Internally uses your downsample_grid /
    upsample_grid + phasecong.

    This is what you'll pass to dask.map_blocks
    """

    I = np.squeeze(block)  # (Y,X)

    # ---------- normalize input to [0,1] ---------- #
    I = np.nan_to_num(I, nan=0.0, posinf=0.0, neginf=0.0)
    norm_val = 1.0

    if np.issubdtype(I.dtype, np.integer):
        norm_val = max(1.0, float(np.iinfo(I.dtype).max))
        I = I.astype(np.float32) / norm_val
    else:
        I = I.astype(np.float32, copy=False)
        if I.max() > 1.0:
            norm_val = I.max() + 1e-8
            I = I / norm_val

    II0 = I

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

        # random odd med size in [med_size_min, med_size_max]
        lo = max(1, med_size_min)
        hi = max(lo, med_size_max)
        med_size = np.random.randint(lo, hi + 1)
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

        # gate PC by mask, scale to 02 (like MATLAB)
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


def downsample_grid(img, factor):
    """
    img: 2D grayscale image
    factor: downsampling factor
    Returns I_stack with shape (h//factor, w//factor, factor^2)
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


def upsample_grid(I_stack, factor):
    """
    I_stack: 3D stack of downsampled images (h/f x w/f x f^2)
    factor: downsampling factor
    Returns reconstructed 2D image of size (h, w)
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
    img: da.Array,  # must by 3-D, axes order Z Y X, chunked as z slices
    bg_thresh: float = 0.004,  # threshold for background mask (like T)
    factor: int = 16,  # down/upsampling grid factor
    diff_thresh: float = 0.007,  # threshold on D to split D0 / D1
    med_size_min: int = 9,  # random median filter size range per tile
    med_size_max: int = 19,
    phase_size: int = 512,  # size for phasecong (square)
    ori_target_deg: float = 90.0,  # stripe orientation
    ori_tol_deg: float = 5.0,  # tolerance around target orientation
) -> da.Array:

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
            f"Incorrect shape or chunking in dask array for destripe, must be Z-slices, with XY chunks as image size"
        )
