"""Destriping module for removing stripe artifacts from volumetric images."""

from __future__ import annotations

import dask.array as da
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_fill_holes
from scipy.signal import medfilt2d
from scipy.signal import convolve2d

from skimage.morphology import binary_dilation,binary_erosion, disk, remove_small_objects
from skimage.transform import resize

from scipy.ndimage import uniform_filter
from skimage.exposure import rescale_intensity
from scipy.signal import convolve2d
from scipy.ndimage import uniform_filter, gaussian_filter


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
    
def _gaussian_kernel(hsize: int, sigma: float) -> np.ndarray:
        hsize = int(hsize)
        if hsize < 1:
            hsize = 1
        if hsize % 2 == 0:
            hsize += 1
        r = hsize // 2
        ax = np.arange(-r, r + 1, dtype=np.float32)
        xx, yy = np.meshgrid(ax, ax, indexing="xy")
        k = np.exp(-(xx * xx + yy * yy) / (2.0 * float(sigma) * float(sigma) + 1e-12))
        k /= (k.sum() + 1e-12)
        return k.astype(np.float32)
        
def guided_filter_gray(guidance: np.ndarray, src: np.ndarray, neigh: int, eps: float) -> np.ndarray:
    # neigh is window size (odd). radius r = (neigh-1)//2; uniform_filter uses size.
    win = int(neigh)
    if win < 1:
        return src
    if win % 2 == 0:
        win += 1

    I = guidance.astype(np.float32, copy=False)
    p = src.astype(np.float32, copy=False)

    mean_I = uniform_filter(I, size=win, mode="reflect")
    mean_p = uniform_filter(p, size=win, mode="reflect")
    mean_II = uniform_filter(I * I, size=win, mode="reflect")
    mean_Ip = uniform_filter(I * p, size=win, mode="reflect")

    var_I = mean_II - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = uniform_filter(a, size=win, mode="reflect")
    mean_b = uniform_filter(b, size=win, mode="reflect")

    q = mean_a * I + mean_b
    return q.astype(np.float32)

def phasecong(
    image,
    nscale=4,
    norient=6,
    min_wave_length=3,
    mult=2,
    sigma_on_f=0.55,
    d_theta_on_sigma=1.2,
    k=2.0,
    cut_off=0.4,
    g=10.0,
    epsilon=1e-4,
):
    """
    Python translation of Peter Kovesi's MATLAB phasecong.m (PC_2 measure),
    matching the original filter construction and FFT shifting behavior.

    Returns:
        phaseCongruency (float32): [0,1]-ish phase congruency map
        orientation_deg (float32): orientation of max energy in degrees
                                   (0..(norient-1))*180/norient
    """
    I = np.asarray(image, dtype=np.float64)
    if I.ndim != 2 or I.shape[0] != I.shape[1]:
        raise ValueError("phasecong: image must be square 2D.")

    rows = cols = I.shape[0]
    thetaSigma = np.pi / norient / d_theta_on_sigma

    imagefft = np.fft.fft2(I)

    totalEnergy = np.zeros((rows, cols), dtype=np.float64)
    totalSumAn  = np.zeros((rows, cols), dtype=np.float64)
    orientation = np.zeros((rows, cols), dtype=np.float64)  # stores (o-1) at max energy

    # Match MATLAB grid construction exactly:
    # x = ones(rows,1)*(-cols/2:(cols/2-1));
    # y = (-rows/2:(rows/2-1))'*ones(1,cols);
    x = np.arange(-cols / 2, cols / 2, dtype=np.float64)
    y = np.arange(-rows / 2, rows / 2, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing="xy")

    radius = np.sqrt(X**2 + Y**2)
    radius[rows // 2, cols // 2] = 1.0  # MATLAB: radius(rows/2+1, cols/2+1)=1
    theta = np.arctan2(-Y, X)           # MATLAB uses -y to get +CCW angles

    maxEnergy = None

    for o in range(1, norient + 1):
        angl = (o - 1) * np.pi / norient

        # Angular distance (wrap-around safe), as MATLAB
        ds = np.sin(theta) * np.cos(angl) - np.cos(theta) * np.sin(angl)
        dc = np.cos(theta) * np.cos(angl) + np.sin(theta) * np.sin(angl)
        dtheta = np.abs(np.arctan2(ds, dc))
        spread = np.exp(-(dtheta**2) / (2.0 * thetaSigma**2))

        wavelength = float(min_wave_length)

        sumE = np.zeros((rows, cols), dtype=np.float64)
        sumO = np.zeros((rows, cols), dtype=np.float64)
        sumAn = np.zeros((rows, cols), dtype=np.float64)
        Energy = np.zeros((rows, cols), dtype=np.float64)

        EO_list = []
        ifftFilt_list = []

        EM_n = None
        maxAn = None

        for s in range(1, nscale + 1):
            fo = 1.0 / wavelength
            rfo = fo / 0.5 * (cols / 2.0)  # MATLAB: fo/0.5*(cols/2)

            # MATLAB: logGabor = exp((-(log(radius/rfo)).^2) / (2 * log(sigmaOnf)^2));
            logGabor = np.exp(-(np.log(radius / rfo) ** 2) / (2.0 * (np.log(sigma_on_f) ** 2)))
            logGabor[rows // 2, cols // 2] = 0.0

            # CRITICAL: MATLAB uses fftshift(filter) before multiplying with fft2(image)
            filt = (logGabor * spread)
            filt = np.fft.fftshift(filt)

            # MATLAB: ifftFilt = real(ifft2(filter))*sqrt(rows*cols);
            ifftFilt = np.real(np.fft.ifft2(filt)) * np.sqrt(rows * cols)
            ifftFilt_list.append(ifftFilt)

            # MATLAB: EO = ifft2(imagefft .* filter);
            EO = np.fft.ifft2(imagefft * filt)
            EO_list.append(EO)

            An = np.abs(EO)
            sumAn += An
            sumE += EO.real
            sumO += EO.imag

            maxAn = An if (maxAn is None) else np.maximum(maxAn, An)

            if s == 1:
                # MATLAB: EM_n = sum(sum(filter.^2));
                EM_n = np.sum(filt**2)

            wavelength *= mult

        # Weighted mean phase vector (MATLAB)
        XEnergy = np.sqrt(sumE**2 + sumO**2) + float(epsilon)
        MeanE = sumE / XEnergy
        MeanO = sumO / XEnergy

        # Energy accumulation across scales (MATLAB dot/cross form)
        for s in range(nscale):
            EOr = EO_list[s].real
            EOi = EO_list[s].imag
            Energy += (EOr * MeanE + EOi * MeanO - np.abs(EOr * MeanO - EOi * MeanE))

        # ---- Noise compensation (MATLAB) ----
        E2_small = np.abs(EO_list[0]) ** 2
        medianE2n = np.median(E2_small)
        meanE2n = -medianE2n / np.log(0.5)
        noisePower = meanE2n / EM_n

        EstSumAn2 = np.zeros((rows, cols), dtype=np.float64)
        for s in range(nscale):
            EstSumAn2 += ifftFilt_list[s] ** 2

        EstSumAiAj = np.zeros((rows, cols), dtype=np.float64)
        for si in range(nscale - 1):
            for sj in range(si + 1, nscale):
                EstSumAiAj += ifftFilt_list[si] * ifftFilt_list[sj]

        EstNoiseEnergy2 = (
            2.0 * noisePower * np.sum(EstSumAn2)
            + 4.0 * noisePower * np.sum(EstSumAiAj)
        )

        tau = np.sqrt(EstNoiseEnergy2 / 2.0)
        EstNoiseEnergy = tau * np.sqrt(np.pi / 2.0)
        EstNoiseEnergySigma = np.sqrt((2.0 - np.pi / 2.0) * tau**2)

        T = (EstNoiseEnergy + k * EstNoiseEnergySigma) / 1.7  # MATLAB PC_2 rescale
        Energy = np.maximum(Energy - T, 0.0)

        # Frequency spread weighting (MATLAB)
        width = (sumAn / (maxAn + float(epsilon))) / float(nscale)
        weight = 1.0 / (1.0 + np.exp((cut_off - width) * g))
        Energy *= weight

        totalSumAn += sumAn
        totalEnergy += Energy

        if o == 1:
            maxEnergy = Energy.copy()
        else:
            change = Energy > maxEnergy
            orientation = (o - 1) * change + orientation * (~change)
            maxEnergy = np.maximum(maxEnergy, Energy)

    phaseCongruency = totalEnergy / (totalSumAn + float(epsilon))
    orientation_deg = orientation * (180.0 / norient)

    return phaseCongruency.astype(np.float32), orientation_deg.astype(np.float32)



def destripe_block(
    block: np.ndarray,
    *,
    bg_thresh: float = 0.004,
    factor: int = 8,
    diff_thresh: float = 0.15,
    med_size_min: int = 11,
    med_size_max: int = 11,
    phase_size: int = 512,
    ori_target_deg: float = 90.0,
    ori_tol_deg: float = 5.0,  
    guided_neighborhood: int = 15,
    guided_smoothing: float = 0.001,
    min_obj_size: int = 30,
    gauss_sigma: float = 3.0,
    gauss_filter_size: int = 5,
    computing_meta: bool = False,  
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
    guided_neighborhood:
        neighrhood size for image guided filter.
    guided_smoothing:
        neighrhood size for image guided filter.

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

    # ---- dask meta path: return an empty array with correct shape/dtype ----
    if computing_meta:
        return np.zeros_like(block, dtype=np.float32)

    # ---- squeeze to 2D ----
    block = np.asarray(block)
    had_leading_one = (block.ndim == 3 and block.shape[0] == 1)
    II0 = np.squeeze(block)
    if II0.ndim != 2:
        raise ValueError(f"destripe_block expects 2D (or 1x2D) block, got shape {block.shape}")

    # ---- preserve original scaling (like your earlier python) ----
    # MATLAB normalizes by max(block(:)) then multiplies back at end.
    # Also handle all-zero safely.
    norm_val = float(np.nanmax(II0)) if II0.size else 0.0
    if not np.isfinite(norm_val) or norm_val <= 0:
        out = np.zeros_like(II0, dtype=np.float32)
        return out[np.newaxis, :, :] if had_leading_one else out

    I0_raw = np.nan_to_num(II0.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    I0 = I0_raw / norm_val  # now roughly [0,1] like MATLAB's block = block/norm_val

    # ---- background mask from *pre-imadjust* normalized image (matches MATLAB) ----
    mask_full = (I0 < float(bg_thresh)).astype(np.float32)

    # ---- imadjust (MATLAB) on I0 ----
    # MATLAB imadjust(I0) stretches min..max to 0..1 (default behavior).
    imin = float(np.min(I0))
    imax = float(np.max(I0))
    if imax > imin:
        I0_adj = (I0 - imin) / (imax - imin)
    else:
        I0_adj = np.zeros_like(I0, dtype=np.float32)

    # ---- downsample into grid tiles ----
    mask_stack = downsample_grid(mask_full, factor)
    I_stack = downsample_grid(I0_adj, factor)
    Y_stack = 0*I_stack

    # choose a deterministic med_size (MATLAB uses fixed 11)
    lo = int(med_size_min)
    hi = int(med_size_max)
    if hi < lo:
        hi = lo
    med_size = _odd((lo + hi) // 2)

    # ---- per-tile processing (serial; one level of parallelism via map_blocks) ----
    num_channels = I_stack.shape[2]
    for idx in range(num_channels):
        I_tile = I_stack[:, :, idx].astype(np.float32, copy=False)
        bg_mask = mask_stack[:, :, idx].astype(bool)

        # MATLAB: bg_mask = imdilate(bg_mask, strel('disk',3))
        bg_mask = binary_dilation(bg_mask, disk(3))

        # median (MATLAB: medfilt2(I,[med_size, med_size]))
        I_med = medfilt2d(I_tile, kernel_size=(med_size, med_size)).astype(np.float32)

        # guided filter using guidance = original tile (pre-med)?? MATLAB uses II0 as guidance (the adjusted tile)
        # In your MATLAB: imguidedfilter(I, II0, ...) where II0 is the (imadjusted) image.
        # Here guidance should be the tile from I_stack before median; we use I_tile.
        I_guided = guided_filter_gray(I_med, I_med, guided_neighborhood, guided_smoothing)
        #I_guided = I_med

        D = I_tile - I_guided

        # MATLAB: I = I + D.*bg_mask;  D = D.*(1-bg_mask)
        I_bg = I_guided + D * bg_mask.astype(np.float32)
        D_bg = D * (~bg_mask).astype(np.float32)

        # split D0 / D1 using thresh=0.15
        mask_big = np.abs(D_bg) >= float(diff_thresh)

        D0 = np.zeros_like(D_bg, dtype=np.float32)
        D1 = np.zeros_like(D_bg, dtype=np.float32)
        D0[~mask_big] = D_bg[~mask_big]
        D1[mask_big] = D_bg[mask_big]

        # MATLAB: D0(D0<0)=0;
        D0 = np.maximum(D0, 0.0)

        II = I_bg + D1

        # phasecong on resized D0
        D0_resized = resize(
            D0,
            (phase_size, phase_size),
            order=3,
            mode="reflect",
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.float32)

        pc_small, ori_small = phasecong(D0_resized)

        ## MATLAB: mask = (orientation == 90)
        #mask_ori = np.isclose(ori_small, float(ori_target_deg))
        
        step = 180.0 / float(getattr(phasecong, "__defaults__", (None, None, None, None, None, None, None, None, None, None))[1] or 6)
        # If the above feels too hacky, just use norient you pass into phasecong. If yours is fixed at 6, step=30.
        # Safer: derive step from the actual values:
        step = 180.0 / 6.0

        tgt_bin = int(np.round(float(ori_target_deg) / step)) % int(np.round(180.0 / step))
        ori_bin = np.round(ori_small / step).astype(np.int32) % int(np.round(180.0 / step))
        mask_ori = (ori_bin == tgt_bin)
        

        # MATLAB: v = bwareaopen(mask, 30); then dilate disk3, fill holes
        #mask_ori = remove_small_objects(mask_ori, min_size=int(min_obj_size))
        #mask_ori = binary_erosion(mask_ori, disk(1))
        #mask_ori = binary_dilation(mask_ori, disk(2))
        #mask_ori = binary_dilation(mask_ori, disk(1))
        mask_ori = binary_dilation(mask_ori, disk(3))
        mask_ori = binary_fill_holes(mask_ori)
        v = mask_ori.astype(np.float32)

        # MATLAB: pc = phaseCongruency .* v; pc = pc/max(pc(:))*2;
        pc_gated = pc_small.astype(np.float32) * v
        m = float(np.max(pc_gated)) if pc_gated.size else 0.0
        if m > 0:
            pc_gated = (pc_gated / m) * 2.0
        else:
            pc_gated = np.zeros_like(pc_gated, dtype=np.float32)

        # MATLAB then clamps: pc(pc<0)=0; pc(pc>1)=1;
        pc_gated = np.clip(pc_gated, 0.0, 1.0)
        
        #g = _gaussian_kernel(gauss_filter_size, gauss_sigma)
        #pc_gated = ndimage.convolve(pc_gated, g, mode="nearest").astype(np.float32)

        # resize pc back to tile size
        pc = resize(
            pc_gated,
            D0.shape,
            order=3,
            mode="reflect",
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.float32)
        

        # MATLAB: Y1 = D0 .* (1 - pc); II = II + Y1;
        Y1 = D0 * (1.0 - pc)
        II = II + Y1
        #II = pc

        I_stack[:, :, idx] = II.astype(np.float32)
        #Y_stack[:,:, idx] = Y1.astype(np.float32)

    # ---- reconstruct full image from tiles ----
    II_recon = upsample_grid(I_stack, factor).astype(np.float32)
    #Y_recon = upsample_grid(Y_stack, factor).astype(np.float32)
    
    
    g = _gaussian_kernel(gauss_filter_size, gauss_sigma)
    #Y_recon = ndimage.convolve(Y_recon, g, mode="nearest").astype(np.float32)
    img_recon = ndimage.convolve(II_recon, g, mode="nearest").astype(np.float32)

    #img_recon = Y_recon
    #img_recon = II_recon+Y_recon
    

    # ---- resize back to original block size (because downsample_grid crops) ----
    if img_recon.shape != I0.shape:
        img_recon = resize(
            img_recon,
            I0.shape,
            order=1,
            mode="reflect",
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.float32)

    # ---- restore original scale ----
    out2d = (img_recon * norm_val).astype(np.float32)

    # restore shape
    if had_leading_one:
        return out2d[np.newaxis, :, :]
    return out2d





#def destripe_block(
#    block: np.ndarray,
#    bg_thresh: float = 0.004,  # threshold for background mask (like T)
#    factor: int = 16,  # down/upsampling grid factor
#    diff_thresh: float = 0.007,  # threshold on D to split D0 / D1
#    med_size_min: int = 9,  # deterministic median filter size range per tile
#    med_size_max: int = 19,
#    phase_size: int = 512,  # size for phasecong (square)
#    ori_target_deg: float = 90.0,  # stripe orientation
#    ori_tol_deg: float = 5.0,  # tolerance around target orientation
#) -> np.ndarray:
#    """
#    De-stripe a single 2D block (typically a z-slice).
#
#    The input block is normalized, background is masked, and stripe-like
#    artifacts are detected via phase congruency in a downsampled grid and
#    then corrected. The result is rescaled back to the original intensity
#    range and has the same shape as the input.
#
#    Parameters
#    ----------
#    block:
#        Input image block as a NumPy array. It is squeezed to 2D
#        (Y, X) before processing. Usually this is the array passed by
#        ``dask.map_blocks``.
#    bg_thresh:
#        Background intensity threshold in the normalized [0, 1] domain.
#        Pixels with intensity lower than this value are treated as
#        background when building the background mask. Increasing this
#        value makes the background mask more aggressive (more pixels
#        are considered background); decreasing it is more conservative.
#    factor:
#        Integer down/upsampling factor for the internal processing grid.
#        The image is divided into a coarse grid scaled by this factor
#        for estimating and correcting stripe patterns. Larger values
#        reduce computational cost and capture broader stripe structure
#        but may miss very fine-scale artifacts; smaller values provide
#        finer sampling at higher computational cost.
#    diff_thresh:
#        Threshold applied to an internal difference map ``D`` used to
#        separate two regimes (e.g., ``D0``/``D1``) when estimating stripe
#        contributions. Higher values make the split more selective
#        (fewer pixels classified as high-difference), while lower values
#        make it more inclusive.
#    med_size_min:
#        Minimum size (in pixels) of the median filter kernel applied per
#        tile during destriping. The actual kernel size used for a given
#        tile is deterministically selected between ``med_size_min`` and
#        ``med_size_max`` (for example, based on the channel index), and
#        odd sizes are enforced internally. Smaller values preserve more
#        fine detail but may leave residual stripe noise.
#    med_size_max:
#        Maximum size (in pixels) of the median filter kernel applied per
#        tile during destriping. Within the deterministically chosen range
#        between ``med_size_min`` and ``med_size_max``, larger kernel
#        sizes yield stronger smoothing and more aggressive stripe removal,
#        at the risk of blurring small structures.
#    phase_size:
#        Size (in pixels) of the square region used for phase congruency
#        analysis. This controls the spatial extent over which oriented
#        features (such as stripes) are detected. Must be large enough to
#        capture several stripe periods; increasing it may improve
#        robustness for broad patterns but increases computation.
#    ori_target_deg:
#        Target stripe orientation in degrees for phase congruency
#        detection (e.g., 90° for vertical stripes in image coordinates).
#        Only features near this orientation are treated as stripe
#        artifacts.
#    ori_tol_deg:
#        Angular tolerance (in degrees) around ``ori_target_deg`` within
#        which features are considered stripe-like. A larger tolerance
#        captures a wider range of orientations (more aggressive
#        destriping), while a smaller tolerance focuses on a narrower
#        band of orientations.
#
#    Returns
#    -------
#    np.ndarray
#        De-striped image block with the same shape as ``block``,
#        rescaled to the original intensity range.
#
#    Notes
#    -----
#    This function operates on in-memory NumPy arrays and is designed to
#    be mapped over a larger volume using ``dask.map_blocks``.
#    """
#
#    image = np.squeeze(block)  # (Y,X)
#
#    # ---------- normalize input to [0,1] ---------- #
#    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
#    norm_val = 1.0
#
#    if np.issubdtype(image.dtype, np.integer):
#        norm_val = max(1.0, float(np.iinfo(image.dtype).max))
#        image = image.astype(np.float32) / norm_val
#    else:
#        image = image.astype(np.float32, copy=False)
#        img_min = image.min()
#        img_max = image.max()
#        # Handle constant images - no normalization needed if max == min
#        if img_max != img_min and img_max > 1.0:
#            norm_val = img_max + 1e-8
#            image = image / norm_val
#
#    II0 = image
#
#    # ---------- background mask & stacking ---------- #
#    mask_full = np.zeros_like(II0, dtype=np.float32)
#    mask_full[II0 < float(bg_thresh)] = 1.0
#
#    mask_stack = downsample_grid(mask_full, factor)
#    I_stack = downsample_grid(II0, factor)
#
#    h_small, w_small, num_channels = I_stack.shape
#
#    # ---------- tile-wise processing (NO parallel inside) ---------- #
#    for idx in range(num_channels):
#        I_tile = I_stack[:, :, idx]
#        bg_mask = mask_stack[:, :, idx]
#
#        # dilate background mask
#        se_bg = disk(3)
#        bg_mask = binary_dilation(bg_mask.astype(bool), se_bg).astype(np.float32)
#
#        I0 = I_tile.copy()
#
#        # deterministic odd med size in [med_size_min, med_size_max]
#        lo = max(1, med_size_min)
#        hi = max(lo, med_size_max)
#        if hi == lo:
#            med_size = lo
#        else:
#            span = hi - lo
#            # use channel index to choose a value in [lo, hi] deterministically
#            med_size = lo + (idx % (span + 1))
#        med_size = _odd(med_size)
#
#        # median filter along stripe direction (vertical kernel)
#        I_med = medfilt2d(I_tile, kernel_size=(med_size, 1))
#
#        D = I0 - I_med
#
#        # apply background mask first
#        I_b = I_med + D * bg_mask
#        D_b = D * (1.0 - bg_mask)
#
#        # split into D0 (small diff) and D1 (large)
#        D0 = np.zeros_like(D_b, dtype=np.float32)
#        D1 = np.zeros_like(D_b, dtype=np.float32)
#
#        mask_big = np.abs(D_b) >= float(diff_thresh)
#        D0[~mask_big] = D_b[~mask_big]
#        D1[mask_big] = D_b[mask_big]
#
#        II = I_b + D1
#
#        # ----- phase congruency on D0 (resized to phase_size x phase_size) ----- #
#        D0_resized = resize(
#            D0.astype(np.float32),
#            (phase_size, phase_size),
#            order=1,
#            preserve_range=True,
#            anti_aliasing=False,
#        ).astype(np.float32)
#
#        pc_small, ori_small_deg = phasecong(D0_resized)  # your CPU version
#
#        # orientation mask: within ori_target_deg ± ori_tol_deg, wrap-safe
#        ori = ori_small_deg.astype(np.float32)
#        diff = np.abs(ori - float(ori_target_deg))
#        diff = np.minimum(diff, 180.0 - diff)
#        mask_ori = diff <= float(ori_tol_deg)
#
#        # morphology to clean orientation mask
#        mask_ori = remove_small_objects(mask_ori, min_size=50)
#        mask_ori = binary_dilation(mask_ori, disk(3))
#        mask_ori = binary_fill_holes(mask_ori)
#        v = mask_ori.astype(np.float32)
#
#        # gate PC by mask, scale to [0, 2] (like MATLAB)
#        pc_gated = pc_small * v
#        m = float(pc_gated.max()) if pc_gated.size else 0.0
#        if m > 0.0:
#            pc_gated = (pc_gated / m) * 2.0
#        else:
#            pc_gated = np.zeros_like(pc_small, dtype=np.float32)
#
#        # upsample PC back to tile shape
#        pc = resize(
#            pc_gated.astype(np.float32),
#            D0.shape,
#            order=1,
#            preserve_range=True,
#            anti_aliasing=False,
#        ).astype(np.float32)
#
#        # reconstruction: Y1 = D0 * (1 - pc), II += Y1
#        Y1 = D0 * np.maximum(0.0, 1.0 - pc)
#        II = II + Y1
#
#        I_out_tile = II.astype(np.float32)
#        I_stack[:, :, idx] = I_out_tile
#
#    # ---------- reconstruct full image from tiles ---------- #
#    img_recon = upsample_grid(I_stack, factor)
#
#    # use original shape and dtype and undo normalization
#    return (img_recon * norm_val).astype(block.dtype).reshape(block.shape)


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
    # Be tolerant to singleton dims (e.g., (1,H,W) or (H,W,1))
    img = np.asarray(img)
    if img.ndim != 2:
        img = np.squeeze(img)
    if img.ndim != 2:
        raise ValueError(f"downsample_grid expects a 2D array after squeeze; got shape {img.shape}")

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
    factor: int = 8,  # down/upsampling grid factor
    diff_thresh: float = 0.007,  # threshold on D to split D0 / D1
    med_size_min: int = 11,  # deterministic median filter size range per tile
    med_size_max: int = 11,
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
