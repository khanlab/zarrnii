"""Patch-based destriping, updated to more closely match the MATLAB implementation.

Main MATLAB-matching choices here:
- Integer input is converted using MATLAB-like im2double semantics
  (uint16 -> /65535, uint8 -> /255), NOT divided by image max.
- MATLAB-like imadjust default is approximated using 1%/99% stretch limits.
- Patch extraction uses 50% overlap and +Y/+X zero padding.
- Patch merging uses max intensity in overlapping regions.
- bwareaopen is approximated with 8-connected remove_small_objects.

Note:
- phasecong is kept as your existing Python translation and is not further adjusted here.
- imguidedfilter is approximated with a standard guided filter implementation.
"""

from __future__ import annotations

import numpy as np
import dask.array as da
from scipy.ndimage import binary_fill_holes, median_filter, uniform_filter
from skimage.morphology import binary_dilation, disk, remove_small_objects
from skimage.transform import resize


# -------------------------------------------------------------------------
# MATLAB-like helpers
# -------------------------------------------------------------------------

def matlab_im2double(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """Approximate MATLAB im2double.

    For integer types, MATLAB im2double maps the full integer range to [0, 1].
    For floating types, MATLAB im2double leaves values essentially unchanged.

    Returns
    -------
    img_double : np.ndarray
        Float32 image after MATLAB-like im2double conversion.
    integer_input : bool
        True if the original image was integer.
    """
    img = np.asarray(img)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        if info.min < 0:
            # MATLAB im2double for signed integers maps full range to [0,1].
            out = (img.astype(np.float32) - float(info.min)) / float(info.max - info.min)
        else:
            out = img.astype(np.float32) / float(info.max)
        return out.astype(np.float32), True

    return img.astype(np.float32, copy=False), False


def matlab_stretchlim(img: np.ndarray, tol: tuple[float, float] = (0.01, 0.99)) -> tuple[float, float]:
    """Approximate MATLAB stretchlim(I), default 1% saturation at low/high."""
    I = np.asarray(img, dtype=np.float32)
    vals = I[np.isfinite(I)]
    if vals.size == 0:
        return 0.0, 1.0

    lo = float(np.quantile(vals, tol[0]))
    hi = float(np.quantile(vals, tol[1]))

    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi):
        hi = 1.0
    if hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
    if hi <= lo:
        return lo, lo + 1.0
    return lo, hi
    
def _odd(n: int) -> int:
    """Return the next odd integer >= n (minimum 1)."""
    n = int(n)
    if n <= 1:
        return 1
    return n if (n % 2 == 1) else n + 1


def matlab_imadjust_default(img: np.ndarray) -> np.ndarray:
    """Approximate MATLAB imadjust(I) for grayscale images.

    MATLAB imadjust(I) uses stretchlim(I) by default, which saturates about
    1% of pixels at low and high intensities, then maps to [0, 1].
    """
    I = np.asarray(img, dtype=np.float32)
    lo, hi = matlab_stretchlim(I)
    J = (I - lo) / (hi - lo)
    J = np.clip(J, 0.0, 1.0)
    return J.astype(np.float32)


def matlab_resize(img: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """Approximate MATLAB imresize default bicubic behavior.

    This is not bit-identical to MATLAB, but closer than arbitrary settings.
    MATLAB's default method is bicubic and uses antialiasing when shrinking.
    """
    in_shape = img.shape
    shrinking = out_shape[0] < in_shape[0] or out_shape[1] < in_shape[1]
    return resize(
        img,
        out_shape,
        order=3,
        mode="reflect",
        preserve_range=True,
        anti_aliasing=shrinking,
    ).astype(np.float32)


# -------------------------------------------------------------------------
# Guided filter approximation of MATLAB imguidedfilter(I, ...)
# -------------------------------------------------------------------------

def guided_filter_gray(guidance: np.ndarray, src: np.ndarray, neigh: int, eps: float) -> np.ndarray:
    """Standard grayscale guided filter.

    MATLAB imguidedfilter is not publicly implemented identically here; this
    function keeps the same high-level operation with local box statistics.
    """
    win = int(neigh)
    if win < 1:
        return src.astype(np.float32, copy=False)
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

    a = cov_Ip / (var_I + float(eps))
    b = mean_p - a * mean_I

    mean_a = uniform_filter(a, size=win, mode="reflect")
    mean_b = uniform_filter(b, size=win, mode="reflect")

    q = mean_a * I + mean_b
    return q.astype(np.float32)


# -------------------------------------------------------------------------
# Existing phasecong translation. Not modified in this pass.
# -------------------------------------------------------------------------

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
    """Python translation of Peter Kovesi's MATLAB phasecong.m."""
    I = np.asarray(image, dtype=np.float64)
    if I.ndim != 2 or I.shape[0] != I.shape[1]:
        raise ValueError("phasecong: image must be square 2D.")

    rows = cols = I.shape[0]
    thetaSigma = np.pi / norient / d_theta_on_sigma
    imagefft = np.fft.fft2(I)

    totalEnergy = np.zeros((rows, cols), dtype=np.float64)
    totalSumAn = np.zeros((rows, cols), dtype=np.float64)
    orientation = np.zeros((rows, cols), dtype=np.float64)

    x = np.arange(-cols / 2, cols / 2, dtype=np.float64)
    y = np.arange(-rows / 2, rows / 2, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing="xy")

    radius = np.sqrt(X**2 + Y**2)
    radius[rows // 2, cols // 2] = 1.0
    theta = np.arctan2(-Y, X)

    maxEnergy = None

    for o in range(1, norient + 1):
        angl = (o - 1) * np.pi / norient
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
            rfo = fo / 0.5 * (cols / 2.0)

            logGabor = np.exp(-(np.log(radius / rfo) ** 2) / (2.0 * (np.log(sigma_on_f) ** 2)))
            logGabor[rows // 2, cols // 2] = 0.0

            filt = np.fft.fftshift(logGabor * spread)
            ifftFilt = np.real(np.fft.ifft2(filt)) * np.sqrt(rows * cols)
            ifftFilt_list.append(ifftFilt)

            EO = np.fft.ifft2(imagefft * filt)
            EO_list.append(EO)

            An = np.abs(EO)
            sumAn += An
            sumE += EO.real
            sumO += EO.imag
            maxAn = An if maxAn is None else np.maximum(maxAn, An)

            if s == 1:
                EM_n = np.sum(filt**2)

            wavelength *= mult

        XEnergy = np.sqrt(sumE**2 + sumO**2) + float(epsilon)
        MeanE = sumE / XEnergy
        MeanO = sumO / XEnergy

        for s in range(nscale):
            EOr = EO_list[s].real
            EOi = EO_list[s].imag
            Energy += EOr * MeanE + EOi * MeanO - np.abs(EOr * MeanO - EOi * MeanE)

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

        EstNoiseEnergy2 = 2.0 * noisePower * np.sum(EstSumAn2) + 4.0 * noisePower * np.sum(EstSumAiAj)
        tau = np.sqrt(EstNoiseEnergy2 / 2.0)
        EstNoiseEnergy = tau * np.sqrt(np.pi / 2.0)
        EstNoiseEnergySigma = np.sqrt((2.0 - np.pi / 2.0) * tau**2)

        T = (EstNoiseEnergy + k * EstNoiseEnergySigma) / 1.7
        Energy = np.maximum(Energy - T, 0.0)

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


# -------------------------------------------------------------------------
# MATLAB-style patch extraction / reconstruction
# -------------------------------------------------------------------------

def downsample_grid(img: np.ndarray, patch_size: int = 1024):
    """Extract 50%-overlapped patches with +Y/+X zero padding.

    Matches MATLAB downsample_grid(img, patchSize) indexing, translated to
    Python 0-based coordinates.
    """
    img = np.asarray(img)
    if img.ndim != 2:
        img = np.squeeze(img)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}")

    h, w = img.shape
    stride = patch_size // 2
    if stride < 1:
        raise ValueError("patch_size must be >= 2")

    h_pad = int(np.ceil((h - patch_size) / stride) * stride + patch_size)
    w_pad = int(np.ceil((w - patch_size) / stride) * stride + patch_size)
    h_pad = max(h_pad, patch_size)
    w_pad = max(w_pad, patch_size)

    img_pad = np.zeros((h_pad, w_pad), dtype=img.dtype)
    img_pad[:h, :w] = img

    y_starts = np.arange(0, h_pad - patch_size + 1, stride, dtype=np.int64)
    x_starts = np.arange(0, w_pad - patch_size + 1, stride, dtype=np.int64)

    num_patch_y = len(y_starts)
    num_patch_x = len(x_starts)
    num_patches = num_patch_y * num_patch_x

    I_stack = np.zeros((patch_size, patch_size, num_patches), dtype=img.dtype)

    p = 0
    for y1 in y_starts:
        for x1 in x_starts:
            I_stack[:, :, p] = img_pad[y1:y1 + patch_size, x1:x1 + patch_size]
            p += 1

    info = {
        "padSize": (h_pad, w_pad),
        "patchSize": patch_size,
        "stride": stride,
        "y_starts": y_starts,
        "x_starts": x_starts,
        "numPatchY": num_patch_y,
        "numPatchX": num_patch_x,
    }
    return I_stack, info


def upsample_grid(I_stack: np.ndarray, info: dict):
    """Merge patches with max intensity in overlapping regions."""
    patch_size = int(info["patchSize"])
    h_pad, w_pad = info["padSize"]
    img_recon = np.zeros((h_pad, w_pad), dtype=I_stack.dtype)

    p = 0
    for y1 in info["y_starts"]:
        for x1 in info["x_starts"]:
            region = img_recon[y1:y1 + patch_size, x1:x1 + patch_size]
            patch = I_stack[:, :, p]
            img_recon[y1:y1 + patch_size, x1:x1 + patch_size] = np.maximum(region, patch)
            p += 1

    return img_recon


# -------------------------------------------------------------------------
# Main per-slice destriping
# -------------------------------------------------------------------------

def destripe_block(
    block: np.ndarray,
    *,
    bg_thresh: float = 0.004,
    patch_size: int = 1024,
    diff_thresh: float = 0.2,
    med_size: int = 61,
    phase_size: int = 512,
    guided_neighborhood: int = 15,
    guided_smoothing: float = 0.01,
    min_obj_size: int = 30,
    return_adjusted_float: bool = True,
    computing_meta: bool = False,
) -> np.ndarray:
    """Destripe one 2D image/slice, closely following the MATLAB script.

    Parameters
    ----------
    return_adjusted_float:
        True matches the MATLAB script more closely: output remains in the
        imadjusted [0,1] float domain. False rescales/casts back to the input
        dtype; use only if needed by an existing pipeline.
    """
    if computing_meta:
        return np.zeros_like(block, dtype=np.float32)

    block_arr = np.asarray(block)
    orig_block_shape = block_arr.shape
    had_leading_one = block_arr.ndim == 3 and block_arr.shape[0] == 1

    II0_in = np.squeeze(block_arr)
    if II0_in.ndim != 2:
        raise ValueError(f"destripe_block expects 2D or singleton-leading 3D, got shape {block_arr.shape}")

    orig_shape = II0_in.shape

    # MATLAB: II0 = im2double(II0);
    II0_double, integer_input = matlab_im2double(II0_in)

    # MATLAB:
    # T = 0.004;
    # mask_full = zeros(size(II0));
    # mask_full(II0<T) = 1;
    mask_full = (II0_double < float(bg_thresh)).astype(np.float32)

    # MATLAB: II0 = imadjust(II0);
    II0_adj = matlab_imadjust_default(II0_double)

    # MATLAB:
    # mask_stack = downsample_grid(mask_full,patchSize);
    # [I_stack,info] = downsample_grid(II0,patchSize);
    mask_stack, _ = downsample_grid(mask_full, patch_size=patch_size)
    I_stack, info = downsample_grid(II0_adj, patch_size=patch_size)

    dZ = I_stack.shape[2]

    for i in range(dZ):
        I = I_stack[:, :, i].astype(np.float32, copy=True)
        bg_mask = mask_stack[:, :, i].astype(bool)

        # MATLAB: bg_mask = double(imdilate(bg_mask, strel('disk',3)));
        bg_mask = binary_dilation(bg_mask, footprint=disk(3)).astype(np.float32)

        # MATLAB: for iter = 1
        I0 = I.copy()

        # MATLAB: I = medfilt2(I,[med_size,1]);
        # MATLAB medfilt2 default pads with zeros. scipy median_filter cval=0 mimics that.
        I = median_filter(I, size=(int(med_size), 1), mode="constant", cval=0.0).astype(np.float32)

        # MATLAB: I = imguidedfilter(I,'NeighborhoodSize',15,'DegreeOfSmoothing',0.01);
        I = guided_filter_gray(I, I, guided_neighborhood, guided_smoothing).astype(np.float32)

        D = I0 - I

        # MATLAB:
        # I = I + D.*bg_mask;
        # D = D.*(1-bg_mask);
        I = I + D * bg_mask
        D = D * (1.0 - bg_mask)

        # MATLAB:
        # D0(abs(D)<thresh) = D(abs(D)<thresh);
        # D1(abs(D)>=thresh) = D(abs(D)>=thresh);
        D0 = np.zeros_like(D, dtype=np.float32)
        D1 = np.zeros_like(D, dtype=np.float32)
        small = np.abs(D) < float(diff_thresh)
        D0[small] = D[small]
        D1[~small] = D[~small]

        # MATLAB: D0(D0<0) = 0;
        D0[D0 < 0] = 0.0

        # MATLAB: II = I + D1;
        II = I + D1

        # MATLAB: [phaseCongruency,orientation]=phasecong(imresize(D0,[512,512]));
        D0_resize = matlab_resize(D0, (int(phase_size), int(phase_size)))
        phaseCongruency, orientation = phasecong(D0_resize)

        # MATLAB:
        # mask = zeros(size(orientation));
        # mask(orientation==90) = 1;
        mask = orientation == 90

        # MATLAB: v = bwareaopen(mask,30); default 2D connectivity is 8-connected.
        v = remove_small_objects(mask.astype(bool), min_size=int(min_obj_size), connectivity=2)

        # MATLAB: v = imdilate(v, strel('disk',3));
        v = binary_dilation(v, footprint=disk(3))

        # MATLAB: v = imfill(v,'holes');
        v = binary_fill_holes(v, structure=np.ones((3, 3), dtype=bool))
        v = v.astype(np.float32)

        # MATLAB:
        # pc = phaseCongruency.*v;
        # pc = pc./max(pc(:))*2;
        pc = phaseCongruency.astype(np.float32) * v
        pc_max = float(np.max(pc)) if pc.size else 0.0
        if pc_max > 0.0:
            pc = pc / pc_max * 2.0
        else:
            pc = np.zeros_like(pc, dtype=np.float32)

        # MATLAB:
        # pc(pc<0) = 0;
        # pc(pc>1) = 1;
        pc = np.clip(pc, 0.0, 1.0).astype(np.float32)

        # MATLAB: pc = imresize(pc,size(II));
        pc = matlab_resize(pc, II.shape)

        # MATLAB:
        # Y1 = D0.*(1-pc);
        # II = II+Y1;
        Y1 = D0 * (1.0 - pc)
        II = II + Y1

        I_stack[:, :, i] = II.astype(np.float32)

    # MATLAB:
    # img_recon = upsample_grid(I_stack, info);
    # img_recon = img_recon(1:Dx,1:Dy);
    img_recon = upsample_grid(I_stack, info).astype(np.float32)
    img_recon = img_recon[:orig_shape[0], :orig_shape[1]]

    if return_adjusted_float:
        out = img_recon.astype(np.float32)
    else:
        # Optional legacy behavior: cast back to original dtype.
        if np.issubdtype(block_arr.dtype, np.integer):
            info_dtype = np.iinfo(block_arr.dtype)
            out = np.clip(img_recon, 0.0, 1.0) * float(info_dtype.max)
            out = out.astype(block_arr.dtype)
        else:
            out = img_recon.astype(block_arr.dtype, copy=False)

    if had_leading_one:
        return out[np.newaxis, :, :]
    return out.reshape(orig_block_shape)


# -------------------------------------------------------------------------
# Dask wrapper
# -------------------------------------------------------------------------

def _has_allowed_chunking(arr: da.Array) -> bool:
    """Validate chunking: trailing axes are (..., Z, Y, X)."""
    if arr.ndim < 3 or arr.ndim > 5:
        return False

    chunks = arr.chunks
    shape = arr.shape
    z_chunks, y_chunks, x_chunks = chunks[-3:]

    if not all(c == 1 for c in z_chunks):
        return False
    if not (len(y_chunks) == 1 and y_chunks[0] == shape[-2]):
        return False
    if not (len(x_chunks) == 1 and x_chunks[0] == shape[-1]):
        return False

    for dim_chunks in chunks[:-3]:
        if not all(c == 1 for c in dim_chunks):
            return False

    return True


def destripe(
    img: da.Array,
    bg_thresh: float = 0.004,
    patch_size: int = 1024,
    diff_thresh: float = 0.2,
    med_size: int = 61,
    phase_size: int = 512,
    guided_neighborhood: int = 15,
    guided_smoothing: float = 0.01,
    min_obj_size: int = 30,
    return_adjusted_float: bool = True,
) -> da.Array:
    """Apply patch-based destriping to each full XY Z-slice of a Dask array."""
    if not _has_allowed_chunking(img):
        raise ValueError(
            "Incorrect shape or chunking in dask array for destripe.\n"
            f"Detected shape: {img.shape}, chunks: {img.chunks}.\n"
            "Required chunking:\n"
            "  - 3D–5D array with trailing axes (..., Z, Y, X)\n"
            "  - Z axis chunk size = 1, meaning one slice per chunk\n"
            "  - Y and X axes each in a single chunk equal to the full image size\n"
            "  - Any leading axes, such as time or channel, chunked with size 1\n"
            "Examples:\n"
            "  img = img.rechunk((1, img.shape[1], img.shape[2]))        # 3D: Z,Y,X\n"
            "  img = img.rechunk((1, 1, img.shape[2], img.shape[3]))     # 4D: C,Z,Y,X\n"
        )

    out_dtype = np.float32 if return_adjusted_float else img.dtype

    return img.map_blocks(
        destripe_block,
        dtype=out_dtype,
        bg_thresh=bg_thresh,
        patch_size=patch_size,
        diff_thresh=diff_thresh,
        med_size=med_size,
        phase_size=phase_size,
        guided_neighborhood=guided_neighborhood,
        guided_smoothing=guided_smoothing,
        min_obj_size=min_obj_size,
        return_adjusted_float=return_adjusted_float,
    )
