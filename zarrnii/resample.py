"""Block-wise resampling of a (large) floating image onto a reference grid.

This module implements the engine behind :meth:`zarrnii.ZarrNii.apply_transform`.

Each output voxel is mapped through a chain of transforms: reference voxel ->
reference physical -> user transforms -> floating voxel. Consecutive affines
are collapsed into one matrix. The work is split into an eager planning step
and one lazy task per output block:

1. **Kernel lattice step.** Inside a task, the chain is evaluated exactly on a
   lattice of control points every ``step`` output voxels (at the block's
   corners for an affine-only chain, which is then exact), and linearly
   upsampled to the block's voxels. ``step`` is chosen by sampling the
   interpolation error at lattice cell centres across the whole grid and
   halving the step until it is below a tolerance (in floating voxels).

2. **Planning.** The chain is evaluated on a coarser planning lattice over the
   whole output grid, and each block's floating-voxel bounding box is taken
   from the lattice points that cover it, padded by twice the planning
   lattice's measured error. The same is done for the input of each
   non-affine transform, so that tasks only carry the part of a displacement
   field they need. Blocks that miss the floating image become constant fill
   blocks with no dependencies. Only bounding boxes are stored, so planning
   memory scales with the number of blocks, not voxels.

3. **Graph.** In ``io="graph"`` mode each output task depends on exactly the
   floating chunks that intersect its bounding box, so the floating image can
   be any dask array (lazy corrections, downsampled levels, channel
   selection), each input chunk is computed once, and nothing calls
   ``compute()`` inside a task. In ``io="direct"`` mode each task carries a
   culled copy of the floating image's graph for its own chunks, so tasks are
   independent (no inter-task transfers, bounded memory, suits
   ``dask.distributed`` with object stores) at the cost of recomputing the
   chunks shared by neighbouring blocks.

4. **Kernel.** The task stitches its floating subvolume, evaluates the cropped
   chain on its lattice points, upsamples to voxel coordinates with separable
   linear interpolation, and samples with
   :func:`scipy.ndimage.map_coordinates`, which releases the GIL so threads
   scale.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import dask.array as da
import numpy as np
from dask.highlevelgraph import HighLevelGraph
from scipy.ndimage import map_coordinates

from .transform import AffineTransform, DisplacementTransform

INTERPOLATION_ORDERS = {"nearest": 0, "linear": 1, "cubic": 3}

# Extra floating voxels loaded around each block's bounding box, per order.
_MARGINS = {0: 1, 1: 1, 3: 4}

# Cap on the kernel lattice step chosen automatically (in output voxels).
_MAX_AUTO_STEP = 64

# Planning lattice: about this many cells per block along each axis.
_PLAN_CELLS_PER_BLOCK = 4

# Bounding boxes are padded by this multiple of the measured lattice error.
_ERROR_SAFETY = 2.0


# ---------------------------------------------------------------------------
# Transform chains
# ---------------------------------------------------------------------------


def compose_chain(transforms: Sequence[Any]) -> List[Any]:
    """Collapse consecutive affines in a chain of transforms.

    Args:
        transforms: Transforms in the order they are applied to points. Each is
            an :class:`AffineTransform`, a 4x4 array, or any object with an
            ``apply_transform(vecs)`` method.

    Returns:
        A list where each run of consecutive affines is replaced by a single
        4x4 numpy matrix (applied first-to-last), and other transforms are kept
        as they are.
    """
    steps: List[Any] = []
    for tfm in transforms:
        if isinstance(tfm, AffineTransform):
            mat = np.asarray(tfm.matrix, dtype=np.float64)
        elif isinstance(tfm, np.ndarray) and tfm.shape == (4, 4):
            mat = tfm.astype(np.float64)
        else:
            steps.append(tfm)
            continue
        if steps and isinstance(steps[-1], np.ndarray):
            steps[-1] = mat @ steps[-1]
        else:
            steps.append(mat)
    return steps


def apply_chain(
    steps: Sequence[Any], pts: np.ndarray, stage_inputs: Optional[list] = None
) -> np.ndarray:
    """Apply a composed chain (see :func:`compose_chain`) to points.

    Args:
        steps: Output of :func:`compose_chain`.
        pts: ``(3, N)`` array of points.
        stage_inputs: If a list is given, the input points of each non-affine
            step are appended to it.

    Returns:
        ``(3, N)`` float64 array of transformed points.
    """
    pts = np.asarray(pts, dtype=np.float64)
    for step in steps:
        if isinstance(step, np.ndarray):
            pts = step[:3, :3] @ pts + step[:3, 3:4]
        else:
            if stage_inputs is not None:
                stage_inputs.append(pts)
            # Generic transforms historically received homogeneous (4, N) input
            homog = np.vstack([pts, np.ones((1, pts.shape[1]))])
            pts = np.asarray(step.apply_transform(homog))[:3]
    return pts


def _is_affine_chain(steps: Sequence[Any]) -> bool:
    return all(isinstance(s, np.ndarray) for s in steps)


class _CroppedDisplacement:
    """A :class:`DisplacementTransform` restricted to a sub-box of its grid.

    Evaluates identically to the full transform for points inside the box
    (including the zero displacement beyond the full field's extent), so tasks
    can carry a small piece of a large displacement field.
    """

    __slots__ = ("field", "offset", "full_shape", "inv_affine")

    def __init__(self, field, offset, full_shape, inv_affine):
        self.field = field  # (3, a, b, c) or None if the box misses the field
        self.offset = offset  # (3,) voxel index of field[:, 0, 0, 0]
        self.full_shape = full_shape  # (3,)
        self.inv_affine = inv_affine  # physical -> full-field voxel

    def apply_transform(self, vecs: np.ndarray) -> np.ndarray:
        vecs = np.asarray(vecs, dtype=np.float64)
        if self.field is None:
            return vecs
        inv = self.inv_affine
        vox = inv[:3, :3] @ vecs[:3] + inv[:3, 3:4]
        n = self.full_shape[:, None]
        outside = np.any((vox < -0.5) | (vox > n - 0.5), axis=0)
        vox -= self.offset[:, None]
        out = vecs.copy()
        for ax in range(3):
            d = map_coordinates(self.field[ax], vox, order=1, mode="nearest")
            d[outside] = 0.0
            out[ax] += d
        return out


def _crop_box(
    step: DisplacementTransform, lo: np.ndarray, hi: np.ndarray, margin: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """Voxel box ``[v0, v1)`` of a displacement field covering a physical box."""
    inv = step._inv_affine()
    shape = np.array(step.field_shape)
    corners = np.array(np.meshgrid(*zip(lo, hi), indexing="ij")).reshape(3, -1)
    vox = inv[:3, :3] @ corners + inv[:3, 3:4]
    v0 = np.clip(np.floor(vox.min(axis=1)).astype(int) - margin, 0, shape)
    v1 = np.clip(np.ceil(vox.max(axis=1)).astype(int) + margin + 1, 0, shape)
    return v0, v1


def _crop_step(step: Any, lo: np.ndarray, hi: np.ndarray) -> Any:
    """Crop a non-affine step to the physical box ``[lo, hi]`` of its inputs."""
    if not isinstance(step, DisplacementTransform):
        return step
    v0, v1 = _crop_box(step, lo, hi)
    sub = (
        None
        if np.any(v1 <= v0)
        else np.ascontiguousarray(step.read_field_box(v0, v1), dtype=np.float32)
    )
    return _CroppedDisplacement(
        sub, v0.astype(np.float64), np.array(step.field_shape), step._inv_affine()
    )


class _ZarrCropSpec:
    """A displacement step whose task reads its crop from a zarr store."""

    __slots__ = ("path", "v0", "v1", "full_shape", "inv_affine")

    def __init__(self, path, v0, v1, full_shape, inv_affine):
        self.path = path
        self.v0, self.v1 = v0, v1
        self.full_shape = full_shape
        self.inv_affine = inv_affine

    def materialize(self) -> "_CroppedDisplacement":
        sub = None
        if np.all(self.v1 > self.v0):
            import zarr

            arr = zarr.open_group(self.path, mode="r")["field"]
            sl = (slice(None),) + tuple(
                slice(int(a), int(b)) for a, b in zip(self.v0, self.v1)
            )
            sub = np.ascontiguousarray(arr[sl], dtype=np.float32)
        return _CroppedDisplacement(
            sub, self.v0.astype(np.float64), self.full_shape, self.inv_affine
        )


class _TiledCropSpec:
    """How to rebuild a :class:`_CroppedDisplacement` from shared field tiles.

    In the graph a cropped displacement step is ``[spec, [tile keys]]``; dask
    replaces the keys with the tiles, and :func:`_materialize_steps` stitches
    them.
    """

    __slots__ = ("v0", "v1", "tile_origins", "full_shape", "inv_affine")

    def __init__(self, v0, v1, tile_origins, full_shape, inv_affine):
        self.v0, self.v1 = v0, v1
        self.tile_origins = tile_origins
        self.full_shape = full_shape
        self.inv_affine = inv_affine


def _materialize_steps(steps: Sequence[Any]) -> List[Any]:
    out = []
    for step in steps:
        if isinstance(step, _ZarrCropSpec):
            out.append(step.materialize())
            continue
        if (
            isinstance(step, list)
            and len(step) == 2
            and isinstance(step[0], _TiledCropSpec)
        ):
            spec, tiles = step
            v0, v1 = spec.v0, spec.v1
            if not tiles:
                sub = None
            else:
                sub = np.empty((3, *(v1 - v0)), dtype=tiles[0].dtype)
                for tile, origin in zip(tiles, spec.tile_origins):
                    src, dst = [], []
                    for a in range(3):
                        t0, t1 = origin[a], origin[a] + tile.shape[1 + a]
                        s0, s1 = max(t0, v0[a]), min(t1, v1[a])
                        src.append(slice(s0 - t0, s1 - t0))
                        dst.append(slice(s0 - v0[a], s1 - v0[a]))
                    sub[(slice(None), *dst)] = tile[(slice(None), *src)]
            step = _CroppedDisplacement(
                sub, v0.astype(np.float64), spec.full_shape, spec.inv_affine
            )
        out.append(step)
    return out


# ---------------------------------------------------------------------------
# Lattices and error estimates
# ---------------------------------------------------------------------------


def lattice_positions(n: int, step: int) -> np.ndarray:
    """Lattice positions (in voxels) every ``step`` from 0, ending at ``n - 1``.

    The last cell may be shorter, so no control point lies outside the grid
    (where a displacement field may not be defined).
    """
    pos = np.arange(0, max(n - 1, 0), step, dtype=np.float64)
    return np.append(pos, float(n - 1)) if n > 0 else pos


def _grid_points(positions: Sequence[np.ndarray]) -> np.ndarray:
    return np.stack(np.meshgrid(*positions, indexing="ij")).reshape(3, -1)


def evaluate_lattice(
    steps, positions: Sequence[np.ndarray], with_stages: bool = False
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Evaluate the chain on the lattice spanned by ``positions``.

    Args:
        steps: Composed chain.
        positions: Positions along each axis (reference voxels).
        with_stages: Also return the input of each non-affine step.

    Returns:
        ``(values, stages)``: values of shape ``(3, L0, L1, L2)`` and, if
        requested, one array of the same shape per non-affine step.
    """
    shape = tuple(len(p) for p in positions)
    n_stages = sum(not isinstance(s, np.ndarray) for s in steps)
    values = np.empty((3, *shape), dtype=np.float64)
    stages = [np.empty_like(values) for _ in range(n_stages if with_stages else 0)]
    # Evaluate slab by slab along axis 0 to bound temporaries
    rows = max(1, (1 << 20) // max(shape[1] * shape[2], 1))
    for s in range(0, shape[0], rows):
        sl = slice(s, s + rows)
        pts = _grid_points([positions[0][sl], positions[1], positions[2]])
        captured: list = [] if with_stages else None
        out = apply_chain(steps, pts, captured)
        sub_shape = (3, len(positions[0][sl]), shape[1], shape[2])
        values[:, sl] = out.reshape(sub_shape)
        for st, cap in zip(stages, captured or []):
            st[:, sl] = cap.reshape(sub_shape)
    return values, stages


def lattice_error(
    steps,
    positions: Sequence[np.ndarray],
    n_samples: int = 4096,
    seed: int = 0,
) -> Tuple[float, List[float]]:
    """Max interpolation error of a lattice, sampled at cell centres.

    Cell centres are where trilinear interpolation of a smooth map is least
    accurate. The lattice values are only evaluated at the sampled cells, so
    this works for lattices far too large to hold in memory.

    Returns:
        ``(error, stage_errors)``: max error of the final coordinates and of the
        input of each non-affine step (in the units of each).
    """
    cells = [max(len(p) - 1, 1) for p in positions]
    rng = np.random.default_rng(seed)
    n_cells = int(np.prod(cells, dtype=np.int64))
    n = min(n_samples, n_cells)
    flat = rng.choice(n_cells, size=n, replace=False) if n < n_cells else np.arange(n)
    idx = np.unravel_index(flat, cells)

    lo_pos, hi_pos = [], []
    for a in range(3):
        p = positions[a]
        lo = np.minimum(idx[a], len(p) - 1)
        hi = np.minimum(lo + 1, len(p) - 1)
        lo_pos.append(p[lo])
        hi_pos.append(p[hi])

    def run(pts):
        cap: list = []
        return apply_chain(steps, pts, cap), cap

    centre = np.stack([(lo_pos[a] + hi_pos[a]) / 2 for a in range(3)])
    exact, exact_stages = run(centre)
    approx = np.zeros_like(exact)
    approx_stages = [np.zeros_like(s) for s in exact_stages]
    for c in np.ndindex(2, 2, 2):
        corner = np.stack([hi_pos[a] if c[a] else lo_pos[a] for a in range(3)])
        val, val_stages = run(corner)
        approx += val / 8
        for acc, v in zip(approx_stages, val_stages):
            acc += v / 8

    def max_err(a, b):
        return float(np.max(np.abs(a - b))) if a.size else 0.0

    return max_err(exact, approx), [
        max_err(e, a) for e, a in zip(exact_stages, approx_stages)
    ]


def _auto_step(steps, ref_spacing: np.ndarray, out_chunks: Sequence[int]) -> List[int]:
    """Initial kernel lattice step (output voxels) for a chain."""
    if _is_affine_chain(steps):
        # An affine map is reproduced exactly by trilinear interpolation
        return [max(int(c), 1) for c in out_chunks]
    disp_spacings = []
    for s in steps:
        if isinstance(s, DisplacementTransform) and s.disp_affine is not None:
            disp_spacings.append(
                np.min(np.linalg.norm(np.asarray(s.disp_affine.matrix)[:3, :3], axis=0))
            )
    if not disp_spacings:
        return [min(8, c) for c in out_chunks]
    # Start at the displacement field's grid spacing
    target = min(disp_spacings)
    return [
        int(
            np.clip(
                np.floor(target / max(sp, 1e-12) + 1e-6),
                1,
                min(_MAX_AUTO_STEP, c),
            )
        )
        for sp, c in zip(ref_spacing, out_chunks)
    ]


def choose_step(
    steps,
    ref_shape: Sequence[int],
    ref_spacing: Sequence[float],
    out_chunks: Sequence[int],
    coord_step: Union[str, int, Sequence[int]] = "auto",
    coord_tol: float = 0.1,
) -> Tuple[List[int], float]:
    """Pick the kernel lattice step and estimate its error (floating voxels)."""

    def err_for(step):
        if all(s == 1 for s in step):
            return 0.0
        pos = [lattice_positions(n, s) for n, s in zip(ref_shape, step)]
        return lattice_error(steps, pos)[0]

    if coord_step != "auto":
        step = [int(coord_step)] * 3 if np.isscalar(coord_step) else list(coord_step)
        err = err_for(step)
        if err > coord_tol:
            warnings.warn(
                f"Coordinate lattice step {tuple(step)} gives an estimated error "
                f"of {err:.3g} floating voxels (> coord_tol={coord_tol}). "
                "Use a smaller coord_step or coord_step='auto'.",
                stacklevel=3,
            )
        return step, err

    step = _auto_step(steps, np.asarray(ref_spacing, dtype=float), out_chunks)
    err = err_for(step)
    while err > coord_tol and any(s > 1 for s in step):
        new_step = [max(s // 2, 1) for s in step]
        new_err = err_for(new_step)
        if new_err > 0.5 * err:
            # Halving the step should cut a smooth map's error by ~4x; if it
            # doesn't, the error comes from a discontinuity (e.g. the edge of
            # a displacement field), which no lattice resolves.
            warnings.warn(
                f"Coordinate lattice error stays at ~{new_err:.3g} floating "
                f"voxels when refining the step {tuple(step)} -> "
                f"{tuple(new_step)}, which suggests a discontinuity in the "
                "transforms (e.g. output voxels beyond the edge of a "
                f"displacement field). Keeping step {tuple(step)}; pass "
                "coord_step=1 for an exact evaluation.",
                stacklevel=3,
            )
            break
        step, err = new_step, new_err
    return step, err


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def _block_ranges(chunks: Sequence[int]) -> List[Tuple[int, int]]:
    bounds = np.concatenate([[0], np.cumsum(chunks)])
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(len(chunks))]


def _covering_ranges(
    positions: np.ndarray, chunks: Sequence[int]
) -> List[Tuple[int, int]]:
    """Inclusive lattice index range covering each block along one axis."""
    out = []
    for s, e in _block_ranges(chunks):
        k0 = int(np.searchsorted(positions, s, side="right") - 1)
        k1 = int(np.searchsorted(positions, e - 1, side="left"))
        out.append((max(k0, 0), max(k0, min(k1, len(positions) - 1))))
    return out


def _block_minmax(values: np.ndarray, k_ranges) -> Tuple[np.ndarray, np.ndarray]:
    """Per-block min and max of lattice values, shape ``(n0, n1, n2, 3)``."""
    lo, hi = values, values
    for a in range(3):
        ax = 1 + a
        lo = np.stack(
            [
                np.take(lo, range(k0, k1 + 1), axis=ax).min(axis=ax)
                for k0, k1 in k_ranges[a]
            ],
            axis=ax,
        )
        hi = np.stack(
            [
                np.take(hi, range(k0, k1 + 1), axis=ax).max(axis=ax)
                for k0, k1 in k_ranges[a]
            ],
            axis=ax,
        )
    return np.moveaxis(lo, 0, -1), np.moveaxis(hi, 0, -1)


@dataclass
class ResamplePlan:
    """Per-block information for a resampling job (no per-voxel data)."""

    steps: List[Any]  # composed chain, reference voxel -> floating voxel
    step: Tuple[int, int, int]  # kernel lattice step (output voxels)
    positions: List[np.ndarray]  # kernel lattice positions along each axis
    out_chunks: Tuple[Tuple[int, ...], ...]  # spatial chunks of the output
    k_ranges: List[List[Tuple[int, int]]]  # kernel lattice range per block
    bbox_lo: np.ndarray  # (n0, n1, n2, 3) floating voxel start
    bbox_hi: np.ndarray  # (n0, n1, n2, 3) floating voxel stop (exclusive)
    empty: np.ndarray  # (n0, n1, n2) bool
    stage_lo: List[np.ndarray] = field(default_factory=list)  # per non-affine step
    stage_hi: List[np.ndarray] = field(default_factory=list)
    lattice_error: float = 0.0
    plan_error: float = 0.0

    def block_steps(self, i: Tuple[int, int, int]) -> List[Any]:
        """The chain for block ``i``, with displacement fields cropped to it."""
        out, s = [], 0
        for step in self.steps:
            if isinstance(step, np.ndarray):
                out.append(step)
            else:
                out.append(_crop_step(step, self.stage_lo[s][i], self.stage_hi[s][i]))
                s += 1
        return out

    def stage_box(self, s: int, i: Tuple[int, int, int]):
        """Physical box of the inputs of non-affine step ``s`` for block ``i``."""
        return self.stage_lo[s][i], self.stage_hi[s][i]

    def block_positions(self, i: Tuple[int, int, int]) -> List[np.ndarray]:
        return [
            self.positions[a][self.k_ranges[a][i[a]][0] : self.k_ranges[a][i[a]][1] + 1]
            for a in range(3)
        ]

    def stats(self, flo_shape: Sequence[int]) -> Dict[str, Any]:
        """Summary of the plan (useful to choose chunk sizes)."""
        sizes = np.prod(self.bbox_hi - self.bbox_lo, axis=-1)
        sizes = np.where(self.empty, 0, sizes)
        out_vox = np.prod([sum(c) for c in self.out_chunks])
        return {
            "n_blocks": int(self.empty.size),
            "n_empty_blocks": int(self.empty.sum()),
            "step": self.step,
            "lattice_error_vox": self.lattice_error,
            "plan_error_vox": self.plan_error,
            "max_block_input_voxels": int(sizes.max()) if sizes.size else 0,
            "input_voxels_read": int(sizes.sum()),
            "input_voxels": int(np.prod(flo_shape)),
            "output_voxels": int(out_vox),
        }


def plan_resample(
    steps,
    ref_shape: Sequence[int],
    out_chunks: Sequence[Sequence[int]],
    flo_shape: Sequence[int],
    ref_spacing: Sequence[float],
    order: int = 1,
    coord_step: Union[str, int, Sequence[int]] = "auto",
    coord_tol: float = 0.1,
) -> ResamplePlan:
    """Choose the kernel lattice and work out what each output block reads.

    Args:
        steps: Composed chain from reference voxel to floating voxel (data
            order), see :func:`compose_chain`.
        ref_shape: Spatial shape of the output grid.
        out_chunks: Spatial dask chunks of the output.
        flo_shape: Spatial shape of the floating image.
        ref_spacing: Output voxel spacing per axis (physical units), used to
            pick the lattice step automatically.
        order: Interpolation order (sets the bounding box margin).
        coord_step: Kernel lattice step in output voxels. ``"auto"`` picks one
            from the transforms and refines it until the error is below
            ``coord_tol``; an int or 3-tuple fixes it (``1`` is exact).
        coord_tol: Max allowed lattice error, in floating voxels, for
            ``coord_step="auto"``.

    Returns:
        A :class:`ResamplePlan`.
    """
    out_chunks = tuple(tuple(int(c) for c in ch) for ch in out_chunks)
    max_chunk = [max(ch) for ch in out_chunks]

    step, err = choose_step(
        steps, ref_shape, ref_spacing, max_chunk, coord_step, coord_tol
    )
    positions = [lattice_positions(n, s) for n, s in zip(ref_shape, step)]
    k_ranges = [_covering_ranges(positions[a], out_chunks[a]) for a in range(3)]

    # Planning lattice: a few cells per block, never finer than the kernel's
    plan_step = [max(s, c // _PLAN_CELLS_PER_BLOCK, 1) for s, c in zip(step, max_chunk)]
    plan_pos = [lattice_positions(n, s) for n, s in zip(ref_shape, plan_step)]
    plan_err, plan_stage_err = (
        (0.0, [0.0] * sum(not isinstance(s, np.ndarray) for s in steps))
        if _is_affine_chain(steps)
        else lattice_error(steps, plan_pos)
    )
    values, stages = evaluate_lattice(steps, plan_pos, with_stages=True)
    plan_ranges = [_covering_ranges(plan_pos[a], out_chunks[a]) for a in range(3)]

    lo, hi = _block_minmax(values, plan_ranges)
    pad = _ERROR_SAFETY * plan_err
    lo, hi = lo - pad, hi + pad

    stage_lo, stage_hi = [], []
    for st, st_err in zip(stages, plan_stage_err):
        s_lo, s_hi = _block_minmax(st, plan_ranges)
        s_pad = _ERROR_SAFETY * st_err
        stage_lo.append(s_lo - s_pad)
        stage_hi.append(s_hi + s_pad)

    flo_shape = np.asarray(flo_shape, dtype=np.int64)
    margin = _MARGINS.get(order, 4)
    # Anything sampled outside [-0.5, n - 0.5] is fill, so such blocks are empty
    empty = np.any((hi < -0.5) | (lo > flo_shape - 0.5), axis=-1)
    bbox_lo = np.clip(np.floor(lo).astype(np.int64) - margin, 0, flo_shape)
    bbox_hi = np.clip(np.ceil(hi).astype(np.int64) + margin + 1, 0, flo_shape)
    empty |= np.any(bbox_hi <= bbox_lo, axis=-1)

    return ResamplePlan(
        steps=list(steps),
        step=tuple(int(s) for s in step),
        positions=positions,
        out_chunks=out_chunks,
        k_ranges=k_ranges,
        bbox_lo=bbox_lo,
        bbox_hi=bbox_hi,
        empty=empty,
        stage_lo=stage_lo,
        stage_hi=stage_hi,
        lattice_error=err,
        plan_error=plan_err,
    )


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


# Voxels per slab when a block is sampled slab by slab (bounds the per-task
# memory for coordinates and sampled values to ~40 MB, whatever the block size).
_SLAB_VOXELS = 1 << 21


class _BlockCoords:
    """Floating voxel coordinates of a block's voxels, produced slab by slab.

    The chain is evaluated once on the block's lattice points (or per slab at
    every voxel, for an exact lattice), and coordinates are upsampled for one
    slab of rows (along the block's first axis) at a time.
    """

    def __init__(self, steps, positions, block_start, block_shape):
        self.steps = steps
        self.voxels = [block_start[a] + np.arange(block_shape[a]) for a in range(3)]
        self.exact = all(
            len(p) == len(v) and np.array_equal(p, v)
            for p, v in zip(positions, self.voxels)
        )
        if not self.exact:
            self.lattice, _ = evaluate_lattice(steps, positions)
            self.axis_coords = [
                np.interp(
                    self.voxels[a],
                    positions[a],
                    np.arange(len(positions[a]), dtype=float),
                )
                for a in range(3)
            ]

    def slab(self, r0: int, r1: int) -> np.ndarray:
        """``(3, r1 - r0, n1, n2)`` float32 coordinates of rows ``r0:r1``."""
        from .core import _interp_linear_separable

        if self.exact:
            vox = [self.voxels[0][r0:r1], self.voxels[1], self.voxels[2]]
            vals = apply_chain(self.steps, _grid_points(vox)).astype(np.float32)
            return vals.reshape(3, r1 - r0, len(vox[1]), len(vox[2]))
        axis_coords = [self.axis_coords[0][r0:r1], *self.axis_coords[1:]]
        return np.stack(
            [_interp_linear_separable(self.lattice[i], axis_coords) for i in range(3)]
        )


def _block_coords(
    steps: Sequence[Any],
    positions: Sequence[np.ndarray],
    block_start: Sequence[int],
    block_shape: Sequence[int],
) -> List[np.ndarray]:
    """Floating voxel coordinates (float32) of each voxel in a block."""
    return list(
        _BlockCoords(steps, positions, block_start, block_shape).slab(
            0, int(block_shape[0])
        )
    )


def _cast(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer) and not np.issubdtype(values.dtype, np.integer):
        info = np.iinfo(dtype)
        values = np.clip(np.rint(values), info.min, info.max)
    elif dtype == np.bool_:
        values = values > 0.5
    return values.astype(dtype, copy=False)


def resample_block(
    subvol: np.ndarray,
    sub_lo: Sequence[int],
    steps: Sequence[Any],
    positions: Sequence[np.ndarray],
    block_start: Sequence[int],
    block_shape: Sequence[int],
    flo_shape: Sequence[int],
    order: int,
    fill_value: float,
    dtype: Any,
) -> np.ndarray:
    """Resample one output block.

    Args:
        subvol: Floating subvolume ``(*lead, s0, s1, s2)``.
        sub_lo: Floating voxel index of ``subvol[..., 0, 0, 0]``.
        steps: Chain from reference voxel to floating voxel for this block
            (displacement fields may be cropped to it).
        positions: Kernel lattice positions covering this block along each
            axis (reference voxels).
        block_start: Reference voxel index of the block's first voxel.
        block_shape: Spatial shape of the block.
        flo_shape: Spatial shape of the whole floating image.
        order: Spline order for :func:`scipy.ndimage.map_coordinates`.
        fill_value: Value for samples outside the floating image.
        dtype: Output dtype.

    Returns:
        Array of shape ``(*lead, *block_shape)``.
    """
    from scipy.ndimage import spline_filter

    block_shape = tuple(int(b) for b in block_shape)
    lead = subvol.shape[:-3]
    coords = _BlockCoords(
        _materialize_steps(steps), positions, block_start, block_shape
    )

    sub = subvol.reshape((-1,) + subvol.shape[-3:])
    work_dtype = sub.dtype if order == 0 else np.result_type(sub.dtype, np.float32)
    if order > 1:
        # prefilter once, not once per slab
        sub = np.stack(
            [
                spline_filter(ch, order=order, mode="nearest", output=np.float32)
                for ch in sub
            ]
        )
    out = np.empty((sub.shape[0],) + block_shape, dtype=np.dtype(dtype))
    lo = np.asarray(sub_lo, dtype=np.float32).reshape(3, 1, 1, 1)
    hi = np.asarray(flo_shape, dtype=np.float32).reshape(3, 1, 1, 1) - 0.5

    rows = max(1, _SLAB_VOXELS // max(block_shape[1] * block_shape[2], 1))
    for r0 in range(0, block_shape[0], rows):
        r1 = min(block_shape[0], r0 + rows)
        c = coords.slab(r0, r1)
        # Samples outside the floating image's extent ([-0.5, n - 0.5] in
        # voxels) are fill. Within it, edges are clamped (nearest), so a voxel
        # centre that lands a rounding error outside [0, n - 1] still gets its
        # value.
        outside = np.any((c < -0.5) | (c > hi), axis=0)
        c -= lo
        for ch in range(sub.shape[0]):
            vals = map_coordinates(
                sub[ch],
                c,
                order=order,
                mode="nearest",
                prefilter=False,
                output=work_dtype,
            )
            if outside.any():
                vals[outside] = fill_value
            out[ch, r0:r1] = _cast(vals, out.dtype)
    return out.reshape(lead + block_shape)


def _paste(sub, sub_lo, sub_hi, chunk, origin) -> None:
    """Copy the part of ``chunk`` (origin ``origin``) inside ``sub``'s box."""
    src, dst = [], []
    for a in range(3):
        c0 = origin[a]
        c1 = c0 + chunk.shape[-3 + a]
        s0, s1 = max(c0, sub_lo[a]), min(c1, sub_hi[a])
        src.append(slice(s0 - c0, s1 - c0))
        dst.append(slice(s0 - sub_lo[a], s1 - sub_lo[a]))
    sub[(Ellipsis, *dst)] = chunk[(Ellipsis, *src)]


def _assemble_and_resample(
    chunks: List[np.ndarray],
    chunk_origins: List[Tuple[int, int, int]],
    sub_lo: Tuple[int, int, int],
    sub_hi: Tuple[int, int, int],
    *args,
) -> np.ndarray:
    """Stitch the needed parts of input chunks, then :func:`resample_block`."""
    lead = chunks[0].shape[:-3]
    sub = np.empty(
        lead + tuple(h - lo for lo, h in zip(sub_lo, sub_hi)), dtype=chunks[0].dtype
    )
    for chunk, origin in zip(chunks, chunk_origins):
        _paste(sub, sub_lo, sub_hi, chunk, origin)
    return resample_block(sub, sub_lo, *args)


class _InlineGraph:
    """Opaque holder for an inlined subgraph.

    Wrapping the dict stops the outer scheduler from treating the keys inside
    it as dependencies of the task.
    """

    __slots__ = ("dsk",)

    def __init__(self, dsk: Dict[Any, Any]):
        self.dsk = dsk


def _inline_and_resample(
    subgraph: _InlineGraph,
    keys: List[Any],
    chunk_origins: List[Tuple[int, int, int]],
    sub_lo: Tuple[int, int, int],
    sub_hi: Tuple[int, int, int],
    *args,
) -> np.ndarray:
    """Compute this block's input chunks from an inlined subgraph, then resample.

    The subgraph is a culled copy of the floating image's graph, so the task
    has no dependencies on other tasks and runs with the synchronous scheduler.
    """
    from dask.local import get_sync

    # one input chunk in memory at a time, besides the stitched subvolume
    sub = None
    for key, origin in zip(keys, chunk_origins):
        chunk = get_sync(subgraph.dsk, key)
        if sub is None:
            sub = np.empty(
                chunk.shape[:-3] + tuple(h - lo for lo, h in zip(sub_lo, sub_hi)),
                dtype=chunk.dtype,
            )
        _paste(sub, sub_lo, sub_hi, chunk, origin)
        del chunk
    return resample_block(sub, sub_lo, *args)


_WARP_TILE = 32


def _tiled_step(dsk, name, i, step, lo, hi):
    """Graph form of a displacement step cropped to ``[lo, hi]`` (physical).

    Adds the field tiles it needs to ``dsk`` (once) and returns
    ``[spec, [tile keys]]``. A field stored in zarr (``step.zarr_path``) is not
    put in the graph: the task reads its crop from the store.
    """
    shape = np.array(step.field_shape)
    v0, v1 = _crop_box(step, lo, hi)
    if step.zarr_path is not None:
        return _ZarrCropSpec(step.zarr_path, v0, v1, shape, step._inv_affine())
    full = step._field_cxyz()
    keys, origins = [], []
    if np.all(v1 > v0):
        t0 = v0 // _WARP_TILE
        t1 = (v1 - 1) // _WARP_TILE
        for t in np.ndindex(*(t1 - t0 + 1)):
            tidx = t0 + np.array(t)
            key = (f"{name}-warp{i}", *(int(v) for v in tidx))
            o = tidx * _WARP_TILE
            if key not in dsk:
                e = np.minimum(o + _WARP_TILE, shape)
                dsk[key] = np.ascontiguousarray(
                    full[:, o[0] : e[0], o[1] : e[1], o[2] : e[2]], dtype=np.float32
                )
            keys.append(key)
            origins.append(o)
    spec = _TiledCropSpec(v0, v1, origins, shape, step._inv_affine())
    return [spec, keys]


def _step_token(step: Any) -> Any:
    if isinstance(step, np.ndarray):
        return step
    if isinstance(step, DisplacementTransform):
        field = step.zarr_path if step.disp_xyz is None else step.disp_xyz
        return (field, np.asarray(step.disp_affine.matrix))
    return id(step)


def _fill_block(shape, fill_value, dtype) -> np.ndarray:
    return np.full(shape, fill_value, dtype=dtype)


def _cull(
    dsk: Dict[Any, Any], keys: List[Any], cache: Dict[Any, set]
) -> Dict[Any, Any]:
    """The part of ``dsk`` needed to compute ``keys``."""
    from dask.core import get_dependencies

    out: Dict[Any, Any] = {}
    stack = list(keys)
    while stack:
        k = stack.pop()
        if k in out:
            continue
        out[k] = dsk[k]
        if k not in cache:
            cache[k] = get_dependencies(dsk, k)
        stack.extend(cache[k])
    return out


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_resampled_array(
    flo: da.Array,
    plan: ResamplePlan,
    order: int,
    fill_value: float,
    dtype: Any,
    io: str = "graph",
    name: Optional[str] = None,
) -> da.Array:
    """Create the lazy output array for a :class:`ResamplePlan`.

    Args:
        flo: Floating image, ``(*lead, s0, s1, s2)``.
        plan: Output of :func:`plan_resample`.
        order: Interpolation order.
        fill_value: Fill for samples outside the floating image.
        dtype: Output dtype.
        io: ``"graph"``: output tasks depend on the floating image's chunk
            tasks, so each input chunk is computed once and shared.
            ``"direct"``: each output task computes the input chunks it needs
            from its own culled copy of the floating image's graph, so tasks
            are independent (no inter-task transfers, bounded memory) but the
            chunks shared by neighbouring blocks are computed more than once.
        name: Optional output name.

    Returns:
        Dask array of shape ``(*lead, *ref_spatial_shape)``.
    """
    from dask.base import tokenize

    if io not in ("graph", "direct"):
        raise ValueError(f"io must be 'graph' or 'direct', got {io!r}")
    dtype = np.dtype(dtype)
    n_lead = flo.ndim - 3
    lead_chunks = flo.chunks[:n_lead]
    flo_spatial = tuple(int(s) for s in flo.shape[-3:])
    if name is None:
        name = "resample-" + tokenize(
            flo.name,
            plan.bbox_lo,
            plan.bbox_hi,
            plan.positions,
            plan.out_chunks,
            [_step_token(s) for s in plan.steps],
            order,
            fill_value,
            str(dtype),
            io,
        )

    # Non-affine steps go into the graph once: displacement fields as float32
    # tiles (each task depends on the tiles its block needs), anything else as
    # a single key.
    dsk: Dict[Any, Any] = {}
    stage_of: Dict[int, int] = {}
    for i, step in enumerate(plan.steps):
        if not isinstance(step, np.ndarray):
            stage_of[i] = len(stage_of)
            if not isinstance(step, DisplacementTransform):
                dsk[(name + "-transform", i)] = step

    def block_chain(bidx):
        chain = []
        for i, step in enumerate(plan.steps):
            if isinstance(step, np.ndarray):
                chain.append(step)
            elif isinstance(step, DisplacementTransform):
                chain.append(
                    _tiled_step(dsk, name, i, step, *plan.stage_box(stage_of[i], bidx))
                )
            else:
                chain.append((name + "-transform", i))
        return chain

    flo_bounds = [np.concatenate([[0], np.cumsum(c)]) for c in flo.chunks[-3:]]
    block_ranges = [_block_ranges(c) for c in plan.out_chunks]
    if io == "direct":
        flo_dsk = dict(flo.__dask_graph__())
        dep_cache: Dict[Any, set] = {}

    for bidx in np.ndindex(*plan.empty.shape):
        block_start = tuple(block_ranges[a][bidx[a]][0] for a in range(3))
        block_shape = tuple(plan.out_chunks[a][bidx[a]] for a in range(3))
        if plan.empty[bidx]:
            for lead_idx in np.ndindex(*[len(c) for c in lead_chunks]):
                lead_shape = tuple(lead_chunks[a][j] for a, j in enumerate(lead_idx))
                dsk[(name, *lead_idx, *bidx)] = (
                    _fill_block,
                    lead_shape + block_shape,
                    fill_value,
                    dtype,
                )
            continue

        lo = tuple(int(v) for v in plan.bbox_lo[bidx])
        hi = tuple(int(v) for v in plan.bbox_hi[bidx])
        steps = block_chain(bidx)
        args = (
            steps,
            plan.block_positions(bidx),
            block_start,
            block_shape,
            flo_spatial,
            order,
            fill_value,
            dtype,
        )
        ranges = []
        for a in range(3):
            b = flo_bounds[a]
            i0 = int(np.searchsorted(b, lo[a], side="right") - 1)
            i1 = int(np.searchsorted(b, hi[a] - 1, side="right") - 1)
            ranges.append(range(i0, i1 + 1))
        origins = [
            tuple(int(flo_bounds[a][ranges[a][ci[a]]]) for a in range(3))
            for ci in np.ndindex(*[len(r) for r in ranges])
        ]
        for lead_idx in np.ndindex(*[len(c) for c in lead_chunks]):
            keys = [
                (flo.name, *lead_idx, *(ranges[a][ci[a]] for a in range(3)))
                for ci in np.ndindex(*[len(r) for r in ranges])
            ]
            key = (name, *lead_idx, *bidx)
            if io == "direct":
                sub = _InlineGraph(_cull(flo_dsk, keys, dep_cache))
                dsk[key] = (_inline_and_resample, sub, keys, origins, lo, hi, *args)
            else:
                dsk[key] = (_assemble_and_resample, keys, origins, lo, hi, *args)

    deps = [flo] if io == "graph" else []
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=deps)
    meta = np.empty((0,) * (n_lead + 3), dtype=dtype)
    return da.Array(
        graph, name, chunks=lead_chunks + plan.out_chunks, dtype=dtype, meta=meta
    )
