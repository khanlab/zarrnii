"""Benchmarking tool for OME-Zarr chunking, sharding, and Dask configuration.

This module provides utilities to help users choose optimal chunking, sharding,
and Dask scheduler settings for their workloads.  It measures wall-clock time,
peak memory, and CPU efficiency for the key operations:

1. Create a synthetic Dask array of configurable shape / dtype.
2. Write the array to an OME-Zarr store (via :func:`zarrnii.save_ngff_image`).
3. Read the array back from the store.

Typical usage
-------------
Run from the command line::

    zarrnii-benchmark --shape 64 256 256 \\
        --chunks 32,32,32 64,64,64 \\
        --shards none 64,64,64 \\
        --dask-configs threads:4 distributed:4:2 \\
        --output-dir ./bench_results

Or use the Python API::

    from zarrnii.benchmark import BenchmarkSuite

    suite = BenchmarkSuite(
        shape=(64, 256, 256),
        dtype="float32",
        chunk_shapes=[(32, 32, 32), (64, 64, 64)],
        dask_configs=[{"scheduler": "threads", "n_threads": 4}],
        output_dir="./bench_results",
    )
    results_df = suite.run()
    suite.generate_report(results_df)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import dask
import dask.array as da
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DaskConfig:
    """Specification for a Dask scheduler setup.

    Attributes:
        scheduler: ``"threads"`` for the built-in threaded scheduler or
            ``"distributed"`` for a ``dask.distributed.LocalCluster``.
        n_threads: Total number of threads/cores available.  For the
            ``"threads"`` scheduler this is passed directly as
            ``num_workers``.  For ``"distributed"`` it is used together
            with *threads_per_worker* to derive the worker count.
        threads_per_worker: Threads allocated to each distributed worker.
            Ignored when *scheduler* is ``"threads"``.
        label: Human-readable identifier used in reports.  Auto-generated
            when not provided.
    """

    scheduler: str = "threads"
    n_threads: int = 4
    threads_per_worker: int = 2
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            if self.scheduler == "distributed":
                n_workers = max(1, self.n_threads // self.threads_per_worker)
                self.label = (
                    f"distributed(workers={n_workers},"
                    f"tpw={self.threads_per_worker})"
                )
            else:
                self.label = f"threads(n={self.n_threads})"


@dataclass
class BenchmarkConfig:
    """Full configuration for one benchmark scenario.

    Attributes:
        shape: Spatial shape of the synthetic array, e.g. ``(64, 256, 256)``.
        dtype: NumPy dtype string, e.g. ``"float32"``.
        chunk_shape: Zarr chunk shape.
        shard_shape: Zarr shard shape (``None`` disables sharding).
        dask_config: Dask scheduler configuration.
    """

    shape: Tuple[int, ...]
    dtype: str
    chunk_shape: Tuple[int, ...]
    shard_shape: Optional[Tuple[int, ...]]
    dask_config: DaskConfig

    @property
    def label(self) -> str:
        """Short human-readable identifier for this config."""
        shard = f"shard={self.shard_shape}" if self.shard_shape else "no_shard"
        return (
            f"shape={self.shape}|dtype={self.dtype}|"
            f"chunk={self.chunk_shape}|{shard}|"
            f"{self.dask_config.label}"
        )


@dataclass
class BenchmarkResult:
    """Timing and resource metrics for one benchmark run.

    Attributes:
        config: The configuration that produced this result.
        repetition: 0-based repetition index.
        write_wall_s: Wall-clock seconds for the OME-Zarr write.
        read_wall_s: Wall-clock seconds for the OME-Zarr read-back.
        write_cpu_s: User+system CPU seconds consumed during write.
        read_cpu_s: User+system CPU seconds consumed during read.
        peak_memory_mb: Peak RSS memory in megabytes during the run.
        error: Non-empty string when the run raised an exception.
    """

    config: BenchmarkConfig
    repetition: int
    write_wall_s: float = 0.0
    read_wall_s: float = 0.0
    write_cpu_s: float = 0.0
    read_cpu_s: float = 0.0
    peak_memory_mb: float = 0.0
    error: str = ""

    # derived -----------------------------------------------------------------

    @property
    def total_wall_s(self) -> float:
        """Total wall time (write + read)."""
        return self.write_wall_s + self.read_wall_s

    @property
    def write_cpu_efficiency(self) -> float:
        """CPU efficiency during write (cpu_s / wall_s / n_threads)."""
        if self.write_wall_s <= 0:
            return 0.0
        return self.write_cpu_s / (
            self.write_wall_s * self.config.dask_config.n_threads
        )

    @property
    def read_cpu_efficiency(self) -> float:
        """CPU efficiency during read (cpu_s / wall_s / n_threads)."""
        if self.read_wall_s <= 0:
            return 0.0
        return self.read_cpu_s / (self.read_wall_s * self.config.dask_config.n_threads)

    def to_dict(self) -> Dict:
        """Serialise to a flat dictionary suitable for CSV / pandas."""
        cfg = self.config
        dc = cfg.dask_config
        return {
            "shape": str(cfg.shape),
            "dtype": cfg.dtype,
            "chunk_shape": str(cfg.chunk_shape),
            "shard_shape": str(cfg.shard_shape) if cfg.shard_shape else "",
            "scheduler": dc.scheduler,
            "n_threads": dc.n_threads,
            "threads_per_worker": dc.threads_per_worker,
            "dask_label": dc.label,
            "repetition": self.repetition,
            "write_wall_s": round(self.write_wall_s, 4),
            "read_wall_s": round(self.read_wall_s, 4),
            "total_wall_s": round(self.total_wall_s, 4),
            "write_cpu_s": round(self.write_cpu_s, 4),
            "read_cpu_s": round(self.read_cpu_s, 4),
            "write_cpu_efficiency": round(self.write_cpu_efficiency, 4),
            "read_cpu_efficiency": round(self.read_cpu_efficiency, 4),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Memory tracker (runs in a background thread)
# ---------------------------------------------------------------------------


class _PeakMemoryTracker:
    """Lightweight background thread that polls RSS memory usage.

    Uses :mod:`psutil` when available, with a fallback to reading
    ``/proc/self/status`` on Linux.
    """

    def __init__(self, interval_s: float = 0.05) -> None:
        self._interval = interval_s
        self._peak_mb: float = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    # ------------------------------------------------------------------
    def _current_rss_mb(self) -> float:
        try:
            import psutil  # type: ignore[import-untyped]

            return psutil.Process().memory_info().rss / 1024**2
        except ImportError:
            pass
        # Fallback: Linux /proc
        try:
            with open("/proc/self/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024
        except OSError:
            pass
        return 0.0

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = self._current_rss_mb()
            if rss > self._peak_mb:
                self._peak_mb = rss
            self._stop.wait(self._interval)

    # ------------------------------------------------------------------
    def __enter__(self) -> "_PeakMemoryTracker":
        self._peak_mb = self._current_rss_mb()
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._stop.set()
        self._thread.join()

    @property
    def peak_mb(self) -> float:
        """Peak RSS in megabytes observed during the tracking window."""
        return self._peak_mb


# ---------------------------------------------------------------------------
# Core benchmarking logic
# ---------------------------------------------------------------------------


def _cpu_time_s() -> float:
    """Return combined user+system CPU time for the current process."""
    t = os.times()
    return t.user + t.system


def _make_synthetic_dask_array(
    shape: Tuple[int, ...],
    dtype: str,
    chunk_shape: Tuple[int, ...],
) -> da.Array:
    """Return a random Dask array with the given shape, dtype, and chunk shape."""
    return da.random.random(shape, chunks=chunk_shape).astype(dtype)


def _write_ome_zarr(
    arr: da.Array,
    store_path: str,
    chunk_shape: Tuple[int, ...],
) -> None:
    """Write *arr* to an OME-Zarr store at *store_path*."""
    import ngff_zarr as nz

    # Build a single-scale NgffImage and save it
    dims = ["z", "y", "x"][-len(arr.shape) :]  # basic spatial dims
    ngff_image = nz.to_ngff_image(arr, dims=dims)
    multiscales = nz.to_multiscales(ngff_image, scale_factors=[], chunks=chunk_shape)
    nz.to_ngff_zarr(store_path, multiscales, version="0.4")


def _read_ome_zarr(store_path: str) -> np.ndarray:
    """Read the OME-Zarr store at *store_path* and return a NumPy array."""
    import ngff_zarr as nz

    multiscales = nz.from_ngff_zarr(store_path)
    # from_ngff_zarr returns an NgffMultiscales; grab the highest-resolution image
    arr = multiscales.images[0].data
    if hasattr(arr, "compute"):
        return arr.compute()
    return np.asarray(arr)


def _run_single(
    config: BenchmarkConfig,
    repetition: int,
    tmp_dir: str,
) -> BenchmarkResult:
    """Execute one write+read benchmark and return a :class:`BenchmarkResult`."""
    from zarrnii.dask_utils import get_dask_client

    result = BenchmarkResult(config=config, repetition=repetition)
    store_path = os.path.join(tmp_dir, f"bench_{repetition}.ome.zarr")

    dc = config.dask_config
    try:
        with get_dask_client(
            scheduler=dc.scheduler,
            threads=dc.n_threads,
            threads_per_worker=dc.threads_per_worker,
        ):
            with _PeakMemoryTracker() as mem:
                # ----- WRITE -----
                arr = _make_synthetic_dask_array(
                    config.shape, config.dtype, config.chunk_shape
                )
                cpu_before_write = _cpu_time_s()
                t0_write = time.perf_counter()
                _write_ome_zarr(arr, store_path, config.chunk_shape)
                result.write_wall_s = time.perf_counter() - t0_write
                result.write_cpu_s = _cpu_time_s() - cpu_before_write

                # ----- READ -----
                cpu_before_read = _cpu_time_s()
                t0_read = time.perf_counter()
                _read_ome_zarr(store_path)
                result.read_wall_s = time.perf_counter() - t0_read
                result.read_cpu_s = _cpu_time_s() - cpu_before_read

            result.peak_memory_mb = mem.peak_mb

    except Exception as exc:  # noqa: BLE001
        result.error = str(exc)
        logger.exception("Benchmark run failed: %s", config.label)

    return result


# ---------------------------------------------------------------------------
# BenchmarkSuite
# ---------------------------------------------------------------------------


class BenchmarkSuite:
    """Orchestrate a set of ZarrNii read/write benchmarks.

    Parameters
    ----------
    shape:
        Spatial shape of the synthetic 3-D array, e.g. ``(64, 256, 256)``.
    dtype:
        NumPy dtype string, e.g. ``"float32"``.
    chunk_shapes:
        List of chunk shapes to sweep.
    shard_shapes:
        List of shard shapes (use ``[None]`` to disable sharding).
    dask_configs:
        List of :class:`DaskConfig` objects (or plain dicts with keys
        ``scheduler``, ``n_threads``, ``threads_per_worker``).
    n_reps:
        Number of repetitions per configuration (results are averaged).
    output_dir:
        Directory where CSV results and HTML report are written.  Defaults
        to the current working directory.

    Examples
    --------
    >>> from zarrnii.benchmark import BenchmarkSuite, DaskConfig
    >>> suite = BenchmarkSuite(
    ...     shape=(32, 128, 128),
    ...     dtype="float32",
    ...     chunk_shapes=[(16, 64, 64), (32, 128, 128)],
    ...     dask_configs=[DaskConfig(scheduler="threads", n_threads=4)],
    ...     n_reps=1,
    ...     output_dir="/tmp/bench",
    ... )
    >>> df = suite.run()
    >>> suite.generate_report(df)
    """

    def __init__(
        self,
        shape: Tuple[int, ...] = (64, 256, 256),
        dtype: str = "float32",
        chunk_shapes: Optional[Sequence[Tuple[int, ...]]] = None,
        shard_shapes: Optional[Sequence[Optional[Tuple[int, ...]]]] = None,
        dask_configs: Optional[Sequence[Union[DaskConfig, Dict]]] = None,
        n_reps: int = 3,
        output_dir: str = ".",
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.chunk_shapes: List[Tuple[int, ...]] = list(
            chunk_shapes or [tuple(s // 2 for s in shape)]
        )
        self.shard_shapes: List[Optional[Tuple[int, ...]]] = list(
            shard_shapes or [None]
        )
        self.n_reps = n_reps
        self.output_dir = Path(output_dir)

        # Normalise dask_configs to DaskConfig objects
        raw_configs = dask_configs or [DaskConfig(scheduler="threads", n_threads=4)]
        self.dask_configs: List[DaskConfig] = [
            DaskConfig(**c) if isinstance(c, dict) else c for c in raw_configs
        ]

    # ------------------------------------------------------------------
    def _build_configs(self) -> List[BenchmarkConfig]:
        configs: List[BenchmarkConfig] = []
        for dc in self.dask_configs:
            for chunk in self.chunk_shapes:
                for shard in self.shard_shapes:
                    configs.append(
                        BenchmarkConfig(
                            shape=self.shape,
                            dtype=self.dtype,
                            chunk_shape=chunk,
                            shard_shape=shard,
                            dask_config=dc,
                        )
                    )
        return configs

    # ------------------------------------------------------------------
    def run(self) -> "pd.DataFrame":  # noqa: F821
        """Run all benchmark configurations and return a :class:`pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            One row per (configuration, repetition) with timing and resource
            columns.  See :class:`BenchmarkResult` for column descriptions.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for benchmark reporting") from exc

        configs = self._build_configs()
        total = len(configs) * self.n_reps
        logger.info(
            "Starting benchmark: %d configs × %d reps = %d runs",
            len(configs),
            self.n_reps,
            total,
        )

        rows = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            for i, cfg in enumerate(configs):
                for rep in range(self.n_reps):
                    run_num = i * self.n_reps + rep + 1
                    logger.info(
                        "[%d/%d] %s (rep %d/%d)",
                        run_num,
                        total,
                        cfg.label,
                        rep + 1,
                        self.n_reps,
                    )
                    result = _run_single(cfg, rep, tmp_dir)
                    rows.append(result.to_dict())
                    _print_result(result)

        df = pd.DataFrame(rows)
        return df

    # ------------------------------------------------------------------
    def generate_report(self, df: "pd.DataFrame") -> None:  # noqa: F821
        """Write CSV and HTML summary reports to :attr:`output_dir`.

        Parameters
        ----------
        df:
            DataFrame returned by :meth:`run`.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for benchmark reporting") from exc

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Raw CSV
        csv_path = self.output_dir / "benchmark_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Raw results saved to %s", csv_path)

        # JSON summary (averages per config key)
        summary_df = (
            df[df["error"] == ""]
            .groupby(
                [
                    "shape",
                    "dtype",
                    "chunk_shape",
                    "shard_shape",
                    "scheduler",
                    "n_threads",
                    "threads_per_worker",
                    "dask_label",
                ],
                dropna=False,
            )
            .agg(
                write_wall_s_mean=("write_wall_s", "mean"),
                write_wall_s_std=("write_wall_s", "std"),
                read_wall_s_mean=("read_wall_s", "mean"),
                read_wall_s_std=("read_wall_s", "std"),
                total_wall_s_mean=("total_wall_s", "mean"),
                write_cpu_efficiency_mean=("write_cpu_efficiency", "mean"),
                read_cpu_efficiency_mean=("read_cpu_efficiency", "mean"),
                peak_memory_mb_max=("peak_memory_mb", "max"),
                n_reps=("repetition", "count"),
            )
            .reset_index()
        )

        summary_csv = self.output_dir / "benchmark_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        logger.info("Summary saved to %s", summary_csv)

        # HTML report
        html_path = self.output_dir / "benchmark_report.html"
        _write_html_report(df, summary_df, html_path)
        logger.info("HTML report saved to %s", html_path)

        # Print best configs
        _print_best_configs(summary_df)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _print_result(result: BenchmarkResult) -> None:
    """Print a one-line summary of a single benchmark result."""
    if result.error:
        print(f"  ERROR: {result.error}")
        return
    print(
        f"  write={result.write_wall_s:.2f}s "
        f"read={result.read_wall_s:.2f}s "
        f"mem={result.peak_memory_mb:.0f}MB "
        f"cpu_eff_w={result.write_cpu_efficiency:.2%} "
        f"cpu_eff_r={result.read_cpu_efficiency:.2%}"
    )


def _print_best_configs(summary_df: "pd.DataFrame") -> None:  # noqa: F821
    """Print the fastest write and read configurations to stdout."""
    if summary_df.empty:
        return
    try:
        fastest_write = summary_df.loc[summary_df["write_wall_s_mean"].idxmin()]
        fastest_read = summary_df.loc[summary_df["read_wall_s_mean"].idxmin()]
        fastest_total = summary_df.loc[summary_df["total_wall_s_mean"].idxmin()]
        print("\n=== Benchmark Summary ===")
        print(
            f"Fastest WRITE : {fastest_write['dask_label']}  "
            f"chunk={fastest_write['chunk_shape']}  "
            f"shard={fastest_write['shard_shape'] or 'none'}  "
            f"{fastest_write['write_wall_s_mean']:.3f}s"
        )
        print(
            f"Fastest READ  : {fastest_read['dask_label']}  "
            f"chunk={fastest_read['chunk_shape']}  "
            f"shard={fastest_read['shard_shape'] or 'none'}  "
            f"{fastest_read['read_wall_s_mean']:.3f}s"
        )
        print(
            f"Fastest TOTAL : {fastest_total['dask_label']}  "
            f"chunk={fastest_total['chunk_shape']}  "
            f"shard={fastest_total['shard_shape'] or 'none'}  "
            f"{fastest_total['total_wall_s_mean']:.3f}s"
        )
    except Exception:  # noqa: BLE001
        pass


def _write_html_report(
    df: "pd.DataFrame",  # noqa: F821
    summary_df: "pd.DataFrame",  # noqa: F821
    html_path: Path,
) -> None:
    """Write an HTML benchmark report to *html_path*."""
    title = "ZarrNii Benchmark Report"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    # Best config callouts
    best_write_row = best_read_row = best_total_row = None
    if not summary_df.empty:
        try:
            best_write_row = summary_df.loc[
                summary_df["write_wall_s_mean"].idxmin()
            ].to_dict()
            best_read_row = summary_df.loc[
                summary_df["read_wall_s_mean"].idxmin()
            ].to_dict()
            best_total_row = summary_df.loc[
                summary_df["total_wall_s_mean"].idxmin()
            ].to_dict()
        except Exception:  # noqa: BLE001
            pass

    def _callout(label: str, row: Optional[Dict]) -> str:
        if row is None:
            return ""
        return (
            f'<div class="callout"><strong>{label}</strong><br>'
            f'Scheduler: {row.get("dask_label", "?")}<br>'
            f'Chunk: {row.get("chunk_shape", "?")}<br>'
            f'Shard: {row.get("shard_shape", "") or "none"}<br>'
            f'Mean time: {row.get(label.lower().replace(" ", "_") + "_wall_s_mean", 0):.3f}s'
            f"</div>"
        )

    callout_write = _callout("write_wall_s", best_write_row)
    callout_read = _callout("read_wall_s", best_read_row)

    summary_html = (
        summary_df.to_html(index=False, classes="table", border=0)
        if not summary_df.empty
        else "<p>No successful runs.</p>"
    )
    raw_html = (
        df.to_html(index=False, classes="table table-sm", border=0)
        if not df.empty
        else "<p>No data.</p>"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>
    body {{ font-family: sans-serif; margin: 2em; background: #f9f9f9; }}
    h1 {{ color: #2c3e50; }}
    h2 {{ color: #34495e; border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
    .callout {{ display:inline-block; background:#eaf4fb; border-left:4px solid #2980b9;
                padding:8px 14px; margin:6px; border-radius:4px; }}
    .table {{ border-collapse: collapse; width: 100%; font-size: 0.85em; }}
    .table th {{ background: #2c3e50; color: white; padding: 6px 10px; text-align:left; }}
    .table td {{ padding: 5px 10px; border-bottom: 1px solid #ddd; }}
    .table tr:hover {{ background: #f1f1f1; }}
    .table-sm td, .table-sm th {{ padding: 3px 8px; font-size:0.8em; }}
    details {{ margin-top:1em; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>Generated: {timestamp}</p>

  <h2>Best Configurations</h2>
  {callout_write}
  {callout_read}

  <h2>Averaged Summary</h2>
  {summary_html}

  <h2>Raw Results</h2>
  <details>
    <summary>Show all {len(df)} rows</summary>
    {raw_html}
  </details>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _parse_chunk(value: str) -> Tuple[int, ...]:
    """Parse ``"32,64,64"`` into ``(32, 64, 64)``."""
    try:
        return tuple(int(x.strip()) for x in value.split(","))
    except ValueError as exc:
        raise ValueError(f"Cannot parse chunk spec {value!r}: {exc}") from exc


def _parse_shard(value: str) -> Optional[Tuple[int, ...]]:
    """Parse a shard spec into a tuple or ``None``.

    Accepted values:

    * ``"none"`` / ``"None"`` → ``None`` (disable sharding)
    * ``"32,64,64"``          → ``(32, 64, 64)``
    """
    if value.strip().lower() == "none":
        return None
    return _parse_chunk(value)


def _parse_dask_config(value: str) -> DaskConfig:
    """Parse a dask config spec string into a :class:`DaskConfig`.

    Accepted formats:

    * ``threads:N``          – threaded scheduler with *N* threads
    * ``distributed:N:M``   – distributed with *N* total threads, *M* per worker
    """
    parts = value.strip().split(":")
    scheduler = parts[0].lower()
    if scheduler not in ("threads", "distributed"):
        raise ValueError(
            f"Unknown scheduler {scheduler!r}; expected 'threads' or 'distributed'"
        )
    n_threads = int(parts[1]) if len(parts) > 1 else 4
    tpw = int(parts[2]) if len(parts) > 2 else 2
    return DaskConfig(scheduler=scheduler, n_threads=n_threads, threads_per_worker=tpw)


def benchmark_cli(argv: Optional[List[str]] = None) -> None:
    """Entry point for the ``zarrnii-benchmark`` command-line tool.

    Parameters
    ----------
    argv:
        Argument list (uses ``sys.argv[1:]`` when ``None``).
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="zarrnii-benchmark",
        description=(
            "Benchmark OME-Zarr write/read performance across different "
            "chunking and Dask configurations."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  zarrnii-benchmark
  zarrnii-benchmark --shape 64 256 256 --chunks 32,32,32 64,64,64
  zarrnii-benchmark --shape 64 256 256 --chunks 32,32,32 --shards none 64,64,64
  zarrnii-benchmark --dask-configs threads:4 distributed:8:2
  zarrnii-benchmark --shape 128 512 512 --chunks 64,64,64 \\
      --shards none 128,128,128 \\
      --dask-configs threads:8 distributed:8:2 \\
      --output-dir ./results --n-reps 5
""",
    )
    parser.add_argument(
        "--shape",
        nargs=3,
        type=int,
        default=[64, 256, 256],
        metavar=("Z", "Y", "X"),
        help="Shape of the synthetic 3-D array (default: 64 256 256)",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help="NumPy dtype for the synthetic array (default: float32)",
    )
    parser.add_argument(
        "--chunks",
        nargs="+",
        default=None,
        metavar="Z,Y,X",
        help=(
            "One or more chunk shapes as comma-separated integers, "
            "e.g. --chunks 32,32,32 64,64,64  (default: half the array shape)"
        ),
    )
    parser.add_argument(
        "--shards",
        nargs="+",
        default=None,
        metavar="Z,Y,X|none",
        help=(
            "One or more shard shapes as comma-separated integers, or 'none' to "
            "disable sharding, e.g. --shards none 128,128,128  (default: none)"
        ),
    )
    parser.add_argument(
        "--dask-configs",
        nargs="+",
        default=["threads:4"],
        metavar="SPEC",
        help=(
            "One or more Dask scheduler specs.  "
            "Format: threads:N or distributed:N:M  "
            "(default: threads:4)"
        ),
    )
    parser.add_argument(
        "--n-reps",
        type=int,
        default=3,
        help="Number of repetitions per configuration (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for CSV and HTML report output (default: current dir)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    shape = tuple(args.shape)
    chunk_shapes = [_parse_chunk(c) for c in args.chunks] if args.chunks else None
    shard_shapes = [_parse_shard(s) for s in args.shards] if args.shards else None
    dask_configs = [_parse_dask_config(s) for s in args.dask_configs]

    suite = BenchmarkSuite(
        shape=shape,
        dtype=args.dtype,
        chunk_shapes=chunk_shapes,
        shard_shapes=shard_shapes,
        dask_configs=dask_configs,
        n_reps=args.n_reps,
        output_dir=args.output_dir,
    )

    print(f"ZarrNii Benchmark")
    print(f"  Shape:       {shape}")
    print(f"  Dtype:       {args.dtype}")
    print(f"  Chunks:      {suite.chunk_shapes}")
    print(f"  Shards:      {suite.shard_shapes}")
    print(f"  Dask configs:{[dc.label for dc in dask_configs]}")
    print(f"  Reps:        {args.n_reps}")
    print(f"  Output dir:  {args.output_dir}")
    print()

    df = suite.run()
    suite.generate_report(df)
    print(f"\nReport written to: {args.output_dir}")
