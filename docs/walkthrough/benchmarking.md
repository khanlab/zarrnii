# Benchmarking Chunking and Dask Configuration

ZarrNii ships with a built-in benchmarking tool that helps you choose the
optimal **chunk size**, **shard size**, and **Dask scheduler setup** for your
specific dataset and hardware.  The tool measures:

- **Wall-clock time** for writing an OME-Zarr store and reading it back.
- **Peak RSS memory** consumed during the run.
- **CPU efficiency** – how well your cores are utilised relative to wall time.

Results are saved as a CSV file and an HTML report that is easy to open in
any browser.

---

## Quick Start

### Command-line interface

The simplest way to run a benchmark is via the `zarrnii-benchmark` command:

```bash
zarrnii-benchmark
```

This runs a small default sweep (shape `64×256×256`, a couple of chunk sizes,
threaded scheduler with 4 threads, 3 repetitions) and writes the results to
the current directory.

### Customise the sweep

```bash
zarrnii-benchmark \
    --shape 128 512 512 \
    --dtype float32 \
    --chunks 32,32,32 64,64,64 128,128,128 \
    --dask-configs threads:4 threads:8 distributed:8:2 \
    --n-reps 5 \
    --output-dir ./bench_results
```

| Flag | Description | Default |
|------|-------------|---------|
| `--shape Z Y X` | Spatial dimensions of the synthetic array | `64 256 256` |
| `--dtype` | NumPy dtype (e.g. `float32`, `uint16`) | `float32` |
| `--chunks` | One or more chunk specs as `Z,Y,X` | half the array shape |
| `--dask-configs` | One or more scheduler specs (see below) | `threads:4` |
| `--n-reps` | Repetitions per configuration | `3` |
| `--output-dir` | Directory for CSV/HTML output | `.` (current dir) |
| `-v` / `--verbose` | Enable debug logging | off |

### Dask configuration specs

The `--dask-configs` flag accepts one or more space-separated spec strings:

| Spec | Scheduler | Example |
|------|-----------|---------|
| `threads:N` | Threaded scheduler with *N* workers | `threads:8` |
| `distributed:N:M` | Distributed (LocalCluster) with *N* total threads, *M* per worker | `distributed:8:2` |

The number of distributed workers is derived as `⌊N / M⌋`.

---

## Python API

You can also drive the benchmarks programmatically:

```python
from zarrnii.benchmark import BenchmarkSuite, DaskConfig

suite = BenchmarkSuite(
    shape=(64, 256, 256),
    dtype="float32",
    chunk_shapes=[(32, 32, 32), (64, 64, 64)],
    shard_shapes=[None],                   # None = no sharding
    dask_configs=[
        DaskConfig(scheduler="threads", n_threads=4),
        DaskConfig(scheduler="distributed", n_threads=8, threads_per_worker=2),
    ],
    n_reps=3,
    output_dir="./bench_results",
)

# Run all configurations – returns a pandas DataFrame
df = suite.run()

# Write CSV + HTML report
suite.generate_report(df)
```

The returned `DataFrame` has one row per *(configuration, repetition)* and
includes the columns below.

| Column | Description |
|--------|-------------|
| `shape` | Array shape |
| `dtype` | Data type |
| `chunk_shape` | Zarr chunk shape used |
| `shard_shape` | Zarr shard shape (empty = none) |
| `scheduler` | `threads` or `distributed` |
| `n_threads` | Total threads/cores |
| `threads_per_worker` | Threads per distributed worker |
| `dask_label` | Human-readable Dask config label |
| `repetition` | 0-based repetition index |
| `write_wall_s` | Wall-clock seconds for OME-Zarr write |
| `read_wall_s` | Wall-clock seconds for OME-Zarr read |
| `total_wall_s` | Sum of write + read wall time |
| `write_cpu_s` | User+system CPU seconds during write |
| `read_cpu_s` | User+system CPU seconds during read |
| `write_cpu_efficiency` | `cpu_s / (wall_s × n_threads)` during write |
| `read_cpu_efficiency` | `cpu_s / (wall_s × n_threads)` during read |
| `peak_memory_mb` | Peak RSS memory (MB) during the run |
| `error` | Non-empty when the run raised an exception |

---

## Output files

After calling `generate_report()` (or finishing a CLI run) the following files
are created in `output_dir`:

| File | Description |
|------|-------------|
| `benchmark_results.csv` | Full raw results (one row per run) |
| `benchmark_summary.csv` | Per-configuration averages |
| `benchmark_report.html` | Self-contained HTML report with best-config callouts |

Open `benchmark_report.html` in your browser for a quick visual overview.

---

## Interpreting results

### Chunk size

- Smaller chunks improve random-access read performance and reduce peak
  memory per task, but increase metadata overhead and scheduler overhead
  for sequential access patterns.
- Larger chunks are better for sequential reads (e.g. whole-volume
  processing) but require more memory per task.
- A common starting point for 3-D biomedical images is **64×64×64** voxels.

### Dask scheduler

| Scheduler | Best for |
|-----------|----------|
| `threads` | Single-machine workloads; low overhead; shared memory |
| `distributed` | Multi-core machines; enables the Dask dashboard and detailed task graphs |

- For **I/O-bound** operations (reading/writing Zarr) more workers rarely
  helps beyond saturating the storage bandwidth; start with `threads:N`
  where *N* equals the number of physical cores.
- **CPU efficiency** close to `1.0` means all cores are busy throughout the
  operation.  Low values suggest I/O bottlenecks or excessive scheduler
  overhead.

### Memory

Peak memory scales roughly with `chunk_size × dtype_bytes × n_tasks_in_flight`.
If you are hitting memory limits, try smaller chunks or fewer workers.

---

## Dask Distributed dashboard

When `scheduler="distributed"` is used, a local Dask dashboard is started on
port **8788** (configurable via `dask_utils.get_dask_client`).  Open
`http://localhost:8788` in your browser while the benchmark is running to
observe task scheduling in real time.

The HTML report captures summary statistics after the run so you do not need
to monitor the dashboard live.

---

## API reference

::: zarrnii.benchmark
    handler: python
    options:
      show_root_heading: false
      show_object_full_path: true
      show_category_heading: true
      members_order: source
      merge_init_into_class: true
      show_docstring_description: true
      docstring_section_style: list
      show_source: true
      filters:
        - "!^_"
