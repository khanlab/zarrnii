"""Tests for zarrnii.dask_utils."""

import dask
import pytest

from zarrnii import get_dask_client


def test_threads_scheduler_yields_none():
    """threads scheduler yields None and sets dask config."""
    with get_dask_client("threads", threads=4) as client:
        assert client is None
        assert dask.config.get("scheduler") == "threads"
        assert dask.config.get("num_workers") == 4


def test_threads_scheduler_respects_thread_count():
    """threads scheduler passes num_workers through correctly."""
    with get_dask_client("threads", threads=8) as client:
        assert client is None
        assert dask.config.get("num_workers") == 8


dask_distributed = pytest.importorskip(
    "dask.distributed", reason="dask.distributed not installed"
)


def test_distributed_scheduler_yields_client():
    """distributed scheduler yields a Client and closes it afterwards."""
    from dask.distributed import Client

    with get_dask_client("distributed", threads=4, threads_per_worker=2) as client:
        assert isinstance(client, Client)
        assert client.status == "running"
