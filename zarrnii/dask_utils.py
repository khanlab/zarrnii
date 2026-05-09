"""Dask scheduler setup utilities."""

import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def get_dask_client(scheduler, threads, threads_per_worker=2):
    """Context manager to set up the Dask scheduler.

    Parameters
    ----------
    scheduler : str
        Scheduler type: ``"threads"`` for the built-in threaded scheduler, or
        ``"distributed"`` for a ``dask.distributed.LocalCluster``.
    threads : int
        Total number of threads/cores available (from ``snakemake.threads``).
    threads_per_worker : int, optional
        Number of threads per worker when using the distributed scheduler.
        Ignored when *scheduler* is ``"threads"``.  Default is 2.

    Yields
    ------
    client : dask.distributed.Client or None
        The distributed ``Client`` when *scheduler* is ``"distributed"``,
        otherwise ``None``.
    """
    if scheduler == "distributed":
        from dask.distributed import Client, LocalCluster

        n_workers = max(1, int(threads // threads_per_worker))
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit="auto",
            dashboard_address=":8788",
        )
        client = Client(cluster)
        logger.info("Dask dashboard: %s", cluster.dashboard_link)
        try:
            yield client
        finally:
            client.close()
            cluster.close()
    else:
        import dask

        dask.config.set(scheduler="threads", num_workers=threads)
        yield None
