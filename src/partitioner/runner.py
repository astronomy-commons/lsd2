"""Wrapper around partitioner execution environments."""

import logging
import time

import partitioner.dask_runner as dask_runner
import partitioner.single_runner as single_runner
from partitioner.arguments import PartitionArguments

LOGGER = logging.getLogger(__name__)


def run(args: PartitionArguments):
    """Pick a runtime environment and run the partitioner"""

    start_time = time.perf_counter()
    if not args:
        raise ValueError("partitioning arguments are required")
    if not isinstance(args, PartitionArguments):
        raise ValueError("args must be type PartitionArguments")

    if args.runtime == "single":
        single_runner.run(args)
    elif args.runtime == "dask":
        dask_runner.run(args)
    else:
        raise NotImplementedError(f"unknown runtime ({args.runtime})")

    end_time = time.perf_counter()
    LOGGER.info("Runner finished in %i seconds", end_time - start_time)
