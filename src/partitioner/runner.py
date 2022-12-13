"""Wrapper around partitioner execution environments."""

import partitioner.dask_runner as dask_runner
import partitioner.single_runner as single_runner
from partitioner.arguments import PartitionArguments


def run(args: PartitionArguments):
    """Pick a runtime environment and run the partitioner"""
    if not args:
        raise ValueError("partitioning arguments are required")
    if not isinstance(args, PartitionArguments):
        raise ValueError("args must be type PartitionArguments")

    match args.runtime:
        case "single":
            single_runner.run(args)
            return
        case "dask":
            dask_runner.run(args)
            return
        case _:
            raise NotImplementedError(f"unknown runtime ({args.runtime})")
