"""Main method to enable command line execution"""

import sys

import dask_runner
import single_runner
from arguments import PartitionArguments


def run():
    """Pick a runtime environment and run the partitioner"""
    match args.runtime:
        case "single":
            single_runner.run(args)
            return
        case "dask":
            dask_runner.run(args)
            return


if __name__ == "__main__":
    args = PartitionArguments()
    args.from_command_line(sys.argv[1:])
    run()
