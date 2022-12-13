"""Main method to enable command line execution"""

import sys

import runner
from arguments import PartitionArguments

if __name__ == "__main__":
    args = PartitionArguments()
    args.from_command_line(sys.argv[1:])
    runner.run(args)
