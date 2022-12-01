"""Main method to enable command line execution"""
import sys

from arguments import PartitionArguments

if __name__ == "__main__":
    args = PartitionArguments()
    args.from_command_line(sys.argv[1:])
    # TODO: everything else!
