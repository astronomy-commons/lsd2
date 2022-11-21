from arguments import PartitionArguments
import sys

if __name__ == "__main__":
    args = PartitionArguments()
    args.from_command_line(sys.argv[1:])
    # TODO: everything else!
