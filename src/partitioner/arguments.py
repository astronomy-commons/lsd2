import argparse
import os


class PartitionArguments:
    def from_command_line(self):
        """Parse arguments from the command line"""

        parser = argparse.ArgumentParser(
            prog="LSD2 Partitioner",
            description="Instantiate a partitioned catalog from unpartitioned sources",
        )
        parser.add_argument(
            "catalog_name",
            help="short name for the catalog that will be used for the output directory",
        )
        parser.add_argument(
            "input_path", help="path prefix for unpartitioned input files"
        )
        parser.add_argument(
            "output_path", help="path prefix for partitioned output and metadata files"
        )
        args = parser.parse_args()

        self.catalog_name = args.catalog_name
        self.input_path = args.input_path
        self.output_path = args.output_path

        self.check_arguments()
        self.check_paths()

    def check_arguments(self):
        """Check existence and consistency of argument values"""
        if self.catalog_name == "":
            raise ValueError("catalog_name is required")

    def check_paths(self):
        """Check existence and permissions on provided path arguments"""
        # TODO: handle non-posix files/paths
        if self.input_path == "":
            raise ValueError("input_path is required")
        if not os.path.exists(self.input_path):
            raise ValueError("input_path not found on local storage")

        if self.output_path == "":
            raise ValueError("output_path is required")
        if not os.path.exists(self.output_path):
            raise ValueError("output_path not found on local storage")
