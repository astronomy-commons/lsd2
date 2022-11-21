import argparse
import os

"""
input		catalog name
input		column mappings (ra_kw, dec_kw, ra_err_kw, dec_err_kw, id_kw)
input		debug - input files
input		input file format
input		input files
output	    output directory
output	    overwrite flag
output	    debug - keep order/pixel column in output
output	    debug - verbose logging
output	    debug - write debug metadata file
histogram	lowest (/highest?) healpix order
histogram	per-pixel threshold
execution	dask - client, distributed cluster, num workers, etc
execution	dask - temp directory
execution	dask or serial
execution	debug - stats only
"""


class PartitionArguments:
    def from_command_line(self, cl_args):
        """Parse arguments from the command line"""

        parser = argparse.ArgumentParser(
            prog="LSD2 Partitioner",
            description="Instantiate a partitioned catalog from unpartitioned sources",
        )

        """
        ===========           INPUT ARGUMENTS           ===========
        catalog name
        input path
        debug input files
        input file format
        column mappings
                ra
                dec
                ra_err
                dec_err
                id
        """
        group = parser.add_argument_group("INPUT")
        group.add_argument(
            "-c",
            "--catalog_name",
            help="short name for the catalog that will be used for the output directory",
            default=None,
            type=str,
        )
        group.add_argument(
            "-i",
            "--input_path",
            help="path prefix for unpartitioned input files",
            default=None,
            type=str,
        )
        group = parser.add_argument_group(
            "INPUT COLUMNS",
            "Column names in the input source that correspond to spatial attributes used in partitioning",
        )
        group.add_argument(
            "-ra",
            "--ra_column",
            help="column name for the ra (rate of ascension)",
            default=None,
            type=str,
        )
        group.add_argument(
            "-dec",
            "--dec_column",
            help="column name for the dec (declination)",
            default=None,
            type=str,
        )
        group.add_argument(
            "-rae",
            "--ra_error_column",
            help="column name for the error in the ra (rate of ascension)",
            default=None,
            type=str,
        )
        group.add_argument(
            "-dece",
            "--dec_error_column",
            help="column name for the error in the dec (declination)",
            default=None,
            type=str,
        )
        group.add_argument(
            "-id",
            "--id_column",
            help="column name for the object id",
            default=None,
            type=str,
        )
        """
        ===========           OUTPUT ARGUMENTS          ===========
        output directory
        overwrite flag
        debug - keep order/pixel column in output
        debug - verbose logging
        debug - write debug metadata file
        """
        group = parser.add_argument_group("OUTPUT")
        group.add_argument(
            "-o",
            "--output_path",
            help="path prefix for partitioned output and metadata files",
            default=None,
            type=str,
        )
        group.add_argument(
            "--overwrite",
            help="if set, the any existing catalog data will be overwritten",
            action="store_true",
        )
        group.add_argument(
            "--no_overwrite",
            help="if set, the pipeline will exit if existing output is found",
            dest="overwrite",
            action="store_false",
        )

        """
        ===========           STATS ARGUMENTS           ===========
        lowest (/highest?) healpix order
        per-pixel threshold
        debug - stats only
        """
        group = parser.add_argument_group("STATS")
        group.add_argument(
            "-ho",
            "--healpix_order",
            help="the most dense healpix order (7-10 is a good range for this)",
            default=None,
            type=int,
        )
        group.add_argument(
            "-pt",
            "--pixel_threshold",
            help="maximum objects allowed in a single pixel",
            default=None,
            type=int,
        )
        group.add_argument(
            "--debug_stats_only",
            help="DEBUGGING FLAG - if set, the pipeline will only fetch statistics about the origin data and will not generate partitioned output",
            action="store_true",
        )
        group.add_argument(
            "--no_debug_stats_only",
            help="DEBUGGING FLAG - if set, the pipeline will generate partitioned output",
            dest="debug_stats_only",
            action="store_false",
        )
        """
        ===========         EXECUTION ARGUMENTS         ===========
        dask - client, distributed cluster, num workers, etc
        dask - temp directory
        dask or serial
        """
        group = parser.add_argument_group("EXECUTION")
        group.add_argument(
            "-dt",
            "--dask_tmp",
            help="directory for storing temporary files generated by dask engine",
            default=None,
            type=str,
        )

        args = parser.parse_args(cl_args)

        self.catalog_name = args.catalog_name
        self.input_path = args.input_path
        self.output_path = args.output_path

        self.check_arguments()
        self.check_paths()

    def from_params(self, catalog_name="", input_path="", output_path=""):
        self.catalog_name = catalog_name
        self.input_path = input_path
        self.output_path = output_path

        self.check_arguments()
        self.check_paths()

    def check_arguments(self):
        """Check existence and consistency of argument values"""
        if self.catalog_name == "":
            raise ValueError("catalog_name is required")

    def check_paths(self):
        """Check existence and permissions on provided path arguments"""
        # TODO: handle non-posix files/paths
        if not self.input_path:
            raise ValueError("input_path is required")
        if not os.path.exists(self.input_path):
            raise ValueError("input_path not found on local storage")

        if not self.output_path:
            raise ValueError("output_path is required")
        if not os.path.exists(self.output_path):
            raise ValueError("output_path not found on local storage")
