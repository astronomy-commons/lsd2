"""Test rull exection of the un-parallelized runner"""


import partitioner.single_runner as sr
import tests.data_paths as dc
from partitioner.arguments import PartitionArguments


def test_small_sky():
    """Test loading the small sky catalog and partitioning each object into the same large bucket"""
    args = PartitionArguments()
    args.from_params(
        catalog_name="small_sky",
        input_path=dc.TEST_SMALL_SKY_DATA_DIR,
        input_format="csv",
        output_path=dc.TEST_TMP_DIR,
        highest_healpix_order=0,
        ra_column="ra",
        dec_column="dec",
    )

    sr.run(args)

    # TODO - test that all files are written to output directory
