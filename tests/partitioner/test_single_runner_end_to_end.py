"""Test full exection of the un-parallelized runner"""


import tempfile

import data_paths as dc

import partitioner.single_runner as sr
from partitioner.arguments import PartitionArguments


def test_small_sky():
    """Test loading the small sky catalog and partitioning each object into the same large bucket"""
    tmp_dir = tempfile.mkdtemp()
    print(tmp_dir)
    args = PartitionArguments()
    args.from_params(
        catalog_name="small_sky",
        input_path=dc.TEST_SMALL_SKY_PARTS_DATA_DIR,
        input_format="csv",
        output_path=tmp_dir,
        highest_healpix_order=0,
        ra_column="ra",
        dec_column="dec",
    )

    print(args.input_paths)

    sr.run(args)

    # TODO - test that all files are written to output directory
