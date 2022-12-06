"""Test full exection of the un-parallelized runner"""


import os

import data_paths as dc
import file_testing as ft

import partitioner.single_runner as sr
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

    # Check that the legacy metadata file exists, and contains correct object data
    expected_lines = [
        "{",
        '    "cat_name": "small_sky",',
        '    "ra_kw": "ra",',
        '    "dec_kw": "dec",',
        '    "id_kw": "id",',
        '    "n_sources": 131,',
        '    "pix_threshold": 1000000,',
        r'    "urls": \[',
        r'        ".*/tests/partitioner/data/small_sky/catalog.csv"',
        "    ],",
        '    "hips": {',
        r'        "0": \[',
        "            11",
        "        ]",
        "    }",
        "}",
    ]
    metadata_filename = os.path.join(dc.TEST_TMP_DIR, "small_sky_meta.json")
    ft.assert_text_file_matches(expected_lines, metadata_filename)

    # TODO - test that all other files are written to output directory
