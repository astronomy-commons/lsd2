"""Test full exection of the un-parallelized runner"""


import os
import tempfile

import data_paths as dc
import file_testing as ft

import partitioner.single_runner as sr
from partitioner.arguments import PartitionArguments


def test_small_sky():
    """Test loading the small sky catalog and partitioning each object into the same large bucket"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = PartitionArguments()
        args.from_params(
            catalog_name="small_sky",
            input_path=dc.TEST_SMALL_SKY_PARTS_DATA_DIR,
            input_format="csv",
            output_path=tmp_dir,
            highest_healpix_order=1,
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
            r'        ".*/small_sky_parts/catalog.*.csv"',
            r'        ".*/small_sky_parts/catalog.*.csv"',
            r'        ".*/small_sky_parts/catalog.*.csv"',
            r'        ".*/small_sky_parts/catalog.*.csv"',
            r'        ".*/small_sky_parts/catalog.*.csv"',
            "    ],",
            '    "hips": {',
            r'        "0": \[',
            "            11",
            "        ]",
            "    }",
            "}",
        ]
        metadata_filename = os.path.join(args.catalog_path, "small_sky_meta.json")
        ft.assert_text_file_matches(expected_lines, metadata_filename)

        # Check that the catalog parquet file exists and contains correct object IDs
        output_file = os.path.join(
            args.catalog_path, "Norder0/Npix11", "catalog.parquet"
        )

        expected_ids = [*range(700, 831)]
        ft.assert_parquet_file_ids(output_file, "id", expected_ids)
