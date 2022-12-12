"""Tests of file IO (reads and writes)"""


import os
import tempfile

import data_paths as dc
import file_testing as ft
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import partitioner.io_utils as io
from partitioner.arguments import PartitionArguments


def test_read_empty_filename():
    """Empty file name"""
    with pytest.raises(FileNotFoundError):
        io.read_dataframe("", "")


def test_read_directory():
    """Provide directory, not file"""
    with pytest.raises(FileNotFoundError):
        io.read_dataframe(dc.TEST_DATA_DIR, "csv")


def test_read_bad_fileformat():
    """Unsupported file format"""
    with pytest.raises(NotImplementedError):
        io.read_dataframe(dc.TEST_BLANK_CSV, "foo")


def test_read_empty_file():
    """Totally empty file can't be parsed"""
    with pytest.raises(pd.errors.EmptyDataError):
        io.read_dataframe(dc.TEST_BLANK_CSV, "csv")


def test_read_single_csv():
    """Success case - CSV file that exists being read as CSV"""
    result = io.read_dataframe(dc.TEST_SMALL_SKY_CSV, "csv")
    assert len(result) == 131


def test_read_wrong_fileformat():
    """CSV file attempting to be read as parquet"""
    with pytest.raises(pa.lib.ArrowInvalid):
        io.read_dataframe(dc.TEST_SMALL_SKY_CSV, "parquet")


def test_write_json_file():
    """Test of arbitrary json dictionary with strings and numbers"""

    expected_lines = [
        "{\n",
        '    "first_english": "a",',
        '    "first_greek": "alpha",',
        '    "first_number": 1,',
        r'    "first_five_fib": \[',
        "        1,",
        "        1,",
        "        2,",
        "        3,",
        "        5",
        "    ]",
        "}",
    ]

    dictionary = {}
    dictionary["first_english"] = "a"
    dictionary["first_greek"] = "alpha"
    dictionary["first_number"] = 1
    dictionary["first_five_fib"] = [1, 1, 2, 3, 5]

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_filename = os.path.join(tmp_dir, "dictionary.json")
        io.write_json_file(dictionary, json_filename)
        ft.assert_text_file_matches(expected_lines, json_filename)


def test_write_catalog_info():
    """Test that we accurately write out partition metadata"""
    expected_lines = [
        "{",
        '    "cat_name": "small_sky",',
        r'    "version": "\d+",',  # version matches digits
        r'    "generation_date": "\d+",',  # date matches date format
        '    "ra_kw": "ra",',
        '    "dec_kw": "dec",',
        '    "id_kw": "id",',
        '    "total_objects": 131,',
        '    "pixel_threshold": 1000000',
        "}",
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        args = PartitionArguments()
        args.from_params(
            catalog_name="small_sky",
            input_path=dc.TEST_SMALL_SKY_DATA_DIR,
            input_format="csv",
            output_path=tmp_dir,
            highest_healpix_order=0,
            ra_column="ra",
            dec_column="dec",
        )
        initial_histogram = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 131])

        io.write_catalog_info(args, initial_histogram)
        metadata_filename = os.path.join(tmp_dir, "small_sky", "catalog_info.json")
        ft.assert_text_file_matches(expected_lines, metadata_filename)


def test_write_partition_info():
    """Test that we accurately write out the individual partition stats"""
    expected_lines = [
        "order,pixel,num_objects",
        "0,11,131",
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = PartitionArguments()
        args.from_params(
            catalog_name="small_sky",
            input_path=dc.TEST_SMALL_SKY_DATA_DIR,
            input_format="csv",
            output_path=tmp_dir,
            highest_healpix_order=0,
            ra_column="ra",
            dec_column="dec",
        )
        pixel_map = np.full(12, None)
        pixel_map[11] = (0, 11, 131)
        io.write_partition_info(args, pixel_map)
        metadata_filename = os.path.join(tmp_dir, "small_sky", "partition_info.csv")
        ft.assert_text_file_matches(expected_lines, metadata_filename)


def test_write_legacy_metadata_file():
    """Test that we can write out the older version of the partition metadata"""
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
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = PartitionArguments()
        args.from_params(
            catalog_name="small_sky",
            input_path=dc.TEST_SMALL_SKY_DATA_DIR,
            input_format="csv",
            output_path=tmp_dir,
            highest_healpix_order=0,
            ra_column="ra",
            dec_column="dec",
        )
        initial_histogram = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 131])
        pixel_map = np.full(12, None)
        pixel_map[11] = (0, 11, 131)

        io.write_legacy_metadata(args, initial_histogram, pixel_map)

        metadata_filename = os.path.join(tmp_dir, "small_sky", "small_sky_meta.json")

        ft.assert_text_file_matches(expected_lines, metadata_filename)


def test_concatenate_parquet_files_1input():
    """Test that concatenating a single parquet file gets the same IDs"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = os.path.join(tmp_dir, "combined.parquet")

        rows_written = io.concatenate_parquet_files(
            [dc.TEST_PARQUET_SHARDS_PART0], output_file
        )

        assert rows_written == 7

        expected_ids = [780, 787, 792, 794, 795, 797, 801]
        ft.assert_parquet_file_ids(output_file, "id", expected_ids)


def test_concatenate_parquet_files_2input():
    """Test that concatenating two parquet files gets both sets of IDs"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = os.path.join(tmp_dir, "combined.parquet")

        rows_written = io.concatenate_parquet_files(
            [dc.TEST_PARQUET_SHARDS_PART2, dc.TEST_PARQUET_SHARDS_PART0],
            output_file_name=output_file,
        )

        assert rows_written == 15
        # fmt: off
        expected_ids = [758, 760, 766, 768, 771, 772,
                        775, 776, 780, 787, 792, 794,
                        795, 797, 801]
        # fmt: on
        ft.assert_parquet_file_ids(output_file, "id", expected_ids)


def test_concatenate_parquet_files_2input_sorting():
    """Test that concatenating two parquet files gets both sets of IDs, with sorting"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = os.path.join(tmp_dir, "combined.parquet")

        rows_written = io.concatenate_parquet_files(
            [dc.TEST_PARQUET_SHARDS_PART0, dc.TEST_PARQUET_SHARDS_PART2],
            output_file_name=output_file,
            sorting="id",
        )

        assert rows_written == 15

        # fmt:off
        expected_ids = [758, 760, 766, 768, 771, 772, 775, 776, 780, 787,
                        792, 794, 795, 797, 801]
        # fmt:on
        ft.assert_parquet_file_ids(output_file, "id", expected_ids)


def test_concatenate_parquet_files_2input_sorting_desc():
    """
    Test that concatenating two parquet files gets both sets of IDs,
    sorting the IDs in descending order (largest to smallest)
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = os.path.join(tmp_dir, "combined.parquet")

        rows_written = io.concatenate_parquet_files(
            [dc.TEST_PARQUET_SHARDS_PART0, dc.TEST_PARQUET_SHARDS_PART2],
            output_file_name=output_file,
            sorting=[("id", "descending")],
        )

        assert rows_written == 15
        # fmt:off
        expected_ids = [801, 797, 795, 794, 792, 787, 780, 776, 775, 772,
                        771, 768, 766, 760, 758]
        # fmt:on
        ft.assert_parquet_file_ids(output_file, "id", expected_ids)


def test_concatenate_parquet_files_directory_input():
    """Test that concatenating entire directory results in all IDs"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = os.path.join(tmp_dir, "combined.parquet")

        rows_written = io.concatenate_parquet_files(
            [dc.TEST_PARQUET_SHARDS_DATA_DIR],
            output_file_name=output_file,
            sorting="id",
        )

        assert rows_written == 42

        # fmt:off
        expected_ids = [703, 707, 716, 718, 723, 729, 730, 733, 734, 735, 736,
                        738, 739, 747, 748, 750, 758, 760, 766, 768, 771, 772,
                        775, 776, 780, 787, 792, 794, 795, 797, 801, 804, 807,
                        810, 811, 815, 816, 817, 818, 822, 826, 830]
        # fmt:on
        ft.assert_parquet_file_ids(output_file, "id", expected_ids)
