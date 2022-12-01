"""Tests of file IO (reads and writes)"""


import os
import re

import data_paths as dc
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


def assert_file_matches(expected_lines, file_name):
    """
    Convenience method to read a text file
    and compare the contents, line for line,
    against regular expressions.

    It can be easier to see differences in indivudual lines
    when file contents grow to be large.
    """
    assert os.path.exists(file_name)
    metadata_file = open(
        file_name,
        "r",
        encoding="utf-8",
    )

    contents = metadata_file.readlines()

    assert len(expected_lines) == len(contents)
    for i, expected in enumerate(expected_lines):
        assert re.match(expected, contents[i])

    metadata_file.close()


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

    json_filename = os.path.join(dc.TEST_TMP_DIR, "dictionary.json")
    io.write_json_file(dictionary, json_filename)
    assert_file_matches(expected_lines, json_filename)


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
    initial_histogram = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 131])

    io.write_catalog_info(args, initial_histogram)
    metadata_filename = os.path.join(dc.TEST_TMP_DIR, "catalog_info.json")
    assert_file_matches(expected_lines, metadata_filename)


def test_write_partition_info():
    """Test that we accurately write out the individual partition stats"""
    expected_lines = [
        "order,pixel,num_objects",
        "0,11,131",
    ]

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
    pixel_map = np.full(12, None)
    pixel_map[11] = (0, 11, 131)
    io.write_partition_info(args, pixel_map)
    metadata_filename = os.path.join(dc.TEST_TMP_DIR, "small_sky", "partition_info.csv")
    assert_file_matches(expected_lines, metadata_filename)


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
        '        "/home/delucchi/git/lsd2/tests/partitioner/data/small_sky/catalog.csv"',
        "    ],",
        '    "hips": {',
        r'        "0": \[',
        "            11",
        "        ]",
        "    }",
        "}",
    ]

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
    initial_histogram = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 131])
    pixel_map = np.full(12, None)
    pixel_map[11] = (0, 11, 131)

    io.write_legacy_metadata(args, initial_histogram, pixel_map)

    metadata_filename = os.path.join(dc.TEST_TMP_DIR, "small_sky_meta.json")

    assert_file_matches(expected_lines, metadata_filename)
