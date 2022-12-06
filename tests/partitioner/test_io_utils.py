"""Tests of file IO (reads and writes)"""

import os

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

    json_filename = os.path.join(dc.TEST_TMP_DIR, "dictionary.json")
    io.write_json_file(dictionary, json_filename)
    ft.assert_text_file_matches(expected_lines, json_filename)


def test_write_legacy_metadata_file():
    """Test that we can write out the older version of the partiion metadata"""
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

    ft.assert_text_file_matches(expected_lines, metadata_filename)
