"""Tests of file IO (reads and writes)"""


import os

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


def test_write_legacy_metadata_file():
    """Test that we can write out the older version of the partiion metadata"""
    expected_lines = [
        "{\n",
        '    "cat_name": "small_sky",\n',
        '    "ra_kw": "ra",\n',
        '    "dec_kw": "dec",\n',
        '    "id_kw": "id",\n',
        '    "n_sources": 131,\n',
        '    "pix_threshold": 1000000,\n',
        '    "urls": [\n',
        '        "/home/delucchi/git/lsd2/tests/partitioner/data/small_sky/catalog.csv"\n',
        "    ],\n",
        '    "hips": {\n',
        '        "0": [\n',
        "            11\n",
        "        ]\n",
        "    }\n",
        "}\n",
    ]
    expected = "".join(expected_lines)

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
    assert os.path.exists(metadata_filename)
    metadata_file = open(
        metadata_filename,
        "r",
        encoding="utf-8",
    )
    contents = metadata_file.read()

    assert contents == expected

    metadata_file.close()
