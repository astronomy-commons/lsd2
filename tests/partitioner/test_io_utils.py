"""Tests of file IO (reads and writes)"""

import data_paths as dc
import pandas as pd
import pyarrow as pa
import pytest

import partitioner.io_utils as io


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
