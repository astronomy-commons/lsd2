"""Tests of file IO (reads and writes)"""

import sys
import unittest

# This doesn't feel good, but I'm tired of fighting.
sys.path.insert(0, "../")

import data.constants as dc
import pandas as pd
import pyarrow as pa

import partitioner.io_utils as io


class TestIOUtils(unittest.TestCase):
    """Test file read/writes"""

    def test_read_empty_filename(self):
        """Empty file name"""
        with self.assertRaises(FileNotFoundError):
            io.read_dataframe("", "")

    def test_read_directory(self):
        """Provide directory, not file"""
        with self.assertRaises(FileNotFoundError):
            io.read_dataframe(dc.TEST_DATA_DIR, "csv")

    def test_read_bad_fileformat(self):
        """Unsupported file format"""
        with self.assertRaises(NotImplementedError):
            io.read_dataframe(dc.TEST_BLANK_CSV, "foo")

    def test_read_empty_file(self):
        """Totally empty file can't be parsed"""
        with self.assertRaises(pd.errors.EmptyDataError):
            io.read_dataframe(dc.TEST_BLANK_CSV, "csv")

    def test_read_single_csv(self):
        """Success case - CSV file that exists being read as CSV"""
        result = io.read_dataframe(dc.TEST_SMALL_SKY_CSV, "csv")
        self.assertEqual(len(result), 131)

    def test_read_wrong_fileformat(self):
        """CSV file attempting to be read as parquet"""
        with self.assertRaises(pa.lib.ArrowInvalid):
            io.read_dataframe(dc.TEST_SMALL_SKY_CSV, "parquet")
