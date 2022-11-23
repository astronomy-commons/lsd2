"""Tests of file IO (reads and writes)"""


import pandas as pd
import pyarrow as pa
import pytest

import partitioner.io_utils as io
import tests.constants as dc


class TestIOUtils:
    """Test file read/writes"""

    def test_read_empty_filename(self):
        """Empty file name"""
        with pytest.raises(FileNotFoundError):
            io.read_dataframe("", "")

    def test_read_directory(self):
        """Provide directory, not file"""
        with pytest.raises(FileNotFoundError):
            io.read_dataframe(dc.TEST_DATA_DIR, "csv")

    def test_read_bad_fileformat(self):
        """Unsupported file format"""
        with pytest.raises(NotImplementedError):
            io.read_dataframe(dc.TEST_BLANK_CSV, "foo")

    def test_read_empty_file(self):
        """Totally empty file can't be parsed"""
        with pytest.raises(pd.errors.EmptyDataError):
            io.read_dataframe(dc.TEST_BLANK_CSV, "csv")

    def test_read_single_csv(self):
        """Success case - CSV file that exists being read as CSV"""
        result = io.read_dataframe(dc.TEST_SMALL_SKY_CSV, "csv")
        assert len(result) == 131

    def test_read_wrong_fileformat(self):
        """CSV file attempting to be read as parquet"""
        with pytest.raises(pa.lib.ArrowInvalid):
            io.read_dataframe(dc.TEST_SMALL_SKY_CSV, "parquet")
