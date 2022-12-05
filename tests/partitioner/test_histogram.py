"""Tests of histogram calculations"""

import data_paths as dc
import numpy.testing as npt
import pytest

import partitioner.histogram as hist


def test_small_sky_same_pixel():
    """Test loading the small sky catalog and partitioning each object into the same large bucket"""
    result = hist.generate_partial_histogram(
        file_path=dc.TEST_SMALL_SKY_CSV,
        highest_order=0,
        file_format="csv",
        ra_column="ra",
        dec_column="dec",
    )

    assert len(result) == 12

    expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 131]
    npt.assert_array_equal(result, expected)
    assert (result == expected).all()


def test_column_names_error():
    """Test loading file with non-default column names (without specifying column names)"""
    with pytest.raises(ValueError):
        hist.generate_partial_histogram(
            file_path=dc.TEST_FORMATS_HEADERS_CSV,
            highest_order=0,
            file_format="csv",
            ra_column="ra",
            dec_column="dec",
        )


def test_column_names():
    """Test loading file with non-default column names"""
    result = hist.generate_partial_histogram(
        file_path=dc.TEST_FORMATS_HEADERS_CSV,
        highest_order=0,
        file_format="csv",
        ra_column="ra_mean",
        dec_column="dec_mean",
    )

    assert len(result) == 12

    expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8]
    npt.assert_array_equal(result, expected)
    assert (result == expected).all()
