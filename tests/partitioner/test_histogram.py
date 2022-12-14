"""Tests of histogram calculations"""

import data_paths as dc
import healpy as hp
import numpy as np
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


def test_alignment_wrong_size():
    """Check that the method raises error when the input histogram is not the expected size."""
    initial_histogram = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 131])
    with pytest.raises(ValueError):
        hist.generate_alignment(initial_histogram, 0, 250)

def test_alignment_exceeds_threshold_order0():
    """Check that the method raises error when some pixel exceeds the threshold."""
    initial_histogram = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 131])
    with pytest.raises(ValueError):
        hist.generate_alignment(initial_histogram, 0, 20)

def test_alignment_exceeds_threshold_order2():
    """Check that the method raises error when some pixel exceeds the threshold."""
    initial_histogram = hist.empty_histogram(2)
    filled_pixels = [4, 11, 14, 13, 5, 7, 8, 9, 11, 23, 4, 4, 17, 0, 1, 0]
    initial_histogram[176:] = filled_pixels[:]
    with pytest.raises(ValueError):
        hist.generate_alignment(initial_histogram, 2, 20)

def test_alignment_small_sky_order0():
    """Create alignment from small sky's distribution at order 0"""
    initial_histogram = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 131])
    result = hist.generate_alignment(initial_histogram, 0, 250)

    expected = np.full(12, None)
    expected[11] = (0, 11, 131)

    npt.assert_array_equal(result, expected)


def test_alignment_small_sky_order1():
    """Create alignment from small sky's distribution at order 1"""
    initial_histogram = hist.empty_histogram(1)
    filled_pixels = [42, 29, 42, 18]
    initial_histogram[44:] = filled_pixels[:]
    result = hist.generate_alignment(initial_histogram, 1, 250)

    expected = np.full(48, None)
    expected[44:] = [(0, 11, 131), (0, 11, 131), (0, 11, 131), (0, 11, 131)]

    npt.assert_array_equal(result, expected)


def test_alignment_small_sky_order2():
    """Create alignment from small sky's distribution at order 2"""
    initial_histogram = hist.empty_histogram(2)
    filled_pixels = [4, 11, 14, 13, 5, 7, 8, 9, 11, 23, 4, 4, 17, 0, 1, 0]
    initial_histogram[176:] = filled_pixels[:]
    result = hist.generate_alignment(initial_histogram, 2, 250)

    expected = np.full(hp.order2npix(2), None)
    tuples = [
        (0, 11, 131),
        (0, 11, 131),
        (0, 11, 131),
        (0, 11, 131),
        (0, 11, 131),
        (0, 11, 131),
        (0, 11, 131),
        (0, 11, 131),
        (0, 11, 131),
        (0, 11, 131),
        (0, 11, 131),
        (0, 11, 131),
        (0, 11, 131),
        (0, 11, 131),
        (0, 11, 131),
        (0, 11, 131),
    ]
    expected[176:192] = tuples

    npt.assert_array_equal(result, expected)


def test_alignment_even_sky():
    """Create alignment from an even distribution at order 8"""
    initial_histogram = np.full(hp.order2npix(8), 10)
    result = hist.generate_alignment(initial_histogram, 8, 1_000)
    # everything maps to order 5, given the density
    for mapping in result:
        assert mapping[0] == 5


def test_destination_pixel_map_order1():
    """Create destination pixel map for small sky at order 1"""

    alignment = np.full(48, None)
    alignment[44:] = [(0, 11, 131), (0, 11, 131), (0, 11, 131), (0, 11, 131)]

    initial_histogram = hist.empty_histogram(1)
    filled_pixels = [51, 29, 51, 0]
    initial_histogram[44:] = filled_pixels[:]

    expected = {tuple([0, 11, 131]): [44, 45, 46]}

    result = hist.generate_destination_pixel_map(initial_histogram, alignment)

    npt.assert_array_equal(result, expected)
