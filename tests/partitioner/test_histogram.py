"""Tests of histogram calculations"""

import unittest

import numpy.testing as npt

import partitioner.histogram as hist
import tests.constants as dc
from partitioner.arguments import PartitionArguments


class TestHistograms(unittest.TestCase):
    """Test histogram calculations"""

    def test_small_sky_threshold_same_pixel(self):
        """Test loading the small sky catalog and partitioning each object into the same large bucket"""
        args = PartitionArguments()
        args.from_params(
            catalog_name="catalog",
            input_path=dc.TEST_SMALL_SKY_DATA_DIR,
            input_format="csv",
            output_path=dc.TEST_TMP_DIR,
            highest_healpix_order=0,
        )
        result = hist.generate_histogram(args)
        self.assertEqual(len(result), 12)

        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 131]
        npt.assert_array_equal(result, expected)
        self.assertTrue((result == expected).all())

    def test_small_sky_parts_same_pixel(self):
        """Test loading the small sky catalog and partitioning each object into the same large bucket"""
        args = PartitionArguments()
        args.from_params(
            catalog_name="catalog",
            input_path=dc.TEST_SMALL_SKY_PARTS_DATA_DIR,
            input_format="csv",
            output_path=dc.TEST_TMP_DIR,
            highest_healpix_order=0,
        )
        result = hist.generate_histogram(args)
        self.assertEqual(len(result), 12)

        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 131]
        npt.assert_array_equal(result, expected)
        self.assertTrue((result == expected).all())

    def test_column_names_error(self):
        """Test loading file with non-default column names"""
        args = PartitionArguments()
        args.from_params(
            catalog_name="catalog",
            debug_input_files=dc.TEST_FORMATS_HEADERS_CSV,
            input_format="csv",
            output_path=dc.TEST_TMP_DIR,
            highest_healpix_order=0,
        )
        with self.assertRaises(ValueError):
            hist.generate_histogram(args)

    def test_column_names(self):
        """Test loading file with non-default column names"""
        args = PartitionArguments()
        args.from_params(
            catalog_name="catalog",
            debug_input_files=dc.TEST_FORMATS_HEADERS_CSV,
            ra_column="ra_mean",
            dec_column="dec_mean",
            ra_error_column="ra_mean_error",
            dec_error_column="dec_mean_error",
            input_format="csv",
            output_path=dc.TEST_TMP_DIR,
            highest_healpix_order=0,
        )
        result = hist.generate_histogram(args)
        self.assertEqual(len(result), 12)

        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8]
        npt.assert_array_equal(result, expected)
        self.assertTrue((result == expected).all())
