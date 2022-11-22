"""Tests of argument validation, in the absense of command line parsing"""

import unittest

import tests.constants as dc
from partitioner.arguments import PartitionArguments


class TestArguments(unittest.TestCase):
    """Test argument validation from parameters"""

    def test_none(self):
        """No arguments provided. Should error for required args."""
        args = PartitionArguments()
        with self.assertRaises(ValueError):
            args.from_params()

    def test_invalid_path(self):
        """Required arguments are provided, but paths aren't found."""
        args = PartitionArguments()
        with self.assertRaises(ValueError):
            args.from_params(
                catalog_name="catalog", input_path="path", output_path="path"
            )

    def test_good_paths(self):
        """Required arguments are provided, and paths are found."""
        args = PartitionArguments()
        args.from_params(
            catalog_name="catalog",
            input_path=dc.TEST_BLANK_DATA_DIR,
            input_format="csv",
            output_path=dc.TEST_TMP_DIR,
        )
        self.assertEqual(args.input_path, dc.TEST_BLANK_DATA_DIR)
        self.assertEqual(len(args.input_paths), 1)
        self.assertEqual(args.input_paths[0], dc.TEST_BLANK_CSV)
        self.assertEqual(args.output_path, dc.TEST_TMP_DIR)

    def test_multiple_files_in_path(self):
        """Required arguments are provided, and paths are found."""
        args = PartitionArguments()
        args.from_params(
            catalog_name="catalog",
            input_path=dc.TEST_SMALL_SKY_PARTS_DATA_DIR,
            input_format="csv",
            output_path=dc.TEST_TMP_DIR,
        )
        self.assertEqual(args.input_path, dc.TEST_SMALL_SKY_PARTS_DATA_DIR)
        self.assertEqual(len(args.input_paths), 5)

    def test_single_debug_file(self):
        """Required arguments are provided, and paths are found."""
        args = PartitionArguments()
        args.from_params(
            catalog_name="catalog",
            debug_input_files=dc.TEST_FORMATS_HEADERS_CSV,
            input_format="csv",
            output_path=dc.TEST_TMP_DIR,
        )
        self.assertEqual(len(args.input_paths), 1)
        self.assertEqual(args.input_paths[0], dc.TEST_FORMATS_HEADERS_CSV)


if __name__ == "__main__":
    unittest.main()
