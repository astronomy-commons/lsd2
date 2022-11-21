"""Tests of argument validation, in the absense of command line parsing"""

import os

from partitioner.arguments import PartitionArguments
import unittest
import argparse

TEST_DIR = os.path.dirname(__file__)


class TestArguments(unittest.TestCase):
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
            catalog_name="catalog", input_path=TEST_DIR, output_path=TEST_DIR
        )
        self.assertEqual(args.input_path, TEST_DIR)
        self.assertEqual(args.output_path, TEST_DIR)


if __name__ == "__main__":
    unittest.main()
