"""Tests of command line argument validation"""

import os

from partitioner.arguments import PartitionArguments
import unittest
import argparse

TEST_DIR = os.path.dirname(__file__)


class TestArguments(unittest.TestCase):
    def test_none(self):
        """No arguments provided. Should error for required args."""
        empty_args = []
        args = PartitionArguments()
        with self.assertRaises(SystemExit):
            args.from_command_line(empty_args)

    def test_invalid_path(self):
        """Required arguments are provided, but paths aren't found."""
        empty_args = ["catalog", "path", "path"]
        args = PartitionArguments()
        with self.assertRaises(ValueError):
            args.from_command_line(empty_args)

    def test_good_paths(self):
        """Required arguments are provided, and paths are found."""
        empty_args = ["catalog", TEST_DIR, TEST_DIR]
        args = PartitionArguments()
        args.from_command_line(empty_args)
        self.assertEqual(args.input_path, TEST_DIR)
        self.assertEqual(args.output_path, TEST_DIR)


if __name__ == "__main__":
    unittest.main()
