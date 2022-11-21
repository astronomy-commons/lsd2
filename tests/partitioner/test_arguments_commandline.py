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
        with self.assertRaises(ValueError):
            args.from_command_line(empty_args)

    def test_invalid_arguments(self):
        """Arguments are ill-formed."""
        bad_form_args = ["catalog", "path", "path"]
        args = PartitionArguments()
        with self.assertRaises(SystemExit):
            args.from_command_line(bad_form_args)

    def test_invalid_path(self):
        """Required arguments are provided, but paths aren't found."""
        bad_path_args = ["-c", "catalog", "-i", "path", "-o", "path"]
        args = PartitionArguments()
        with self.assertRaises(ValueError):
            args.from_command_line(bad_path_args)

    def test_good_paths(self):
        """Required arguments are provided, and paths are found."""
        good_args = [
            "--catalog_name",
            "catalog",
            "--input_path",
            TEST_DIR,
            "--output_path",
            TEST_DIR,
        ]
        args = PartitionArguments()
        args.from_command_line(good_args)
        self.assertEqual(args.input_path, TEST_DIR)
        self.assertEqual(args.output_path, TEST_DIR)

    def test_good_paths_short_names(self):
        """Required arguments are provided, using short names for arguments."""
        good_args = ["-c", "catalog", "-i", TEST_DIR, "-o", TEST_DIR]
        args = PartitionArguments()
        args.from_command_line(good_args)
        self.assertEqual(args.input_path, TEST_DIR)
        self.assertEqual(args.output_path, TEST_DIR)


if __name__ == "__main__":
    unittest.main()
